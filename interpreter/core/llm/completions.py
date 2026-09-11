"""The wire call: litellm.completion with the fork's fixes and retries around it."""

import json
import os

from .errors import AccessDeniedError, FunctionCallingNotSupportedError, ModelNotFoundError


def _litellm():
    """Import litellm on first use: it costs over a second and most CLI paths never need it."""
    import litellm

    litellm.suppress_debug_info = True
    litellm.REPEATED_STREAMING_CHUNK_LIMIT = 99999999
    return litellm


def fixed_litellm_completions(**params):
    """
    Just uses a dummy API key, since we use litellm without an API key sometimes.
    Hopefully they will fix this!
    """
    litellm = _litellm()

    if "local" in params.get("model"):
        # Kinda hacky, but this helps sometimes
        params["stop"] = ["<|assistant|>", "<|end|>", "<|eot_id|>"]

    if params.get("model") == "i" and "conversation_id" in params:
        litellm.drop_params = False  # If we don't do this, litellm will drop this param!
    else:
        litellm.drop_params = True

    params["model"] = params["model"].replace(":latest", "")

    # Run completion
    # Keep provider retries explicit in respond.py so user choices ("n = stop")
    # are the single source of truth for retry behavior.
    params["num_retries"] = 0
    tried_dummy_key = False

    # DeepSeek models via OpenRouter have thinking mode enabled by default and require
    # reasoning_content on EVERY prior assistant message in the request — even turns
    # where the model chose not to think (empty string is accepted). Without this
    # the API returns 400 "reasoning_content must be passed back to the API".
    #
    # Primary fix: convert_to_openai_messages propagates reasoning_content to every
    # assistant message in a turn — text preamble, every tool-call message, and tool
    # output turns alike.  It survives tool/function responses in between so that a
    # multi-tool-call turn never leaves a tool-call message without the reasoning it
    # actually generated.  SiliconFlow (used as an OpenRouter backend) strictly
    # enforces this — passing "" on a tool_calls message triggers 400 when the model
    # actually returned reasoning.
    #
    # This is a secondary/belt-and-suspenders pass that catches any remaining assistant
    # messages that slipped through (e.g. synthetic messages injected by process_messages).
    # It makes sure every assistant message carries the current turn's reasoning (or ""
    # for turns where the model did not think), since DeepSeek requires the field to be
    # present.  We skip this only when reasoning was explicitly disabled by the caller.
    _model = params.get("model", "")
    _extra_body = params.get("extra_body", {})
    _reasoning_explicitly_disabled = (
        _extra_body.get("include_reasoning") is False or params.get("include_reasoning") is False
    )
    _uses_deepseek_reasoning_history = _model.startswith("deepseek/") or (
        _model.startswith("openrouter/") and "deepseek" in _model.lower()
    )
    if _uses_deepseek_reasoning_history and not _reasoning_explicitly_disabled:
        # DeepSeek's thinking mode docs require that, for requests carrying `tools`,
        # every assistant tool_calls message in the history carry `reasoning_content`
        # back to the API (api-docs.deepseek.com/guides/thinking_mode). The direct
        # DeepSeek API tolerates an empty string, but the OpenRouter BYOK relay used
        # here validates strictly: BOTH "" and a missing field on a tool_calls message
        # trigger a 400 — only a non-empty value is accepted. The model genuinely
        # produces no reasoning on some trivial tool-call turns (e.g. "ok" -> execute
        # code) and that reasoning is unrecoverable, so we must not emit "". We
        # synthesize a neutral placeholder instead so the request succeeds. A fabricated
        # thought is semantically wrong but harmless compared to a hard 400; the model
        # already sees the tool_calls and can reconstruct intent from them.
        #
        # This is a well-known cross-ecosystem failure mode, not a quirk of this repo:
        # the same 400 ("The `reasoning_content` in the thinking mode must be passed
        # back to the API") was reported and worked around in litellm
        # (github.com/BerriAI/litellm/issues/26395, #28045, #27439 — #27439 documents
        # that the litellm fix only covers the `deepseek/` prefix and never fires for
        # `openrouter/` routes), langchain (github.com/langchain-ai/langchain/issues/37174),
        # Qwen Code (github.com/QwenLM/qwen-code/issues/3579), Spring AI
        # (github.com/spring-projects/spring-ai/issues/6026), and OpenCode
        # (github.com/anomalyco/opencode/issues/24722). The ecosystem consensus is to
        # inject a placeholder when no reasoning was produced; Qwen Code's maintainers
        # explicitly chose "an empty string or space", but OpenRouter's BYOK relay for
        # the `~deepseek/...-latest` tilde alias rejects even "" and requires non-empty.
        # Semantically inert placeholder: the API only requires a non-empty value on
        # tool_calls messages, but the text is fed back to the model as its own prior
        # reasoning (DeepSeek's interleaved thinking mode). A phrase like the previous
        # "Executing the requested command." reads as a completed action and gets echoed
        # back as real reasoning — which taught the model that narrating an action without
        # executing it is acceptable. A lone period has no content the model can imitate.
        _no_reasoning_placeholder = "."
        # Pre-change placeholder that may still be stored in old conversations. Treat it as
        # "no reasoning" too so contaminated history heals itself on the next request.
        _legacy_no_reasoning_placeholder = "Executing the requested command."
        last_reasoning = None
        for msg in params.get("messages", []):
            if msg.get("role") == "assistant":
                if "reasoning_content" in msg:
                    rc = msg.get("reasoning_content")
                    if (rc or "").strip() in (
                        _no_reasoning_placeholder,
                        _legacy_no_reasoning_placeholder,
                    ):
                        # No real reasoning on this turn (either a legacy placeholder echoed
                        # into the stored conversation, or the inert marker from a previous
                        # request). Reset the propagation chain — a placeholder must never be
                        # forwarded onto later messages as if the model had thought it.
                        last_reasoning = None
                        # Only tool_calls messages need a non-empty value (DeepSeek 400s on
                        # "" there); plain assistant messages accept "".
                        if msg.get("tool_calls"):
                            msg["reasoning_content"] = _no_reasoning_placeholder
                        else:
                            msg["reasoning_content"] = ""
                    else:
                        last_reasoning = rc
                        # A tool-call message that already carries "" (e.g. attached by
                        # convert_to_openai_messages) is just as fatal as a missing field —
                        # DeepSeek 400s on empty reasoning for tool_calls.  Normalize it to a
                        # placeholder so the request never goes out with "".
                        if not (rc or "").strip() and msg.get("tool_calls"):
                            msg["reasoning_content"] = _no_reasoning_placeholder
                else:
                    # Propagate the turn's real reasoning when the model thought; only fall
                    # back to a placeholder for turns with no thinking.  Passing "" where the
                    # model actually reasoned is exactly what triggers the 400 on tool-call turns.
                    if msg.get("tool_calls"):
                        msg["reasoning_content"] = last_reasoning if last_reasoning else _no_reasoning_placeholder
                    else:
                        msg["reasoning_content"] = ""
            elif msg.get("role") == "user":
                last_reasoning = None

    # Debug: dump the exact outgoing request params (model, messages, tools,
    # extra_body, stream_options) to a JSONL file before litellm sends them.
    # Opt-in via OI_LOG_LITELLM_REQUESTS=1; the literal dict handed to
    # litellm.completion() is what becomes the wire request, so this captures
    # what the provider actually receives (modulo litellm's internal transforms).
    # Each line is one request: {"ts": ..., "model": ..., "messages": [...], ...}.
    if os.environ.get("OI_LOG_LITELLM_REQUESTS") == "1":
        try:
            import datetime as _dt

            dump_dir = os.path.expanduser("~/.config/open-interpreter/logs")
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, "litellm_requests.jsonl")
            with open(dump_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
                            "model": params.get("model"),
                            "messages": params.get("messages"),
                            "tools": params.get("tools"),
                            "extra_body": params.get("extra_body"),
                            "stream_options": params.get("stream_options"),
                            "include_reasoning": params.get("include_reasoning"),
                            "reasoning_effort": params.get("reasoning_effort"),
                            "temperature": params.get("temperature"),
                            "max_tokens": params.get("max_tokens"),
                        },
                        default=str,
                    )
                    + "\n"
                )
            print(f"\n[Dumped outgoing request to {dump_path}]", flush=True)
        except Exception:
            pass

    while True:
        try:
            yield from litellm.completion(**params)
            return  # If the completion is successful, exit the function
        except KeyboardInterrupt:
            # Re-raise so terminal_interface.py's outer handler can cancel the
            # current response and return to the prompt, rather than exiting.
            raise
        except Exception as e:
            # Diagnostic: DeepSeek's "reasoning_content must be passed back" 400 is hard to
            # reproduce without the exact request, so when it fires, dump a compact view of
            # the outgoing messages (reasoning status per message) to a file for debugging.
            if "reasoning_content" in str(e):
                try:
                    import datetime as _dt
                    import os as _os

                    dump_dir = _os.path.expanduser("~/.config/open-interpreter/logs")
                    _os.makedirs(dump_dir, exist_ok=True)
                    dump_path = _os.path.join(
                        dump_dir, f"reasoning_400_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    with open(dump_path, "w") as f:
                        json.dump(
                            [
                                {
                                    "role": m.get("role"),
                                    "has_function_call": bool(m.get("function_call")),
                                    "has_tool_calls": bool(m.get("tool_calls")),
                                    "reasoning_len": len(m["reasoning_content"])
                                    if isinstance(m.get("reasoning_content"), str)
                                    else ("missing" if "reasoning_content" not in m else m.get("reasoning_content")),
                                    "content": (m.get("content") or "")[:200],
                                }
                                for m in params.get("messages", [])
                            ],
                            f,
                            indent=2,
                        )
                    print(f"\n[Dump of failing request written to {dump_path}]", flush=True)
                except Exception:
                    pass
            # Check if this is a function-calling-not-supported error.
            # Only check for this if we're actually trying to use function calling.
            if "tools" in params:
                error_message = str(e).lower()
                if any(
                    phrase in error_message
                    for phrase in [
                        "no endpoints found that support tool use",
                        "tool use",
                        "function calling",
                        "tool calling",
                    ]
                ):
                    raise FunctionCallingNotSupportedError(str(e)) from e

            # LiteLLM sometimes requires an api_key parameter even when the backend
            # provider ignores it. Retry exactly once with a dummy key, then surface
            # the error so respond.py's retry prompt controls subsequent retries.
            if (
                isinstance(e, litellm.exceptions.AuthenticationError)
                and "api_key" not in params
                and not tried_dummy_key
            ):
                print(
                    "LiteLLM requires an API key. Trying again with a dummy API key. In the future, if this fixes it, please set a dummy API key to prevent this message. (e.g `interpreter --api_key x` or `self.api_key = 'x'`)"
                )
                params["api_key"] = "x"
                tried_dummy_key = True
                continue

            # Bubble up all provider errors to respond.py for user-facing handling.
            raise
