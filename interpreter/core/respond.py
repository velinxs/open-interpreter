import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import html2text
import openai
from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel

from ..terminal_interface.utils.display_markdown_message import display_markdown_message
from .run_code import run_pending_code
from .toolbox.web.web import ApiKeyError, WebToolboxError
from .tools.file_edit import dry_run_edit, run_edit
from .utils.assemble_system_message import assemble_system_message
from .utils.prompt_choice import (
    NoInteractiveInput,
    prompt_choice,
)


def _litellm_optional_api_exceptions():
    import litellm

    return tuple(
        getattr(litellm.exceptions, name)
        for name in ("ServiceUnavailableError", "InternalServerError")
        if hasattr(litellm.exceptions, name)
    )


def _html_error_to_renderable(error_str):
    """
    If error_str contains an HTML error body (e.g. a provider 502 page),
    convert it to a Rich Markdown renderable and return it. Otherwise return
    None.

    The exception string has a plain-text prefix before the HTML (e.g.
    "litellm.APIError: OpenrouterException - <!DOCTYPE html>..."), so we slice
    from the first HTML tag before handing off to html2text, which converts it
    to Markdown for Rich to render.
    """
    # Quick bail-out: not HTML at all
    lower = error_str.lower()
    doctype_idx = lower.find("<!doctype")
    html_idx = lower.find("<html")
    if doctype_idx == -1 and html_idx == -1:
        return None

    # Slice from whichever HTML marker appears first
    indices = [i for i in (doctype_idx, html_idx) if i >= 0]
    html_part = error_str[min(indices) :]

    try:
        h2t = html2text.HTML2Text()
        h2t.body_width = 0  # Let Rich handle reflowing
        md = h2t.handle(html_part).strip()
    except Exception:
        return None

    if not md:
        return None

    # Leading newline so the first line of content is not clipped by the panel title bar
    return Markdown("\n" + md)


def _is_temporary_provider_error(error):
    error_message = str(error).lower()
    temporary_markers = (
        "temporarily",
        "temporary",
        "retry shortly",
        "try again later",
        "rate-limit",
        "rate limited",
        "429",
        "503",
        "504",
        "timeout",
        "timed out",
        "overloaded",
        "unavailable",
    )
    return any(marker in error_message for marker in temporary_markers)


def _temporary_error_signature(error):
    """
    Normalize temporary provider errors so repeated retries with the same
    upstream failure don't spam duplicate full error panels.
    """
    return re.sub(r"\s+", " ", str(error)).strip()


def _render_temporary_retry_status(retry_count):
    """
    Update a single in-place status line for temporary upstream retries.
    """
    dots = "." * ((retry_count - 1) % 3 + 1)
    sys.stdout.write(f"\r  ▌ Temporary upstream provider error; retrying{dots}")
    sys.stdout.flush()


@dataclass
class LoopState:
    """State that survives one pass of the loop, shared with the code runner."""

    last_unsupported_code: str = ""


def respond(interpreter):
    """
    Yields chunks.
    Responds until it decides not to run any more code or say anything else.
    """
    import litellm

    state = LoopState()
    insert_loop_message = False
    loop_message = None  # set from interpreter.loop_message when loop mode re-prompts
    always_retry_provider_errors = False
    temporary_provider_error_retries = 0
    last_temporary_provider_error_signature = None

    while True:
        ## RENDER SYSTEM MESSAGE ##

        ## Rendering ↓
        rendered_system_message = assemble_system_message(interpreter)
        ## Rendering ↑

        # Store the actual rendered system message for %info command (before converting to dict)
        interpreter._last_rendered_system_message = rendered_system_message

        rendered_system_message = {
            "role": "system",
            "type": "message",
            "content": rendered_system_message,
        }

        # Create the version of messages that we'll send to the LLM
        messages_for_llm = [m for m in interpreter.messages.copy() if m.get("role") != "system"]
        messages_for_llm = [rendered_system_message] + messages_for_llm

        if insert_loop_message:
            messages_for_llm.append(
                {
                    "role": "user",
                    "type": "message",
                    "content": loop_message,
                }
            )
            # Yield two newlines to separate the LLMs reply from previous messages.
            yield {"role": "assistant", "type": "message", "content": "\n\n"}
            insert_loop_message = False

        ### RUN THE LLM ###

        assert len(interpreter.messages) > 0, (
            "User message was not passed in. You need to pass in at least one message."
        )

        if interpreter.messages[-1]["type"] not in ("code", "edit"):  # If it is, we run below
            try:
                for chunk in interpreter.llm.run(messages_for_llm):
                    yield {"role": "assistant", **chunk}

            except litellm.exceptions.BudgetExceededError:
                interpreter.display_message(
                    f"""> Max budget exceeded

                    **Session spend:** ${litellm._current_cost}
                    **Max budget:** ${interpreter.max_budget}

                    Press CTRL-C then run `interpreter --max_budget [higher USD amount]` to proceed.
                """
                )
                break

            except Exception as e:
                # Continue with existing error handling
                error_message = str(e).lower()

                # Check for API errors - display them in a panel without tracebacks
                # Also check for our formatted errors (containing |||)
                error_str = str(e)
                # Normalize provider/API errors (LiteLLM + OpenAI) so we can render
                # them consistently using Rich, regardless of which exception class
                # LiteLLM chose for the underlying provider (e.g. OpenRouter 5xx).
                if isinstance(
                    e,
                    (
                        # LiteLLM exception variants
                        getattr(litellm, "APIError", Exception),
                        getattr(litellm, "OpenAIError", Exception),
                        litellm.exceptions.APIError,
                        litellm.exceptions.OpenAIError,
                        litellm.exceptions.NotFoundError,
                        litellm.exceptions.BadRequestError,
                        litellm.exceptions.RateLimitError,
                        litellm.exceptions.AuthenticationError,
                        getattr(litellm.exceptions, "APIConnectionError", Exception),
                        *_litellm_optional_api_exceptions(),
                        # OpenAI Python client variants (defensive, in case they leak through)
                        getattr(openai, "APIError", Exception),
                        getattr(openai, "OpenAIError", Exception),
                    ),
                ):
                    is_temporary_error = _is_temporary_provider_error(e)
                    panel_border_style = "yellow" if is_temporary_error else "red"
                    panel_title = "Warning" if is_temporary_error else "Error"
                    temporary_error_signature = _temporary_error_signature(e) if is_temporary_error else None
                    if is_temporary_error and temporary_error_signature == last_temporary_provider_error_signature:
                        temporary_provider_error_retries += 1
                        _render_temporary_retry_status(temporary_provider_error_retries)
                        time.sleep(2)
                        continue
                    # Format with Rich Panel with red border for errors
                    # Check if this is an error with JSON structure that can be parsed
                    if "{" in error_str and "}" in error_str:
                        # Parse the JSON structure and format it nicely
                        try:
                            # Extract JSON from error string
                            json_start = error_str.find("{")
                            json_end = error_str.rfind("}") + 1
                            error_data = json.loads(error_str[json_start:json_end])

                            # Build text content with nested structure
                            lines = ["[bold]OpenRouterException:[/bold]"]
                            lines.append("")

                            def format_value(val, indent=0):
                                """Format a value with proper indentation"""
                                prefix = "  " * indent
                                if isinstance(val, dict):
                                    result = []
                                    for key, value in val.items():
                                        if isinstance(value, dict):
                                            result.append(f"{prefix}• {key}:")
                                            result.extend(format_value(value, indent + 1))
                                        elif isinstance(value, list):
                                            result.append(f"{prefix}• {key}:")
                                            for item in value:
                                                result.extend(format_value(item, indent + 1))
                                        else:
                                            # For certain fields, show the value directly if it's already a helpful message
                                            if key == "raw" or (
                                                key == "message" and isinstance(value, str) and len(value) < 100
                                            ):
                                                result.append(f"{prefix}• {key}: {value}")
                                            else:
                                                result.append(f"{prefix}• {key}: {value}")
                                    return result
                                else:
                                    return [f"{prefix}• {val}"]

                            lines.extend(format_value(error_data))

                            formatted_error = "\n".join(lines)
                            panel = Panel(
                                formatted_error, border_style=panel_border_style, title=panel_title, title_align="left"
                            )
                            # Yield a special chunk to stop Live display before printing error panel
                            # This prevents Live display from overwriting the error panel
                            yield {"type": "stop_live_display"}
                            print("")  # Newline so panel top border is not cut off
                            rich_print(panel)
                            print("")  # Add space after error
                        except Exception:
                            # Fallback if JSON parsing fails (e.g. body is HTML)
                            display = _html_error_to_renderable(error_str) or error_str
                            yield {"type": "stop_live_display"}
                            print("")  # Newline so panel top border is not cut off
                            panel = Panel(
                                display, border_style=panel_border_style, title=panel_title, title_align="left"
                            )
                            rich_print(panel)
                            print("")
                    else:
                        # Format all other API errors in a Panel. If the body is HTML (e.g. provider 502 page),
                        # convert with html2text and render as Rich Markdown.
                        display = _html_error_to_renderable(error_str) or error_str
                        yield {"type": "stop_live_display"}
                        print("")  # Newline so panel top border is not cut off by previous output
                        panel = Panel(display, border_style=panel_border_style, title=panel_title, title_align="left")
                        rich_print(panel)
                        print("")  # Add space after error

                    # Temporary provider errors (including upstream rate limits)
                    # are retried automatically to avoid blocking on user input.
                    if is_temporary_error:
                        last_temporary_provider_error_signature = temporary_error_signature
                        temporary_provider_error_retries += 1
                        _render_temporary_retry_status(temporary_provider_error_retries)
                        time.sleep(2)
                        continue

                    # For non-temporary provider errors, offer manual retry.
                    if always_retry_provider_errors:
                        print("")
                        interpreter.display_message("> Retrying...")
                        time.sleep(2)
                        continue

                    try:
                        retry_choice = interpreter.prompter(
                            "  Retry? (y = retry once, a = keep retrying, n = stop)\n\n  ",
                            ("y", "a", "n"),
                        )
                    except NoInteractiveInput:
                        retry_choice = None  # nobody to ask: fall through and raise below

                    if retry_choice == "a":
                        always_retry_provider_errors = True
                        interpreter.display_message("> Retrying...")
                        time.sleep(2)
                        continue
                    if retry_choice == "y":
                        interpreter.display_message("> Retrying...")
                        time.sleep(2)
                        continue

                        interpreter._stopped_retrying = True
                        return

                    interpreter._stopped_retrying = True
                    return

                if interpreter.offline == False and ("auth" in error_message or "api key" in error_message):
                    # Provide extra information on how to change API keys, if
                    # we encounter that error (Many people writing GitHub
                    # issues were struggling with this)
                    output = traceback.format_exc()

                    # Generic hint: a hard-coded llm.api_key can conflict with CLI-provided model/provider
                    api_key_in_config = bool(getattr(interpreter.llm, "api_key", None))
                    provider_hint = ""
                    if api_key_in_config:
                        provider_hint = (
                            "\n\nHint: You have `llm.api_key` set in your profile/config. "
                            "If you pass `--model` (or `--api_key`) on the command line and they don't match the same provider, you'll get 401 Unauthorized. "
                            "Either remove `llm.api_key` from your profile/default.yaml so your command-line selection takes effect, "
                            "or pass both `--model` and `--api_key` together on the command line to ensure they match."
                        )

                    raise Exception(
                        f"{output}\n\nThere might be an issue with your API key(s).{provider_hint}\n\n"
                        "To reset your API key (we'll use OPENAI_API_KEY for this example, but you may need to reset your ANTHROPIC_API_KEY, HUGGINGFACE_API_KEY, etc):\n        Mac/Linux: 'export OPENAI_API_KEY=your-key-here'. Update your ~/.zshrc on MacOS or ~/.bashrc on Linux with the new key if it has already been persisted there.,\n        Windows: 'setx OPENAI_API_KEY your-key-here' then restart terminal.\n\n"
                    )
                elif isinstance(e, litellm.exceptions.RateLimitError) and (
                    "exceeded" in str(e).lower() or "insufficient_quota" in str(e).lower()
                ):
                    display_markdown_message(
                        """ > You ran out of current quota for OpenAI's API, please check your plan and billing details. You can either wait for the quota to reset or upgrade your plan.

                        To check your current usage and billing details, visit the [OpenAI billing page](https://platform.openai.com/settings/organization/billing/overview).

                        You can also use `interpreter --max_budget [higher USD amount]` to set a budget for your sessions.
                        """
                    )

                elif interpreter.offline == False and "not have access" in str(e).lower():
                    # Check for invalid model in error message and then fallback.
                    if "invalid model" in error_message or "model does not exist" in error_message:
                        provider_message = f"\n\nThe model '{interpreter.llm.model}' does not exist or is invalid. Please check the model name and try again.\n\nWould you like to try Open Interpreter's hosted `i` model instead? (y/n)\n\n  "
                    elif "groq" in error_message:
                        provider_message = f"\n\nYou do not have access to {interpreter.llm.model}. Please check with Groq for more details.\n\nWould you like to try Open Interpreter's hosted `i` model instead? (y/n)\n\n  "
                    else:
                        provider_message = f"\n\nYou do not have access to {interpreter.llm.model}. If you are using an OpenAI model, you may need to add a payment method and purchase credits for the OpenAI API billing page (this is different from ChatGPT Plus).\n\nhttps://platform.openai.com/account/billing/overview\n\nWould you like to try Open Interpreter's hosted `i` model instead? (y/n)\n\n"

                    print(provider_message)

                    try:
                        response = interpreter.prompter("  ", ("y", "n"))
                    except NoInteractiveInput:
                        # Reached in server mode too, where there is no TTY.
                        # Quietly switching to a hosted model that trains on the
                        # conversation is not an assumption to make on the user's
                        # behalf, so fall through to the re-raise below and let
                        # the real provider error surface.
                        response = "n"

                    if response == "y":
                        interpreter.llm.model = "i"
                        interpreter.display_message("> Model set to `i`")
                        interpreter.display_message(
                            "***Note:*** *Conversations with this model will be used to train our open-source model.*\n"
                        )

                    else:
                        raise
                elif interpreter.offline and not interpreter.os:
                    raise
                else:
                    raise

            else:
                if temporary_provider_error_retries > 0:
                    print("")
                temporary_provider_error_retries = 0
                last_temporary_provider_error_signature = None

        # Inject image from view_image tool call (tool appends result first, then we add user image)
        pending_path = getattr(interpreter, "_pending_view_image_path", None)
        if pending_path is not None:
            delattr(interpreter, "_pending_view_image_path")
            pending_shrink = getattr(interpreter, "_pending_view_image_shrink", None)
            if pending_shrink is not None:
                delattr(interpreter, "_pending_view_image_shrink")
            img_msg = {
                "role": "user",
                "type": "image",
                "format": "path",
                "content": pending_path,
            }
            if pending_shrink is not None:
                img_msg["shrink"] = pending_shrink
            interpreter.messages.append(img_msg)

        ### RUN FILE EDIT (if it's there) ###

        if interpreter.messages[-1]["type"] == "edit":
            edit_msg = interpreter.messages[-1]
            language = edit_msg["format"].lower().strip()
            code = edit_msg["content"]
            target = edit_msg.get("target", "")

            if interpreter.verbose:
                print("Running edit:", edit_msg)

            try:
                dry_run_output = None
                dry_run_ok = True
                if not interpreter.auto_run:
                    try:
                        preview = dry_run_edit(language, code, target)
                        if preview is not None:
                            dry_run_output = preview["output"]
                            dry_run_ok = preview["ok"]
                    except Exception as e:
                        # Validation errors (missing file, bad path, etc.)
                        dry_run_output = str(e)
                        dry_run_ok = False

                # Yield confirmation so the terminal can prompt y/n (respects auto_run).
                # format: "edit" distinguishes this from a code execution confirmation.
                try:
                    confirmation_content = {
                        "format": language,
                        "content": code,
                        "target": target,
                    }
                    if dry_run_output is not None:
                        confirmation_content["dry_run_output"] = dry_run_output
                        confirmation_content["dry_run_ok"] = dry_run_ok
                    yield {
                        "role": "computer",
                        "type": "confirmation",
                        "format": "edit",
                        "content": confirmation_content,
                    }
                except GeneratorExit:
                    break

                # Re-read in case the user edited target/content (unlikely for edit, but consistent)
                edit_msg = [m for m in interpreter.messages if m["type"] == "edit"][-1]
                language = edit_msg["format"].lower().strip()
                code = edit_msg["content"]
                target = edit_msg.get("target", target)

                output = run_edit(language, code, target)

            except KeyboardInterrupt:
                break
            except Exception as e:
                output = traceback.format_exc() if interpreter.debug else str(e)

            # Yield as role:computer type:console so the result is:
            # (a) shown to the user in the terminal output panel, and
            # (b) converted to role:function name:edit by convert_to_openai_messages
            #     (which tracks last_tool_name="edit") so the AI sees it exactly
            #     once as the tool response — same pattern as execute.
            # There is no separate role:tool message; process_messages handles pairing.
            yield {
                "role": "computer",
                "type": "console",
                "format": "output",
                "content": output,
            }
            # Signal end-of-execution so core.py closes the output block.
            yield {
                "role": "computer",
                "type": "console",
                "format": "active_line",
                "content": None,
            }
            continue

        ### RUN CODE (if it's there) ###

        if interpreter.messages[-1]["type"] == "code":
            outcome = yield from run_pending_code(interpreter, state)
            if outcome == "break":
                break
            continue

        else:
            ## LOOP MESSAGE
            # This makes it utter specific phrases if it doesn't want to be told to "Proceed."

            # If an image was just shown (screenshot from display.view or approved view_image tool),
            # continue so the LLM gets another turn to see and comment on it.
            if (
                interpreter.messages
                and interpreter.messages[-1].get("type") == "image"
                and interpreter.messages[-1].get("role") in ("computer", "user")
            ):
                continue

            # If the last message is a tool response (unsupported function call error, etc.)
            # continue the loop so the LLM gets another turn with the result in context.
            if (
                interpreter.messages
                and interpreter.messages[-1].get("role") == "tool"
                and interpreter.messages[-1].get("type") == "message"
            ):
                continue

            loop_message = interpreter.loop_message
            if interpreter.os:
                loop_message = loop_message.replace(
                    "If the entire task I asked for is done,",
                    "If the entire task I asked for is done, take a screenshot to verify it's complete, or if you've already taken a screenshot and verified it's complete,",
                )
            loop_breakers = interpreter.loop_breakers

            if (
                interpreter.loop
                and interpreter.messages
                and interpreter.messages[-1].get("role", "") == "assistant"
                and not any(task_status in interpreter.messages[-1].get("content", "") for task_status in loop_breakers)
            ):
                # Remove past loop_message messages
                interpreter.messages = [
                    message for message in interpreter.messages if message.get("content", "") != loop_message
                ]
                # Combine adjacent assistant messages, so hopefully it learns to just keep going!
                combined_messages = []
                for message in interpreter.messages:
                    if (
                        combined_messages
                        and message["role"] == "assistant"
                        and combined_messages[-1]["role"] == "assistant"
                        and message["type"] == "message"
                        and combined_messages[-1]["type"] == "message"
                    ):
                        combined_messages[-1]["content"] += "\n" + message["content"]
                    else:
                        combined_messages.append(message)
                interpreter.messages = combined_messages

                # Send model the loop_message:
                insert_loop_message = True

                continue

            # Doesn't want to run code. We're done!
            break

    return
