"""Running a model that calls tools, and turning the call back into LMC chunks.

The stream is accumulated rather than acted on as it arrives: a provider may
send a tool call's arguments in pieces, and partial JSON cannot be trusted to
say which language or which file is meant. Reasoning is streamed through as it
comes so the user sees the model think, then the completed call is handed to
tool_dispatch, which decides what it means.
"""

import json
import os
import re

from ..terminal.base_language import format_execute_language_description
from ..tools.file_edit import EDIT_LANGUAGES
from .tool_dispatch import dispatch_function_call
from .tool_messages import process_messages
from .tool_schema import (
    VIEW_IMAGE_ALLOWED_EXTENSIONS,
    build_request_tools,
    edit_tool_schema,
    tool_schema,
    view_image_tool_schema,
)
from .utils.merge_deltas import merge_deltas, normalize_delta_to_dict
from .utils.parse_partial_json import parse_partial_json
from .utils.stream_usage import record_stream_chunk_usage


def run_tool_calling_llm(llm, request_params):
    ## Setup

    # Check verbose flag for debug output
    verbose = llm.interpreter.verbose or llm.interpreter.debug

    if verbose:
        print(f"[DEBUG] run_tool_calling_llm called with model: {llm.model}", flush=True)
        if "reasoning" in request_params:
            print(f"[DEBUG] reasoning parameter: {request_params['reasoning']}", flush=True)

    request_params["tools"] = build_request_tools(llm.interpreter, messages=request_params["messages"])

    # Append tool-calling-specific instructions to the system message (analogous to
    # how run_text_llm appends execution_instructions in markdown/no-functions mode).
    if llm.tool_calling_instructions:
        request_params["messages"][0]["content"] += "\n" + llm.tool_calling_instructions

    llm.interpreter._last_rendered_system_message = request_params["messages"][0]["content"]

    request_params["messages"] = process_messages(request_params["messages"], model=llm.model)

    ## Convert output to LMC format

    accumulated_deltas = {}
    language = None
    code = ""
    function_call_detected = False
    accumulated_review = ""
    review_category = None
    buffer = ""
    content_yielded_during_streaming = False  # Track if content was already yielded
    has_reasoning_content = False  # Track if we have reasoning_content (to delay content output)
    reasoning_streamed = False  # True if we yielded reasoning during the stream (so post-stream only yields remainder)
    reasoning_replace_yielded = False  # True after we yield the replace chunk (must happen before first content chunk)

    for chunk in llm.completions(**request_params):
        record_stream_chunk_usage(llm, chunk)

        if "choices" not in chunk or len(chunk["choices"]) == 0:
            # This happens sometimes
            continue

        raw_delta = chunk["choices"][0]["delta"]

        # Normalize delta to dict immediately - LiteLLM may return Pydantic objects
        # This ensures all code paths work with plain dicts consistently
        delta = normalize_delta_to_dict(raw_delta)

        # Mark if we see tool_calls (but don't try to parse incomplete streaming data)
        if "tool_calls" in delta and delta["tool_calls"]:
            function_call_detected = True

        # Accumulate deltas
        # Note: merge_deltas now handles lists (like tool_calls) properly
        # We accumulate everything during streaming, but only parse after stream completes
        accumulated_deltas = merge_deltas(accumulated_deltas, delta)

        # Track if we have reasoning_content (even if incomplete) - this will delay content output
        if "reasoning_content" in accumulated_deltas and accumulated_deltas.get("reasoning_content"):
            has_reasoning_content = True

        # Stream reasoning_content token-by-token as it arrives. After the stream
        # completes a replace chunk is yielded so the rich display and stored message
        # get clean blockquote formatting. Plain-text mode skips the replace chunk and
        # uses the streamed tokens directly.
        if "reasoning_content" in delta and delta["reasoning_content"]:
            new_chunk = delta["reasoning_content"]
            if isinstance(new_chunk, str):
                yield {"role": "assistant", "type": "message", "format": "reasoning", "content": new_chunk}
                reasoning_streamed = True

        if "content" in delta and delta["content"]:
            if function_call_detected:
                # More content after a code block? This is a code review by a judge layer.

                # print("Code safety review:", delta["content"])

                if review_category == None:
                    accumulated_review += delta["content"]

                    if "<unsafe>" in accumulated_review:
                        review_category = "unsafe"
                    if "<warning>" in accumulated_review:
                        review_category = "warning"
                    if "<safe>" in accumulated_review:
                        review_category = "safe"

                # If we have review tags, process as review
                if review_category != None:
                    for tag in [
                        "<safe>",
                        "</safe>",
                        "<warning>",
                        "</warning>",
                        "<unsafe>",
                        "</unsafe>",
                    ]:
                        delta["content"] = delta["content"].replace(tag, "")

                    if re.search("</.*>$", accumulated_review):
                        buffer += delta["content"]
                        continue
                    elif buffer:
                        yield {
                            "type": "review",
                            "format": review_category,
                            "content": buffer + delta["content"],
                        }
                        buffer = ""
                    else:
                        yield {
                            "type": "review",
                            "format": review_category,
                            "content": delta["content"],
                        }
                        buffer = ""
                else:
                    # function_call_detected is True but no review tags found
                    # This might be regular content, not a review - yield it as message
                    # But only if we don't have actual tool_calls (might be false positive)
                    if not accumulated_deltas.get("tool_calls") and not accumulated_deltas.get("function_call"):
                        # Yield replace chunk before first content so reasoning block is closed with blockquotes before the response.
                        if has_reasoning_content and reasoning_streamed and not reasoning_replace_yielded:
                            full_raw = accumulated_deltas.get("reasoning_content") or ""
                            if isinstance(full_raw, str) and full_raw.strip():
                                yield {
                                    "role": "assistant",
                                    "type": "message",
                                    "format": "reasoning",
                                    "content": full_raw.rstrip() + "\n\n",
                                    "replace": True,
                                }
                            reasoning_replace_yielded = True
                        # No actual tool calls, so this is just regular content. Stream it;
                        # reasoning (if any) was already streamed first by the provider.
                        yield {"role": "assistant", "type": "message", "content": delta["content"]}
                        content_yielded_during_streaming = True

            else:
                # Yield replace chunk before first content so reasoning block is closed with blockquotes before the response.
                if has_reasoning_content and reasoning_streamed and not reasoning_replace_yielded:
                    full_raw = accumulated_deltas.get("reasoning_content") or ""
                    if isinstance(full_raw, str) and full_raw.strip():
                        yield {
                            "role": "assistant",
                            "type": "message",
                            "format": "reasoning",
                            "content": full_raw.rstrip() + "\n\n",
                            "replace": True,
                        }
                    reasoning_replace_yielded = True
                # Stream content as it arrives; reasoning (if any) already streamed first.
                yield {"role": "assistant", "type": "message", "content": delta["content"]}
                content_yielded_during_streaming = True

        if (
            accumulated_deltas.get("function_call")
            and "name" in accumulated_deltas["function_call"]
            and (
                accumulated_deltas["function_call"]["name"] == "python"
                or accumulated_deltas["function_call"]["name"] == "functions"
            )
        ):
            if language is None:
                language = "python"

            # Pull the code string straight out of the "arguments" string
            arguments_str = accumulated_deltas["function_call"]["arguments"]
            # Ensure arguments is a string before slicing
            if isinstance(arguments_str, str):
                code_delta = arguments_str[len(code) :]
                # Update the code
                code = arguments_str
                # Yield the delta
                if code_delta:
                    yield {
                        "role": "assistant",
                        "type": "code",
                        "format": language,
                        "content": code_delta,
                    }

        if (
            accumulated_deltas.get("function_call")
            and "arguments" in accumulated_deltas["function_call"]
            and accumulated_deltas["function_call"]["arguments"]
        ):
            if "arguments" in accumulated_deltas["function_call"]:
                arguments = accumulated_deltas["function_call"]["arguments"]
                arguments = parse_partial_json(arguments)

                # Ensure arguments is a dictionary, not a string or None
                if not isinstance(arguments, dict):
                    arguments = None

                if arguments:
                    if (
                        language is None
                        and "language" in arguments
                        and "code"
                        in arguments  # <- This ensures we're *finished* typing language, as opposed to partially done
                        and arguments["language"]
                    ):
                        language = arguments["language"]

                    if language is not None and "code" in arguments:
                        # Ensure code is a string (some models may return other types)
                        code_value = arguments["code"]
                        if not isinstance(code_value, str):
                            # If code is not a string, skip this chunk
                            continue
                        # Calculate the delta (new characters only)
                        code_delta = code_value[len(code) :]
                        # Update the code
                        code = code_value
                        # Yield the delta
                        if code_delta:
                            yield {
                                "role": "assistant",
                                "type": "code",
                                "format": language,
                                "content": code_delta,
                            }
                else:
                    if llm.interpreter.verbose:
                        print("Arguments not a dict.")

    # After stream completes, convert tool_calls to function_call format if needed
    # Don't try to parse incomplete tool_calls during streaming

    # Debug: Always check what we have after stream
    has_tool_calls = "tool_calls" in accumulated_deltas and accumulated_deltas["tool_calls"]
    has_function_call = bool(accumulated_deltas.get("function_call"))
    has_content = "content" in accumulated_deltas and accumulated_deltas.get("content")

    # NOTE: llm.interpreter.verbose sometimes returns False even when --verbose flag is passed.
    # As a workaround, we check the debug attribute which is typically set alongside verbose.
    # If debug is True, we also enable verbose output for consistency.
    # This appears to be related to interpreter instance handling during profile loading.
    verbose = llm.interpreter.verbose or llm.interpreter.debug

    # Debug info only in verbose mode
    if verbose:
        print(
            f"[DEBUG] After stream - has_tool_calls: {has_tool_calls}, has_function_call: {has_function_call}, has_content: {bool(has_content)}",
            flush=True,
        )
        print(f"[DEBUG] accumulated_deltas keys: {list(accumulated_deltas.keys())}", flush=True)
        # NOTE: Provider detection removed - OpenRouter routes to different providers (DeepInfra, Together)
        # but this information is only available in OpenRouter's API response metadata, not in LiteLLM chunks.
        # DeepInfra returns reasoning_content as separate field, Together mixes it into content.
        if has_tool_calls:
            print(
                f"[DEBUG] tool_calls type: {type(accumulated_deltas['tool_calls'])}, value: {json.dumps(accumulated_deltas['tool_calls'], default=str)[:1000]}",
                flush=True,
            )
        if has_function_call:
            print(
                f"[DEBUG] function_call: {json.dumps(accumulated_deltas['function_call'], default=str)[:500]}",
                flush=True,
            )
        if has_content:
            content_preview = str(accumulated_deltas.get("content", ""))[:200]
            print(f"[DEBUG] content preview: {repr(content_preview)}", flush=True)
        if "reasoning_content" in accumulated_deltas:
            reasoning_preview = str(accumulated_deltas.get("reasoning_content", ""))[:200]
            print(f"[DEBUG] reasoning_content preview: {repr(reasoning_preview)}", flush=True)

    # POST-STREAM PROCESSING: Yield in order: reasoning → content → code
    # This ensures the model's thought process is shown before actions

    # 1. REASONING: Yield reasoning_content if present (in block quotes)
    if has_reasoning_content and "reasoning_content" in accumulated_deltas and accumulated_deltas["reasoning_content"]:
        if reasoning_streamed:
            # Replace was already yielded before first content chunk; only yield if we never got any content (no content delta in stream).
            if not reasoning_replace_yielded:
                full_raw = accumulated_deltas.get("reasoning_content") or ""
                if isinstance(full_raw, str) and full_raw.strip():
                    yield {
                        "role": "assistant",
                        "type": "message",
                        "format": "reasoning",
                        "content": full_raw.rstrip() + "\n\n",
                        "replace": True,
                    }
        else:
            # Provider sent reasoning only at end (e.g. no per-chunk reasoning_content); yield full block
            reasoning_content = accumulated_deltas["reasoning_content"]
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                if verbose:
                    print(
                        f"[DEBUG] reasoning_content length: {len(reasoning_content)}, preview: {repr(reasoning_content[:200])}",
                        flush=True,
                    )
                    if "content" in accumulated_deltas:
                        print(
                            f"[DEBUG] content length: {len(accumulated_deltas['content'])}, preview: {repr(accumulated_deltas['content'][:200])}",
                            flush=True,
                        )

                yield {
                    "role": "assistant",
                    "type": "message",
                    "format": "reasoning",
                    "content": reasoning_content.rstrip() + "\n\n",
                }

    # 2. CONTENT: Yield accumulated_review or regular content
    if accumulated_review and review_category == None:
        if accumulated_review.strip():
            yield {"role": "assistant", "type": "message", "content": accumulated_review}
    elif "content" in accumulated_deltas and accumulated_deltas["content"]:
        content = accumulated_deltas["content"]
        if not accumulated_deltas.get("function_call") and not accumulated_deltas.get("tool_calls"):
            if content.strip() and not content_yielded_during_streaming:
                yield {"role": "assistant", "type": "message", "content": content}

    # 3. Finally, process and yield code blocks (function_call/tool_calls)
    tool_call_id_for_error = None  # Store tool_call_id in case we need to yield error as tool response
    if "tool_calls" in accumulated_deltas and accumulated_deltas["tool_calls"]:
        if not accumulated_deltas.get("function_call"):
            # Try to convert tool_calls to function_call format now that stream is complete
            tool_calls = accumulated_deltas["tool_calls"]

            # Debug: log what we received (only in verbose mode)
            if llm.interpreter.verbose:
                print(
                    f"[DEBUG] Converting tool_calls after stream. tool_calls type: {type(tool_calls)}, value: {json.dumps(tool_calls, default=str)[:500]}",
                    flush=True,
                )

            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                # Extract tool_call_id for potential error response
                if isinstance(tool_call, dict) and "id" in tool_call:
                    tool_call_id_for_error = tool_call["id"]
                elif hasattr(tool_call, "id"):
                    tool_call_id_for_error = tool_call.id

                converted = False
                if isinstance(tool_call, dict) and "function" in tool_call:
                    if isinstance(tool_call["function"], dict):
                        accumulated_deltas["function_call"] = {
                            "name": tool_call["function"].get("name"),
                            "arguments": tool_call["function"].get("arguments"),
                        }
                        function_call_detected = True
                        converted = True
                        if llm.interpreter.verbose:
                            print(
                                f"[DEBUG] Converted tool_call to function_call: name={accumulated_deltas['function_call']['name']}",
                                flush=True,
                            )
                elif hasattr(tool_call, "function"):
                    accumulated_deltas["function_call"] = {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                    function_call_detected = True
                    converted = True
                    if llm.interpreter.verbose:
                        print(
                            f"[DEBUG] Converted tool_call (object) to function_call: name={accumulated_deltas['function_call']['name']}",
                            flush=True,
                        )

                # If we still couldn't convert, raise an error with details
                if not converted:
                    raise Exception(
                        f"Unsupported tool_call format. Type: {type(tool_call)}, "
                        f"Has 'function' attr: {hasattr(tool_call, 'function')}, "
                        f"Is dict: {isinstance(tool_call, dict)}, "
                        f"Dict keys if dict: {list(tool_call.keys()) if isinstance(tool_call, dict) else 'N/A'}"
                    )

    yield from dispatch_function_call(
        llm, accumulated_deltas, request_params, tool_call_id_for_error, verbose, language
    )

    if os.getenv("INTERPRETER_REQUIRE_AUTHENTICATION", "False").lower() == "true":
        print("function_call_detected", function_call_detected)
        print("accumulated_review", accumulated_review)
        if function_call_detected and not accumulated_review:
            print("WTF!!!!!!!!!")
            # import pdb
            # pdb.set_trace()
            raise Exception("Judge layer required but did not run.")
