import json
import os
import re

from ..terminal.base_language import format_execute_language_description
from ..tools.file_edit import EDIT_LANGUAGES
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

    # Process the converted function_call (if any) to yield code
    if accumulated_deltas.get("function_call"):
        function_call = accumulated_deltas["function_call"]
        function_name = function_call.get("name", "")

        # If we don't have tool_call_id yet, try to extract it from the last assistant message
        if not tool_call_id_for_error:
            # Look at the last message in the conversation to find tool_call_id
            messages = request_params.get("messages", [])
            for message in reversed(messages):
                if message.get("role") == "assistant" and "tool_calls" in message:
                    tool_calls = message["tool_calls"]
                    if isinstance(tool_calls, list) and len(tool_calls) > 0:
                        tool_call = tool_calls[0]
                        if isinstance(tool_call, dict) and "id" in tool_call:
                            tool_call_id_for_error = tool_call["id"]
                            break
                        elif hasattr(tool_call, "id"):
                            tool_call_id_for_error = tool_call.id
                            break

        # Ensure tool_call_id is a non-empty string if we have it
        if tool_call_id_for_error and not isinstance(tool_call_id_for_error, str):
            tool_call_id_for_error = str(tool_call_id_for_error)
        if tool_call_id_for_error == "":
            tool_call_id_for_error = None

        # Only "execute" is supported as a direct tool call
        # Other functions (like toolbox.web.search) must be called from within Python code
        if function_name == "execute":
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                arguments = parse_partial_json(arguments)

            # Validate arguments and yield code, or yield error as tool response
            if isinstance(arguments, dict):
                if language is None and "language" in arguments and "code" in arguments and arguments["language"]:
                    language = arguments["language"]

                if language is not None and "code" in arguments:
                    code_value = arguments["code"]
                    if isinstance(code_value, str):
                        # Yield the full code (since we converted after stream, code variable is empty)
                        if code_value:
                            yield {
                                "role": "assistant",
                                "type": "code",
                                "format": language,
                                "content": code_value,
                            }
                        else:
                            # Empty code - yield error as tool response
                            error_msg = "Invalid execute call: code is empty"
                            if (
                                tool_call_id_for_error
                                and isinstance(tool_call_id_for_error, str)
                                and tool_call_id_for_error.strip()
                            ):
                                yield {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id_for_error,
                                    "type": "message",
                                    "content": error_msg,
                                }
                            elif verbose:
                                print(
                                    f"[ERROR] Cannot yield tool response: missing tool_call_id. Error: {error_msg}",
                                    flush=True,
                                )
                            if verbose:
                                print(
                                    f"[ERROR] {error_msg}. Arguments: {json.dumps(arguments, default=str)}", flush=True
                                )
                    else:
                        # Code is not a string - yield error as tool response
                        error_msg = f"Invalid execute call: code must be a string, got {type(code_value).__name__}"
                        if (
                            tool_call_id_for_error
                            and isinstance(tool_call_id_for_error, str)
                            and tool_call_id_for_error.strip()
                        ):
                            yield {
                                "role": "tool",
                                "tool_call_id": tool_call_id_for_error,
                                "type": "message",
                                "content": error_msg,
                            }
                        elif verbose:
                            print(
                                f"[ERROR] Cannot yield tool response: missing tool_call_id. Error: {error_msg}",
                                flush=True,
                            )
                        if verbose:
                            print(f"[ERROR] {error_msg}. Arguments: {json.dumps(arguments, default=str)}", flush=True)
                else:
                    # Missing language or code - yield error as tool response
                    error_msg = f"Invalid execute call: missing required fields. Got: {list(arguments.keys())}"
                    if verbose:
                        print(f"[ERROR] {error_msg}. Arguments: {json.dumps(arguments, default=str)}", flush=True)
                        print(
                            f"[ERROR] tool_call_id_for_error: {repr(tool_call_id_for_error)}, type: {type(tool_call_id_for_error)}",
                            flush=True,
                        )

                    if (
                        tool_call_id_for_error
                        and isinstance(tool_call_id_for_error, str)
                        and tool_call_id_for_error.strip()
                    ):
                        tool_response = {
                            "role": "tool",
                            "tool_call_id": tool_call_id_for_error,
                            "type": "message",
                            "content": error_msg,
                        }
                        if verbose:
                            print(
                                f"[ERROR] Yielding tool response: {json.dumps(tool_response, default=str)}", flush=True
                            )
                        yield tool_response
                    else:
                        # No tool_call_id available - this should not happen, but log it
                        if verbose:
                            print(
                                f"[ERROR] Cannot yield tool response: missing tool_call_id. Error: {error_msg}",
                                flush=True,
                            )
                            print(f"[ERROR] tool_call_id_for_error value: {repr(tool_call_id_for_error)}", flush=True)
                        # Still yield as assistant message so user sees the error
                        yield {"role": "assistant", "type": "message", "content": f"**Error:** {error_msg}"}
            else:
                # Arguments is not a dict - yield error as tool response
                error_msg = f"Invalid execute call: arguments must be a dict, got {type(arguments).__name__}"
                if (
                    tool_call_id_for_error
                    and isinstance(tool_call_id_for_error, str)
                    and tool_call_id_for_error.strip()
                ):
                    yield {
                        "role": "tool",
                        "tool_call_id": tool_call_id_for_error,
                        "type": "message",
                        "content": error_msg,
                    }
                elif verbose:
                    print(f"[ERROR] Cannot yield tool response: missing tool_call_id. Error: {error_msg}", flush=True)
                if verbose:
                    print(f"[ERROR] {error_msg}. Function call: {json.dumps(function_call, default=str)}", flush=True)
        elif function_name == "view_image":
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                arguments = parse_partial_json(arguments)
            path = isinstance(arguments, dict) and arguments.get("path")
            if not path or not isinstance(path, str):
                content = "view_image: path is required and must be a string."
            elif not os.path.isabs(path):
                content = "view_image: path must be absolute (e.g. C:\\Users\\... on Windows, /home/... on Linux/Mac)."
            elif not os.path.exists(path):
                content = f"view_image: file not found: {path}"
            else:
                ext = os.path.splitext(path)[1].lstrip(".").lower()
                if ext not in VIEW_IMAGE_ALLOWED_EXTENSIONS:
                    content = (
                        f"view_image: unsupported file format '.{ext}'. "
                        f"Supported formats: {', '.join(sorted(VIEW_IMAGE_ALLOWED_EXTENSIONS))}. "
                        "PDF and other document formats are not supported."
                    )
                else:
                    # Store the assistant's view_image call before the approval prompt.
                    # Without this, interpreter.messages has an orphaned role:tool response
                    # with no preceding assistant+tool_calls, causing process_messages to
                    # insert a synthetic execute call that the LLM echoes on the next turn.
                    yield {
                        "type": "view_image_call",
                        "tool_call_id": tool_call_id_for_error,
                        "path": path,
                    }
                    yield {
                        "type": "view_image_approval",
                        "paths": [path],
                    }
                    # f/r/n from terminal (single prompt): full res / resize / decline. "y" = legacy full res.
                    approval = getattr(llm.interpreter, "_view_image_approval", "n")
                    if approval in ("f", "r", "y"):
                        llm.interpreter._pending_view_image_path = path
                        llm.interpreter._pending_view_image_shrink = approval == "r"
                        content = "Image added; you will see it when you continue."
                    else:
                        content = "User declined to show image."
            if tool_call_id_for_error:
                yield {
                    "role": "tool",
                    "tool_call_id": tool_call_id_for_error,
                    "type": "message",
                    "content": content,
                }
            else:
                yield {"role": "assistant", "type": "message", "content": content}
        elif function_name == "edit":
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                arguments = parse_partial_json(arguments)

            if isinstance(arguments, dict):
                edit_language = arguments.get("language")
                edit_code = arguments.get("code")
                edit_target = arguments.get("target")

                valid_edit_languages = EDIT_LANGUAGES
                if not edit_language or edit_language not in valid_edit_languages:
                    error_msg = (
                        f"edit: invalid language {edit_language!r}. "
                        f"Must be one of: {', '.join(sorted(valid_edit_languages))}"
                    )
                elif edit_code is None:
                    error_msg = "edit: 'code' is required."
                elif not isinstance(edit_code, str):
                    error_msg = f"edit: 'code' must be a string, got {type(edit_code).__name__}"
                elif not edit_code.strip() and edit_language != "write":
                    error_msg = f"edit: 'code' cannot be empty for language {edit_language!r}."
                elif not edit_target:
                    error_msg = "edit: 'target' is required."
                elif not os.path.isabs(edit_target):
                    error_msg = f"edit: 'target' must be an absolute path, got: {edit_target!r}"
                else:
                    error_msg = None

                if error_msg:
                    if (
                        tool_call_id_for_error
                        and isinstance(tool_call_id_for_error, str)
                        and tool_call_id_for_error.strip()
                    ):
                        yield {
                            "role": "tool",
                            "tool_call_id": tool_call_id_for_error,
                            "type": "message",
                            "content": error_msg,
                        }
                    else:
                        yield {"role": "assistant", "type": "message", "content": f"**Error:** {error_msg}"}
                else:
                    yield {
                        "role": "assistant",
                        "type": "edit",
                        "format": edit_language,
                        "content": edit_code,
                        "target": edit_target,
                    }
            else:
                error_msg = f"edit: arguments must be a JSON object, got: {type(arguments).__name__}"
                if (
                    tool_call_id_for_error
                    and isinstance(tool_call_id_for_error, str)
                    and tool_call_id_for_error.strip()
                ):
                    yield {
                        "role": "tool",
                        "tool_call_id": tool_call_id_for_error,
                        "type": "message",
                        "content": error_msg,
                    }
                else:
                    yield {"role": "assistant", "type": "message", "content": f"**Error:** {error_msg}"}

        elif function_name:
            # Unsupported function call - yield error as tool response to maintain proper message ordering
            # The API expects: assistant (with tool_call) → tool (response) → user
            error_msg = (
                f"Unsupported function call: '{function_name}'. "
                f"Only 'execute', 'edit', and 'view_image' (vision models only) are supported as direct tool calls. "
                f"To use '{function_name}', call it from within Python code using the execute function. "
                f"For example: `toolbox.web.search('your query')`"
            )

            # Yield error as tool response so the model sees it and message ordering stays correct (assistant → tool → …).
            # Any assistant message content the model sent before this tool call is already yielded above with role "assistant".
            if tool_call_id_for_error:
                yield {"role": "tool", "tool_call_id": tool_call_id_for_error, "type": "message", "content": error_msg}
            else:
                yield {"role": "assistant", "type": "message", "content": f"**Error:** {error_msg}"}

            if verbose:
                print(f"[ERROR] {error_msg}", flush=True)
                print(f"[ERROR] Function call details: {json.dumps(function_call, default=str)}", flush=True)
                if tool_call_id_for_error:
                    print(
                        f"[ERROR] Yielding error as tool response with tool_call_id: {tool_call_id_for_error}",
                        flush=True,
                    )

    if os.getenv("INTERPRETER_REQUIRE_AUTHENTICATION", "False").lower() == "true":
        print("function_call_detected", function_call_detected)
        print("accumulated_review", accumulated_review)
        if function_call_detected and not accumulated_review:
            print("WTF!!!!!!!!!")
            # import pdb
            # pdb.set_trace()
            raise Exception("Judge layer required but did not run.")
