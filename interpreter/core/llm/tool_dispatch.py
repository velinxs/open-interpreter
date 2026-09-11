"""Turning a completed tool call into LMC chunks.

Once the stream is done, exactly one function call may be pending: execute
(run code), edit (apply a file edit) or view_image (show the model a file).
Anything else, or bad arguments, comes back as a tool response the model can
read and correct, rather than an exception that ends the turn.
"""

import json
import os

from ..tools.file_edit import EDIT_LANGUAGES
from .tool_schema import VIEW_IMAGE_ALLOWED_EXTENSIONS
from .utils.parse_partial_json import parse_partial_json


def dispatch_function_call(llm, accumulated_deltas, request_params, tool_call_id_for_error, verbose, language):
    """Yield the chunks for the pending function call, if there is one."""
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
