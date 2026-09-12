"""Turning a completed tool call into LMC chunks.

Once the stream is done, exactly one function call may be pending: execute
(run code), edit (apply a file edit) or view_image (show the model a file).
Anything else, or bad arguments, comes back as a tool response the model can
read and correct, rather than an exception that ends the turn.
"""

import json
import os

from ..tools.file_edit import EDIT_LANGUAGES
from .tool_messages import generate_tool_id
from .tool_schema import VIEW_IMAGE_ALLOWED_EXTENSIONS
from .utils.parse_partial_json import parse_partial_json


def _error_chunk(tool_call_id_for_error, error_msg):
    """Build the tool-response chunk every malformed-call branch below yields.

    dispatch_function_call mints a tool_call_id before any of these branches
    run (see the block right after id normalisation), so there is no longer a
    "no id" case to fall back from here — every call into this function has
    one. That matters because respond() only gives the model another turn
    when the last message has role == "tool"; a role: assistant message would
    end the turn with the user seeing the error and the model never reading
    it, which is the failure this whole function exists to avoid.
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id_for_error,
        "type": "message",
        "content": error_msg,
    }


def _tool_call_chunk(tool_call_id, function_call):
    """Record the tool call the model actually made, ahead of the response to it.

    Without this, ``interpreter.messages`` holds a ``role: tool`` response with
    no assistant tool call before it, and process_messages has to invent one to
    satisfy the provider's pairing rule. What it invents is
    ``execute(code="pass  # (synthetic; do not run)")``, so the model reads its
    own history as "I called execute with `pass`" immediately followed by "that
    call was invalid" — two statements that contradict each other about a call
    it never made. In the nameless-function case it was worse still: the model
    was shown a call *named* execute and then told its function name was
    missing. Recording the real call is what makes the corrective turn readable.

    ``arguments`` is kept exactly as the model sent it, unparseable JSON
    included. It is paired with an error that says the JSON was bad, so
    "repairing" it here would put a call the model never made in front of that
    error and reintroduce the contradiction.
    """
    arguments = function_call.get("arguments")
    if not isinstance(arguments, (str, dict, list, int, float, bool, type(None))):
        # Conversations are saved as JSON; an exotic provider object here would
        # make the whole history unsaveable. str() is the closest honest thing.
        arguments = str(arguments)
    return {
        "role": "assistant",
        "type": "tool_call",
        "tool_call_id": tool_call_id,
        # Empty when the provider sent no function name. That is the honest
        # record: the paired error says the name was missing, and naming a tool
        # here would be the fabrication described above.
        "name": function_call.get("name") or "",
        "arguments": arguments,
    }


def _malformed_call(tool_call_id_for_error, function_call, error_msg):
    """Every chunk a malformed tool call produces, in the order history needs.

    The call the model really made, then the error answering it. Branches must
    yield both together: one that yields only the error leaves the response
    unpaired, and process_messages then invents the assistant message.
    """
    yield _tool_call_chunk(tool_call_id_for_error, function_call)
    yield _error_chunk(tool_call_id_for_error, error_msg)


def _mint_tool_call_id(request_params, model):
    """Mint a tool_call_id for a call the provider didn't attach one to.

    Delegates the format to generate_tool_id (tool_messages.py owns the
    Mistral 9-char-alphanumeric rule; hand-rolling a second format here would
    just be a new place for that rule to drift out of sync). The candidate is
    checked against every id already present in request_params["messages"] —
    both on an assistant's tool_calls and on any tool response — and bumped
    until it is unique, so this never hands out an id that collides with one
    already in flight for this request.
    """
    existing_ids = set()
    for message in request_params.get("messages") or []:
        for tool_call in message.get("tool_calls") or []:
            call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
            if call_id:
                existing_ids.add(call_id)
        call_id = message.get("tool_call_id")
        if call_id:
            existing_ids.add(call_id)

    n = 1
    candidate = generate_tool_id(n, model)
    while candidate in existing_ids:
        n += 1
        candidate = generate_tool_id(n, model)
    return candidate


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

        # Ensure tool_call_id is a real, non-blank string, or None. Whitespace-only
        # ids are treated as absent too: a bare truthiness check would let "   "
        # through as if it were usable, and a blank id in a tool response is the
        # exact pairing failure this normalisation exists to prevent.
        if tool_call_id_for_error is not None and not isinstance(tool_call_id_for_error, str):
            tool_call_id_for_error = str(tool_call_id_for_error)
        if isinstance(tool_call_id_for_error, str) and not tool_call_id_for_error.strip():
            tool_call_id_for_error = None

        # The provider gave us no usable id. Mint one rather than leaving this
        # call id-less: every branch below needs a real tool_call_id to answer
        # with a properly paired role:tool message, which is what lets respond()
        # give the model a corrective turn instead of ending in silence.
        if tool_call_id_for_error is None:
            tool_call_id_for_error = _mint_tool_call_id(request_params, getattr(llm, "model", None))

        # Only "execute" is supported as a direct tool call
        # Other functions (like toolbox.web.search) must be called from within Python code
        if function_name == "execute":
            raw_arguments = function_call.get("arguments")
            arguments = raw_arguments
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
                            error_msg = (
                                "Invalid execute call: code is empty. "
                                "execute requires a non-empty 'code' string."
                            )
                            yield from _malformed_call(tool_call_id_for_error, function_call, error_msg)
                            if verbose:
                                print(
                                    f"[ERROR] {error_msg}. Arguments: {json.dumps(arguments, default=str)}", flush=True
                                )
                    else:
                        # Code is not a string - yield error as tool response
                        error_msg = (
                            f"Invalid execute call: code must be a string, got {type(code_value).__name__}. "
                            "execute requires 'code' as a string."
                        )
                        yield from _malformed_call(tool_call_id_for_error, function_call, error_msg)
                        if verbose:
                            print(f"[ERROR] {error_msg}. Arguments: {json.dumps(arguments, default=str)}", flush=True)
                else:
                    # Missing language or code - yield error as tool response
                    error_msg = (
                        f"Invalid execute call: missing required fields. "
                        f"execute requires 'language' and 'code'. Got: {list(arguments.keys())}"
                    )
                    if verbose:
                        print(f"[ERROR] {error_msg}. Arguments: {json.dumps(arguments, default=str)}", flush=True)
                        print(
                            f"[ERROR] tool_call_id_for_error: {repr(tool_call_id_for_error)}, type: {type(tool_call_id_for_error)}",
                            flush=True,
                        )
                    yield from _malformed_call(tool_call_id_for_error, function_call, error_msg)
            else:
                # Arguments is not a dict. Distinguish a syntax problem (the JSON
                # could not be parsed or repaired, so parse_partial_json handed
                # back its None sentinel) from a shape problem (it parsed fine but
                # to something other than an object) — a model told "got NoneType"
                # for a syntax error has no way to know it sent bad JSON at all,
                # and will likely resend the same broken payload.
                if isinstance(raw_arguments, str) and arguments is None:
                    error_msg = (
                        "Invalid execute call: arguments were not valid JSON and could not be repaired. "
                        "execute requires a JSON object with 'language' and 'code', "
                        'e.g. {"language": "python", "code": "print(1)"}.'
                    )
                else:
                    error_msg = (
                        f"Invalid execute call: arguments must be a dict, got {type(arguments).__name__}. "
                        "execute requires a JSON object with 'language' and 'code'."
                    )
                yield from _malformed_call(tool_call_id_for_error, function_call, error_msg)
                if verbose:
                    print(f"[ERROR] {error_msg}. Function call: {json.dumps(function_call, default=str)}", flush=True)
        elif function_name == "view_image":
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                arguments = parse_partial_json(arguments)
            path = isinstance(arguments, dict) and arguments.get("path")
            if not path or not isinstance(path, str):
                yield from _malformed_call(
                    tool_call_id_for_error, function_call, "view_image: path is required and must be a string."
                )
            elif not os.path.isabs(path):
                yield from _malformed_call(
                    tool_call_id_for_error,
                    function_call,
                    "view_image: path must be absolute (e.g. C:\\Users\\... on Windows, /home/... on Linux/Mac).",
                )
            elif not os.path.exists(path):
                yield from _malformed_call(
                    tool_call_id_for_error, function_call, f"view_image: file not found: {path}"
                )
            else:
                ext = os.path.splitext(path)[1].lstrip(".").lower()
                if ext not in VIEW_IMAGE_ALLOWED_EXTENSIONS:
                    yield from _malformed_call(
                        tool_call_id_for_error,
                        function_call,
                        f"view_image: unsupported file format '.{ext}'. "
                        f"Supported formats: {', '.join(sorted(VIEW_IMAGE_ALLOWED_EXTENSIONS))}. "
                        "PDF and other document formats are not supported.",
                    )
                else:
                    # Store the assistant's view_image call before the approval prompt,
                    # for the same reason every malformed branch records its call: an
                    # orphaned role:tool response makes process_messages invent an
                    # assistant message, and what it invents is an execute() call the
                    # model never made. This used to be its own "view_image_call" chunk
                    # type; it is the general tool_call record now, because one mechanism
                    # covering both is what stops the two drifting apart.
                    yield _tool_call_chunk(tool_call_id_for_error, function_call)
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
                    # Not an error — approval outcomes go straight out as role:tool,
                    # unlike the branches above which route through _malformed_call.
                    yield {
                        "role": "tool",
                        "tool_call_id": tool_call_id_for_error,
                        "type": "message",
                        "content": content,
                    }
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
                    yield from _malformed_call(tool_call_id_for_error, function_call, error_msg)
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
                yield from _malformed_call(tool_call_id_for_error, function_call, error_msg)

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
            yield from _malformed_call(tool_call_id_for_error, function_call, error_msg)

            if verbose:
                print(f"[ERROR] {error_msg}", flush=True)
                print(f"[ERROR] Function call details: {json.dumps(function_call, default=str)}", flush=True)
                # No `if tool_call_id_for_error:` guard here: an id is always minted
                # above, so the condition was dead and read as if it could be absent.
                print(
                    f"[ERROR] Yielding error as tool response with tool_call_id: {tool_call_id_for_error}",
                    flush=True,
                )
        else:
            # function_call present but its name is missing or empty. Every branch
            # above is keyed on function_name, so without this arm the call falls
            # off the end of the chain silently: nothing yielded, nothing logged,
            # and the model never learns its call was dropped.
            error_msg = (
                "Malformed tool call: function name is missing. "
                "Call one of 'execute', 'edit', or 'view_image' (vision models only)."
            )
            yield from _malformed_call(tool_call_id_for_error, function_call, error_msg)
            if verbose:
                print(f"[ERROR] {error_msg}. Function call: {json.dumps(function_call, default=str)}", flush=True)
