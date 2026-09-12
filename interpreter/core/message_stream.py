"""Turning the loop's chunks into saved messages.

respond() yields a stream of small chunks; this assembles them into the
conversation's messages, adds the start/end flags the display needs, keeps
ephemeral chunks (active line, review) out of the history, and truncates
console output to max_output while archiving the full text.
"""

import time

from .respond import respond
from .utils.compact_output import compact_console_output
from .utils.execution_allowlist import should_require_execution_confirmation
from .utils.truncate_output import truncate_output


def respond_and_store(interpreter):
    """
    Pulls from the respond stream, adding delimiters. Some things, like active_line, console, confirmation... these act specially.
    Also assembles new messages and adds them to `interpreter.messages`.
    """
    # NOTE: There used to be a line here that set interpreter.verbose = False, which was wrong.
    # The verbose setting should be preserved from the user's configuration.

    # Utility function
    def is_ephemeral(chunk):
        """
        Ephemeral = this chunk doesn't contribute to a message we want to save.
        """
        if "format" in chunk and chunk["format"] == "active_line":
            return True
        if chunk["type"] == "review":
            return True
        return False

    last_flag_base = None

    try:
        for chunk in respond(interpreter):
            # For async usage
            if hasattr(interpreter, "stop_event") and interpreter.stop_event.is_set():
                print("Open Interpreter stopping.")
                break

            # Skip empty content, except for console output - empty command output is
            # meaningful (e.g. grep with no matches) and must be added so the LLM
            # sees that the command ran, preventing it from re-proposing the same code.
            if chunk.get("content") == "" and not (chunk.get("type") == "console" and chunk.get("format") == "output"):
                continue

            # If active_line is None, we finished running code.
            if chunk.get("format") == "active_line" and chunk.get("content", "") == None:
                # If output wasn't yet produced, add an empty output
                if interpreter.messages[-1]["role"] != "computer":
                    interpreter.messages.append(
                        {
                            "role": "computer",
                            "type": "console",
                            "format": "output",
                            "content": "",
                        }
                    )

            # Handle special chunks that don't need normal processing
            if chunk.get("type") == "stop_live_display":
                # Pass through to terminal interface for handling
                yield chunk
                continue

            # Handle the special "confirmation" chunk, which neither triggers a flag or creates a message
            if chunk["type"] == "confirmation":
                # Emit a end flag for the last message type, and reset last_flag_base
                if last_flag_base:
                    yield {**last_flag_base, "end": True}
                    last_flag_base = None

                if should_require_execution_confirmation(interpreter, chunk):
                    yield chunk

                # We want to append this now, so even if content is never filled, we know that the execution didn't produce output.
                # ... rethink this though.
                # interpreter.messages.append(
                #     {
                #         "role": "computer",
                #         "type": "console",
                #         "format": "output",
                #         "content": "",
                #     }
                # )
                continue

            # tool_call: records the tool call the model actually made (view_image, or
            # any malformed call) so convert_to_openai_messages can rebuild it as a real
            # assistant+tool_calls. Without it process_messages invents an assistant
            # message to pair with the tool response, and what it invents is an execute()
            # call the model never made, contradicting the error right below it.
            if chunk.get("type") == "tool_call":
                interpreter.messages.append(chunk)
                continue

            # notice: display-only. A malformed tool call is answered with a role:tool
            # message the user never sees (below), so without this line the user gets a
            # silent pause and another turn — a model stuck in a malformed-call loop
            # looks like a hang. It must NOT be stored: the model already has the tool
            # response and a second, assistant-shaped account of the same failure would
            # be one more message it has to reconcile.
            if chunk.get("type") == "notice":
                if last_flag_base:
                    yield {**last_flag_base, "end": True}
                    last_flag_base = None
                yield chunk
                continue

            # role:tool messages are API-internal (pairing for tool_call records,
            # unsupported function calls, etc.) and must not be displayed to the user.
            if chunk.get("role") == "tool" and chunk.get("type") == "message":
                if last_flag_base:
                    yield {**last_flag_base, "end": True}
                    last_flag_base = None
                interpreter.messages.append(chunk)
                continue

            # view_image_approval: AI wants to show image(s); user approves in terminal, result stored on interpreter
            if chunk.get("type") == "view_image_approval":
                if last_flag_base:
                    yield {**last_flag_base, "end": True}
                    last_flag_base = None
                yield chunk
                continue

            # Replace streamed reasoning with blockquote-formatted version (reasoning streamed raw, then replaced when complete)
            if chunk.get("replace") and chunk.get("format") == "reasoning":
                for i in range(len(interpreter.messages) - 1, -1, -1):
                    if interpreter.messages[i].get("format") == "reasoning":
                        interpreter.messages[i]["content"] = chunk["content"]
                        break
                yield chunk  # Terminal replaces active_block content while block is still active
                if last_flag_base:
                    yield {**last_flag_base, "end": True}
                last_flag_base = None
                continue

            # Check if the chunk's role, type, and format (if present) match the last_flag_base
            if (
                last_flag_base
                and "role" in chunk
                and "type" in chunk
                and last_flag_base["role"] == chunk["role"]
                and last_flag_base["type"] == chunk["type"]
                and (
                    "format" not in last_flag_base
                    or ("format" in chunk and chunk["format"] == last_flag_base["format"])
                )
            ):
                # If they match, append the chunk's content to the current message's content
                # (Except active_line, which shouldn't be stored)
                if not is_ephemeral(chunk):
                    if any(
                        [
                            (property in interpreter.messages[-1])
                            and (interpreter.messages[-1].get(property) != chunk.get(property))
                            for property in ["role", "type", "format"]
                        ]
                    ):
                        interpreter.messages.append(chunk)
                    else:
                        interpreter.messages[-1]["content"] += chunk["content"]
            else:
                # If they don't match, yield a end message for the last message type and a start message for the new one
                if last_flag_base:
                    yield {**last_flag_base, "end": True}

                last_flag_base = {"role": chunk["role"], "type": chunk["type"]}

                # Don't add format to type: "console" flags, to accommodate active_line AND output formats
                if "format" in chunk and chunk["type"] != "console":
                    last_flag_base["format"] = chunk["format"]

                yield {**last_flag_base, "start": True}

                # Add the chunk as a new message
                if not is_ephemeral(chunk):
                    interpreter.messages.append(chunk)

            # Yield the chunk itself
            yield chunk

            # Truncate output if it's console output
            if chunk["type"] == "console" and chunk["format"] == "output":
                # Squeeze out escape codes, redrawn progress frames and repeat
                # runs first: they are re-sent with every later request.
                interpreter.messages[-1]["content"] = compact_console_output(interpreter.messages[-1]["content"])
                spill_note = interpreter._record_full_output(interpreter.messages[-1], chunk["content"])
                interpreter.messages[-1]["content"] = truncate_output(
                    interpreter.messages[-1]["content"],
                    interpreter.max_output,
                    spill_note=spill_note,
                )

        # Yield a final end flag
        if last_flag_base:
            yield {**last_flag_base, "end": True}
    except GeneratorExit:
        raise  # gotta pass this up!
    except SystemExit:
        # Don't yield final end flag when exiting due to sys.exit()
        # This prevents duplicate output when error panel is displayed
        raise
