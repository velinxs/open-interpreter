"""Shaping the message list for function-calling providers.

Providers disagree about tool-call ids, about pairing each call with a
response, and about where an image may sit relative to tool messages;
process_messages() normalizes all of that before the request goes out.
"""

import json
import re


def generate_tool_id(tool_id_num, model=None):
    """
    Generate a tool call ID. For Mistral models, uses 9-character alphanumeric format.
    For other models, uses the original format.

    Mistral requires tool call IDs to match: ^[a-zA-Z0-9]{9}$
    See: https://github.com/mistralai/mistral-common/blob/21ee9f6cee3441e9bb1e6ed2d10173f90bd9b94b/src/mistral_common/protocol/instruct/validator.py#L309
    """
    # Check if this is a Mistral model
    is_mistral = model and ("mistral" in model.lower() or "devstral" in model.lower())

    if is_mistral:
        # Mistral requires exactly 9 alphanumeric characters
        import string

        # Base36: 0-9, a-z (36 characters total)
        base36_chars = string.digits + string.ascii_lowercase
        num = tool_id_num
        suffix = ""
        for _ in range(5):  # 5 digits to make total 9 chars (4 for "tool" + 5 for number)
            suffix = base36_chars[num % 36] + suffix
            num //= 36
        return f"tool{suffix}"
    else:
        # Original format for other models
        return f"toolu_{tool_id_num}"


def _inline_user_image_in_turn_after_last_assistant_text(messages):
    """
    True if any user message after the last assistant *text* reply includes an image_url
    part. Those images are already on the API request; offering view_image spuriously leads
    models to hallucinate filesystem paths (e.g. /mnt/data/image.png) and fail vision.
    """
    last_text_i = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            last_text_i = i
            break
    for m in messages[last_text_i + 1 :]:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for part in c:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
    return False


def process_messages(messages, model=None):
    processed_messages = []
    last_tool_id = 0

    i = 0
    while i < len(messages):
        message = messages[i]

        if message.get("function_call"):
            last_tool_id += 1
            tool_id = generate_tool_id(last_tool_id, model)

            # Convert function_call to tool_calls
            function = message.pop("function_call")
            # Some providers (e.g. Alibaba via OpenRouter) require function.arguments to be valid JSON string.
            args = function.get("arguments")
            if isinstance(args, dict):
                function = {**function, "arguments": json.dumps(args)}
            message["tool_calls"] = [{"id": tool_id, "type": "function", "function": function}]
            processed_messages.append(message)

            # Process the next message if it's a function response
            if i + 1 < len(messages) and messages[i + 1].get("role") == "function":
                next_message = messages[i + 1].copy()
                next_message["role"] = "tool"
                next_message["tool_call_id"] = tool_id
                processed_messages.append(next_message)
                i += 1  # Skip the next message as we've already processed it
            else:
                # Add an empty tool response if there isn't one
                processed_messages.append({"role": "tool", "tool_call_id": tool_id, "content": ""})

        elif message.get("role") == "function":
            # This handles orphaned function responses
            last_tool_id += 1
            tool_id = generate_tool_id(last_tool_id, model)

            # Add a tool call before this orphaned tool response. Providers like Alibaba require
            # function.arguments to be valid JSON; use execute-shaped payload to avoid API errors.
            processed_messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": "execute",
                                "arguments": json.dumps(
                                    {
                                        "language": "python",
                                        "code": "# Automated tool call to fetch more output, triggered by the user.",
                                    }
                                ),
                            },
                        }
                    ],
                }
            )

            # Process the function response
            message["role"] = "tool"
            message["tool_call_id"] = tool_id
            processed_messages.append(message)

        elif message.get("role") == "tool":
            # Tool message must follow an assistant message with tool_calls (OpenRouter/Alibaba etc.).
            # Unsupported/invalid tool calls yield a tool response but we never store an assistant
            # with tool_calls for that call, so we can get assistant (content) then tool (error).
            # Insert a synthetic assistant with tool_calls using this message's tool_call_id.
            prev = processed_messages[-1] if processed_messages else None
            if not prev or "tool_calls" not in prev or not prev.get("tool_calls"):
                tool_id = message.get("tool_call_id") or generate_tool_id(last_tool_id + 1, model)
                processed_messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": "execute",
                                    "arguments": json.dumps(
                                        {
                                            "language": "python",
                                            "code": "pass  # (synthetic; do not run)",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                )
            processed_messages.append(message)

        else:
            # For non-tool-related messages, just add them as is
            processed_messages.append(message)

        i += 1

    return processed_messages
