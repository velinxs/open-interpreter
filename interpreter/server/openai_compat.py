"""The OpenAI-compatible /openai/chat/completions surface.

Translates OpenAI-style requests into LMC messages, streams the loop's
chunks back as SSE deltas, and carries the yes/no code-approval protocol a
plain chat client can drive.
"""

import asyncio
import json
import time
from typing import Any

import shortuuid
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from ..core.utils.execution_allowlist import (
    should_require_execution_confirmation,
    should_require_execution_confirmation_for_code,
)

try:
    from fastapi import APIRouter
    from fastapi.responses import StreamingResponse
except ImportError:  # server extras not installed
    APIRouter = None
    StreamingResponse = None

last_start_time = 0

# Blank line before the closing --- so Chatbox does not treat it as a setext heading.
OPENAI_CODE_APPROVAL_PROMPT = (
    "\n\n---\n"
    "**[Open Interpreter]** Execution is paused. "
    "Reply with exactly **yes** to run this code or **no** to skip.\n"
    "\n---\n"
)

OPENAI_CODE_APPROVAL_DECLINED = "\n\n---\n**[Open Interpreter]** Okay, I won't run that code.\n\n---\n"

OPENAI_CODE_APPROVAL_INVALID_TEMPLATE = (
    "\n\n---\n**[Open Interpreter]** There is code waiting for your approval. "
    'Reply with exactly **yes** or **no** (you sent: "{reply}").\n'
    "\n---\n"
)

OPENAI_SHELL_OUTPUT_NOTE = "Note: Shell command output will be shown after completion."


def _normalize_openai_code_approval_reply(content):
    """Return ``yes``, ``no``, or ``None`` for a pending-code approval turn."""
    if not isinstance(content, str):
        return None
    normalized = content.lower().strip().strip(".!?,")
    if normalized == "yes":
        return "yes"
    if normalized == "no":
        return "no"
    return None


def _openai_server_has_pending_code(async_interpreter):
    return bool(
        async_interpreter.messages
        and async_interpreter.messages[-1].get("type") == "code"
        and should_require_execution_confirmation_for_code(
            async_interpreter,
            async_interpreter.messages[-1].get("format", ""),
            async_interpreter.messages[-1].get("content", ""),
        )
    )


def _openai_server_awaiting_approval(async_interpreter):
    if async_interpreter.auto_run_mode == "all":
        return False
    if getattr(async_interpreter, "_server_awaiting_code_approval", False):
        return True
    return _openai_server_has_pending_code(async_interpreter)


def _clear_openai_code_approval_wait(async_interpreter):
    async_interpreter._server_awaiting_code_approval = False


def _cancel_pending_code(async_interpreter):
    _clear_openai_code_approval_wait(async_interpreter)
    if async_interpreter.messages and async_interpreter.messages[-1].get("type") == "code":
        async_interpreter.messages.append(
            {
                "role": "user",
                "type": "message",
                "content": "[User moved on without approving the pending code.]",
                "source": "server",
            }
        )


def _format_openai_console_output(content, language="bash"):
    if not content or not str(content).strip():
        return None
    text = str(content)
    if OPENAI_SHELL_OUTPUT_NOTE in text and text.strip() == OPENAI_SHELL_OUTPUT_NOTE.strip():
        return None
    text = text.replace(OPENAI_SHELL_OUTPUT_NOTE, "").strip()
    if not text:
        return None
    fence_lang = (language or "text").lower()
    if fence_lang == "shell":
        fence_lang = "bash"
    return f"\n\n```{fence_lang}\n{text}\n```\n\n"


def _openai_apply_request_messages(async_interpreter, request, last_message):
    """Keep server-side LMC state; Chatbox replays plain-text history each POST."""
    _, client_system = _openai_messages_to_lmc(request.messages)
    async_interpreter._server_request_system = client_system

    if _openai_server_awaiting_approval(async_interpreter):
        return

    if not async_interpreter.messages:
        async_interpreter.messages, _ = _openai_messages_to_lmc(request.messages)
        return

    if isinstance(last_message.content, str):
        user_msg = {
            "role": "user",
            "type": "message",
            "content": last_message.content,
        }
        if (
            async_interpreter.messages
            and async_interpreter.messages[-1].get("role") == "user"
            and async_interpreter.messages[-1].get("content") == user_msg["content"]
        ):
            return
        async_interpreter.messages.append(user_msg)
    elif isinstance(last_message.content, list):
        for part in last_message.content:
            if part.get("type") == "text":
                user_msg = {
                    "role": "user",
                    "type": "message",
                    "content": part.get("text", ""),
                }
                if (
                    async_interpreter.messages
                    and async_interpreter.messages[-1].get("role") == "user"
                    and async_interpreter.messages[-1].get("content") == user_msg["content"]
                ):
                    continue
                async_interpreter.messages.append(user_msg)


def _is_openai_auxiliary_title_request(content):
    """Open WebUI and similar UIs send title prompts as normal chat completions."""
    if not isinstance(content, str):
        return False
    lower = content.lower()
    markers = (
        "give this conversation a name",
        "name this conversation",
        "conversation a name",
        "short title for this conversation",
        "generating a short title",
    )
    return any(marker in lower for marker in markers)


def _openai_messages_to_lmc(openai_messages):
    """Map an OpenAI-style messages array to LMC user/assistant turns.

    Client ``system`` messages are returned separately; ``respond()`` already
    prepends Open Interpreter's system message and ``llm.run`` forbids a second.
    """
    lmc = []
    client_system_parts = []
    for msg in openai_messages:
        role = msg.role
        content = msg.content
        if role == "system":
            if isinstance(content, str) and content.strip():
                client_system_parts.append(content.strip())
            continue
        if role == "assistant":
            text = content if isinstance(content, str) else str(content)
            if text:
                lmc.append({"role": "assistant", "type": "message", "content": text})
        elif role == "user":
            if isinstance(content, str):
                lmc.append({"role": "user", "type": "message", "content": content})
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        lmc.append(
                            {
                                "role": "user",
                                "type": "message",
                                "content": part.get("text", ""),
                            }
                        )
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if "base64," not in url:
                            raise ValueError('Image must be "data:image/jpeg;base64,{data}"')
                        data = url.split("base64,")[1]
                        fmt = "base64." + url.split(";")[0].split("/")[1]
                        lmc.append(
                            {
                                "role": "user",
                                "type": "image",
                                "format": fmt,
                                "content": data,
                            }
                        )
    client_system = "\n\n".join(client_system_parts) if client_system_parts else None
    return lmc, client_system


def _new_openai_completion_id():
    return f"chatcmpl-{shortuuid.uuid()}"


def _openai_sse_chunk(
    completion_id,
    created,
    delta_content=None,
    finish_reason=None,
    *,
    role=None,
):
    delta = {}
    if role is not None:
        delta["role"] = role
    if delta_content is not None:
        delta["content"] = delta_content
    choice = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return (
        "data: "
        + json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "open-interpreter",
                "choices": [choice],
            }
        )
        + "\n\n"
    )


def _lmc_chunk_to_openai_delta(chunk, interpreter, *, pending_code_language=None):
    if chunk.get("format") == "reasoning":
        return None
    if chunk.get("type") == "confirmation" and should_require_execution_confirmation(interpreter, chunk):
        return OPENAI_CODE_APPROVAL_PROMPT
    if chunk.get("type") == "message" and "content" in chunk:
        return chunk["content"]
    if chunk.get("type") == "code" and "start" in chunk:
        return "\n\n```" + chunk["format"] + "\n"
    if chunk.get("type") == "code" and "content" in chunk:
        return chunk["content"]
    if chunk.get("type") == "code" and "end" in chunk:
        return "\n```\n\n"
    if (
        chunk.get("role") == "computer"
        and chunk.get("type") == "console"
        and chunk.get("format") == "output"
        and chunk.get("content")
    ):
        return _format_openai_console_output(
            chunk["content"],
            language=pending_code_language or "bash",
        )
    return None


def _pending_code_language(async_interpreter):
    for message in reversed(async_interpreter.messages):
        if message.get("type") == "code":
            fmt = message.get("format") or "bash"
            if fmt == "shell":
                return "bash"
            return fmt
    return "bash"


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False


def create_openai_router(async_interpreter):
    router = APIRouter()

    async def _stream_openai_assistant_text(text):
        completion_id = _new_openai_completion_id()
        created = int(time.time())
        yield _openai_sse_chunk(completion_id, created, delta_content=text, role="assistant")
        yield _openai_sse_chunk(completion_id, created, finish_reason="stop")
        yield "data: [DONE]\n\n"

    def _openai_assistant_text_response(text, model):
        return {
            "id": _new_openai_completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        }

    async def _stream_openai_title(content_str):
        completion_id = _new_openai_completion_id()
        created = int(time.time())
        title_messages = [
            {
                "role": "system",
                "type": "message",
                "content": ("Output only a short conversation title (max 10 words). No quotes, no explanation."),
            },
            {"role": "user", "type": "message", "content": content_str},
        ]
        async for chunk in iterate_in_threadpool(
            async_interpreter.llm.run(title_messages, auxiliary_title_request=True)
        ):
            if chunk.get("format") == "reasoning":
                continue
            if chunk.get("type") == "message" and chunk.get("content"):
                yield _openai_sse_chunk(completion_id, created, delta_content=chunk["content"])
        yield _openai_sse_chunk(completion_id, created, finish_reason="stop")
        yield "data: [DONE]\n\n"

    async def openai_compatible_generator(run_code):
        completion_id = _new_openai_completion_id()
        created = int(time.time())
        sent_role = False

        async def emit_delta(text):
            nonlocal sent_role
            role = "assistant" if not sent_role else None
            if role:
                sent_role = True
            return _openai_sse_chunk(completion_id, created, delta_content=text, role=role)

        pending_lang = _pending_code_language(async_interpreter)

        if run_code:
            print("Running code.\n")
            chunk_iter = async_interpreter._respond_and_store()
        elif async_interpreter.context_mode:
            # 01 / context-mode clients: legacy nudge loop when the model stays silent.
            made_chunk = False
            for message in [
                ".",
                "Just say something, anything.",
                "Hello? Answer please.",
                "Are you there?",
                "Can you respond?",
                "Please reply.",
            ]:
                async for chunk in iterate_in_threadpool(
                    async_interpreter.chat(message=message, stream=True, display=False)
                ):
                    made_chunk = True
                    output_content = _lmc_chunk_to_openai_delta(
                        chunk,
                        async_interpreter,
                        pending_code_language=pending_lang,
                    )
                    if output_content:
                        yield await emit_delta(output_content)
                    if async_interpreter.stop_event.is_set():
                        break
                if made_chunk:
                    break
            yield _openai_sse_chunk(completion_id, created, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return
        else:
            async_interpreter.last_messages_count = len(async_interpreter.messages)
            chunk_iter = async_interpreter._respond_and_store()

        async for chunk in iterate_in_threadpool(chunk_iter):
            if run_code and "content" in chunk:
                print(chunk.get("content", ""), end="")
            if run_code and "start" in chunk:
                print("\n")

            if chunk.get("type") == "confirmation":
                # "yes" approves only the one pending block; further code in this
                # response must prompt again (same as terminal y/n per block).
                if run_code:
                    run_code = False
                    continue
                output_content = _lmc_chunk_to_openai_delta(
                    chunk,
                    async_interpreter,
                    pending_code_language=pending_lang,
                )
                if output_content:
                    yield await emit_delta(output_content)
                if should_require_execution_confirmation(async_interpreter, chunk):
                    async_interpreter._server_awaiting_code_approval = True
                    break
                continue

            output_content = _lmc_chunk_to_openai_delta(
                chunk,
                async_interpreter,
                pending_code_language=pending_lang,
            )
            if output_content:
                yield await emit_delta(output_content)

            if async_interpreter.stop_event.is_set():
                break

        if not _openai_server_has_pending_code(async_interpreter):
            _clear_openai_code_approval_wait(async_interpreter)

        yield _openai_sse_chunk(completion_id, created, finish_reason="stop")
        yield "data: [DONE]\n\n"

    @router.post("/openai/chat/completions")
    async def chat_completion(request: ChatCompletionRequest):
        global last_start_time

        # Convert to LMC
        last_message = request.messages[-1]

        if last_message.role != "user":
            raise ValueError("Last message must be from the user.")

        if last_message.content == "{STOP}":
            # Handle special STOP token
            async_interpreter.stop_event.set()
            await asyncio.sleep(5)
            async_interpreter.stop_event.clear()
            return

        if last_message.content in ["{CONTEXT_MODE_ON}", "{REQUIRE_START_ON}"]:
            async_interpreter.context_mode = True
            return

        if last_message.content in ["{CONTEXT_MODE_OFF}", "{REQUIRE_START_OFF}"]:
            async_interpreter.context_mode = False
            return

        if last_message.content == "{AUTO_RUN_ON}":
            async_interpreter.auto_run = True
            return

        if last_message.content == "{AUTO_RUN_OFF}":
            async_interpreter.auto_run = False
            return

        content_str = last_message.content if isinstance(last_message.content, str) else None

        if content_str and _is_openai_auxiliary_title_request(content_str):
            if request.stream:
                return StreamingResponse(
                    _stream_openai_title(content_str),
                    media_type="text/event-stream",
                )
            title_messages = [
                {
                    "role": "system",
                    "type": "message",
                    "content": ("Output only a short conversation title (max 10 words). No quotes, no explanation."),
                },
                {"role": "user", "type": "message", "content": content_str},
            ]

            def collect_title():
                content = ""
                for chunk in async_interpreter.llm.run(title_messages, auxiliary_title_request=True):
                    if chunk.get("format") == "reasoning":
                        continue
                    if chunk.get("type") == "message" and chunk.get("content"):
                        content += chunk["content"]
                return content

            content = await run_in_threadpool(collect_title)
            completion_id = _new_openai_completion_id()
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            }

        run_code = False
        skip_message_replace = False
        if _openai_server_awaiting_approval(async_interpreter) and content_str is not None:
            approval = _normalize_openai_code_approval_reply(content_str)
            if approval == "yes":
                run_code = True
                skip_message_replace = True
                print(">", content_str, "(approve code)")
            elif approval == "no":
                _clear_openai_code_approval_wait(async_interpreter)
                async_interpreter.messages.append(
                    {
                        "role": "user",
                        "type": "message",
                        "content": "[User declined to run this code.]",
                        "source": "server",
                    }
                )
                print(">", content_str, "(decline code)")
                if request.stream:
                    return StreamingResponse(
                        _stream_openai_assistant_text(OPENAI_CODE_APPROVAL_DECLINED),
                        media_type="text/event-stream",
                    )
                return _openai_assistant_text_response(OPENAI_CODE_APPROVAL_DECLINED, request.model)
            else:
                retry_text = OPENAI_CODE_APPROVAL_INVALID_TEMPLATE.format(reply=content_str.strip()[:200])
                print(">", content_str, "(invalid approval reply)")
                if request.stream:
                    return StreamingResponse(
                        _stream_openai_assistant_text(retry_text),
                        media_type="text/event-stream",
                    )
                return _openai_assistant_text_response(retry_text, request.model)
        if not skip_message_replace:
            if isinstance(last_message.content, (str, list)):
                _openai_apply_request_messages(async_interpreter, request, last_message)
                print(">", content_str or last_message.content)
            elif async_interpreter.context_mode:
                if last_message.content == "{START}":
                    if async_interpreter.messages[-1]["content"] == "{START}":
                        async_interpreter.messages = async_interpreter.messages[:-1]
                    last_start_time = time.time()
                    if async_interpreter.messages and async_interpreter.messages[-1].get("role") != "user":
                        return
                else:
                    current_time = time.time()
                    if current_time - last_start_time > 6:
                        return
            elif last_message.content == "{START}":
                async_interpreter.messages = async_interpreter.messages[:-1]
                return

        async_interpreter.stop_event.set()
        await asyncio.sleep(0.1)
        async_interpreter.stop_event.clear()

        if request.stream:
            return StreamingResponse(
                openai_compatible_generator(run_code),
                media_type="text/event-stream",
            )
        else:
            async_interpreter.last_messages_count = len(async_interpreter.messages)
            pending_lang = _pending_code_language(async_interpreter)

            def collect():
                content = ""
                for chunk in async_interpreter._respond_and_store():
                    if chunk.get("type") == "confirmation" and run_code:
                        continue
                    delta = _lmc_chunk_to_openai_delta(
                        chunk,
                        async_interpreter,
                        pending_code_language=pending_lang,
                    )
                    if delta:
                        content += delta
                    if (
                        chunk.get("type") == "confirmation"
                        and not run_code
                        and should_require_execution_confirmation(async_interpreter, chunk)
                    ):
                        break
                return content

            content = await run_in_threadpool(collect)
            completion_id = _new_openai_completion_id()
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            }

    return router
