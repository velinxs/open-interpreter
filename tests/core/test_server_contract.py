"""Contract of the HTTP server as it behaves today, LLM faked.

The interp.local sandbox talks to /openai/chat/completions; a previous
postmortem found that endpoint dead for five reasons at once. These tests
lock the streaming and non-streaming shapes, the code-approval round trip,
the websocket protocol, and the insecure /run route so the Phase 2 split of
async_core cannot change them silently.
"""

import json

import pytest
from fastapi.testclient import TestClient

from interpreter.core.async_core import (
    OPENAI_CODE_APPROVAL_DECLINED,
    OPENAI_CODE_APPROVAL_PROMPT,
    AsyncInterpreter,
)
from tests.support.fake_llm import install_fake_llm

CODE_REPLY = "Running it.\n```python\nprint(6 * 7)\n```"
DONE_REPLY = "It printed 42."


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("INTERPRETER_INSECURE_ROUTES", "true")
    monkeypatch.delenv("INTERPRETER_API_KEY", raising=False)
    ai = AsyncInterpreter()
    ai.offline = True
    ai.disable_telemetry = True
    client = TestClient(ai.server.app)
    yield ai, client
    try:
        ai.terminal.terminate()
    except Exception:
        pass


def _chat(client, content, stream=False):
    return client.post(
        "/openai/chat/completions",
        json={
            "model": "fake",
            "stream": stream,
            "messages": [{"role": "user", "content": content}],
        },
    )


def _sse_content(response):
    text = ""
    for line in response.iter_lines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: ") :])
        delta = chunk["choices"][0].get("delta", {})
        text += delta.get("content") or ""
    return text


def test_heartbeat(server):
    """Health check used by oi-update.sh."""
    _, client = server
    assert client.get("/heartbeat").json() == {"status": "alive"}


def test_chat_completion_non_stream_runs_code_when_auto_run(server):
    """With auto_run on, one request runs the code and returns text plus console output."""
    ai, client = server
    install_fake_llm(ai, [CODE_REPLY, DONE_REPLY])
    ai.auto_run = True

    r = _chat(client, "what is 6*7")

    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "42" in content
    assert DONE_REPLY in content


def test_chat_completion_stream_runs_code_when_auto_run(server):
    """Streaming returns SSE chunks whose concatenated deltas carry the same content."""
    ai, client = server
    install_fake_llm(ai, [CODE_REPLY, DONE_REPLY])
    ai.auto_run = True

    r = _chat(client, "what is 6*7", stream=True)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    content = _sse_content(r)

    assert "42" in content
    assert DONE_REPLY in content


def test_chat_completion_pauses_for_approval_then_runs_on_yes(server):
    """Without auto_run the reply ends with the approval prompt; 'yes' runs the code."""
    ai, client = server
    install_fake_llm(ai, [CODE_REPLY, DONE_REPLY])
    ai.auto_run = False

    first = _chat(client, "what is 6*7").json()["choices"][0]["message"]["content"]
    assert OPENAI_CODE_APPROVAL_PROMPT.strip() in first
    assert "42" not in first

    second = _chat(client, "yes").json()["choices"][0]["message"]["content"]
    assert "42" in second
    assert DONE_REPLY in second


def test_chat_completion_declines_on_no(server):
    """'no' skips the code and answers with the declined notice."""
    ai, client = server
    install_fake_llm(ai, [CODE_REPLY])
    ai.auto_run = False

    _chat(client, "what is 6*7")
    r = _chat(client, "no").json()["choices"][0]["message"]["content"]

    assert OPENAI_CODE_APPROVAL_DECLINED.strip() in r


def test_run_route_executes_code_when_insecure_routes_enabled(server):
    """/run exists only with INTERPRETER_INSECURE_ROUTES=true and returns the output."""
    _, client = server
    r = client.post("/run", json={"language": "python", "code": "print(3 + 4)"})
    assert r.status_code == 200
    assert "7" in json.dumps(r.json()["output"])


def test_get_setting(server):
    """/settings/{name} returns the attribute as JSON text."""
    ai, client = server
    ai.auto_run = True
    r = client.get("/settings/auto_run")
    assert r.status_code == 200
    assert json.loads(r.json())["auto_run"] is True


def test_websocket_round_trip(server):
    """LMC start/content/end in, LMC chunks out, terminated by the complete status."""
    ai, client = server
    install_fake_llm(ai, ["Hello from the fake."])
    ai.auto_run = True

    with client.websocket_connect("/") as ws:
        ws.send_json({"auth": "anything"})
        assert ws.receive_json() == {"auth": True}
        ws.send_json({"role": "user", "type": "message", "start": True})
        ws.send_json({"role": "user", "type": "message", "content": "hi"})
        ws.send_json({"role": "user", "type": "message", "end": True})

        received = []
        for _ in range(200):
            msg = ws.receive_json()
            received.append(msg)
            if msg.get("role") == "server" and msg.get("content") == "complete":
                break

    text = "".join(
        m.get("content", "")
        for m in received
        if m.get("role") == "assistant" and m.get("type") == "message" and isinstance(m.get("content"), str)
    )
    assert "Hello from the fake." in text
    assert received[-1]["content"] == "complete"


def test_streaming_turn_does_not_block_the_event_loop(server):
    """A heartbeat answered while a turn is streaming proves the loop is free.

    The turn generator used to be iterated inside the async handler, so every
    other request waited for the model and the code to finish.
    """
    import asyncio
    import time as _time

    import httpx

    ai, _ = server

    class SlowFake:
        def __init__(self):
            self.calls = []

        def __call__(self, **params):
            self.calls.append(params)
            for piece in ("slow ", "reply ", "here"):
                _time.sleep(0.3)
                yield {"choices": [{"delta": {"content": piece}}]}
            yield {"choices": [{"delta": {}}]}

    install_fake_llm(ai, [])
    ai.llm.completions = SlowFake()
    ai.auto_run = True

    async def scenario():
        transport = httpx.ASGITransport(app=ai.server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            body = {"model": "fake", "stream": True, "messages": [{"role": "user", "content": "hi"}]}

            async def stream():
                async with client.stream("POST", "/openai/chat/completions", json=body) as r:
                    return "".join([chunk async for chunk in r.aiter_text()])

            task = asyncio.create_task(stream())
            await asyncio.sleep(0.15)  # the turn is now inside its first slow chunk
            t0 = _time.monotonic()
            hb = await client.get("/heartbeat")
            heartbeat_latency = _time.monotonic() - t0
            text = await task
            return hb.json(), heartbeat_latency, text

    hb, latency, raw = asyncio.run(scenario())
    assert hb == {"status": "alive"}
    text = ""
    for line in raw.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            text += json.loads(line[len("data: ") :])["choices"][0].get("delta", {}).get("content") or ""
    assert "slow reply here" in text
    assert latency < 0.2, f"heartbeat waited {latency:.2f}s behind the streaming turn"
