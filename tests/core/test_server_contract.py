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
