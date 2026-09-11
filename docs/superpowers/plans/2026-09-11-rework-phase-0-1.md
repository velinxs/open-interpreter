# Fork Rework, Phases 0 and 1: Safety Net and Cleanse

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the fork a CI, an offline-green test baseline, characterization tests for the server and CLI, and a token-measurement harness (Phase 0); then remove dead code and unused dependencies, move packaging to PEP 621 + hatchling + uv on Python 3.12+, adopt ruff, and strip every import-time side effect (Phase 1).

**Architecture:** Nothing moves between modules in these phases. Phase 0 only adds files under `tests/` and `.github/`. Phase 1 deletes files, rewrites `pyproject.toml`, rewrites `interpreter/__init__.py` around a PEP 562 module `__getattr__`, and pushes heavy imports (litellm, selenium, tiktoken, ...) down into the functions that use them.

**Tech Stack:** Python 3.12+, pytest, ruff, uv, hatchling, FastAPI TestClient, tiktoken.

**Spec:** `docs/superpowers/specs/2026-09-11-fork-rework-design.md`

## Global Constraints

- Python floor: `requires-python = ">=3.12,<4"`.
- Every commit: `ruff check` clean on the files it touches, `pytest -m "not integration"` green offline.
- One idea per commit, Conventional Commits, follow-up fixes folded into the commit they fix.
- No feature-bearing code is removed. Only the dead files listed in the spec go.
- Public import paths keep working: `interpreter.interpreter`, `interpreter.OpenInterpreter`, `interpreter.AsyncInterpreter`, `interpreter.core.core.OpenInterpreter`, `interpreter.core.async_core.AsyncInterpreter` and `.Server`, console entry points `interpreter`, `i`, `interpreter-classic`, `wtf`.
- Commit trailer on every commit: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Baseline on `integration` (2026-09-11): 12 failed, 258 passed, 30 skipped; ruff 6 errors. Phase 0 must end at 0 failed.
- Work happens on branch `rework`. Run everything with the project venv: `.venv/bin/python -m pytest ...` until Task 8 switches to `uv run pytest ...`.

---

## Phase 0: safety net

### Task 1: CI that runs for the fork

**Files:**
- Modify: `.github/workflows/python-package.yml` (replace whole file)

**Interfaces:**
- Produces: a workflow named `Lint and Test` with jobs `lint` and `test`; Task 8 later edits the install lines to use uv.

- [ ] **Step 1: Replace the workflow**

```yaml
name: Lint and Test

on:
  push:
    branches: [main, integration, rework]
  pull_request:
    branches: [main, integration, rework]

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Ruff
        run: |
          pip install ruff
          ruff check interpreter tests scripts

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.12", "3.14"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[server]"
          pip install pytest websockets requests
      - name: Test
        run: pytest -m "not integration" -q
```

- [ ] **Step 2: Validate the YAML parses**

Run: `.venv/bin/python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/python-package.yml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/python-package.yml
git commit -m "ci: run lint and tests for rework, integration and main

The workflow only fired on main, so nothing ever ran for the branches we
actually work on. It now runs ruff and the offline test suite on 3.12 and
3.14 for every push and pull request to rework, integration and main. The
coverage upload is dropped: the fork has no Codecov token and the job failed
closed without one.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 2: Deterministic offline baseline

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/core/test_llm_provider_routing.py:126-180`
- Modify: `tests/core/test_profile_reasoning_settings.py` (three tests)
- Modify: `tests/core/test_web.py:13,77,88`
- Modify: `tests/core/test_terminal_languages.py:136,160,190` (three tests)

**Interfaces:**
- Produces: pytest marker `network` (skipped unless `RUN_NETWORK_TESTS=1`).

- [ ] **Step 1: Register the marker and the skip rule in conftest**

Replace `tests/conftest.py` with:

```python
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires an LLM API key (not run in default CI)"
    )
    config.addinivalue_line(
        "markers",
        "network: makes real HTTP calls to a provider (run with RUN_NETWORK_TESTS=1)",
    )


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(
        reason="OPENAI_API_KEY not set; skipping integration test"
    )
    skip_network = pytest.mark.skip(
        reason="RUN_NETWORK_TESTS not set; test makes real provider calls"
    )
    for item in items:
        if "integration" in item.keywords and not os.environ.get("OPENAI_API_KEY"):
            item.add_marker(skip_integration)
        if "network" in item.keywords and not os.environ.get("RUN_NETWORK_TESTS"):
            item.add_marker(skip_network)
```

- [ ] **Step 2: Mark the six provider tests**

These six stub the OpenRouter registry but still reach `litellm.completion`, which makes a real HTTPS call and fails with 401 offline. They are not hermetic; Phase 2 rewrites them against the split LLM module. Add `@pytest.mark.network` directly above each `def`:

- `tests/core/test_llm_provider_routing.py`: `test_openrouter_qwen37_vision_detected_when_registry_stale`, `test_openrouter_qwen37_text_only_not_vision`, `test_openrouter_vision_helper_skips_non_openrouter_models`
- `tests/core/test_profile_reasoning_settings.py`: `test_mandatory_reasoning_ignores_include_reasoning_false`, `test_mandatory_reasoning_still_sends_supported_effort`, `test_unsupported_effort_dropped_with_warning`

Both files already `import pytest`; check with `grep -n "^import pytest" <file>` and add it if missing.

- [ ] **Step 3: Skip the linkup tests when linkup is not installed**

In `tests/core/test_web.py`, add `import pytest` at the top, and as the first line inside each of `test_structured_output_linkup`, `test_structured_output_pydantic_flexibility`, and `test_structured_output_no_backend_available`:

```python
        pytest.importorskip("linkup", reason="linkup-sdk not installed")
```

(`test_structured_output_no_backend_available` fails without linkup because the error it gets is `Install linkup-sdk: pip install linkup-sdk`, not the aggregate message it asserts on.)

- [ ] **Step 4: Skip the three PowerShell-dependent tests when pwsh is absent**

In `tests/core/test_terminal_languages.py` add `import shutil` and `import unittest` to the imports if missing, define once near the top:

```python
_HAS_PWSH = bool(shutil.which("pwsh") or os.environ.get("INTERPRETER_POWERSHELL"))
```

and decorate `test_active_line_injection_disabled_when_env_false`, `test_active_line_injection_present_when_env_true`, and `test_powershell_line_postprocessor_filters_prompt_and_continuation` with:

```python
    @unittest.skipUnless(_HAS_PWSH, "pwsh not installed")
```

- [ ] **Step 5: Run the whole offline suite**

Run: `.venv/bin/python -m pytest -q -m "not integration" -p no:cacheprovider tests`
Expected: `0 failed`, the 12 former failures now counted under skipped.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/core/test_llm_provider_routing.py tests/core/test_profile_reasoning_settings.py tests/core/test_web.py tests/core/test_terminal_languages.py
git commit -m "test: make the offline suite deterministic

Twelve tests failed on every machine without provider keys, pwsh, or the
optional linkup-sdk, so a red run carried no information. Six tests that
reach a real provider are marked network and skipped unless
RUN_NETWORK_TESTS=1; the three linkup tests importorskip linkup; the three
tests that construct PowerShell skip when pwsh is absent. The suite is now
green offline, which is the baseline every later commit is held to.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 3: Fake LLM test support

**Files:**
- Create: `tests/support/__init__.py` (empty)
- Create: `tests/support/fake_llm.py`
- Modify: `tests/conftest.py` (add fixture)
- Create: `tests/core/test_fake_llm.py`

**Interfaces:**
- Produces: `FakeCompletions(replies: list[str])` callable as `llm.completions(**params)`, with `.calls: list[dict]` recording every request's params; `install_fake_llm(interpreter, replies) -> FakeCompletions`; pytest fixture `offline_interpreter` returning an `OpenInterpreter` with telemetry off, `auto_run=True`, `offline=True`, and a helper method `offline_interpreter.script(replies)` that installs the fake and returns it.

- [ ] **Step 1: Write the failing test**

`tests/core/test_fake_llm.py`:

```python
"""The fake LLM must drive a full turn, including code execution, with no network."""

from tests.support.fake_llm import install_fake_llm


def test_fake_llm_runs_code_and_records_requests(offline_interpreter):
    """A scripted reply with a python block is executed and its output fed back.

    Locks the injection point every other characterization test relies on:
    `llm.completions` is the only seam between the loop and the provider.
    """
    fake = install_fake_llm(
        offline_interpreter,
        ["Sure.\n```python\nprint(21 * 2)\n```", "The answer is 42."],
    )

    messages = offline_interpreter.chat("what is 21*2", display=False, stream=False)

    outputs = [m["content"] for m in messages if m.get("type") == "console"]
    assert any("42" in str(o) for o in outputs)
    assert messages[-1]["content"] == "The answer is 42."
    assert len(fake.calls) == 2
    assert fake.calls[0]["messages"][0]["role"] == "system"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest -q tests/core/test_fake_llm.py -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.support'` or `fixture 'offline_interpreter' not found`.

- [ ] **Step 3: Write the support module**

`tests/support/fake_llm.py`:

```python
"""A scripted stand-in for Llm.completions.

Llm.run() only ever touches the provider through `llm.completions(**params)`,
which normally is litellm.completion. Replacing that one attribute lets the
whole loop (system message assembly, trimming, code execution, output
feedback) run for real while the model's replies come from a list.
"""


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # params dict per request, in order

    def __call__(self, **params):
        self.calls.append(params)
        if not self.replies:
            raise AssertionError("FakeCompletions ran out of scripted replies")
        text = self.replies.pop(0)
        step = max(1, len(text) // 3)  # a few chunks, like a real stream
        for i in range(0, len(text), step):
            yield {"choices": [{"delta": {"content": text[i : i + step]}}]}
        yield {"choices": [{"delta": {}}]}


def install_fake_llm(interpreter, replies):
    fake = FakeCompletions(replies)
    llm = interpreter.llm
    llm.completions = fake
    llm.model = "openai/fake"
    llm.api_key = "fake"
    llm.api_base = "http://127.0.0.1:9"
    llm.supports_functions = False
    llm.supports_vision = False
    llm.context_window = 32000
    llm.max_tokens = 4096
    llm._is_loaded = True
    interpreter.offline = True
    interpreter.disable_telemetry = True
    interpreter.auto_run = True
    return fake
```

Create `tests/support/__init__.py` empty. Append to `tests/conftest.py`:

```python
@pytest.fixture
def offline_interpreter():
    """An OpenInterpreter that never talks to a provider or telemetry."""
    from interpreter.core.core import OpenInterpreter
    from tests.support.fake_llm import install_fake_llm

    interp = OpenInterpreter()
    interp.offline = True
    interp.disable_telemetry = True
    interp.auto_run = True
    interp.script = lambda replies: install_fake_llm(interp, replies)
    yield interp
    try:
        interp.terminal.stop()
    except Exception:
        pass
```

Check `OpenInterpreter.terminal` has a `stop()` method: `grep -n "def stop" interpreter/core/terminal/terminal.py`. If it is named differently (for example `terminate`), use that name.

- [ ] **Step 4: Run the test until it passes**

Run: `.venv/bin/python -m pytest -q tests/core/test_fake_llm.py -p no:cacheprovider -x`
Expected: PASS. If `Llm.run` raises inside trimming because the model name is unknown to tokentrim, the fix is in the fake, not the code: keep `context_window` set (that path skips model lookup) and re-run.

- [ ] **Step 5: Commit**

```bash
git add tests/support tests/conftest.py tests/core/test_fake_llm.py
git commit -m "test: add a scripted fake LLM and an offline interpreter fixture

Every characterization test that follows needs to drive a real turn without
a provider. The fake replaces llm.completions, the single seam Llm.run uses,
records each request's params, and replays scripted replies as stream
chunks, so system-message assembly, trimming, and code execution all run
for real.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 4: Server characterization tests

**Files:**
- Create: `tests/core/test_server_contract.py`

**Interfaces:**
- Consumes: `install_fake_llm` from Task 3; `AsyncInterpreter`, `OPENAI_CODE_APPROVAL_PROMPT`, `OPENAI_CODE_APPROVAL_DECLINED` from `interpreter.core.async_core`.
- Produces: the behavioral contract for `/heartbeat`, `/openai/chat/completions` (stream and non-stream, approval round-trip), the websocket, `/run`, and `/settings/{setting}`.

These are characterization tests: they lock what the server does today. If an assertion below does not match observed behavior and the observed behavior is not an error, change the assertion to the observed value and say so in the commit body.

- [ ] **Step 1: Write the tests**

```python
"""Contract of the HTTP server as it behaves today, LLM faked.

The interp.local sandbox talks to /openai/chat/completions; a previous
postmortem found that endpoint dead for five reasons at once. These tests
lock the streaming and non-streaming shapes, the code-approval round trip,
the websocket protocol, and the insecure /run route so the Phase 2 split of
async_core cannot change them silently.
"""

import json
import os

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
        ai.terminal.stop()
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

    with _chat(client, "what is 6*7", stream=True) as r:
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
```

- [ ] **Step 2: Run them**

Run: `.venv/bin/python -m pytest -q tests/core/test_server_contract.py -p no:cacheprovider -x`
Expected: PASS, or a mismatch that reflects real current behavior. If the websocket rejects the `auth` message because no key is configured, drop the two auth lines and note it in the commit body; if the first approval response does not contain the prompt text verbatim, assert on `"Reply with exactly"` instead. Do not change server code in this task.

- [ ] **Step 3: Commit**

```bash
git add tests/core/test_server_contract.py
git commit -m "test: characterize the HTTP server before restructuring it

Locks the shapes the sandbox depends on: heartbeat, the OpenAI-compatible
endpoint in streaming and non-streaming form, the yes/no code-approval
round trip, the insecure /run route, /settings reads, and the websocket
LMC protocol through to the complete status. All with the LLM faked.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 5: CLI smoke test

**Files:**
- Create: `tests/terminal_interface/test_cli_smoke.py`
- Create: `tests/support/fake_profile.py`

**Interfaces:**
- Produces: a subprocess-driven check that `interpreter --stdin -y --profile <fake_profile>` reaches the model, runs code, and prints its output.

- [ ] **Step 1: Write the fake profile**

`tests/support/fake_profile.py` (a python profile; the loader strips the `from interpreter import interpreter` line and exec's the rest with `interpreter` in scope):

```python
"""Profile that replaces the model with a two-reply script. Used by the CLI smoke test."""

from interpreter import interpreter

_replies = ["Sure.\n```python\nprint(21 * 2)\n```", "The answer is 42."]


def _fake_completions(**params):
    text = _replies.pop(0)
    yield {"choices": [{"delta": {"content": text}}]}
    yield {"choices": [{"delta": {}}]}


interpreter.llm.completions = _fake_completions
interpreter.llm.model = "openai/fake"
interpreter.llm.api_key = "fake"
interpreter.llm.supports_functions = False
interpreter.llm.supports_vision = False
interpreter.llm.context_window = 32000
interpreter.llm.max_tokens = 4096
interpreter.llm._is_loaded = True
interpreter.offline = True
interpreter.disable_telemetry = True
interpreter.auto_run = True
```

- [ ] **Step 2: Write the test**

```python
"""The CLI must start, take a message on stdin, run code, and print the result."""

import os
import subprocess
import sys
from pathlib import Path

PROFILE = Path(__file__).resolve().parents[1] / "support" / "fake_profile.py"


def test_cli_stdin_mode_runs_a_turn(tmp_path):
    """interpreter --stdin reads one line, drives a turn, and prints console output.

    Runs the real entry point in a subprocess with a profile that fakes the
    model, so argument parsing, profile loading, rendering, and execution are
    all exercised end to end without a provider.
    """
    code = (
        "import sys;"
        "sys.argv=['interpreter','--stdin','-y','--disable_telemetry','--plain','--profile',sys.argv[1]];"
        "from interpreter.terminal_interface.start_terminal_interface import main;"
        "main()"
    )
    env = dict(os.environ, HOME=str(tmp_path), TERM="dumb", NO_COLOR="1")
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(PROFILE)],
        input="what is 21*2\n",
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "42" in result.stdout, result.stdout
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/python -m pytest -q tests/terminal_interface/test_cli_smoke.py -p no:cacheprovider -x`
Expected: PASS. If it hangs, run the same command by hand with `timeout 60` and read where it stops (`interpreter -h` lists `--plain` and `--stdin`; the profile-migration prompt is the usual culprit and is suppressed by `HOME=tmp_path`). If it fails because `--profile` does not accept an absolute path, check `get_profile` in `interpreter/terminal_interface/profiles/profiles.py:73`; `os.path.join(profile_dir, "/abs/path")` returns the absolute path, so it should.

- [ ] **Step 4: Commit**

```bash
git add tests/support/fake_profile.py tests/terminal_interface/test_cli_smoke.py
git commit -m "test: smoke-test the CLI end to end with a faked model

Drives the real console entry point in a subprocess: argument parsing,
profile loading, stdin mode, plain rendering, code execution, and output.
The profile replaces llm.completions so no provider is needed.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 6: Token harness and recorded baseline

**Files:**
- Create: `tests/support/scripted_session.py`
- Create: `tests/core/test_token_budget.py`
- Create: `docs/superpowers/specs/2026-09-11-baseline.md`

**Interfaces:**
- Consumes: `install_fake_llm` from Task 3.
- Produces: `SCRIPT: list[tuple[str, list[str]]]` (user message, scripted replies); `run_scripted_session(interpreter) -> list[int]` (prompt tokens per request, in order); `count_prompt_tokens(messages) -> int`; constant `TOKEN_BUDGET` in the test, tightened in Phase 3.

- [ ] **Step 1: Write the harness**

`tests/support/scripted_session.py`:

```python
"""A fixed six-turn session for measuring prompt tokens per request.

Turns: run python, produce long console output, hit an error and recover,
write a file, read it back, and a plain chat reply. The fake replies are
constant, so any change in token counts comes from the code, not the model.
"""

import tiktoken

from tests.support.fake_llm import install_fake_llm

SCRIPT = [
    ("print hello", ["```python\nprint('hello')\n```", "Done."]),
    (
        "print 400 numbered lines",
        ["```python\nfor i in range(400):\n    print('line', i)\n```", "That was long."],
    ),
    (
        "import a module that does not exist, then recover",
        [
            "```python\nimport nonexistent_module_xyz\n```",
            "```python\nprint('recovered')\n```",
            "Recovered.",
        ],
    ),
    (
        "write the word ok to notes.txt",
        ["```python\nopen('notes.txt', 'w').write('ok')\n```", "Written."],
    ),
    ("read notes.txt back", ["```python\nprint(open('notes.txt').read())\n```", "It says ok."]),
    ("thanks", ["You're welcome."]),
]

_enc = tiktoken.get_encoding("cl100k_base")


def count_prompt_tokens(messages):
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(_enc.encode(content))
        elif isinstance(content, list):  # vision-style parts
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(_enc.encode(part["text"]))
    return total


def run_scripted_session(interpreter, workdir):
    """Drive SCRIPT through the interpreter; return prompt tokens per request."""
    import os

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        fake = install_fake_llm(interpreter, [r for _, replies in SCRIPT for r in replies])
        for user_message, _ in SCRIPT:
            interpreter.chat(user_message, display=False, stream=False)
    finally:
        os.chdir(cwd)
    return [count_prompt_tokens(call["messages"]) for call in fake.calls]
```

- [ ] **Step 2: Write the test with a placeholder-free budget procedure**

`tests/core/test_token_budget.py`:

```python
"""Upper bound on prompt tokens sent during the scripted session.

Phase 0 sets TOKEN_BUDGET to the measured total plus ten percent so any
accidental growth fails; Phase 3 lowers it as the prompts shrink.
"""

from tests.support.scripted_session import SCRIPT, run_scripted_session

TOKEN_BUDGET = 10**9  # replaced in Step 3 with the measured value


def test_scripted_session_stays_under_token_budget(offline_interpreter, tmp_path):
    per_request = run_scripted_session(offline_interpreter, tmp_path)

    expected_requests = sum(len(replies) for _, replies in SCRIPT)
    assert len(per_request) == expected_requests

    total = sum(per_request)
    print(f"\nprompt tokens per request: {per_request}\ntotal: {total}")
    assert total <= TOKEN_BUDGET, f"{total} prompt tokens exceeds budget {TOKEN_BUDGET}"
```

- [ ] **Step 3: Measure, then set the budget**

Run: `.venv/bin/python -m pytest -q tests/core/test_token_budget.py -p no:cacheprovider -s`
Read the printed `total`. Set `TOKEN_BUDGET = <ceil(total * 1.1)>` in the file, and run again.
Expected: PASS.

- [ ] **Step 4: Record the baseline document**

Measure startup:

```bash
for i in 1 2 3; do /usr/bin/time -f "%e s import interpreter" .venv/bin/python -c "import interpreter" ; done
for i in 1 2 3; do /usr/bin/time -f "%e s interpreter --help" .venv/bin/python -c "import sys; sys.argv=['interpreter','--help']; from interpreter.terminal_interface.start_terminal_interface import main; main()" >/dev/null; done
.venv/bin/python -X importtime -c "import interpreter" 2>&1 | sort -t'|' -k2 -n | tail -5
```

Write `docs/superpowers/specs/2026-09-11-baseline.md` with: the test counts before and after Task 2, the ruff error count, the per-request token list and total from Step 3, the system-prompt sizes from the spec (2,201 / 4,125 tokens), and the median of the three startup timings for each command.

- [ ] **Step 5: Commit**

```bash
git add tests/support/scripted_session.py tests/core/test_token_budget.py docs/superpowers/specs/2026-09-11-baseline.md
git commit -m "test: measure prompt tokens over a fixed scripted session

A six-turn script (run code, long output, error and recovery, write a
file, read it, chat) drives the loop with the fake model and counts prompt
tokens per request. The budget is the measured total plus ten percent so
growth fails now and Phase 3 can ratchet it down. The baseline document
records tests, lint, tokens, and startup timings for comparison.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Phase 1: cleanse

### Task 7: Delete dead files

**Files:**
- Delete: `interpreter/core/archived_server_1.py`, `interpreter/core/archived_server_2.py`, `interpreter/computer_use/unused_markdown.py`, `test_refactoring.py`, `test_state.py`, `test_web_answer.py`, `test_web_fetch.py`, `test_web_search.py`, `test_web_search_engines.py`, `file.txt`, `.devin/config.local.json`, `.cursor/rules/conda-oi-classic.mdc`, `scripts/dashscope_usage_probe.py`

- [ ] **Step 1: Prove nothing references them**

Run:
```bash
grep -rnE "archived_server|unused_markdown|dashscope_usage_probe|test_web_answer|test_web_fetch|test_web_search|test_refactoring|test_state" --include=*.py --include=*.md --include=*.toml --include=*.yml --include=*.yaml . | grep -v "^./.venv" | grep -v "^./docs/superpowers"
```
Expected: no output.

- [ ] **Step 2: Delete and verify**

```bash
git rm -q interpreter/core/archived_server_1.py interpreter/core/archived_server_2.py interpreter/computer_use/unused_markdown.py test_refactoring.py test_state.py test_web_answer.py test_web_fetch.py test_web_search.py test_web_search_engines.py file.txt .devin/config.local.json .cursor/rules/conda-oi-classic.mdc scripts/dashscope_usage_probe.py
.venv/bin/python -m pytest -q -m "not integration" -p no:cacheprovider tests
```
Expected: same pass count as after Task 6, 0 failed.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove dead files

Two archived server implementations nothing imports, an unused markdown
helper, six scratch scripts at the repo root, a stray file.txt, editor
config for tools this fork does not use, and a one-off DashScope probe.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 8: Packaging: PEP 621, hatchling, uv, Python 3.12+

**Files:**
- Modify: `pyproject.toml` (replace whole file)
- Delete: `poetry.lock`
- Create: `uv.lock` (generated)
- Modify: `.gitignore` (remove the `poetry.lock` line at the end)
- Modify: `.github/workflows/python-package.yml` (install via uv)

**Interfaces:**
- Produces: `uv sync --group dev` creates `.venv`; `uv run pytest` and `uv run ruff` work.

- [ ] **Step 1: Write the new pyproject.toml**

```toml
[project]
name = "open-interpreter"
version = "0.4.3"
description = "Let language models run code"
readme = "README.md"
requires-python = ">=3.12,<4"
authors = [
    { name = "Killian Lucas", email = "killian@openinterpreter.com" },
]
dependencies = [
    "inquirer>=3.1.3,<4.0.0",
    "pyyaml>=6.0.1,<7.0.0",
    "rich>=13.4.2,<14.0.0",
    "tokentrim>=0.1.13,<0.2.0",
    "wget>=3.2,<4.0",
    "psutil>=5.9.6,<6.0.0",
    "pyreadline3>=3.4.1,<4.0.0; sys_platform == 'win32'",
    "html2image>=2.0.4.3,<3.0.0",
    "send2trash>=1.8.2,<2.0.0",
    "ipykernel>=6.26.0,<7.0.0",
    "jupyter-client>=8.6.0,<9.0.0",
    "matplotlib>=3.8.2,<4.0.0",
    "toml>=0.10.2,<0.11.0",
    "tiktoken>=0.8.0,<1.0.0",
    "platformdirs>=4.2.0,<5.0.0",
    "pydantic>=2.6.4,<3.0.0",
    "pyperclip>=1.9.0,<2.0.0",
    "yaspin>=3.0.2,<4.0.0",
    "shortuuid>=1.0.13,<2.0.0",
    "litellm>=1.41.26,<2.0.0",
    "bc-detect-secrets>=1.5,<2.0",
    "starlette>=0.42,<0.46",
    "html2text>=2024.2.26,<2025.0.0",
    "markdown-it-py>=3.0.0,<4.0.0",
    "selenium>=4.24.0,<5.0.0",
    "webdriver-manager>=4.0.2,<5.0.0",
    "anthropic>=0.37.1,<0.38.0",
    "pyautogui>=0.9.54,<0.10.0",
    "fastapi>=0.115.8,<0.116.0",
    "uvicorn>=0.30.1,<0.31.0",
    "janus>=1.0.0,<2.0.0",
    "chardet>=5.0,<6.0",
    "babel>=2.12,<3.0",
    "requests>=2.31,<3.0",
]

[project.optional-dependencies]
os = [
    "opencv-python>=4.8.1.78,<5.0.0.0",
    "pyautogui>=0.9.54,<0.10.0",
    "plyer>=2.1.0,<3.0.0",
    "pywinctl>=0.3,<0.4",
    "pytesseract>=0.3.10,<0.4.0",
    "sentence-transformers>=2.5.1,<3.0.0",
    "ipywidgets>=8.1.2,<9.0.0",
    "timm>=0.9.16,<0.10.0",
    "screeninfo>=0.8.1,<0.9.0",
]
safe = [
    "semgrep>=1.52.0,<2.0.0",
]
local = [
    "opencv-python>=4.8.1.78,<5.0.0.0",
    "pytesseract>=0.3.10,<0.4.0",
    "torch>=2.2.1,<3.0.0",
    "transformers==4.41.2",
    "einops>=0.8.0,<0.9.0",
    "torchvision>=0.18.0,<0.19.0",
    "easyocr>=1.7.1,<2.0.0",
]
# fastapi, uvicorn, janus are core deps; kept so `pip install ".[server]"` keeps working
server = [
    "fastapi>=0.115.8,<0.116.0",
    "uvicorn>=0.30.1,<0.31.0",
    "janus>=1.0.0,<2.0.0",
]

[project.scripts]
i = "interpreter.terminal_interface.start_terminal_interface:main"
interpreter = "interpreter.terminal_interface.start_terminal_interface:main"
wtf = "scripts.wtf:main"
interpreter-classic = "interpreter.terminal_interface.start_terminal_interface:main"

[dependency-groups]
dev = [
    "pytest>=8",
    "ruff>=0.16",
    "websockets>=13.1",
    "pre-commit>=3.5",
]

[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["interpreter", "scripts"]

[tool.ruff]
line-length = 120
target-version = "py312"
exclude = ["examples"]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F"]
ignore = [
    "E401",
    "E402",
    "E711",
    "E712",
    "E713",
    "E721",
    "E722",
    "E741",
    "F401",
    "F541",
    "F811",
    "F841",
]

[tool.ruff.lint.per-file-ignores]
"interpreter/core/computer/display/display.py" = ["F821"]
"interpreter/core/computer/display/point/point.py" = ["F821"]
"interpreter/core/llm/utils/convert_to_openai_messages.py" = ["F821"]
"interpreter/core/respond.py" = ["F821"]
"tests/test_interpreter.py" = ["F821"]
```

(The ruff tables are copied unchanged; Task 10 rewrites them.)

- [ ] **Step 2: Lock and sync**

```bash
git rm -q poetry.lock
sed -i '/^poetry\.lock$/d' .gitignore
rm -rf .venv
~/.local/bin/uv sync --group dev --extra server
~/.local/bin/uv run python -c "import interpreter; print('import ok')"
```
Expected: `uv.lock` created, `.venv` rebuilt, `import ok`. If `uv sync` cannot find a Python 3.12+, run `~/.local/bin/uv python install 3.14` first.

- [ ] **Step 3: Run the suite under uv**

Run: `~/.local/bin/uv run pytest -q -m "not integration" -p no:cacheprovider tests`
Expected: 0 failed, same pass count as Task 7.

- [ ] **Step 4: Switch CI to uv**

In `.github/workflows/python-package.yml`, replace the `Install` and `Test` steps of the `test` job with:

```yaml
      - uses: astral-sh/setup-uv@v6
      - name: Install
        run: uv sync --group dev --extra server --python ${{ matrix.python-version }}
      - name: Test
        run: uv run pytest -m "not integration" -q
```

and the `Ruff` step of the `lint` job with:

```yaml
      - uses: astral-sh/setup-uv@v6
      - name: Ruff
        run: |
          uv sync --group dev --python 3.12
          uv run ruff check interpreter tests scripts
```

Validate: `~/.local/bin/uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/python-package.yml')); print('ok')"`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .github/workflows/python-package.yml
git commit -m "build: PEP 621 metadata, hatchling, uv lock, Python 3.12 floor

Drops the duplicated [tool.poetry] tables and poetry.lock in favor of a
single [project] table, hatchling as the build backend, and uv.lock. Six
dependencies nothing imports are removed (astor, git-python, six,
google-generativeai, typer, setuptools) and requests, which the code uses
but never declared, is added. The Python floor moves to 3.12; the sandbox
runs 3.14 and every current distro ships 3.12. CI installs with uv.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 9: Installers install the fork with uv

The spec also wants the docs' install commands switched to uv; that lands with the Phase 5 docs rewrite, not here.

**Files:**
- Modify: `installers/oi-linux-installer.sh`, `installers/oi-mac-installer.sh`, `installers/oi-windows-installer.ps1`, `installers/oi-windows-installer-conda.ps1`

- [ ] **Step 1: Find the install lines**

Run: `grep -n "pip install\|open-interpreter" installers/*`

- [ ] **Step 2: Replace each PyPI install of `open-interpreter` with the fork**

Shell installers: replace the line that runs `pip install open-interpreter` (any extras, any flags) with:

```bash
uv tool install "open-interpreter[server] @ git+https://github.com/velinxs/open-interpreter.git@integration"
```

and, directly before it, ensure uv exists:

```bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
```

PowerShell installers: replace the equivalent `pip install open-interpreter` line with:

```powershell
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" }
uv tool install "open-interpreter[server] @ git+https://github.com/velinxs/open-interpreter.git@integration"
```

Leave everything else in the scripts (conda setup, PATH notes) as it is.

- [ ] **Step 3: Syntax-check the shell scripts**

Run: `bash -n installers/oi-linux-installer.sh && bash -n installers/oi-mac-installer.sh && echo ok`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add installers
git commit -m "build: installers install the fork from GitHub with uv

The scripts installed the PyPI package, which is the upstream project, not
this fork. They now install from velinxs/open-interpreter at integration
with uv tool install, bootstrapping uv when it is missing.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 10: Ruff as the single linter and formatter

**Files:**
- Modify: `pyproject.toml` (ruff tables; drop `[tool.black]`, `[tool.isort]`)
- Modify: `.pre-commit-config.yaml` (replace whole file)
- Modify: `interpreter/core/toolbox/display/display.py:478`
- Modify: `interpreter/core/toolbox/display/point/point.py:446-452`
- Modify: `interpreter/terminal_interface/terminal_interface.py:520-527`
- Modify: many files via autofix

Four commits: the two undefined names and the duplicate keys; the rule change with safe autofixes; pyupgrade; format.

- [ ] **Step 1: Fix the six real errors**

`display.py:478`: change `monitors = get_monitors()` to `monitors = screeninfo.get_monitors()` (the module already does `screeninfo = lazy_import("screeninfo")` at line 39).

`point.py`: directly after `fast_model = True` (line 446) add:

```python
# Where the fine-tuned ViT weights are cached when fast_model is off.
model_path = os.path.join(platformdirs.user_cache_dir("open-interpreter"), "point_vit_siglip.pth")
```

and add `import platformdirs` next to the file's other top-level imports (line 7 area). `os` is imported at line 453; move that `import os` up with the top-level imports so the new line can use it.

`terminal_interface.py:520-527`: delete the second `"cmd": ".bat",` and the second `"bash": ".sh",` lines.

Run: `.venv/bin/ruff check interpreter tests scripts` (or `uv run ruff check ...`)
Expected: `All checks passed!`

```bash
git add interpreter/core/toolbox/display/display.py interpreter/core/toolbox/display/point/point.py interpreter/terminal_interface/terminal_interface.py
git commit -m "fix: resolve the undefined names and duplicate keys ruff reported

get_displays() called get_monitors without the screeninfo prefix the rest
of the module uses; point.py used model_path without ever defining it; the
editor extension map listed cmd and bash twice.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 2: Enable E, F, W, I with safe autofixes**

Replace the ruff tables in `pyproject.toml` with:

```toml
[tool.ruff]
line-length = 120
target-version = "py312"
exclude = ["examples", "docs"]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "W", "I", "UP"]
ignore = [
    "E402",  # imports mid-file are deliberate in several modules (kernel setup, lazy heavy deps)
    "E722",  # bare except: cleaned per module in Phase 2
    "E731",
    "E741",
    "F401",  # unused imports: cleaned per module in Phase 2 (some are re-exports)
    "F811",
    "F841",
    "E711",
    "E712",
    "E713",
    "E721",
]
```

Delete the `[tool.black]` and `[tool.isort]` tables and the `[tool.ruff.lint.per-file-ignores]` table.

Run, in order:
```bash
uv run ruff check --select I --fix interpreter tests scripts
uv run pytest -q -m "not integration" -p no:cacheprovider tests
```
Expected: import blocks re-sorted; tests 0 failed. If a test breaks, the cause is an import-order dependency; fix it by leaving that block as it was with `# isort: skip` on the offending import line and note the file in the commit body.

Then:
```bash
uv run ruff check interpreter tests scripts
```
Expected: `All checks passed!`, or a short list of `W` warnings; fix those by hand (they are whitespace and newline issues).

```bash
git add -A
git commit -m "style: enable ruff import sorting and warnings, drop black and isort config

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 3: Apply pyupgrade rules for 3.12**

```bash
uv run ruff check --select UP --fix interpreter tests scripts
uv run ruff check --select UP --unsafe-fixes --fix interpreter tests scripts
uv run pytest -q -m "not integration" -p no:cacheprovider tests
```
Expected: `Optional[X]` becomes `X | None`, `typing.List` becomes `list`, redundant `object` bases and encodings go; tests 0 failed. Review the diff with `git diff --stat` and skim any file with more than 50 changed lines for a semantic change (there should be none; UP fixes are syntactic).

```bash
git add -A
git commit -m "style: apply pyupgrade fixes for the Python 3.12 floor

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 4: Format once, and make pre-commit use ruff**

```bash
uv run ruff format interpreter tests scripts
uv run pytest -q -m "not integration" -p no:cacheprovider tests
```

Replace `.pre-commit-config.yaml` with:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.16.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

```bash
git add -A
git commit -m "style: format the codebase with ruff format

One-time mechanical reformat; no logic changes. Pre-commit now runs ruff
and ruff format instead of black and isort.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 11: A package import with no side effects

**Files:**
- Modify: `interpreter/__init__.py` (replace whole file)
- Modify: `interpreter/terminal_interface/start_terminal_interface.py` (add `_run_computer_use_mode`, call it first in `main`)
- Create: `tests/test_import_hygiene.py`

**Interfaces:**
- Produces: module-level `__getattr__` in `interpreter/__init__.py` serving `interpreter`, `toolbox`, `ai2`, `OpenInterpreter`, `AsyncInterpreter`, `BaseLanguage` on demand; `start_terminal_interface._run_computer_use_mode()`.

- [ ] **Step 1: Write the failing tests**

`tests/test_import_hygiene.py`:

```python
"""Importing the package must be cheap and must not run anything.

Before this, `import interpreter` constructed OpenInterpreter, and with
`--os` on argv it made an HTTP call to PyPI and started the computer-use
loop before any CLI code ran.
"""

import subprocess
import sys


def _run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)


def test_import_does_not_construct_the_singleton():
    r = _run("import sys, interpreter; assert 'interpreter.core.core' not in sys.modules; print('ok')")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_singleton_is_built_on_first_access():
    r = _run("from interpreter import interpreter; print(type(interpreter).__name__)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "OpenInterpreter"


def test_public_names_still_importable():
    r = _run(
        "from interpreter import OpenInterpreter, AsyncInterpreter, BaseLanguage, toolbox, ai2; print('ok')"
    )
    assert r.returncode == 0, r.stderr


def test_os_flag_is_ignored_by_import():
    r = _run("import sys; sys.argv=['x','--os']; import interpreter; print('ok')")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q tests/test_import_hygiene.py -p no:cacheprovider`
Expected: the first and fourth fail (the fourth may hang; the 60 s timeout turns that into a failure).

- [ ] **Step 3: Rewrite `interpreter/__init__.py`**

```python
"""Open Interpreter.

Importing this package has no side effects. The module-level singleton
(`from interpreter import interpreter`) and the public classes are created
on first access via PEP 562 so `import interpreter` stays cheap and never
touches the network or starts a mode.
"""

import importlib
import os
import warnings

# OpenRouter (via LiteLLM): optional HTTP-Referer and app title for openrouter.ai rankings.
os.environ.setdefault("OR_SITE_URL", "https://github.com/velinxs/open-interpreter")
os.environ.setdefault("OR_APP_NAME", "Open Interpreter")

# Suppress pydantic warning from litellm about fields being removed in V2
warnings.filterwarnings(
    "ignore",
    message="Valid config keys have changed in V2:*",
    module="pydantic.*",
)

_CLASSES = {
    "OpenInterpreter": ("interpreter.core.core", "OpenInterpreter"),
    "AsyncInterpreter": ("interpreter.core.async_core", "AsyncInterpreter"),
    "BaseLanguage": ("interpreter.core.terminal.base_language", "BaseLanguage"),
}

_singleton = None


def _get_singleton():
    global _singleton
    if _singleton is None:
        module = importlib.import_module("interpreter.core.core")
        _singleton = module.OpenInterpreter()
    return _singleton


def __getattr__(name):
    if name == "interpreter":
        return _get_singleton()
    if name == "toolbox":
        return _get_singleton().toolbox
    if name == "ai2":
        return _get_singleton().toolbox.ai2
    if name in _CLASSES:
        module_name, attr = _CLASSES[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module 'interpreter' has no attribute {name!r}")


__all__ = ["interpreter", "toolbox", "ai2", "OpenInterpreter", "AsyncInterpreter", "BaseLanguage"]
```

- [ ] **Step 4: Move the `--os` block into the CLI entry**

In `interpreter/terminal_interface/start_terminal_interface.py`, add above `def main():`:

```python
def _run_computer_use_mode():
    """`interpreter --os`: the computer-use loop, previously started from interpreter/__init__.py at import time."""
    import sys

    from rich import print as rich_print
    from rich.markdown import Markdown
    from rich.rule import Rule

    def print_markdown(message):
        for line in message.split("\n"):
            line = line.strip()
            if line == "":
                print("")
            elif line == "---":
                rich_print(Rule(style="white"))
            else:
                try:
                    rich_print(Markdown(line))
                except UnicodeEncodeError:
                    print("Error displaying line:", line)
        if "\n" not in message and message.startswith(">"):
            print("")

    def check_for_update():
        from importlib.metadata import version as installed_version

        import requests
        from packaging import version

        response = requests.get("https://pypi.org/pypi/open-interpreter/json", timeout=5)
        latest = response.json()["info"]["version"]
        return version.parse(latest) > version.parse(installed_version("open-interpreter"))

    try:
        if check_for_update():
            print_markdown(
                "> **A new version of Open Interpreter is available.**\n>Please run: `pip install --upgrade open-interpreter`\n\n---"
            )
    except Exception:
        pass  # offline or PyPI unreachable: the check is advisory

    if "--voice" in sys.argv:
        print("Coming soon...")

    from interpreter.computer_use.loop import run_async_main

    run_async_main()
```

and make the first statement of `main()`:

```python
    if "--os" in sys.argv:
        _run_computer_use_mode()
        return
```

(`sys` is already imported at the top of that module; confirm with `grep -n "^import sys" interpreter/terminal_interface/start_terminal_interface.py`.) The `packaging` import comes with litellm; leave it.

- [ ] **Step 5: Run the new tests and the whole suite**

Run: `uv run pytest -q tests/test_import_hygiene.py -p no:cacheprovider && uv run pytest -q -m "not integration" -p no:cacheprovider tests`
Expected: all four pass; 0 failed overall. The profile defaults that do `from interpreter import interpreter` keep working because the loader strips that line, and `interpreter/core/terminal/terminal.py:125` and `toolbox/ai/ai.py:158` import inside functions, which resolve through `__getattr__`.

- [ ] **Step 6: Commit**

```bash
git add interpreter/__init__.py interpreter/terminal_interface/start_terminal_interface.py tests/test_import_hygiene.py
git commit -m "refactor: make importing the package side-effect free

interpreter/__init__.py constructed OpenInterpreter at import, and when
--os was on argv it called PyPI and started the computer-use loop before
the CLI parsed anything. The singleton and the public classes are now
served on first access through a module __getattr__, and the --os path
runs from the console entry point where it belongs. Public names are
unchanged.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 12: Heavy packages import lazily

**Files:**
- Modify: `tests/test_import_hygiene.py` (add one test)
- Modify: `interpreter/core/respond.py:10-30`
- Modify: `interpreter/core/llm/llm.py:9,20` and the functions that use `litellm` / `tt`
- Modify: `interpreter/core/toolbox/ai2.py:5`
- Modify: `interpreter/core/toolbox/ai/ai.py:3`
- Modify: `interpreter/core/llm/utils/cache_aware_trim.py:3`
- Modify: `interpreter/core/toolbox/browser/browser.py:6-7`, `interpreter/core/toolbox/browser/browser_next.py:8-11`
- Modify: `interpreter/core/terminal/languages/jupyter_language.py:21`

**Interfaces:**
- Produces: `OpenInterpreter()` can be constructed without importing litellm, selenium, pyautogui, matplotlib, tiktoken, anthropic, cv2, or torch.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_import_hygiene.py`:

```python
HEAVY = ("litellm", "selenium", "pyautogui", "matplotlib", "tiktoken", "anthropic", "cv2", "torch")


def test_constructing_interpreter_does_not_import_heavy_packages():
    """Heavy packages load at their point of use, not when the object is built.

    litellm alone costs over a second; selenium and pyautogui need a display
    stack. None of them are needed to construct the interpreter or to serve
    --help, and the sandbox is headless.
    """
    code = (
        "import sys; from interpreter import OpenInterpreter; OpenInterpreter();"
        f"print([m for m in {HEAVY!r} if m in sys.modules])"
    )
    r = _run(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", r.stdout
```

- [ ] **Step 2: Run it to see the offenders**

Run: `uv run pytest -q tests/test_import_hygiene.py::test_constructing_interpreter_does_not_import_heavy_packages -p no:cacheprovider`
Expected: FAIL printing a list such as `['litellm', 'selenium', 'tiktoken']`.

- [ ] **Step 3: Push each import into the functions that use it**

Known module-level sites and their fix:

- `interpreter/core/respond.py:10` `import litellm`, and lines 26-30 build `_TEMPORARY_ERRORS` from `litellm.exceptions` at import. Replace with a function:

```python
def _temporary_error_types():
    import litellm

    names = ("ServiceUnavailableError", "InternalServerError")
    return tuple(getattr(litellm.exceptions, n) for n in names if hasattr(litellm.exceptions, n))
```

and inside `_is_temporary_provider_error` use `isinstance(error, _temporary_error_types())`. Any other `litellm.` reference in the file gets a local `import litellm` at the top of its function.

- `interpreter/core/llm/llm.py:9,20`: delete the module-level `import litellm` and `import tokentrim as tt`; add `import litellm` as the first line of every method or function in the file that references `litellm.` (`run`, `load`, `fixed_litellm_completions`, the registry helpers), and `import tokentrim as tt` at the top of the function that calls `tt.trim`. Find them with `grep -n "litellm\.\|tt\." interpreter/core/llm/llm.py`. Module-level statements such as `litellm.suppress_debug_info = True` move into `Llm.__init__` guarded by a local import.
- `interpreter/core/toolbox/ai2.py:5` and `interpreter/core/toolbox/ai/ai.py:3` and `interpreter/core/llm/utils/cache_aware_trim.py:3`: same treatment for `litellm` and `tiktoken`.
- `interpreter/core/toolbox/browser/browser.py:6-7` and `browser_next.py:8-11`: move the selenium imports inside the methods that create the driver (`grep -n "webdriver\.\|Service(\|Options(\|By\.\|Keys\." <file>` lists them).
- `interpreter/core/terminal/languages/jupyter_language.py:21` `import litellm  # noqa: F401`: read the comment above it. If it exists only to pre-import litellm for the kernel, delete it; if the kernel code string needs litellm importable inside the kernel, the module-level import in this process does nothing for that and it is still deleted.

After each file, re-run Step 2's command. When the list is empty, run `uv run python -X importtime -c "from interpreter import OpenInterpreter; OpenInterpreter()" 2>&1 | sort -t'|' -k2 -n | tail -8` and record the top entries in `docs/superpowers/specs/2026-09-11-baseline.md` under a "after Task 12" heading together with the new `import interpreter` timing (same three-run procedure as Task 6).

- [ ] **Step 4: Run everything**

Run: `uv run ruff check interpreter tests scripts && uv run pytest -q -m "not integration" -p no:cacheprovider tests`
Expected: clean; 0 failed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "perf: import litellm, selenium, tiktoken and friends at point of use

Constructing OpenInterpreter pulled in litellm (over a second), selenium
and pyautogui (display stack) and tiktoken before anything needed them.
Each now loads inside the function that uses it, so import and --help are
fast and a headless host without a display stack still starts.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Phase exit

- [ ] `uv run pytest -q -m "not integration" tests` reports 0 failed.
- [ ] `uv run ruff check interpreter tests scripts` is clean.
- [ ] `docs/superpowers/specs/2026-09-11-baseline.md` has before and after numbers for tests, lint, tokens, and startup.
- [ ] `git log --oneline integration..rework` reads as one idea per commit.
- [ ] Fast-forward `integration` to `rework` only after the operator has read the log: `git checkout integration && git merge --ff-only rework && git checkout rework`.
