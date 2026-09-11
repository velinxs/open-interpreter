# Fork Rework, Phase 2a: Runtime Core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop leaking kernels at exit, split the server into a package with a non-blocking event loop, split the LLM module into provider, reasoning and completions modules with hermetic tests, and add the headless session seam (`core/session.py`) plus a prompter hook that the server is rebuilt on and the channels will use.

**Architecture:** Every split is a `git mv` or a symbol-by-symbol move with the minimum edits to keep tests green, followed by a separate cleanup commit. Public paths keep working through thin shims. The Phase 0 contract tests (`tests/core/test_server_contract.py`, `tests/core/test_fake_llm.py`, `tests/terminal_interface/test_cli_smoke.py`, `tests/core/test_token_budget.py`) are the acceptance gate for every task.

**Tech Stack:** Python 3.12+, pytest, ruff, FastAPI/starlette, jupyter_client, psutil.

**Spec:** `docs/superpowers/specs/2026-09-11-fork-rework-design.md` (Phase 2)

## Global Constraints

- Every commit: `.venv/bin/ruff check interpreter tests scripts` clean; `.venv/bin/python -m pytest -q -m "not integration" -p no:cacheprovider tests` reports 0 failed (283 passed at the start of this plan).
- One idea per commit, Conventional Commits, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Follow-up fixes are folded with `git commit --fixup <sha>` then `GIT_SEQUENCE_EDITOR=true git rebase -q -i --autosquash <sha>~1` on a clean tree.
- Public import paths keep working: `interpreter.core.async_core.AsyncInterpreter`, `.Server`, `.create_router`, `.OPENAI_CODE_APPROVAL_PROMPT`, `.OPENAI_CODE_APPROVAL_DECLINED`; `interpreter.core.llm.llm.Llm`; `interpreter.core.core.OpenInterpreter`.
- Deliberate behavior changes in this plan, and no others: kernels and shell subprocesses die when the Python process exits; the server's async handlers no longer block the event loop while a turn runs; the two provider-error prompts in the loop go through `interpreter.prompter`.
- Deviations from the spec, on purpose: no `core/approval.py` (the allow/deny logic already lives in `core/utils/execution_allowlist.py`; wrapping it would add a layer, not remove one); `core/session.py` yields the existing LMC chunk dicts rather than a new typed event class; the `llm/utils/` modules stay where they are instead of being merged into `messages.py`/`trim.py`.
- Never `pkill -f <pattern>` from the tool shell; it matches the shell itself. Kill by PID from `pgrep -f '[i]pykernel_launcher'`.

---

### Task 1: Kernels and shell subprocesses die with the process

**Files:**
- Modify: `interpreter/core/terminal/terminal.py:61-66` (`Terminal.__init__`)
- Create: `tests/core/terminal/test_process_lifetime.py`

**Interfaces:**
- Produces: module-level `_LIVE_TERMINALS: weakref.WeakSet[Terminal]` and `_terminate_live_terminals()` registered with `atexit` in `terminal.py`. No public API change.

- [ ] **Step 1: Write the failing test**

```python
"""Language runtimes must not outlive the interpreter's process.

Terminal.stop() only interrupts running code; only Terminal.terminate() kills
the Jupyter kernel and shell subprocesses, and nothing called it at exit, so
every CLI session and every test left an ipykernel_launcher behind.
"""

import json
import subprocess
import sys
import time

import psutil

SCRIPT = """
import json, sys
import psutil
from interpreter import OpenInterpreter
i = OpenInterpreter()
i.offline = True
i.disable_telemetry = True
i.terminal.run("python", "print(1)")
i.terminal.run("shell", "echo 1")
children = [p.pid for p in psutil.Process().children(recursive=True)]
print(json.dumps(children))
sys.exit(0)
"""


def _alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_runtimes_are_terminated_at_interpreter_exit():
    """After a normal exit, every child the terminal spawned is gone within five seconds."""
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    children = json.loads(r.stdout.strip().splitlines()[-1])
    assert children, "the script should have spawned a kernel and a shell"

    deadline = time.time() + 5
    while time.time() < deadline and any(_alive(pid) for pid in children):
        time.sleep(0.2)
    survivors = [pid for pid in children if _alive(pid)]
    for pid in survivors:  # never leave them behind even when the test fails
        psutil.Process(pid).kill()
    assert not survivors, f"processes survived interpreter exit: {survivors}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest -q tests/core/terminal/test_process_lifetime.py -p no:cacheprovider`
Expected: FAIL with `processes survived interpreter exit: [...]`.

- [ ] **Step 3: Register termination at exit**

In `interpreter/core/terminal/terminal.py` add near the top imports:

```python
import atexit
import weakref
```

and directly above `class Terminal:`:

```python
# Every live Terminal, so the process can shut their kernels and shells down at
# exit. A WeakSet keeps this registry from extending any Terminal's lifetime.
_LIVE_TERMINALS = weakref.WeakSet()


def _terminate_live_terminals():
    for terminal in list(_LIVE_TERMINALS):
        try:
            terminal.terminate()
        except Exception:
            pass  # exit must never fail because a runtime was already gone


atexit.register(_terminate_live_terminals)
```

and as the last line of `Terminal.__init__`:

```python
        _LIVE_TERMINALS.add(self)
```

- [ ] **Step 4: Run the test and the suite**

Run: `.venv/bin/python -m pytest -q tests/core/terminal/test_process_lifetime.py -p no:cacheprovider && .venv/bin/python -m pytest -q -m "not integration" -p no:cacheprovider tests | tail -1`
Expected: PASS; 0 failed. Then `pgrep -f '[i]pykernel_launcher' | wc -l` prints `0`.

- [ ] **Step 5: Commit**

```bash
git add interpreter/core/terminal/terminal.py tests/core/terminal/test_process_lifetime.py
git commit -F - <<'MSG'
fix: terminate kernels and shell subprocesses when the process exits

Terminal.stop() only interrupts running code; Terminal.terminate() is what
shuts the Jupyter kernel and the shell subprocesses down, and nothing called
it at exit. Every CLI session and every test that ran code left an
ipykernel_launcher (and a bash, started in its own session) behind. Live
terminals are now tracked in a WeakSet and terminated from an atexit hook.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 2: Server package

**Files:**
- Move: `interpreter/core/async_core.py` -> `interpreter/server/app.py` (via `git mv`)
- Create: `interpreter/server/__init__.py`, `interpreter/server/interpreter.py`, `interpreter/server/auth.py`, `interpreter/server/openai_compat.py`
- Create: `interpreter/core/async_core.py` (shim)
- Modify: `tests/core/test_async_core.py` (imports)
- Modify: `interpreter/terminal_interface/start_terminal_interface.py` (only if it imports from `core.async_core`; `grep -n async_core` says)

**Interfaces:**
- Produces: `interpreter.server.AsyncInterpreter`, `interpreter.server.Server`, `interpreter.server.create_router`; `interpreter.server.openai_compat.create_openai_router(async_interpreter) -> APIRouter`; `interpreter.server.auth.authenticate_function`.

Two commits: the move, then the event-loop fix.

- [ ] **Step 1: Move the file and carve it into modules**

```bash
mkdir -p interpreter/server
git mv interpreter/core/async_core.py interpreter/server/app.py
```

Then, by symbol (current line numbers in the moved file):

- `interpreter/server/interpreter.py`: `class AsyncInterpreter` (55-276) with the imports it needs (`asyncio`, `json`, `os`, `threading`, `time`, `traceback`, `deque`, `datetime`, `janus` inside the same try/except as today, `OpenInterpreter` from `..core.core`, `should_require_execution_confirmation` from `..core.utils.execution_allowlist`, and `Server` imported lazily inside `__init__` as `from .app import Server` to avoid the import cycle).
- `interpreter/server/auth.py`: `authenticate_function` (277-292).
- `interpreter/server/openai_compat.py`: the four `OPENAI_*` constants (293-312), every `_openai*`/`_normalize*`/`_format_openai*`/`_is_openai*`/`_new_openai*`/`_lmc_chunk_to_openai_delta`/`_pending_code_language`/`_cancel_pending_code`/`_clear_openai_code_approval_wait` helper (314-555), the `ChatMessage` and `ChatCompletionRequest` models (947-956) moved to module level, and a new

  ```python
  def create_openai_router(async_interpreter):
      router = APIRouter()
      ...  # _openai_assistant_text_response, _stream_openai_assistant_text, _stream_openai_title,
           # openai_compatible_generator, and the @router.post("/openai/chat/completions") endpoint,
           # moved verbatim from create_router (958-1241)
      return router
  ```

  `last_start_time` (the `global` used by the endpoint) moves here as a module global.
- `interpreter/server/app.py` keeps: the imports block, `complete_message`, `create_router` with the remaining routes (`/heartbeat`, `/`, websocket, `POST /`, `/settings`, `/settings/{setting}`, `/run`, `/upload`, `/download/{filename}`) and, at its end, `router.include_router(create_openai_router(async_interpreter))`; and `class Server`. `should_require_execution_confirmation_for_code` moves with whichever module still references it (`grep -n` after the carve).
- `interpreter/server/__init__.py`:

  ```python
  from .app import Server, create_router
  from .interpreter import AsyncInterpreter

  __all__ = ["AsyncInterpreter", "Server", "create_router"]
  ```
- `interpreter/core/async_core.py` (new shim):

  ```python
  """Compatibility path. The server lives in interpreter.server since the rework."""

  from ..server import AsyncInterpreter, Server, create_router
  from ..server.auth import authenticate_function
  from ..server.openai_compat import (
      OPENAI_CODE_APPROVAL_DECLINED,
      OPENAI_CODE_APPROVAL_INVALID_TEMPLATE,
      OPENAI_CODE_APPROVAL_PROMPT,
      OPENAI_SHELL_OUTPUT_NOTE,
  )

  __all__ = [
      "AsyncInterpreter", "Server", "create_router", "authenticate_function",
      "OPENAI_CODE_APPROVAL_DECLINED", "OPENAI_CODE_APPROVAL_INVALID_TEMPLATE",
      "OPENAI_CODE_APPROVAL_PROMPT", "OPENAI_SHELL_OUTPUT_NOTE",
  ]
  ```
- `tests/core/test_async_core.py`: change the underscore-helper imports to `from interpreter.server.openai_compat import (...)` and keep `AsyncInterpreter, Server` from `interpreter.core.async_core` (that exercises the shim).
- `interpreter/__init__.py`: `_CLASSES["AsyncInterpreter"]` becomes `("interpreter.server", "AsyncInterpreter")`.

- [ ] **Step 2: Verify**

Run: `.venv/bin/ruff check --fix interpreter/server interpreter/core/async_core.py tests/core/test_async_core.py && .venv/bin/ruff format interpreter/server interpreter/core/async_core.py && .venv/bin/python -m pytest -q -m "not integration" -p no:cacheprovider tests | tail -1 && wc -l interpreter/server/*.py`
Expected: 0 failed; no server module over 600 lines (`app.py` about 450, `openai_compat.py` about 500, `interpreter.py` about 230).

- [ ] **Step 3: Commit the move**

```bash
git add -A
git commit -F - <<'MSG'
refactor: split the server into interpreter/server

async_core.py held the async wrapper, auth, the OpenAI-compatible
translation layer, every route, and the uvicorn Server in 1,329 lines. It
is now a package: interpreter.py (AsyncInterpreter), auth.py,
openai_compat.py (constants, LMC<->OpenAI helpers, request models, and the
/openai/chat/completions router), app.py (the remaining routes and Server).
interpreter.core.async_core stays as a re-export shim. No behavior change.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

- [ ] **Step 4: Write the failing event-loop test**

Append to `tests/core/test_server_contract.py`:

```python
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

    hb, latency, text = asyncio.run(scenario())
    assert hb == {"status": "alive"}
    assert "slow reply here" in text
    assert latency < 0.2, f"heartbeat waited {latency:.2f}s behind the streaming turn"
```

Run: `.venv/bin/python -m pytest -q tests/core/test_server_contract.py::test_streaming_turn_does_not_block_the_event_loop -p no:cacheprovider`
Expected: FAIL on the latency assertion (about 0.9 s).

- [ ] **Step 5: Take blocking work off the loop**

In `interpreter/server/openai_compat.py`:

- `from starlette.concurrency import iterate_in_threadpool, run_in_threadpool` next to the other starlette imports.
- In `chat_completion`: replace `time.sleep(5)` with `await asyncio.sleep(5)` and `time.sleep(0.1)` with `await asyncio.sleep(0.1)`.
- In `openai_compatible_generator`: `for chunk in chunk_iter:` becomes `async for chunk in iterate_in_threadpool(chunk_iter):`; the context-mode nudge loop's `for chunk in async_interpreter.chat(...)` becomes `async for chunk in iterate_in_threadpool(async_interpreter.chat(...))`. Drop the `await asyncio.sleep(0)` lines that only existed to yield control.
- In the non-stream branch of `chat_completion`: wrap the `for chunk in async_interpreter._respond_and_store(): ...` accumulation in a local function `collect()` that returns `content`, and call `content = await run_in_threadpool(collect)`.
- The title branch's `for chunk in async_interpreter.llm.run(...)` likewise moves into a local function run through `run_in_threadpool`.

- [ ] **Step 6: Verify and commit**

Run: `.venv/bin/ruff check interpreter/server tests && .venv/bin/python -m pytest -q -m "not integration" -p no:cacheprovider tests | tail -1`
Expected: 0 failed, the new test included.

```bash
git add -A
git commit -F - <<'MSG'
perf: keep the server's event loop free while a turn runs

The OpenAI-compatible handler iterated the turn generator inside the async
function and slept with time.sleep, so every other request (heartbeat, a
second client, the websocket) waited for the model and the code to finish.
Turns now run through starlette's threadpool iterators and the sleeps are
awaited. A regression test streams a slow turn and requires a concurrent
heartbeat to answer within 200 ms.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 3: LLM package

**Files:**
- Move: `interpreter/core/llm/run_text_llm.py` -> `interpreter/core/llm/text.py`; `interpreter/core/llm/run_tool_calling_llm.py` -> `interpreter/core/llm/tool_calling.py`
- Create: `interpreter/core/llm/completions.py`, `interpreter/core/llm/providers.py`, `interpreter/core/llm/reasoning.py`
- Modify: `interpreter/core/llm/llm.py`, `interpreter/core/utils/system_debug_info.py:89`, `tests/core/test_text_llm_parser.py:10`, `tests/core/test_llm_provider_routing.py`, `tests/core/test_profile_reasoning_settings.py`

**Interfaces:**
- Produces: `completions.fixed_litellm_completions(**params)`; `providers.configure(llm)` (the body of `Llm.load()` after the `_is_loaded` guard, mutating `llm` exactly as before), `providers.openrouter_model_entry(model)`, `providers.openrouter_supports_vision(model)` with the module caches `_openrouter_model_entries`, `_warned_mandatory_reasoning`, `_warned_unsupported_effort`; `reasoning.apply_reasoning_params(llm, params, stream_options, model)` returning nothing and mutating `params`/`stream_options` exactly as lines 386-473 of `run()` do today.

Four commits: renames; completions; providers; reasoning (with the hermetic test rewrite).

- [ ] **Step 1: Rename the runners**

```bash
git mv interpreter/core/llm/run_text_llm.py interpreter/core/llm/text.py
git mv interpreter/core/llm/run_tool_calling_llm.py interpreter/core/llm/tool_calling.py
```

Update the three importers (`llm.py:18,34`, `system_debug_info.py:89`, `test_text_llm_parser.py:10`) to `.text` / `.tool_calling`; keep the imported names `run_text_llm` and `run_tool_calling_llm` unchanged so `monkeypatch.setattr(llm_mod, "run_text_llm", ...)` in the reasoning tests still binds.

Run: `.venv/bin/python -m pytest -q -m "not integration" -p no:cacheprovider tests | tail -1`; commit `refactor: rename the LLM runner modules to text.py and tool_calling.py`.

- [ ] **Step 2: Move the completions wrapper**

Move `fixed_litellm_completions` (llm.py 746 to end) and any helper it alone uses to `interpreter/core/llm/completions.py`, with `_litellm()` duplicated there as a private helper (it is four lines). In `llm.py`: `from .completions import fixed_litellm_completions`.

Run the suite; commit `refactor: move the litellm completions wrapper to llm/completions.py`.

- [ ] **Step 3: Move provider routing**

In `providers.py`:

```python
"""Provider-specific routing: DashScope and DeepSeek defaults, Ollama tags and
num_ctx, the OpenRouter model registry and its vision/reasoning metadata."""
```

Move the body of `Llm.load()` (everything after `if self._is_loaded: return`) into `def configure(llm):` with `self` renamed to `llm`; `Llm.load()` becomes:

```python
    def load(self):
        if self._is_loaded:
            return
        configure(self)
```

Move `_openrouter_model_entry` and `_openrouter_supports_vision` to module functions `openrouter_model_entry(model)` / `openrouter_supports_vision(model)` together with the module-level caches they use (`grep -n "_openrouter_model_entries\|_warned_" interpreter/core/llm/llm.py` lists them). Keep two one-line methods on `Llm` that delegate, so `run()` does not change.

Run the suite; commit `refactor: move provider routing to llm/providers.py`.

- [ ] **Step 4: Move reasoning parameters and make the six provider tests hermetic**

Move lines 386-473 of `run()` (from `# Reasoning tokens:` through the end of the `reasoning_effort` handling) into `reasoning.apply_reasoning_params(llm, params, stream_options, model)`; `run()` calls it at the same point.

Rewrite the six `@pytest.mark.network` tests so they never reach a provider: each installs `install_fake_llm(interpreter, ["ok"])` first (from `tests.support.fake_llm`) and then sets the specific `model`, `supports_vision` and `_is_loaded = False` the test needs, so `load()`/`configure()` runs but `completions` is the fake. The reasoning tests assert on `fake.calls[0]` params instead of patching `run_text_llm`. Remove the `network` markers from those six tests. Keep the `network` marker registered in conftest for future use.

Run: `.venv/bin/python -m pytest -q tests/core/test_llm_provider_routing.py tests/core/test_profile_reasoning_settings.py -p no:cacheprovider`
Expected: all pass, none skipped. Then the full suite, then `wc -l interpreter/core/llm/llm.py` at most 600.

Commit:

```bash
git add -A
git commit -F - <<'MSG'
refactor: move reasoning parameters to llm/reasoning.py; make provider tests hermetic

Llm.run() assembled the reasoning request parameters inline across ninety
lines; they now live in apply_reasoning_params(). The six provider-routing
and reasoning tests that were marked network reached a real provider
through litellm because they patched the wrong seam; they now install the
fake completions and assert on the captured request params.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 4: Prompter hook and the headless session seam

**Files:**
- Modify: `interpreter/core/core.py:111-236` (`__init__`: add `self.prompter`)
- Modify: `interpreter/core/respond.py:305-322,372-380` (use `interpreter.prompter`)
- Create: `interpreter/core/session.py`
- Create: `tests/core/test_session.py`
- Modify: `interpreter/server/openai_compat.py` (rebuild the two turn paths on `session.drive`)

**Interfaces:**
- Produces: `interpreter.prompter: Callable[[str, tuple[str, ...]], str]`, default `prompt_choice`, raising `NoInteractiveInput` when nobody can answer; `session.RUN`, `session.SKIP`, `session.PAUSE`; `session.drive(interpreter, message=None, *, approve=None) -> Iterator[dict]`; `session.DECLINED_NOTICE`, `session.EDIT_DECLINED_NOTICE`.

- [ ] **Step 1: The prompter hook**

In `OpenInterpreter.__init__`, after `self.loop_message = loop_message`:

```python
        # Who answers the loop's own questions (retry a failed provider call,
        # switch to the hosted model). prompt_choice asks the terminal and raises
        # NoInteractiveInput when nobody can answer; a server or channel installs
        # its own callable here.
        from .utils.prompt_choice import prompt_choice

        self.prompter = prompt_choice
```

In `respond.py`:

- replace the block

  ```python
                    if _stdin_is_interactive():
                        retry_choice = prompt_choice(
                            "  Retry? (y = retry once, a = keep retrying, n = stop)\n\n  ",
                            ("y", "a", "n"),
                        )
  ```

  with

  ```python
                    try:
                        retry_choice = interpreter.prompter(
                            "  Retry? (y = retry once, a = keep retrying, n = stop)\n\n  ",
                            ("y", "a", "n"),
                        )
                    except NoInteractiveInput:
                        retry_choice = None
                    if retry_choice is not None:
  ```

  and re-indent the `if retry_choice == "a": ... if retry_choice == "y": ...` lines that followed by one level (they were inside the `if _stdin_is_interactive():` block already; keep them inside the new `if retry_choice is not None:`).
- replace `response = prompt_choice("  ", ("y", "n"))` with `response = interpreter.prompter("  ", ("y", "n"))`.
- delete the `_stdin_is_interactive = stdin_is_interactive` alias and the now-unused imports (`ruff` reports them).

Run the suite (the CLI smoke test and `tests/core/test_prompt_choice.py` cover this); commit `refactor: route the loop's provider prompts through interpreter.prompter`.

- [ ] **Step 2: Write the failing session tests**

`tests/core/test_session.py`:

```python
"""core.session.drive: one headless turn, approvals decided by a callback.

The server and the channels share this seam. Before it, the server kept its
own copy of the approval state machine and the CLI another.
"""

from interpreter.core import session
from interpreter.core.session import PAUSE, RUN, SKIP, drive

CODE_REPLY = "Sure.\n```python\nprint(6 * 7)\n```"


def _console(chunks):
    return "".join(c.get("content", "") for c in chunks if c.get("type") == "console" and isinstance(c.get("content"), str))


def test_run_executes_pending_code_and_finishes_the_turn(offline_interpreter):
    fake = offline_interpreter.script([CODE_REPLY, "It printed 42."])
    offline_interpreter.auto_run = False

    chunks = list(drive(offline_interpreter, "what is 6*7", approve=lambda chunk: RUN))

    assert "42" in _console(chunks)
    assert offline_interpreter.messages[-1]["content"] == "It printed 42."
    assert len(fake.calls) == 2


def test_skip_records_the_decline_and_does_not_call_the_model_again(offline_interpreter):
    fake = offline_interpreter.script([CODE_REPLY])
    offline_interpreter.auto_run = False

    chunks = list(drive(offline_interpreter, "what is 6*7", approve=lambda chunk: SKIP))

    assert "42" not in _console(chunks)
    assert chunks[-1]["type"] == "confirmation"
    assert offline_interpreter.messages[-1]["content"] == session.DECLINED_NOTICE
    assert len(fake.calls) == 1


def test_pause_leaves_the_code_pending_and_a_later_run_executes_it(offline_interpreter):
    fake = offline_interpreter.script([CODE_REPLY, "It printed 42."])
    offline_interpreter.auto_run = False

    first = list(drive(offline_interpreter, "what is 6*7", approve=lambda chunk: PAUSE))
    assert first[-1]["type"] == "confirmation"
    assert offline_interpreter.messages[-1]["type"] == "code"
    assert len(fake.calls) == 1

    second = list(drive(offline_interpreter, None, approve=lambda chunk: RUN))
    assert "42" in _console(second)
    assert offline_interpreter.messages[-1]["content"] == "It printed 42."
    assert len(fake.calls) == 2


def test_auto_run_never_asks(offline_interpreter):
    offline_interpreter.script([CODE_REPLY, "Done."])
    offline_interpreter.auto_run = True
    asked = []

    chunks = list(drive(offline_interpreter, "go", approve=lambda chunk: asked.append(chunk) or SKIP))

    assert asked == []
    assert "42" in _console(chunks)
```

Run: `.venv/bin/python -m pytest -q tests/core/test_session.py -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'interpreter.core.session'`.

- [ ] **Step 3: Write `interpreter/core/session.py`**

```python
"""One headless turn of the interpreter, approvals decided by a callback.

The loop (respond.py) yields a "confirmation" chunk before running code or
applying an edit when the auto-run policy says a human must decide. The
terminal answers with a prompt; the server answers on the next HTTP request;
a channel answers with the next chat message. drive() is the one place that
knows how to continue, decline, or pause around that chunk.
"""

RUN = "run"  # continue: the loop executes the pending code
SKIP = "skip"  # record a decline and end the turn; the code is not run
PAUSE = "pause"  # end the turn with the code still pending; drive(None) resumes it

DECLINED_NOTICE = "[User declined to run this code.]"
EDIT_DECLINED_NOTICE = "[User declined to apply this edit.]"


def drive(interpreter, message=None, *, approve=None):
    """Yield the turn's LMC chunks. approve(chunk) -> RUN | SKIP | PAUSE.

    message: the user's message for a new turn, or None to resume a turn that
    was paused with code pending (the loop skips the model call when the last
    message is code, so resuming re-yields the confirmation).
    """
    approve = approve or (lambda chunk: RUN)
    if message is None:
        interpreter.last_messages_count = len(interpreter.messages)
        chunks = interpreter._respond_and_store()
    else:
        chunks = interpreter.chat(message, display=False, stream=True)
    try:
        for chunk in chunks:
            if chunk.get("type") != "confirmation":
                yield chunk
                continue
            decision = approve(chunk)
            if decision == RUN:
                continue  # resuming the generator runs the code
            yield chunk
            if decision == SKIP:
                notice = EDIT_DECLINED_NOTICE if chunk.get("format") == "edit" else DECLINED_NOTICE
                interpreter.messages.append({"role": "user", "type": "message", "content": notice, "source": "session"})
            return
    finally:
        chunks.close()
```

Run: `.venv/bin/python -m pytest -q tests/core/test_session.py -p no:cacheprovider`
Expected: 4 passed. If `test_pause...` fails because the resumed loop calls the model instead of re-yielding the confirmation, check `interpreter.messages[-1]["type"]` after the pause: `_streaming_chat`'s `finally` must not have altered it; the contract test `test_chat_completion_pauses_for_approval_then_runs_on_yes` already depends on this resume behavior working.

Commit `feat: add core.session.drive, the headless turn driver`.

- [ ] **Step 4: Rebuild the OpenAI endpoint on drive**

In `interpreter/server/openai_compat.py`:

- `from ..core import session`.
- In `openai_compatible_generator(run_code)`: replace the `chunk_iter` selection and the `for chunk in chunk_iter:` loop (everything from `if run_code:` to the `if not _openai_server_has_pending_code(...)` check, except the context-mode nudge branch which stays) with:

  ```python
        remaining_approvals = 1 if run_code else 0

        def approve(chunk):
            nonlocal remaining_approvals
            if remaining_approvals > 0:
                remaining_approvals -= 1
                return session.RUN
            return session.PAUSE

        if not run_code:
            async_interpreter.last_messages_count = len(async_interpreter.messages)
        async for chunk in iterate_in_threadpool(session.drive(async_interpreter, None, approve=approve)):
            if chunk.get("type") == "confirmation":
                async_interpreter._server_awaiting_code_approval = True
            output_content = _lmc_chunk_to_openai_delta(chunk, async_interpreter, pending_code_language=pending_lang)
            if output_content:
                yield await emit_delta(output_content)
            if async_interpreter.stop_event.is_set():
                break
  ```

  (`drive` sets `last_messages_count` itself when `message is None`; the explicit line above is kept only for the `run_code` branch's prior behavior of not resetting it. Read the old code once more before deleting it and keep the `print(...)` lines that echo output to the server console when `run_code` is true.)
- The non-stream branch's `collect()` uses the same `approve` construction and `for chunk in session.drive(async_interpreter, None, approve=approve)`.
- Delete `_openai_server_has_pending_code` and `_cancel_pending_code` if nothing references them afterwards (`grep -rn`).

Run: `.venv/bin/python -m pytest -q tests/core/test_server_contract.py tests/core/test_async_core.py -p no:cacheprovider && .venv/bin/python -m pytest -q -m "not integration" -p no:cacheprovider tests | tail -1`
Expected: all pass; the approval round-trip tests are the proof.

Commit:

```bash
git add -A
git commit -F - <<'MSG'
refactor: drive the OpenAI-compatible endpoint through core.session

The endpoint carried its own approval state machine (run_code juggling in
two places, pending-code checks, flag clearing). Both the streaming and the
non-streaming paths now call session.drive with an approve callback that
runs the one approved block and pauses on the next. Behavior is unchanged
and covered by the server contract tests.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

---

## Phase 2a exit

- [ ] `.venv/bin/python -m pytest -q -m "not integration" tests` reports 0 failed and `pgrep -f '[i]pykernel_launcher' | wc -l` prints `0` afterwards.
- [ ] `.venv/bin/ruff check interpreter tests scripts` is clean.
- [ ] `find interpreter -name '*.py' | xargs wc -l | awk '$1>600'` lists only: `toolbox/web/web.py`, `llm/tool_calling.py`, `terminal_interface/terminal_interface.py`, `core/core.py`, `profiles/profiles.py`, `core/respond.py`, `start_terminal_interface.py`, `jupyter_language.py`, `point.py`, `file_edit.py` (all handled by plan 2b except the last two).
- [ ] Push `rework`; CI green.
