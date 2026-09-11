# Fork Rework, Phase 2b: Structure of the CLI, Web Toolbox, Core Loop and Kernel Language

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring every module except the two exemptions under 600 lines by splitting along existing seams, delete dead code found on the way, add a characterization test for tool-calling mode before touching it, and lock the size budget with a test.

**Architecture:** Every task is a symbol-by-symbol move with minimal edits, verified by the existing suite (including the Phase 0 contract tests and Phase 2a session tests), then a cleanup commit where needed. Public names tests or profiles import keep working via re-exports named in each task.

**Tech Stack:** Python 3.12+, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-11-fork-rework-design.md` (Phase 2)

## Global Constraints

- Every commit: `.venv/bin/ruff check interpreter tests scripts` clean; `.venv/bin/python -m pytest -q -m "not integration" -p no:cacheprovider tests` reports 0 failed (295 passed at the start of this plan).
- One idea per commit, Conventional Commits, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- After each task: `find interpreter -name '*.py' | xargs wc -l | awk '$1>600 && $2!="total"'` must no longer list the file the task targeted.
- Size budget exemptions (spec): `interpreter/core/toolbox/display/point/point.py`, `interpreter/core/tools/file_edit.py`.
- Behavior changes in this plan, and no others: none. Dead code removal (commented-out blocks, unreachable branches) is not a behavior change.
- Never `pkill -f <pattern>` from the tool shell. Kill by PID from `pgrep -f '[k]ernel_launcher'`.

---

### Task 1: CLI argument table

**Files:**
- Create: `interpreter/terminal_interface/arguments.py`
- Modify: `interpreter/terminal_interface/start_terminal_interface.py` (remove lines 42-346 `arguments = [...]`, 373-393 `deprecated_flags` + loop, 630-648 `set_attributes`/`get_argument_dictionary`; import them)

**Interfaces:**
- Produces: `arguments.build_arguments(interpreter) -> list[dict]` (the table; it references `interpreter.llm` etc. for `attribute` targets, so it stays a function taking the interpreter), `arguments.DEPRECATED_FLAGS: dict[str, str]`, `arguments.set_attributes(args, arguments)`, `arguments.get_argument_dictionary(arguments, key)`.

- [ ] **Step 1:** Move the table into `def build_arguments(interpreter): return [ ... ]` in `arguments.py` (check with `grep -n "interpreter\." <start_terminal_interface.py> | awk -F: '$1>=42 && $1<=346'` which names the table references; `_DEFAULT_PROFILE` moves with it). Move `set_attributes` and `get_argument_dictionary` verbatim. Add `DEPRECATED_FLAGS = {"--debug_mode": "--verbose"}`.
- [ ] **Step 2:** In `start_terminal_interface()`: `arguments = build_arguments(interpreter)` where the literal was; `for old_flag, new_flag in DEPRECATED_FLAGS.items():`; `from .arguments import DEPRECATED_FLAGS, build_arguments, get_argument_dictionary, set_attributes` at the top. Keep `_DEFAULT_PROFILE` importable from `start_terminal_interface` (`from .arguments import _DEFAULT_PROFILE`) since the sandbox notes reference it.
- [ ] **Step 3:** Run: `.venv/bin/python -m pytest -q tests/terminal_interface/test_cli_smoke.py -p no:cacheprovider && .venv/bin/python -c "import sys; sys.argv=['interpreter','--help']; from interpreter.terminal_interface.start_terminal_interface import main; main()" | head -3` then the full suite. `wc -l` both files: `start_terminal_interface.py` at most 450.
- [ ] **Step 4:** Commit `refactor: move the CLI flag table to terminal_interface/arguments.py`.

### Task 2: Profile migration

**Files:**
- Create: `interpreter/terminal_interface/profiles/migrate.py`
- Modify: `interpreter/terminal_interface/profiles/profiles.py`

- [ ] **Step 1:** Move `migrate_profile` (297-641), `determine_user_version` (778), `migrate_app_directory` (803), `migrate_user_app_directory` (848) and the constants only they use (`grep -n "^[A-Z_]* = " profiles.py`, check each name's uses) to `migrate.py`. `profiles.py` imports them by name so `profiles.migrate_profile` etc. still resolve; `migrate.py` imports whatever it needs from `profiles.py` only if that does not create a cycle (`OI_VERSION`, `profile_dir` live in profiles.py; if migrate needs them, move those constants to a new `profiles/paths.py` and import from there in both).
- [ ] **Step 2:** Suite; `wc -l profiles.py` at most 600.
- [ ] **Step 3:** Commit `refactor: move profile migration to profiles/migrate.py`.

### Task 3: Python preprocessing out of the kernel language

**Files:**
- Create: `interpreter/core/terminal/languages/python_preprocess.py`
- Modify: `interpreter/core/terminal/languages/jupyter_language.py`, `tests/core/test_terminal_languages.py:10-13`

- [ ] **Step 1:** Move the module-level functions and class from line 512 to the end (`strip_redundant_imports`, `preprocess_python`, `add_active_line_prints`, `AddLinePrints`, `wrap_in_try_except`, `string_to_python`) to `python_preprocess.py`; `jupyter_language.py` imports the names it calls. Update the test import to `from interpreter.core.terminal.languages.python_preprocess import strip_redundant_imports` and keep `JupyterLanguage` from `jupyter_language`.
- [ ] **Step 2:** Suite; `wc -l jupyter_language.py` at most 600.
- [ ] **Step 3:** Commit `refactor: move python code preprocessing out of jupyter_language.py`.

### Task 4: Tool-calling runner

**Files:**
- Create: `tests/core/test_tool_calling_runner.py`
- Create: `interpreter/core/llm/tool_schema.py`, `interpreter/core/llm/tool_messages.py`, `interpreter/core/llm/tool_dispatch.py`
- Modify: `interpreter/core/llm/tool_calling.py`, `interpreter/core/utils/system_debug_info.py:89`

Four commits: the characterization test; dead code; schema and messages; dispatch.

- [ ] **Step 1: Characterization test first.** With `install_fake_llm`, set `llm.supports_functions = True` and feed litellm-style tool-call deltas:

```python
"""Tool-calling mode: OpenAI-style tool_call deltas become code, edit, and view_image chunks."""

import json

from tests.support.fake_llm import install_fake_llm


def _tool_call_stream(name, arguments, call_id="call_1"):
    payload = json.dumps(arguments)
    head, tail = payload[: len(payload) // 2], payload[len(payload) // 2 :]
    yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": {"name": name, "arguments": head}}]}}]}
    yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": tail}}]}}]}
    yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}


def _text_stream(text):
    yield {"choices": [{"delta": {"content": text}}]}
    yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


class ScriptedStreams:
    def __init__(self, streams):
        self.streams = list(streams)
        self.calls = []

    def __call__(self, **params):
        self.calls.append(params)
        return self.streams.pop(0)


def test_execute_tool_call_runs_code(offline_interpreter):
    """An execute(language, code) tool call is executed and its output fed back."""
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.llm.completions = ScriptedStreams(
        [_tool_call_stream("execute", {"language": "python", "code": "print(6 * 7)"}), _text_stream("Done: 42.")]
    )

    messages = offline_interpreter.chat("what is 6*7", display=False, stream=False)

    assert any(m.get("type") == "console" and "42" in str(m.get("content")) for m in messages)
    assert messages[-1]["content"] == "Done: 42."


def test_unknown_tool_call_is_answered_with_a_tool_error(offline_interpreter):
    """A function the runner does not support is reported back as a tool response, not raised."""
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.llm.completions = ScriptedStreams(
        [_tool_call_stream("toolbox.web.search", {"query": "x"}), _text_stream("Understood.")]
    )

    messages = offline_interpreter.chat("search", display=False, stream=False)

    assert messages[-1]["content"] == "Understood."
    assert any(m.get("role") == "tool" for m in messages)
```

Run it; adjust assertions to observed behavior only where the behavior is not an error (say so in the commit body). Commit `test: characterize tool-calling mode before splitting the runner`.

- [ ] **Step 2: Dead code.** Delete the commented-out block at `tool_calling.py:301-355` (three obsolete message-rewriting passes). Suite; commit `chore: drop the commented-out message rewriting in tool_calling.py`.
- [ ] **Step 3: Schema and messages.** `tool_schema.py` gets `tool_schema`, `EDIT_LANGUAGES_ENUM`, `VIEW_IMAGE_ALLOWED_EXTENSIONS`, `view_image_tool_schema`, `edit_tool_schema` (lines 9-100) and `build_request_tools` (262-278). `tool_messages.py` gets `generate_tool_id`, `_inline_user_image_in_turn_after_last_assistant_text`, `process_messages` (103-261). `tool_calling.py` imports what `run_tool_calling_llm` uses; `system_debug_info.py:89` imports `build_request_tools` from `tool_schema`. Suite; commit `refactor: move tool schemas and message shaping out of tool_calling.py`.
- [ ] **Step 4: Dispatch.** Move the post-stream function-call handling (`# Process the converted function_call (if any) to yield code`, line 712, through the end of the `elif function_name:` branch, line 1001) into `tool_dispatch.py::dispatch_function_call(llm, accumulated_deltas, request_params, tool_call_id_for_error, verbose)` as a generator that yields the same chunks; `run_tool_calling_llm` does `yield from dispatch_function_call(...)`. Locals it reads are exactly the parameters listed (confirm with `ruff --select F821` after the move). Suite; `wc -l tool_calling.py` at most 600; commit `refactor: move tool-call dispatch to llm/tool_dispatch.py`.

### Task 5: Web toolbox package

**Files:**
- Create: `interpreter/core/toolbox/web/results.py`, `interpreter/core/toolbox/web/backends/__init__.py`, `.../backends/brave.py`, `serper.py`, `serpapi.py`, `tavily.py`, `linkup.py`
- Modify: `interpreter/core/toolbox/web/web.py`, `interpreter/core/toolbox/web/__init__.py`

- [ ] **Step 1: results.py.** Move `_default_locale_from_environment`, `_normalize_locale_language_for_hl`, `_normalize_locale_country_for_gl` (29-66), `ApiKeyError`, `WebToolboxError`, `SearchResult`, `FetchResult`, `AnswerResult`, `StructuredOutputResult` (67-312), `_normalize_tavily_single_page` (313-329). `web.py` re-exports `SearchResult, FetchResult, AnswerResult, StructuredOutputResult, WebToolboxError, ApiKeyError` (tests import `StructuredOutputResult, Web, WebToolboxError` from `web.web`).
- [ ] **Step 2: backends as mixins.** Each backend module defines a mixin class whose methods are moved verbatim (they use `self._check_api_key`, `self._handle_import_error`, `self._normalize_result_item`, `self._get_locale_defaults` from `Web`): `BraveBackend._search_brave`; `SerperBackend._search_serper, _fetch_serper`; `SerpApiBackend._search_serpapi`; `TavilyBackend._search_tavily, _answer_tavily, _fetch_tavily`; `LinkupBackend._search_linkup, _answer_linkup, _structured_output_linkup, _fetch_linkup`. `class Web(BraveBackend, SerperBackend, SerpApiBackend, TavilyBackend, LinkupBackend)` keeps `__init__`, the shared helpers, `_check_backend_available`, `_build_no_backends_error`, and the public `search`, `answer`, `structured_output`, `fetch`. `_normalize_result_item` and `_create_normalized_response` (408-458) move to `results.py` as functions if `web.py` is still over 600 after the mixins.
- [ ] **Step 3:** Run `.venv/bin/python -m pytest -q tests/core/test_web.py -p no:cacheprovider` (the linkup tests skip; the others must pass), then the suite; `wc -l` every file in the package at most 600.
- [ ] **Step 4:** Commit `refactor: split the web toolbox into results, backends and the Web facade`.

### Task 6: Core: conversation titles and the spill archive

**Files:**
- Create: `interpreter/core/conversation_title.py`, `interpreter/core/utils/spill.py`
- Modify: `interpreter/core/core.py`

- [ ] **Step 1:** `conversation_title.py` gets `_CONVERSATION_TITLE_TRANSCRIPT_OMITTED_MARKER`, `_conversation_title_transcript_trim_to_cap` (46-92) and the bodies of `_is_user_message_for_conversation_title`, `_is_assistant_message_for_conversation_title`, `_clip_conversation_title_text`, `_conversation_auto_title_transcript`, `_sanitize_conversation_title_slug`, `_run_llm_for_conversation_title_slug`, `rename_conversation_file_from_llm_title`, `_maybe_upgrade_conversation_title` (285-482) as module functions taking `interpreter` first. `OpenInterpreter` keeps each as a one-line delegating method (tests call `interpreter._conversation_auto_title_transcript(...)` and `rename_conversation_file_from_llm_title(...)`; `core.py` re-exports the two module-level names `tests/core/test_conversation_rename.py` imports).
- [ ] **Step 2:** `utils/spill.py` gets `spill_file_path(interpreter)`, `open_spill(interpreter, path)`, `record_full_output(interpreter, message, chunk_content)` from `_spill_file_path`, `_open_spill`, `_record_full_output` (808-890). `OpenInterpreter` keeps the three underscore methods as delegates (`tests/core/test_spill_output.py` calls them on a fake).
- [ ] **Step 3:** Suite; `wc -l core.py` at most 600; commit `refactor: move conversation titling and the spill archive out of core.py`.

### Task 7: The loop: code execution and provider errors

**Files:**
- Create: `interpreter/core/run_code.py`, `interpreter/core/provider_errors.py`
- Modify: `interpreter/core/respond.py`

- [ ] **Step 1: run_code.py.** Move the body of `if interpreter.messages[-1]["type"] == "code":` (498-788, including its `except KeyboardInterrupt` / `except Exception` handlers) into `def run_pending_code(interpreter):` as a generator that yields the same chunks; `respond()` does `yield from run_pending_code(interpreter)` in that branch. The module-level helpers only that block uses (`grep -n "^def " respond.py` and check each name's uses) move with it. Suite (the fake-LLM, session and contract tests all execute code through this path); commit `refactor: move code execution out of respond.py into run_code.py`.
- [ ] **Step 2: provider_errors.py.** Move the body of `except Exception as e:` (177-393) into `def handle_provider_error(interpreter, error, state) -> str` where `state` is a small `@dataclass class RetryState: temporary_retries: int = 0; last_signature: str | None = None; always_retry: bool = False` created once before the `while True:` in `respond()`. The function returns `"retry"` wherever the old code did `continue` (after performing the same sleep/print), and raises wherever the old code raised (re-raise `error`); the caller does `if handle_provider_error(interpreter, e, state) == "retry": continue`. The `else:` clause of the `try` (394-398) resets `state.temporary_retries = 0; state.last_signature = None`. `_html_error_to_renderable`, `_is_temporary_provider_error`, `_temporary_error_signature`, `_render_temporary_retry_status`, `_litellm_optional_api_exceptions` (26-113) move with it. Suite; `wc -l respond.py` at most 600; commit `refactor: move provider error handling out of respond.py`.

### Task 8: The terminal interface

**Files:**
- Create: `interpreter/terminal_interface/approval.py`
- Modify: `interpreter/terminal_interface/terminal_interface.py`

- [ ] **Step 1:** Move the confirmation block (`if chunk["type"] == "confirmation":` at 313, through the `continue` at 624) into `approval.py::handle_confirmation(interpreter, chunk, active_block) -> tuple[object | None, str]` returning the (possibly new) `active_block` and `"continue"` or `"break"`; the loop does `active_block, outcome = handle_confirmation(interpreter, chunk, active_block); if outcome == "break": break; continue`. `_prompt_or_skip`, `_display_edit_dry_run`, `NO_APPROVER_NOTICE` and the imports only they need move with it (`persist_allowlist_rule`, `should_require_execution_confirmation`, `prompt_choice`, `NoInteractiveInput`, `scan_code`, `CodeBlock`, `Panel`, the rich console helpers).
- [ ] **Step 2:** Move the pre-loop mode banner (112-143) into `_print_mode_banner(interpreter)` and the message preparation (178-260: empty message, `%` magic, the two "many users do this" rewrites, image-path detection) into `_prepare_message(interpreter, message, interactive) -> str | None` (None means "skip this loop iteration") in the same file.
- [ ] **Step 3:** `.venv/bin/python -m pytest -q tests/terminal_interface tests/core/test_terminal_languages.py -p no:cacheprovider` then the suite; `wc -l terminal_interface.py` at most 600; commit `refactor: split approval prompts and message preparation out of the terminal loop`.

### Task 9: Size budget test

**Files:**
- Create: `tests/test_module_sizes.py`

- [ ] **Step 1:**

```python
"""No module over 600 lines, so every file can be held in one head at a time.

Two files are exempt: the vision-model pointer (point.py) and the file-edit
tool, both single-purpose and left as they are by the rework spec.
"""

from pathlib import Path

LIMIT = 600
EXEMPT = {
    "interpreter/core/toolbox/display/point/point.py",
    "interpreter/core/tools/file_edit.py",
}
ROOT = Path(__file__).resolve().parents[1]


def test_no_module_over_the_line_budget():
    over = {}
    for path in (ROOT / "interpreter").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        lines = sum(1 for _ in path.open(encoding="utf-8"))
        if lines > LIMIT and rel not in EXEMPT:
            over[rel] = lines
    assert not over, f"modules over {LIMIT} lines: {over}"
```

- [ ] **Step 2:** Run it (must pass after Tasks 1-8), suite, commit `test: enforce the 600-line module budget`.

---

## Phase 2b exit

- [ ] Suite 0 failed; ruff clean; `pgrep -f '[k]ernel_launcher' | wc -l` prints 0.
- [ ] `tests/test_module_sizes.py` passes.
- [ ] Push `rework`; CI green.
