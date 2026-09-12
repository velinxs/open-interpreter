# Remaining Defects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the last defects found while raising coverage, and make two classes of failure visible: an error the user should see must reach the terminal, and a malformed tool call must reach the model as an answer it can correct.

**Architecture:** Four independent tasks, each touching one area. No new modules. Every task flips or adds a test; several defects already have characterisation tests pinning the broken behaviour, and those are flipped rather than duplicated.

**Tech Stack:** Python 3.12+, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-11-fork-rework-design.md`

## Global Constraints

- Run everything with `~/.local/bin/uv run`. Never `pip`, never the system Python.
- Every commit: `~/.local/bin/uv run ruff check interpreter tests scripts` clean, and `~/.local/bin/uv run pytest -q -m "not integration" -p no:cacheprovider tests` green. The suite is 761 passed / 36 skipped / 12 deselected before this plan; it takes about 150 seconds.
- Bite-sized commits: one defect per commit, Conventional Commits, and a body that says what was broken, what the user saw, and why the fix is the right shape. End every commit message with exactly:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- Do not push. The controller pushes.
- Tests must pass offline: no API key, no network, no Ollama, no display. Anything needing those is marked `integration` or `network` and skipped by default.
- Tests must never read or write the real `~/.config/open-interpreter` or `~/.cache/open-interpreter`. Redirect with `tmp_path` and `monkeypatch`. The pattern is the autouse fixture at the top of `tests/core/test_execution_allowlist.py`.
- Every test function needs a docstring naming the behaviour it pins and the failure it prevents, matching the style already in the suite.
- After any run that constructs an interpreter, `pgrep -f '[k]ernel_launcher' | wc -l` must print 0. Never `pkill -f` a pattern that could match your own shell.
- Comment the *why*, not the *what*, per AGENTS.md. A fix to a subtle defect carries a comment explaining what went wrong, so the next reader does not reintroduce it.

---

### Task 1: Three small correctness fixes

**Files:**
- Modify: `interpreter/core/toolbox/utils/recipient_utils.py`
- Modify: `interpreter/terminal_interface/utils/find_image_path.py`
- Modify: `interpreter/terminal_interface/contributing_conversations.py`
- Test: the existing characterisation tests for each (find them with `grep -rn parse_for_recipient tests/`, `grep -rn find_image_path tests/`, `grep -rn get_contribute_cache_contents tests/`)

**Interfaces:**
- Produces: no signature changes. `parse_for_recipient` keeps returning `(recipient, message)`; `find_image_path` keeps returning a list of paths; `get_contribute_cache_contents` keeps returning the parsed cache.

Three unrelated one-area fixes, same shape: each has a characterisation test pinning the wrong behaviour, and each fix flips that test.

- [ ] **Step 1: `parse_for_recipient` truncates at the second colon.**

It splits a tagged line with `parts[2].split(":")[1]`, taking only the segment between the first and second colon. A message containing a URL, a timestamp, a Windows path or a dict repr loses everything after its own first colon: `"http://example.com"` becomes `"http"`. Split once, on the first colon only, and keep the remainder intact. Read the function and its existing test first; the test currently asserts the truncated value.

- [ ] **Step 2: `find_image_path` promises order and returns a set.**

Its docstring says the paths come back in the order they appear in the message, and the implementation ends in `list(set(...))`, so the order is arbitrary. Which image is "first" decides which one the model is shown. Deduplicate while preserving first appearance (`dict.fromkeys` over the matches) so the docstring becomes true.

- [ ] **Step 3: the contribution cache depends on telemetry to create its directory.**

`get_contribute_cache_contents` opens `~/.cache/open-interpreter/contribute.json` with `open()`, which does not create the directory. It works today only because importing `interpreter.core.utils.telemetry` happens to create that directory at import time. Make the function create its own parent directory, so it does not depend on an unrelated import's side effect. Do not change what it returns when the file is absent.

- [ ] **Step 4: Verify and commit**

Run the three tests you touched, then the full suite and ruff. Commit each fix separately, three commits, each naming the defect and what the user saw.

---

### Task 2: A mistyped profile key is a silent security downgrade

**Files:**
- Modify: `interpreter/terminal_interface/profiles/profiles.py` (`_validate_profile`, `apply_profile_to_object`)
- Test: `tests/terminal_interface/test_profiles.py`

**Interfaces:**
- Produces: no signature change. `apply_profile_to_object(obj, profile)` still applies a profile to an object; it simply stops inventing attributes.

- [ ] **Step 1: Read the two functions and the test that pins today's behaviour.**

`_validate_profile` warns "This setting will be ignored" for a key the interpreter does not have. `apply_profile_to_object` then calls `setattr` unconditionally, so the key *is* created, as a new attribute nothing reads. The warning is false, and a typo is silent: `auto_run_moed: all` creates a dead attribute while `auto_run_mode` keeps its default, so a user who believes they enabled something has not, and a user who believes they restricted something has not either. That second direction is why this is a security concern and not a cosmetic one.

- [ ] **Step 2: Make the behaviour match the warning.**

An unknown key is reported and skipped, not set. Keep the existing warning text where it is accurate. Preserve the `llm` and `toolbox` sub-dictionaries, which legitimately set attributes on sub-objects, and preserve any key the code sets deliberately that would look unknown (check `version` and `start_script`, which the loader strips and which must not produce a warning).

- [ ] **Step 3: A near-miss should be easy to fix.**

When an unknown key is close to a real one, name the likely intent in the message: `difflib.get_close_matches(key, known, n=1)`. If there is no close match, say only that the key is unknown and was skipped. Keep it to one line per key.

- [ ] **Step 4: Tests**

Flip the characterisation test. Add: a typo'd `auto_run_moed: all` leaves `auto_run_mode` at its default and does not create the attribute; the message names `auto_run_mode` as the probable intent; a valid profile with `llm:` and `toolbox:` blocks still applies fully; `version` and `start_script` produce no warning.

- [ ] **Step 5: Verify and commit**

Full suite, ruff, one commit.

---

### Task 3: Failures the user should see must reach the terminal

**Files:**
- Modify: `interpreter/terminal_interface/local_setup.py`
- Test: `tests/terminal_interface/test_local_setup.py`

**Interfaces:**
- Produces: no signature changes. Failure paths print a diagnosis instead of an empty line.

- [ ] **Step 1: Find every place a failure is swallowed in local setup.**

`grep -n "except" interpreter/terminal_interface/local_setup.py`. The pattern to look for is a handler that prints the exception bare, prints nothing, or prints only a blank line, and then continues as though the step had succeeded. The llamafile download path is the known example: a failure printed an empty line and left the model path `None`, which crashed later on `.split()` with a traceback that named neither the download nor the cause.

- [ ] **Step 2: Make each one say what failed and what the user can do.**

An error the user must act on gets one line naming the operation, one line with the cause, and where there is an obvious next step, one line saying it. Do not print a traceback into a chat session; do not swallow the cause either. Where a failure means the step cannot continue, return or exit rather than proceeding with a value the caller cannot use.

Keep this to `local_setup.py`. Do not refactor the module.

- [ ] **Step 3: Tests**

For at least the download failure and one other path you fixed: the message reaches stdout, names the operation, and the function does not go on to use a `None` it produced. Use `capsys`. Do not assert on exact punctuation; assert the operation and the cause are both present.

- [ ] **Step 4: Verify and commit**

Full suite, ruff, one commit per failure path you fixed if they are independent, otherwise one commit for the group.

---

### Task 4: A malformed tool call must reach the model as an answer

**Files:**
- Modify (only if you find a gap): `interpreter/core/llm/tool_dispatch.py`
- Test: `tests/core/test_tool_dispatch.py` or `tests/core/test_tool_calling_runner.py`

**Interfaces:**
- Consumes: `dispatch_function_call(llm, accumulated_deltas, request_params, tool_call_id_for_error, verbose, language)`, a generator yielding LMC chunks.
- Produces: no signature change.

The model calls exactly two tools, `execute(language, code)` and `edit(language, code, target)`. When it calls one wrongly, the turn must not end in silence or a crash: the error goes back as a `role: tool` message the model reads on its next turn and can correct from. Several of these paths already exist. This task verifies the set is complete and closes what is missing.

- [ ] **Step 1: Enumerate what happens today, in writing.**

Read `dispatch_function_call` and list, in your report file, every way a call can be malformed and what the code currently does for each. At minimum: a function name that is not `execute`, `edit` or `view_image`; arguments that are not valid JSON; valid JSON missing a required key; a key present with the wrong type; `edit` naming a target that does not exist; `view_image` given an extension the model cannot see. For each, record whether the model is told, and whether the message carries a `tool_call_id`.

- [ ] **Step 2: Close the gaps.**

Any case that ends without telling the model gets a `role: tool` message whose content names what was wrong and what a correct call looks like. Say what is required, not just what failed: "execute requires `language` and `code`; got `language` only" is actionable, "invalid arguments" is not. Where a `tool_call_id` is available it must be attached, because a provider that requires the pairing will reject the next request without it.

Do not invent new failure modes to handle. If every case is already covered, say so in the report with the evidence, change no code, and go to Step 3.

- [ ] **Step 3: Tests**

Drive these through the real runner with the scripted-stream helpers in `tests/core/test_tool_calling_runner.py` (`_tool_call_stream`, `ScriptedStreams`), not by calling the dispatcher directly, so the test covers the path the provider actually takes. One test per malformed case you found in Step 1. Each asserts: the turn does not raise, a `role: tool` message is appended, and its content names the specific problem.

- [ ] **Step 4: Verify and commit**

Full suite, ruff, one commit.

---

## Plan exit

- [ ] Suite green, ruff clean, no leaked kernels.
- [ ] Every defect in this plan either fixed with a test, or reported as already correct with evidence.
