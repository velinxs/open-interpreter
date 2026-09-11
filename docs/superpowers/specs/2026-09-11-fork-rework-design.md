# Fork rework design

Date: 2026-09-11. Branch: `rework` (from `integration` at 68c3fd17).

## Goals

1. **Hard fork.** velinxs/open-interpreter diverges from endolith's `classic/develop`
   for good (base snapshot 736e342f plus our 26 commits). No further merges from
   upstream; fixes are cherry-picked by hand if ever wanted.
2. **Same features and support.** Terminal chat CLI, HTTP server with the
   OpenAI-compatible agent endpoint, GUI computer control (OS mode), the toolbox
   (web, files, skills, vision, macOS calendar/mail/SMS/contacts, browser), file
   edit tool, profiles, every execution language. Nothing feature-bearing is removed.
3. **Efficient, simplified, streamlined codebase.** Fewer lines doing the same
   work, one place per concern, no dead code, no import-time side effects, fast
   startup, no module over 600 lines.
4. **Token efficient.** Measurably fewer prompt tokens per turn at the LLM
   boundary, with a stable cacheable prefix, without dropping any rule the model
   needs.
5. **Channels.** A new channel layer so the agent can be driven over a messaging
   app. Signal is built now; the adapter interface makes Telegram, Discord and
   Matrix additions later.
6. **Python 3.12 or newer.**

## Non-goals

- No rename of the package, import name, or commands (`open-interpreter`,
  `interpreter`, `i`, `wtf`). That is a separate decision.
- No rewrite. Behavior is preserved except where this document lists a change.
- No deployment from this work; `integration` is left ready to fast-forward and
  the sandbox update stays the operator's step.

## Current state (measured 2026-09-11 on `integration`)

- 28,057 lines of Python in 162 files under `interpreter/`; 5,617 lines of tests
  in 24 files; 99 files under `docs/`.
- Monoliths: `core/async_core.py` 1,425; `core/toolbox/web/web.py` 1,813;
  `terminal_interface/terminal_interface.py` 1,027 plus
  `start_terminal_interface.py` 740 (45 CLI flags); `core/llm/llm.py` 1,018 plus
  `run_tool_calling_llm.py` 941; `core/core.py` 952 plus `core/respond.py` 911;
  `profiles/profiles.py` 927.
- Dead: `core/archived_server_1.py`, `core/archived_server_2.py`,
  `computer_use/unused_markdown.py`, six `test_*.py` scripts at the repo root,
  `file.txt`, `.devin/`, `.cursor/`, `scripts/dashscope_usage_probe.py`.
- Declared but never imported: astor, git-python, six, google-generativeai, typer,
  setuptools. Used but undeclared: requests (arrives via litellm).
- Import-time side effects in `interpreter/__init__.py`: the `--os` branch does a
  PyPI HTTP call and starts the computer-use loop during import; `OpenInterpreter()`
  is constructed at import.
- The agent loop already yields a `confirmation` chunk and lets the consumer
  decide; only two TTY prompts remain inside `respond.py` (provider retry, and one
  y/n).
- Token baseline (tiktoken cl100k): system prompt 2,201 tokens on the default
  profile, 4,125 with toolbox API docs injected (toolbox docs alone 1,924),
  tool-calling instructions 127, console output capped at 2,800 characters per
  block.
- CI (`.github/workflows/python-package.yml`) triggers only on `main`, so nothing
  runs for this branch. Ruff reports errors on the touched files.

## Phase 0: safety net

- Branch `rework` in this checkout. `integration` stays deployable; each finished
  phase fast-forwards `integration` so the sandbox can take phases one at a time.
- Commit rules: one idea per commit, Conventional Commits, ruff clean and offline
  tests green at every commit, follow-up fixes folded into the commit they fix.
- CI runs ruff and `pytest -m "not integration"` on Python 3.12 and 3.14 for every
  push and pull request to `rework`, `integration`, and `main`.
- Tests that need the network, a model registry, or `pwsh` get explicit skip
  markers so an offline run is green and deterministic. The recorded baseline is
  the comparison point for every later commit.
- Characterization tests, LLM mocked, written before any surgery:
  `/openai/chat/completions` streaming and non-streaming, the code-approval
  round-trip, `/run`, websocket connect and message, and a CLI smoke test that
  drives `interpreter` as a subprocess with a fake model. These are the contract
  for phases 1 to 3.
- A token harness: a recording fake LLM plus a fixed six-turn scripted session
  (write and run Python, a shell command with long output, an error and retry, a
  file edit). It reports prompt tokens per request and becomes a test with an
  upper bound in Phase 3.
- Startup measurement: wall time of `import interpreter` and of `interpreter --help`,
  recorded alongside the test baseline.

## Phase 1: cleanse

Deletions: the dead files listed above. Examples, installers, docs, and all
profile defaults stay.

Dependencies: remove astor, git-python, six, google-generativeai, typer,
setuptools; add requests. The remaining dependency set is unchanged so install
behavior does not change; heavy packages (litellm, selenium, pyautogui,
matplotlib, tiktoken, anthropic) become lazy imports at their point of use so a
missing optional package fails at use, not at import.

Packaging: `requires-python = ">=3.12,<4"`; the `[tool.poetry]` tables go,
hatchling is the build backend, `uv.lock` replaces `poetry.lock`, dev tools live in
a `dev` dependency group. Installer scripts and docs switch their commands to `uv`
with `pip` shown as the alternative.

Lint: ruff target py312; rules E, F, W, I, UP enabled with their fixes applied in
this phase; B and SIM enabled per module as Phase 2 touches it. The per-file F821
ignores are removed by fixing the undefined names.

Import hygiene: `interpreter/__init__.py` loses every side effect. The `--os`
handling and the version check move into the CLI entry. The `interpreter`
singleton stays available as `from interpreter import interpreter` through a
module-level `__getattr__` that constructs it on first access, so
`import interpreter` is cheap.

## Phase 2: structure

Rule for every split: first a `git mv` (or a copy for a partial split) with the
minimum edits to keep tests green, then a separate cleanup commit, so history and
blame survive. Public import paths that documentation or profiles use keep working
through thin re-exports: `interpreter.interpreter`, `interpreter.OpenInterpreter`,
`interpreter.AsyncInterpreter`, `interpreter.core.core.OpenInterpreter`,
`interpreter.core.async_core.AsyncInterpreter` and `.Server`, and the console
entry points.

Target layout (new or changed files only):

```
interpreter/
  __init__.py                      lazy singleton, no side effects
  core/
    core.py                        OpenInterpreter: configuration, chat(), message store
    respond.py                     the turn loop only: llm -> code -> run -> output
    session.py                     NEW headless driver: drive(interpreter, message, policy) -> events
    approval.py                    NEW ExecutionPolicy: auto-run modes, allow/deny lists, prompt callbacks
    system_message.py              default text plus assemble_system_message()
    llm/
      llm.py                       Llm.run() only
      providers.py                 model-to-provider routing, context-window lookup, Ollama num_ctx
      reasoning.py                 reasoning settings and the mandatory-reasoning guard
      tool_calling.py              from run_tool_calling_llm.py
      text.py                      from run_text_llm.py
      messages.py                  LMC <-> OpenAI conversion, delta merging, partial JSON
      trim.py                      cache-aware trim and the tokentrim wrapper
    toolbox/web/
      search.py, fetch.py, answer.py, engines/   from web.py
  server/
    __init__.py                    re-exports AsyncInterpreter, Server, create_router
    interpreter.py                 AsyncInterpreter: async wrapper, input and output queues
    app.py                         create_router: /, /heartbeat, /settings, /run, /upload, /download, websocket
    openai_compat.py               /openai/chat/completions: conversion, SSE, approval state
    auth.py
  terminal_interface/
    start_terminal_interface.py    main(): parse, configure, dispatch
    arguments.py                   the flag table as data, plus apply-to-interpreter
    terminal_interface.py          the render loop
    approval.py                    the run and edit prompts (y/n/e/a) on top of core.approval
    profiles/loader.py             load, migrate, reset
    profiles/catalog.py            list and open
```

`core/session.py` is the seam the server and the channels share. It sends one
user message, iterates chunks, routes every `confirmation` through the
`ExecutionPolicy` (auto-run modes and the allow/deny lists from our existing
commits, or an `ask` callback), and yields typed events: assistant text, code,
console output, approval requests, errors, end of turn. The OpenAI endpoint is
rebuilt on it, which removes the duplicated approval bookkeeping in
`async_core.py`.

Behavior changes in this phase, all deliberate:

- The two TTY prompts inside `respond.py` go through the `ExecutionPolicy`
  callbacks. The headless default behaves as today (skip, or re-raise the real
  error).
- Import has no side effects; OS mode starts from the CLI entry.
- The server no longer keeps its own copy of the approval state machine.

Runtime efficiency targets, verified by the Phase 0 measurements:

- `import interpreter` completes without importing litellm, selenium, pyautogui,
  or matplotlib.
- Async handlers in the server contain no blocking sleeps or synchronous network
  or file I/O on the event loop; anything blocking runs in a thread.
- No polling loops with fixed sleeps where an event or queue exists.
- No module over 600 lines except `toolbox/display/point/point.py` and
  `tools/file_edit.py`, which are left as they are.

## Phase 3: token efficiency

Measured with the Phase 0 harness before and after; the before/after table goes
in the commit message.

- Default system prompt: 2,201 to at most 1,200 tokens. Every rule stays; prose,
  repetition, and examples that restate a rule go. Dynamic parts (working
  directory, date, user info) move to the end of the system message so the prefix
  is byte-identical across turns within a session and prompt caching applies.
- Toolbox API docs: 1,924 to at most 900 tokens by listing terse signatures with
  one-line descriptions, and by including only the modules that are actually
  available on the host.
- Tool-calling mode sends the tool schema or the prose instructions, never both.
- Console output: before the 2,800-character cap is applied, strip ANSI
  sequences, collapse progress-bar carriage-return frames to the last frame, and
  collapse runs of identical lines. The spill notice is shortened to one line
  plus the retrieval command.
- History trimming keeps the cached prefix intact: early messages are never
  rewritten by a later trim.
- The existing boilerplate stripping (redundant imports and `cd` prefixes) stays.
- The harness becomes a regression test with an upper bound on total prompt
  tokens for the scripted session.

## Phase 4: channels

New package `interpreter/channels/`.

- `base.py`: `Channel` interface with `receive()` (an async iterator of
  `Incoming(chat_id, sender, text, attachments, timestamp)`) and
  `send(chat_id, text, attachments=None)`, plus `max_message_length`.
- `runner.py`: `ChannelRunner(channel, config)`. One `OpenInterpreter` per
  `chat_id`, each with its own persisted conversation; one worker per chat so
  messages from the same chat are handled in order. It drives `core.session` and
  turns events into channel messages, splitting at `max_message_length`. When
  the policy needs a human, the runner sends the code block and waits for the
  next reply from that chat: `y`, `n`, or `e` with a replacement; anything else
  cancels after a configurable timeout. Chat commands: `/reset`, `/status`,
  `/stop`. A sender allowlist is mandatory; messages from anyone else are dropped
  and logged.
- `signal.py`: adapter over signal-cli-rest-api (bbernhard). Receive by polling
  `GET /v1/receive/{number}` at a configurable interval; send with
  `POST /v2/send`; attachments as base64. The fork never talks to signal-cli
  directly.
- Configuration lives in the profile under `channels:`: `signal.url`,
  `signal.number`, `signal.allowed_senders`, `signal.poll_interval`,
  `approval_timeout`. Auto-run behavior reuses the profile's `auto_run_mode`
  and allow/deny lists.
- Entry point: `interpreter --channel signal` (a new row in the flag table) and
  `python -m interpreter.channels signal`.
- Tests: an in-memory `FakeChannel` exercises the runner, including the approval
  round-trip, ordering, splitting, allowlist rejection, and commands. The Signal
  adapter is tested against a stub HTTP server.

## Phase 5: identity and docs

- `README.md` rewritten for the fork: what it is, install with `uv`, the three ways
  to run it (CLI, server, channel), where it came from.
- `AGENTS.md` rewritten: working setup that runs, how to test and lint, CI, the
  sandbox note, commit rules kept from upstream.
- Translated READMEs move to `docs/translations/` with a header saying they are
  upstream snapshots and may lag.
- Docs pruned of the PyPI upgrade check and of references to the original
  organization that no longer apply; `OR_SITE_URL` points at the fork.
- Version becomes 0.5.0 and `CHANGELOG.md` starts with this rework.

## Testing strategy

- Every commit: `ruff check` clean, `pytest -m "not integration"` green offline.
- Phase 0 characterization tests are the contract for phases 1 to 3.
- The token harness is a test with an upper bound from Phase 3 on.
- Channels: `FakeChannel` unit tests and a stubbed Signal server.
- After each phase, `integration` fast-forwards; the operator runs `oi-update.sh`
  on the sandbox and smoke-tests the agent endpoint.

## Risks

- Hidden coupling between the terminal interface and the core (shared state on
  the interpreter object). Mitigated by the characterization tests and by
  splitting via `git mv` with separate cleanup commits.
- Shorter prompts changing model behavior. Mitigated by keeping every rule and
  only cutting prose; an A/B with a real model on the scripted session is the
  operator's call before deploying Phase 3.
- Dropping Python 3.9 to 3.11 users. Accepted.

## Decisions taken without asking

- Package and command names unchanged.
- Translations demoted to snapshots rather than maintained or deleted.
- Dependency set unchanged apart from the unused six and the missing requests;
  heavy packages become lazy imports rather than optional extras, so installs
  behave as before.
- Signal through signal-cli-rest-api rather than signal-cli directly.
- No deployment to the sandbox from this work.
