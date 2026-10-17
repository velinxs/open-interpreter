# AGENTS.md

Open Interpreter — Python CLI (3.12+) that lets LLMs execute code locally. Uses LiteLLM, ipykernel/jupyter-client, Rich, FastAPI.

This repo (`endolith/open-interpreter`) is the community-maintained home of OI Classic (Python). Contributions belong here. The original home [openinterpreter/openinterpreter](https://github.com/openinterpreter/openinterpreter) is now an unrelated Rust coding agent (Codex fork). Do not treat it as upstream or copy its architecture. OpenInterpreter.com is now dedicated to an unrelated desktop tool.

## Development setup

This branch is `rework`: `classic/develop` restructured, with `classic/develop`'s
own later work merged back in. The toolchain is **uv + ruff**, not poetry +
black/isort — `requires-python = ">=3.12"`, and `poetry.lock` is gone.

    uv sync --group dev --extra server
    uv run ruff check interpreter tests scripts
    uv run pytest -m "not integration" -q

Tests **do** run here, and CI runs too: `.github/workflows/python-package.yml`
triggers on `main`, `integration` and `rework`, lints with ruff and runs the
suite on 3.12 and 3.14, collecting coverage on 3.14. So a red run is real — do
not dismiss it as "CI does not run on this branch", which was true of
`classic/develop` and is not true here.

Skips are expected and legitimate: they gate on things a given machine does not
have (`pwsh`, `R`, `yq`, `gawk`, `patch`, `poke`), and each says which. The
`integration` marker is excluded on purpose — those tests call a real model and
bill a real API key.

Most unit tests here were written after the fact by an AI, assuming the code was
correct — so a surprising failure is more often a bad test than a regression (see
Testing below). Check which before assuming either way.

## Configuration

Runtime settings come from profiles: YAML, JSON or Python files in the user config
directory (`~/.config/open-interpreter/profiles/` on Linux). `docs/profiles.md` is
the reference — search order, every settable key with its type and default, the
`version:` key a headless deployment must pin, the four `auto_run` modes and their
rule format, precedence between profile/flag/environment, and a list of known rough
edges in the loader. Read it before changing anything under
`interpreter/terminal_interface/profiles/` or the flag table in
`interpreter/terminal_interface/arguments.py`, and update it when those change.

Note that this branch sets `_DEFAULT_PROFILE = "develop.yaml"` in `arguments.py`;
`main` uses `default.yaml`.

## Codebase map

Where things are after the restructure. Several modules that `classic/develop`
kept in one file are split here, and that is the most common reason an import
copied from an older branch fails.

- `interpreter/terminal_interface/start_terminal_interface.py:main` — CLI entry
  (`interpreter` command). The flag table itself lives in
  `terminal_interface/arguments.py` (`build_arguments`, `get_argument_dictionary`,
  `set_attributes`, `_DEFAULT_PROFILE`), not inline in the entry point.
- `interpreter/terminal_interface/terminal_interface.py` — the interactive REPL.
  Typed-line handling is in `user_input.py`, the run prompt in `approval.py`, the
  Escape-to-interrupt watcher in `escape_watch.py`, and the opening panel in
  `startup_banner.py`. `%`-commands are in `magic_commands.py`; profile migration
  in `profiles/migrate.py` and `profiles/paths.py`.
- `interpreter/core/core.py` — the `OpenInterpreter` object (config, `chat()`).
  The chunk-to-message loop is `message_stream.py`; conversation naming is
  `conversation_title.py`; the spill file for truncated output is `utils/spill.py`.
- `interpreter/core/respond.py` — the respond loop. Code execution and the
  boilerplate-strip peek are in `run_code.py`; tool dispatch in
  `llm/tool_dispatch.py`.
- `interpreter/core/llm/` — the LiteLLM seam, split by job: `llm.py` (config and
  request assembly), `completions.py` (`fixed_litellm_completions`, the request
  and response dumps), `providers.py`, `reasoning.py`, `errors.py`,
  `session_env.py`, and the two runners `text.py` and `tool_calling.py`.
  Tool-call plumbing is `tool_messages.py` (`process_messages`,
  `answer_dangling_tool_calls`), `tool_schema.py` and `tool_dispatch.py`.
  **There is no `run_text_llm.py` or `run_tool_calling_llm.py`** — that layout is
  `classic/develop`'s. Message conversion is `utils/convert_to_openai_messages.py`,
  trimming `utils/cache_aware_trim.py`.
- `interpreter/server/` — the async HTTP/WebSocket server (`app.py`,
  `interpreter.py`, `auth.py`, `openai_compat.py`). `core/async_core.py` is a
  compatibility shim re-exporting from it.
- `interpreter/core/terminal/` — code execution. Shell and REPL languages are in
  `terminal/languages/`; Python AST rewriting is `python_preprocess.py` and the
  state snippet `python_state.py`, not inside `jupyter_language.py`. Shared cwd
  tracking, redundant-`cd` stripping and null-redirect normalisation live in the
  `CwdTrackingMixin` in `cwd_tracking.py`.
- `interpreter/core/toolbox/` — desktop and web integration (`mouse/`, `keyboard/`,
  `display/`, `vision/`, `mail/`, `sms/`, `calendar/`, `contacts/`, `clipboard/`,
  `files/`, `browser/`, `web/`, `os/`), plus `actions/` and `ai2.py`. The web
  toolbox is a package: `web/web.py` is the coordinator and each provider is a
  mixin in `web/backends/`. **There is no `core/computer/`** and **no
  `toolbox/skills/`** — skills were retired in favour of `toolbox.actions`.
- `interpreter/core/tools/file_edit.py` — the file-editing tool. No `comby`: it is
  not cross-platform, and a structural rewrite needs a pattern *and* a rewrite,
  which an edit language taking one code block cannot express.
- `interpreter/core/utils/` — small helpers: `truncate_output`, `prompt_choice`
  (y/n prompts; takes only exact single letters and re-prompts on anything else),
  `sanitize_terminal_input` (strips TUI control sequences out of program output
  and typed input), `execution_allowlist`, `shell_chain`, `assemble_system_message`.
- `scripts/wtf.py` — standalone "fix my last terminal error" CLI.
- **Dead or fragile live code**: macOS-only, AppleScript/DB-gated: mail, sms,
  calendar, contacts. `interpreter/computer_use/` at top level is the deprecated
  Anthropic demo — excluded from coverage, do not touch.
- **`api.openinterpreter.com`** is still reached by exactly two things, both
  deliberate: the hosted `i` model's `api_base` (`llm.py`) and the opt-in
  conversation contribution (`contributing_conversations.py`). What was removed
  are the callers that posted the user's *screen* there — the `/point/`,
  `/point/text/` and `/text/` screenshot endpoints behind
  `toolbox.display.find`/`get_text_as_list_of_lists`, and `/browser/search`.
  All four return HTTP 500, so they were dead as well as a privacy problem;
  grounding is done locally by the session's own vision model instead. Do not
  reintroduce them.

## Model providers

Prefixes that route to a specific provider are wired in `Llm.load()`
(`interpreter/core/llm/llm.py`) and `llm/providers.py`:

- `openai/…`, `openrouter/…`, `anthropic/…` etc. — resolved by LiteLLM itself, no OI code.
- `dashscope-us/`, `dashscope-intl/`, `deepseek/` — OI sets `api_base`/`api_key` from env vars.
- `i` — Open Interpreter's own hosted model, special-cased in `Llm.load()`.

Three things to know before adding another prefix:

- **Do not rewrite the model name to make LiteLLM route it.** `self.model` is
  read by every provider gate in the codebase — tool-call format, reasoning
  padding, vision, secret sanitising — so overwriting it with a different
  provider's name makes all of them decide wrongly and quietly. The correct
  shape is: keep `self.model` truthful, and pass the wire name plus
  `custom_llm_provider` in the params inside `run()`.
- **A provider prefix that LiteLLM does not know needs its metadata supplied
  explicitly.** Anything LiteLLM would otherwise answer (context window, vision
  support, tool-call format) comes back as a miss, and a miss is worse than an
  answer when the consequence is silent: an unknown context window means
  trimming to 8000 tokens, and unknown vision means images replaced by text
  descriptions.
- Pointing a model at an `api_base` makes LiteLLM resolve an unset `api_key` from
  the environment, so an ambient `OPENAI_API_KEY` gets sent as `Authorization:
  Bearer` to that host. A prefix with its own credential must set `api_key`
  explicitly or fail loudly.
- `arguments.py` has a hardcoded allowlist of prefixes that must not be
  re-prefixed when `--api_base` is set; a new prefix needs adding there too.

Tool-call message shape is **not** decided per provider here. Everything that
supports function calling goes through `tool_calling.py`, and
`tool_messages.process_messages` normalises the whole history to modern
`tool_calls`/`tool` before the request leaves. That covers every strict
validator, not an allowlist of three.

## Not yet carried over from classic/develop

Real work on that branch that this branch does not have. Each is self-contained;
none is blocked, and the notes say what the port needs.

- **`opencode_go` provider.** Absent at the fork point, so entirely
  `classic/develop`'s: five helpers in its `llm.py`, the `custom_llm_provider`
  wiring, a `profiles/defaults/opencode_go.py`, the `arguments.py` prefix
  allowlist entry, and `tests/core/test_opencode_go.py` (649 lines). It belongs in
  `llm/providers.py` here, not back in `llm.py`.
- **Incremental conversion cache.** `llm/utils/conversion_cache.py` and
  `tests/core/test_conversion_cache.py` are present but not wired.
  `convert_to_openai_messages` already accepts the `resume_state`/`end_state`
  contract it needs, and the cache already calls this branch's own
  `cache_aware_trim`. What remains is for `Llm.run()` to call
  `cached_convert_and_trim(self, derived, token_limit, model)` instead of the
  separate convert (llm.py) + trim pair.
- **Part of the retry machinery.** The temporary/permanent classification, the
  `MAX_TEMPORARY_PROVIDER_RETRIES` cap and `_is_bodiless_400` are ported; the
  bodiless-400 retry-once path, the status-clearing closure and the
  exit-offering retry prompt are not. `tests/core/test_temporary_error_retry.py`
  is the specification — the failing tests in it are the to-do list.
- **The keyless `vanshul` search backend** (`https://mcp.vanshul.com/mcp`, no API
  key). It fits as another mixin in `web/backends/`.

The 44 failing tests in `tests/core/test_opencode_go.py`,
`test_temporary_error_retry.py` and `test_conversion_cache.py` are the
specification for these four. They are the only expected failures; anything else
red is a regression.

## Module size

`tests/test_module_sizes.py` holds every module to 600 lines so each one fits in
one head. Four are currently exempt as a **split backlog**, each pushed over by a
feature this merge brought in, and each capped so it cannot grow further while it
waits: `respond.py` (697), `terminal_interface.py` (635),
`jupyter_language.py` (612), `llm/llm.py` (602). Splitting one removes its entry;
nothing else belongs on that list.

## Code change guidelines

### Testing

- **Always write or update unit tests** when changing code. New functions/methods need tests; bug fixes need regression tests.
- **Every test function must have a docstring** explaining what behavior it verifies and why. Someone who breaks the test must be able to understand what they broke and what the intended behavior is.
- **Pin the rule, not the wording.** A test that asserts an exact prompt phrase breaks on every rewrite while still passing if the behaviour is lost. Assert the contract.

### Documentation

When changing code, update **all** relevant documentation:

- **Code comments and docstrings** — keep them accurate and up-to-date.
- **`docs/` folder** — update any affected documentation pages.
- **README files** — the project has READMEs in multiple languages (`README.md`, `docs/README_ZH.md`, `docs/README_JA.md`, `docs/README_ES.md`, …). If you change something documented in the English README, update the translated versions too.
- **`AGENTS.md`** — update this file if the development guidelines change or there is something non-obvious that an agent needs to know in future jobs.

### Commits

- **Make every commit a small, self-contained, working unit that completes one coherent idea—and nothing else** (i.e., both atomic and logical). Unrelated edits belong in separate commits even when each is small (e.g. a workflow trigger change and a pytest marker are two commits). A commit's idea includes everything that supports it—its tests, documentation, and any CI/workflow changes for it—so keep those in the same commit as the code they describe, not in a later commit for a different feature, and reviewers can read commit-by-commit while `git revert <commit>` undoes one idea cleanly.
- **Fold follow-up fixes into the commit that caused the problem.** If a commit in an unmerged branch needs a small fix (e.g. it broke CI), squash that fix into the original commit with `git rebase -i` (mark it `fixup`) rather than stacking a "fix CI" commit on top. History should read as if each commit was correct the first time, so every commit is one coherent, CI-green unit.  (Actually in `develop` if the fix is several commits later, just leave it clearly marked so it can be folded into the feature PR later.)
- **Write comprehensive commit messages.** The subject line is a concise summary; the body must explain the problem being solved, the chosen approach, and any trade-offs. Provide the *context* that makes the diff understandable—why each change exists and what it achieves. Avoid meta-commentary about the commit itself (e.g., "fixing my commit according to instructions"). Keep process discussion in chat.
- **Use Conventional Commits** (e.g., `feat:`, `fix:`, `docs:`, `test:`, `chore:`) to categorize changes and enable automated changelog generation.

### Comments

- **Code comments** explain *why*: the intent, non-obvious reasoning, edge cases, and business logic. If a comment is needed to restate what the code does, rewrite the code to be clearer instead. Historical context that explains current behavior is acceptable. Remove meta-commentary about the development process (e.g., "fixing my commit according to instructions" or "now following the directions"). Keep process discussions in chat, not in comments or commit messages.
- Don't delete or omit comments while changing things. Comments are just as important as code.

### PRs and Issues

- On this branch, don't create PRs. Features will be merged into `main` as PRs eventually.

#### CodeRabbit review workflow

No PRs, so no CodeRabbit. :D
