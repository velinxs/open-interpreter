# Profiles

A profile is a file that sets attributes on the `OpenInterpreter` object before a
session starts. It is the only way to configure Open Interpreter persistently:
everything a profile can do, a CLI flag or a Python script could also do, but the
profile does it on every run without being typed again.

This document describes the profile system as implemented on this branch. Every
default and behaviour below was read out of the source or verified by running it.
The implementation lives in:

| File | Responsibility |
| --- | --- |
| `interpreter/terminal_interface/profiles/paths.py` | Where profiles live; the format version constant |
| `interpreter/terminal_interface/profiles/profiles.py` | Resolving, loading, validating and applying a profile |
| `interpreter/terminal_interface/profiles/migrate.py` | Upgrading pre-0.2.1 profiles and app directories |
| `interpreter/terminal_interface/profiles/defaults/` | The profiles shipped inside the package |
| `interpreter/terminal_interface/arguments.py` | The CLI flag table and which attribute each flag writes |
| `interpreter/core/core.py` | `OpenInterpreter.__init__` — the top-level attributes |
| `interpreter/core/llm/llm.py` | `Llm.__init__` — the `llm:` attributes |
| `interpreter/core/toolbox/toolbox.py` | `Toolbox.__init__` — the `toolbox:` attributes |
| `interpreter/core/utils/execution_allowlist.py` | The four auto-run modes and the rule format |

---

## 1. Where profiles live

There are exactly two locations.

**User profiles** live in `<config dir>/profiles/`, where `<config dir>` is
`platformdirs.user_config_dir("open-interpreter")`:

| Platform | Path |
| --- | --- |
| Linux | `~/.config/open-interpreter/profiles/` |
| macOS | `~/Library/Application Support/open-interpreter/profiles/` |
| Windows | `%LOCALAPPDATA%\open-interpreter\profiles\` |

On Linux this honours `XDG_CONFIG_HOME`. `interpreter --profiles` opens the
directory in a file manager. The same config directory also holds `conversations/`,
`allowlist.yaml` and `denylist.yaml`.

**Bundled profiles** ship inside the installed package, at
`interpreter/terminal_interface/profiles/defaults/`. They are read-only examples;
you never edit them in place.

### How `--profile NAME` resolves

`profile()` in `profiles.py` resolves `NAME` in this order:

1. **Bundled-name shorthand.** If `NAME` with its extension stripped matches the
   stem of any file in `defaults/`, `NAME` is rewritten to that bundled filename.
   So `--profile fast` becomes `fast.yaml` and `--profile os` becomes `os.py`.
   This is the only case where an extension may be omitted.
2. **Reserved-name takeover.** If the resolved name is a bundled name *other than*
   `default.yaml` or `develop.yaml`, the bundled copy is loaded directly from the
   package and the user directory is not consulted. If a user file of that name
   exists, it is first **renamed on disk** to `<name>_custom.<ext>`. See
   [Known rough edges](#10-known-rough-edges).
3. **User directory.** Otherwise the name is joined onto the profiles directory and
   loaded if the file exists. Because `os.path.join` is used, an **absolute path**
   passed to `--profile` is honoured as-is — `--profile /etc/oi/prod.yaml` works.
   A *relative* path resolves against the profiles directory, never the cwd.
4. **URL.** If nothing exists locally, the string is fetched with `requests.get`.
   The prefixes `i.com/`, `www.i.com/`, `https://i.com/` and `http://i.com/` are
   rewritten to `https://openinterpreter.com/profiles/`, and if the trailing
   segment has no dot, `.json`, `.py` and `.yaml` are tried in that order.
5. **Failure.** For `default`/`default.yaml` the file is reset from the bundled
   copy and retried. For `develop`/`develop.yaml` the bundled copy is copied into
   the user directory and retried. For anything else the exception propagates.

A name that is not a bundled stem and has no extension goes straight to step 4 and
fails with a `requests` `MissingSchema` error. **Give your own profiles their full
filename including the extension.**

### The default profile name

`_DEFAULT_PROFILE` in `interpreter/terminal_interface/arguments.py` is the name
used when `--profile` is absent. On this branch it is:

```python
_DEFAULT_PROFILE = "develop.yaml"
```

This is deliberate and branch-local: `classic/develop` keeps fork settings in
`develop.yaml` so they stay separate from upstream's `default.yaml`. On `main` the
value is `"default.yaml"`. The constant is the single place the default name is
set; nothing else hardcodes it.

Several flags are pure profile shortcuts and overwrite `--profile` in
`start_terminal_interface.py`: `--fast` → `fast.yaml`, `--vision` → `vision.yaml`,
`--os` → `os.py`, `--local` → `local.py`, `--codestral` → `codestral.py`,
`--assistant` → `assistant.py`, `--llama3` → `llama3.py`, `--groq` → `groq.py`.
Combining `--local`/`--codestral`/`--llama3` with `--vision` or `--os` selects the
corresponding `-vision`/`-os` variant.

---

## 2. The three formats

The extension picks the loader:

| Extension | Loader | Notes |
| --- | --- | --- |
| `.yaml`, `.yml`, anything else | `yaml.safe_load` | The fallback branch, so `.yml` works |
| `.json` | `json.load` | Same key structure as YAML |
| `.py` | `ast.parse` → `ast.unparse`, stored as `start_script` | Executed, not parsed as data |

### YAML and JSON

Both produce a plain mapping. Keys are applied recursively onto the interpreter by
`apply_profile_to_object`: a scalar value becomes `setattr(obj, key, value)`, and a
nested mapping recurses into `getattr(obj, key)`. So `llm: {model: "x"}` sets
`interpreter.llm.model`.

```yaml
llm:
  model: "gpt-4.1-mini"
  temperature: 0
toolbox:
  import_toolbox_api: true
auto_run_mode: prompt
version: 0.2.5
```

The JSON equivalent:

```json
{"llm": {"model": "gpt-4.1-mini", "temperature": 0},
 "toolbox": {"import_toolbox_api": true},
 "auto_run_mode": "prompt",
 "version": "0.2.5"}
```

### Python

A `.py` profile is read as source, parsed to an AST, passed through the
`RemoveInterpreter` node transformer, unparsed back to source, and stored under the
key `start_script`. It is then given the synthetic version `OI_VERSION`, so a Python
profile is **always considered current** and never triggers the migration prompt.

`RemoveInterpreter` strips exactly two things:

- `from interpreter import interpreter` (and any `from interpreter import ...`
  whose alias list contains `interpreter`)
- an assignment `interpreter = OpenInterpreter()`

Both are removed because the loader supplies `interpreter` itself. `apply_profile`
runs the script with:

```python
scope = {"interpreter": interpreter}
exec(profile["start_script"], scope, scope)
```

So the name `interpreter` is already bound to the live object. Keeping the import
line in the file is fine and conventional — it makes the file readable and lets
editors resolve the symbol — but it is removed before execution.

```python
from interpreter import interpreter  # stripped by the loader; harmless to keep

interpreter.llm.model = "groq/llama-3.3-70b-versatile"
interpreter.llm.context_window = 110000
interpreter.toolbox.import_toolbox_api = True
interpreter.auto_run = False
```

Arbitrary code is allowed: imports, environment lookups, network calls, and helper
methods such as `interpreter.display_message(markdown)`, `interpreter.get_oi_dir()`
and `interpreter.local_setup()` which exist mainly for this purpose. See
`defaults/template_profile.py` for a commented starting point.

### `start_script` in a YAML or JSON profile

`start_script` is not exclusive to `.py` files — it is just a profile key whose
value is a string of Python source. `apply_profile` execs it **first**, before any
other key is applied. That ordering matters: a `llm.model` key in the same YAML file
overwrites whatever the script set.

```yaml
start_script: |
  import os
  interpreter.llm.api_key = os.environ["MY_GATEWAY_KEY"]
llm:
  model: "openai/my-gateway-model"
version: 0.2.5
```

### Shipped examples

`defaults/` contains four YAML/YML files (`default.yaml`, `develop.yaml`,
`fast.yaml`, `vision.yaml`, `snowpark.yml`) and about twenty Python ones, including
`template_profile.py` (annotated skeleton), `local.py` and `local-os.py` (run the
local-model wizard), `codestral*.py`, `llama3*.py`, `groq.py`, `qwen.py`,
`gemma2.py`, `cerebras.py`, `bedrock-anthropic.py`, `e2b.py`, `obsidian.py`,
`screenpipe.py`, `aws-docs.py`, `llama31-database.py` and `the01.py`. Read them for
patterns, but derive settings from the tables below — several shipped examples set
keys that no longer exist.

---

## 3. Settable keys

Anything the loader can reach with `setattr` is settable. The tables below are the
attributes that actually exist and are actually read. A key that names a
non-existent attribute produces a warning (see [validation](#validation)).

### 3.1 Top-level keys

These are attributes of `OpenInterpreter` (`interpreter/core/core.py`).

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `offline` | bool | `False` | Disables update checks, telemetry and other network features. Does not stop a hosted LLM call. |
| `auto_run` | bool \| str | `False` | Alias that writes `auto_run_mode`; `True` means `"all"`. Reading it returns `auto_run_mode == "all"`. |
| `auto_run_mode` | str | `"prompt"` | One of `prompt`, `all`, `allowlist`, `denylist`. See [section 5](#5-auto-run-modes). |
| `auto_run_allowlist_file` | str | `<config>/allowlist.yaml` | YAML file of extra allowlist rules; also where "add to allowlist" writes. |
| `auto_run_allowlist_rules` | list \| None | `None` | Allowlist rules inline in the profile. |
| `auto_run_allowlist_replace_builtin` | bool | `False` | `True` drops the four built-in allowlist rules. |
| `auto_run_denylist_file` | str | `<config>/denylist.yaml` | YAML file of extra denylist rules. |
| `auto_run_denylist_rules` | list \| None | `None` | Denylist rules inline in the profile. |
| `auto_run_denylist_replace_builtin` | bool | `False` | `True` drops the nine built-in deny rules. |
| `safe_mode` | str | `"off"` | `off`, `ask` or `auto`. `ask`/`auto` run `semgrep` over code first; they require `pip install semgrep`. |
| `max_output` | int | `2800` | Characters of code output shown to the LLM. Overflow is archived to a spill file. |
| `verbose` | bool | `False` | Detailed logging. |
| `debug` | bool | `False` | Developer debug output. |
| `multi_line` | bool | `True` | Accept multi-line input delimited by ```` ``` ````. |
| `highlight_active_line` | bool | `True` | Highlight the executing line inside a code block. |
| `plain_text_display` | bool | `False` | Plain text instead of Rich markdown rendering. |
| `shrink_images` | bool | `False` | Default answer for "resize this image before sending?" when nothing can be asked. |
| `speak_messages` | bool | `False` | macOS only, experimental: reads replies aloud via `say`. |
| `os` | bool | `False` | OS-control mode. |
| `sync_computer` | bool | `False` | Keep the Python kernel's `computer` object in sync with the host object. |
| `disable_telemetry` | bool | `False` | Turns off anonymous usage stats. |
| `contribute_conversation` | bool | `False` | Offer the conversation as open training data. |
| `conversation_history` | bool | `True` | Persist each conversation as JSON. |
| `conversation_history_path` | str | `<config>/conversations` | Where those JSON files go. |
| `conversation_filename` | str \| None | `None` | Force a filename instead of deriving one. |
| `system_message` | str | `default_system_message` | The base prompt. **Do not set this** — the loader prints a warning and sleeps 2s, because overriding it freezes an old prompt in place. Use `custom_instructions`. |
| `custom_instructions` | str | `""` | Appended to the system message. This is the right place for your own guidance. |
| `user_message_template` | str | `"{content}"` | Wrapper applied to user messages. |
| `always_apply_user_message_template` | bool | `False` | Apply that wrapper to every user message, not just the first. |
| `code_output_template` | str | `"Code output: {content}\n\n…"` | How code output is phrased back to the model. |
| `empty_code_output_template` | str | `"The code above was executed…"` | Same, for output-free runs. |
| `code_output_sender` | str | `"user"` | Role attributed to code output. |
| `loop` | bool | `False` | Keep prompting the model until it emits a loop breaker. |
| `loop_message` | str | long "Proceed…" string | The nudge sent each loop iteration. |
| `loop_breakers` | list[str] | 4 phrases | Exact replies that end the loop. |
| `in_terminal_interface` | bool | `False` | Set by the CLI; not meant for profiles. |
| `messages` | list | `[]` | Pre-seed the conversation. Rarely useful in a profile. |

`computer` and `toolbox` are the same object (`interpreter.computer is
interpreter.toolbox` is `True`), kept as an alias for older profiles.

### 3.2 The `llm:` block

These are attributes of `Llm` (`interpreter/core/llm/llm.py`).

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `model` | str | `"gpt-4o-mini"` | LiteLLM model string. Note the shipped `default.yaml`/`develop.yaml` override this to `gpt-4.1-mini`. |
| `temperature` | float | `0` | Sampling temperature. Only sent when truthy, so `0` is left to the provider default. |
| `api_key` | str \| None | `None` | Sent as LiteLLM's `api_key`. Set here, it beats the provider environment variable. |
| `api_base` | str \| None | `None` | OpenAI-compatible endpoint URL. |
| `api_version` | str \| None | `None` | Mainly for Azure. |
| `context_window` | int \| None | `None` | Token budget used for truncation. For `ollama*/` models it is also sent as `num_ctx`. |
| `max_tokens` | int \| None | `None` | Max completion tokens. If larger than `context_window`, it is clamped to 20% of it with a warning. |
| `supports_functions` | bool \| None | `None` | `None` auto-detects via `litellm.supports_function_calling`. `False` falls back to parsing markdown code fences. |
| `supports_vision` | bool \| None | `None` | `None` auto-detects via `litellm.supports_vision`. When `False`, images are described through `toolbox.vision.query` instead. |
| `vision_renderer` | callable | `toolbox.vision.query` | Used only when `supports_vision` is `False`. |
| `execution_instructions` | str \| False | long instruction string | Appended to the system message when `supports_functions` is `False`. Set to `False` to suppress. |
| `tool_calling_instructions` | str \| False | long instruction string | Appended when `supports_functions` is `True`. Set to `False` to suppress. |
| `retention_ratio` | float \| None | `None` | Cache-aware truncation. When the prompt overflows, drop whole oldest turns down to this fraction of the window (0.8 keeps 80%). `None` uses tokentrim's sliding window, which busts the provider KV cache every turn. |
| `max_budget` | float \| None | `None` | LiteLLM budget cap in USD. |
| `sanitize_secrets` | str | `"auto"` | `auto` redacts API keys and passwords for remote models only; `on` always; `off` never. |
| `include_reasoning` | bool \| None | `None` | `None` auto, `True` requests reasoning tokens, `False` disables them (and `reasoning_effort`). Ignored with a note on endpoints where reasoning is mandatory. |
| `reasoning_effort` | str \| None | `None` | `"low"`, `"medium"`, `"high"`. Some models accept only a subset; unsupported values are dropped with a note. |
| `last_completion_usage` | dict \| None | `None` | Runtime state, filled from the final stream chunk. Not a setting. |

Two `llm:` keys are intercepted by the loader rather than applied:

- `llm.max_output` is moved to the top-level `interpreter.max_output`, because
  `max_output` lives on the interpreter, not the LLM.
- `llm.truncation_step` is deprecated and deleted; if `retention_ratio` is not also
  present it is defaulted to `0.8`. The two knobs are not equivalent — the old one
  dropped a fixed token chunk — so there is no exact conversion.

### 3.3 The `toolbox:` block

These are attributes of `Toolbox` (`interpreter/core/toolbox/toolbox.py`). The block
may also be written `computer:` for backward compatibility; the loader renames it.

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `import_toolbox_api` | bool | `False` | Expose the `toolbox` object (browser, files, vision, web, skills, …) inside the Python kernel and describe it in the system message. The shipped defaults set this `True`. |
| `import_computer_api` | bool | `False` | Alias property for the above. |
| `import_skills` | bool | `False` | Import saved skills into the kernel at startup. |
| `save_skills` | bool | `True` | Allow the model to persist new skills. |
| `emit_images` | bool | `True` | Allow tool results to emit images. |
| `offline` | bool | `False` | Toolbox-local offline flag, separate from `interpreter.offline`. |
| `verbose` | bool | `False` | Toolbox-local verbosity. |
| `debug` | bool | `False` | Toolbox-local debug. |
| `api_base` | str | `"https://api.openinterpreter.com/v0"` | Endpoint for the hosted toolbox services. |
| `system_message` | str | generated | Setting it replaces the generated toolbox section of the prompt wholesale. |
| `max_output` | int | mirrors `interpreter.max_output` at construction | Vestigial — the only reference to it is commented out. |

Nested toolbox modules are reachable too, e.g.:

```yaml
toolbox:
  skills:
    path: "/srv/oi/skills"
```

### Languages

Restricting which languages may execute is **special-cased and only works under
`computer:`**:

```yaml
computer:
  languages: ["python", "bash"]
```

The loader intersects this list (case-insensitively) with
`interpreter.terminal.languages` and deletes the key before the generic apply pass.
The full set is: Ruby, python, bash, JavaScript, HTML, AppleScript, R, powershell,
React, Java, perl, augeas.

Writing the same thing under `toolbox:` does **not** work — see
[Known rough edges](#10-known-rough-edges).

### Validation

Before applying, `_validate_profile` warns about keys that do not correspond to an
existing attribute:

```
⚠️  Profile validation warnings:
   Profile has 'llm.nonexistent_key' but this attribute doesn't exist on the Llm class. This setting will be ignored.
   Profile has 'bogus_top_level' but this attribute doesn't exist on the Interpreter class. This setting will be ignored.
```

The keys `llm`, `computer`, `wtf`, `version` and `start_script` are exempt, as is
any key beginning with `_`, and any top-level key whose value is a mapping. The
message overstates: the setting is not in fact ignored, only unused. See
[Known rough edges](#10-known-rough-edges).

---

## 4. The `version:` key

```yaml
version: 0.2.5  # Profile version (do not modify)
```

`version` records which profile *format* the file was written for. The expected
value is the constant `OI_VERSION` in `profiles/paths.py`, currently `0.2.5`. It is
unrelated to the package version in `pyproject.toml` (`0.4.3`).

`apply_profile` compares them. If `version` is missing, or is anything other than
`OI_VERSION`, it prints:

```
We have updated our profile file format. Would you like to migrate your profile file to the new format? No data will be lost.

(y/n)
```

- **`y`** runs `migrate_user_app_directory()`, which upgrades a pre-0.2.0
  (`Open Interpreter`) or 0.2.0 (`Open Interpreter Terminal`) config directory into
  the current one, converting each YAML profile through `migrate_profile` and
  copying `conversations/` and `config.yaml` across. If the profile being loaded is
  `default.yaml`, its `version:` line is rewritten in place and the obsolete models
  `gpt-4` and `gpt-4-turbo-preview` are rewritten to `gpt-4.1`. The profile then
  loads normally.
- **`n`** **abandons the profile entirely** — it prints "Skipping loading profile…"
  and returns the untouched interpreter. Nothing you configured is applied. For
  `default.yaml` only, a `version:` line is appended so the question is not asked
  again.
- **No terminal to ask** raises `NoInteractiveInput`, and `apply_profile` prints an
  explanation and calls `sys.exit(1)`. Guessing "no" would silently discard the
  model, the auto-run mode and every other setting while the run still appeared to
  start, so the code refuses to guess.

### Why a headless deployment must pin it

In a server, container, cron job or CI step there is no TTY. A profile without a
current `version:` line therefore does not degrade — the process **exits 1 before
doing any work**. Pin it:

```yaml
version: 0.2.5  # Profile version (do not modify)
```

This is verified behaviour, not theory. Running the bundled `develop.yaml` with
stdin closed produces:

```
Cannot start: …/profiles/develop.yaml uses an older profile format and
migrating it needs a yes/no answer, but there is no interactive
terminal to ask.
…
    version: 0.2.5  # Profile version (do not modify)
```

Python profiles are exempt: the loader stamps them with `OI_VERSION` at load time.

---

## 5. Auto-run modes

`auto_run_mode` decides which code blocks stop for confirmation. All four modes and
the matching logic live in `interpreter/core/utils/execution_allowlist.py`.

| Mode | Behaviour | Failure direction |
| --- | --- | --- |
| `prompt` | Every code block asks for approval. The default. | Fail-closed |
| `all` | Nothing asks — code *and* file edits run unattended. `-y` / `--auto_run` is this mode. | No gate |
| `allowlist` | A block runs unattended only if a rule matches it; everything else asks. | Fail-closed |
| `denylist` | A block runs unattended unless a rule matches it; a match asks. | **Fail-open** |

`normalize_auto_run_mode` accepts `True`/`"true"`/`"all"` → `all`, and
`False`/`"false"`/`"prompt"`/`None` → `prompt`; anything else must be exactly
`allowlist` or `denylist` or it raises `ValueError`. Because `auto_run` is a
property over the same field, `auto_run: true` and `auto_run_mode: all` are the same
setting.

Two interactions worth knowing:

- `auto_run_mode: all` combined with `safe_mode: ask`/`auto` is incompatible, and
  the CLI silently downgrades the mode to `prompt`. Allowlist and denylist modes are
  left alone, since both still gate something.
- At runtime `%auto_run all|prompt|allowlist|denylist` switches modes mid-session.

### Rule format

A rule is a mapping with exactly three meaningful fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `language` | yes | The code block's language. `bash`, `shell` and `sh` are treated as interchangeable; every other language must match exactly, case-insensitively. |
| `match` | yes | `exact` or `regex`. **These are the only two matchers**; anything else raises `ValueError` at load time. |
| `pattern` | yes | The literal string (for `exact`) or the Python regex (for `regex`). |

`exact` compares `code.strip() == pattern` — the whole block, whitespace-trimmed.
`regex` uses `re.search` with `IGNORECASE | MULTILINE`, so it matches **anywhere in
the block**, which is what lets a denylist rule catch a dangerous line buried in a
long script. An invalid regex raises at load time rather than silently never
matching.

Rules can come from four places, merged in this order with duplicates dropped
(duplicate = same `language`/`match`/`pattern` triple):

1. the built-ins, unless `auto_run_*_replace_builtin: true`
2. `auto_run_*_rules` in the profile
3. the YAML file named by `auto_run_*_file` (a mapping with a top-level `rules:` list)
4. session rules — **allowlist only**. Answering `a` at a confirmation prompt appends
   an `exact` rule for that exact code to both the session and the allowlist file.
   There is deliberately no denylist equivalent: accumulating "allow this once" is
   safe, accumulating "stop blocking this" is not.

### Built-in allowlist rules

`BUILTIN_STRICT_RULES` — four `exact` rules, nothing more:

```yaml
- {language: bash,   match: exact, pattern: "ls"}
- {language: shell,  match: exact, pattern: "ls"}
- {language: cmd,    match: exact, pattern: "dir"}
- {language: python, match: exact, pattern: "help(os)"}
```

### Built-in deny rules

`BUILTIN_DENY_RULES` — nine `regex` rules. Exact matching is useless for a denylist
because you cannot enumerate every spelling of a destructive command.

| Language | Pattern | Catches |
| --- | --- | --- |
| bash | `\brm\s+(-[a-z]*\s+)*-[a-z]*r[a-z]*f\|\brm\s+(-[a-z]*\s+)*-[a-z]*f[a-z]*r` | `rm -rf` in either flag order |
| bash | `\bmkfs(\.\w+)?\b` | Filesystem creation |
| bash | `\bdd\b[^\n]*\bof=\s*/dev/` | `dd` writing to a raw device |
| bash | `>\s*/dev/(sd[a-z]\|nvme\d\|vd[a-z])` | Redirect onto a raw disk |
| bash | `\bch(mod\|own)\s+(-[a-z]*\s+)*-[a-z]*R[a-z]*\s+[^\n]*\s/(\s\|$)` | Recursive chmod/chown rooted at `/` |
| bash | `:\s*\(\s*\)\s*\{.*\|.*&.*\}\s*;\s*:` | Fork bomb |
| bash | `\b(fdisk\|parted\|sgdisk\|wipefs)\b[^\n]*/dev/` | Partition-table rewrites |
| python | `\bshutil\.rmtree\s*\(` | Recursive delete |
| python | `\bos\.(remove\|unlink\|rmdir\|removedirs)\s*\(` | File/directory delete |

Note the coverage gaps this implies: nothing for `curl … \| sh`, nothing for
`python` shelling out via `subprocess`, nothing for any language other than bash and
python. Denylist mode is fail-open by construction; treat the built-ins as a floor,
not a fence.

### Worked example: an allowlist rule

```yaml
auto_run_mode: allowlist
auto_run_allowlist_rules:
  - language: bash
    match: exact
    pattern: "git status"
  - language: bash
    match: regex
    pattern: "^git (status|diff|log)( |$)"
version: 0.2.5
```

Loading this yields six rules (four built-in plus these two). Verified behaviour:

| Code block | Runs unattended? |
| --- | --- |
| `ls` | yes — built-in |
| `git status` | yes — the `exact` rule |
| `git diff --stat` | yes — the `regex` rule |
| `rm -rf /` | no — nothing matches, so it asks |

The `regex` rule here is anchored with `^` on purpose. Without the anchor, `re.search`
would match `git log` appearing anywhere, so a block like
`rm -rf /tmp/x && git log` would run unattended. **Anchor allowlist regexes.**

### Worked example: a denylist rule

```yaml
auto_run_mode: denylist
auto_run_denylist_rules:
  - language: bash
    match: regex
    pattern: "\\b(git\\s+push|kubectl\\s+(apply|delete)|terraform\\s+apply)\\b"
  - language: python
    match: exact
    pattern: "os.system('reboot')"
version: 0.2.5
```

Loading this yields eleven rules (nine built-in plus these two). Verified behaviour:

| Language | Code block | Asks for confirmation? |
| --- | --- | --- |
| bash | `ls -la` | no — runs unattended |
| bash | `git push origin main` | yes — the profile regex |
| bash | `echo hi && rm -rf /tmp/x` | yes — built-in `rm -rf`, matched mid-block |
| shell | `git push` | yes — `shell` and `bash` are interchangeable |
| python | `os.system('reboot')` | yes — the profile `exact` rule |
| python | `shutil.rmtree("/tmp/x")` | yes — built-in |

Note the backslash doubling: YAML double-quoted scalars process escapes, so `\b`
must be written `\\b`. Single-quoted or plain YAML scalars do not, so
`pattern: '\b(git\s+push)\b'` is equivalent and easier to read.

The rule file form, for `auto_run_allowlist_file` / `auto_run_denylist_file`:

```yaml
rules:
  - language: bash
    match: exact
    pattern: "git status"
```

---

## 6. Precedence

The startup order in `start_terminal_interface.py` is:

1. `set_attributes(args, arguments)` — CLI flags applied, so a Python profile can
   read what the user typed.
2. `profile(interpreter, …)` — the profile is applied, overwriting step 1.
3. `set_attributes(args, arguments)` again — **CLI flags applied a second time, so
   they win over the profile.**
4. `disable_telemetry` is OR-ed with the `DISABLE_TELEMETRY` environment variable.
5. Model-specific defaults are filled in *only where the value is still `None`*
   (context window and max tokens for `gpt-4*` and `gpt-3.5-turbo*`).

So, in general: **CLI flag > profile > built-in default.**

`set_attributes` skips any argument whose parsed value is `None`. Every settings
flag defaults to `None`, so an omitted flag leaves the profile alone — *except* for
`--safe_mode` (default `"off"`) and `--sanitize_secrets` (default `"auto"`), which
are always non-`None` and therefore always clobber the profile. See
[Known rough edges](#10-known-rough-edges).

### Environment variables

Environment variables are the weakest layer for LLM credentials and the strongest
for telemetry.

| Variable | Read by | Interaction with the profile |
| --- | --- | --- |
| `DISABLE_TELEMETRY` | `start_terminal_interface.py` | OR-ed in. Can only turn telemetry **off**; a profile cannot turn it back on. |
| `OPENAI_API_KEY` | `validate_llm_settings.py`, LiteLLM | Only consulted when neither `llm.api_key` nor `llm.api_base` is set. Profile wins. |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_API_BASE` | `llm/providers.py` | Applied only `if llm.api_key is None` / `api_base is None`. Profile wins. |
| `DASHSCOPE_API_KEY` | `llm/providers.py` | Same. Profile wins. |
| `OLLAMA_HOST` | `llm/providers.py` | Used only when `llm.api_base` is unset. Profile wins. |
| `OR_SITE_URL`, `OR_APP_NAME` | `llm/providers.py` | OpenRouter attribution headers; no profile equivalent. |
| `INTERPRETER_BASH`, `INTERPRETER_POWERSHELL`, `INTERPRETER_MPL_BACKEND`, `INTERPRETER_ACTIVE_LINE_DETECTION`, `INTERPRETER_TOOLBOX_API` | `core/terminal/…` | Execution-layer knobs with no profile key at all. Environment is the only way to set them. |

The help text for `--api_key`, `--api_base` and `--api_version` says they "override
environment variables", and that is accurate for both the flag and the profile key:
when set, they are passed to LiteLLM explicitly as request parameters.

---

## 7. Worked example: a hosted model

Verified on this machine against the real OpenAI API.

```yaml
# Hosted model, allowlist auto-run.

llm:
  model: "gpt-4.1-mini"
  temperature: 0
  supports_functions: true      # skip the litellm probe; this model does tool calls
  supports_vision: true
  context_window: 1000000
  max_tokens: 16384
  retention_ratio: 0.8          # cache-aware trimming, keeps the KV prefix warm
  max_budget: 5.0               # hard USD ceiling for the session
  sanitize_secrets: "on"        # always redact keys/passwords from outbound messages

toolbox:
  import_toolbox_api: true

custom_instructions: "Prefer Python. Show the command before running it."
max_output: 4000

auto_run_mode: allowlist
auto_run_allowlist_rules:
  - language: bash
    match: exact
    pattern: "git status"
  - language: bash
    match: regex
    pattern: "^git (status|diff|log)( |$)"

version: 0.2.5  # Profile version (do not modify)
```

Why each non-obvious choice:

- **No `api_key`.** `OPENAI_API_KEY` in the environment is enough for an OpenAI
  model. Put the key in the profile only when you want it to override the
  environment, and remember the file is world-readable unless you `chmod 600` it.
- **`supports_functions: true`** skips `litellm.supports_function_calling`. For a
  model LiteLLM knows, the probe returns the right answer anyway; asserting it makes
  the profile independent of the registry's freshness.
- **`retention_ratio: 0.8`.** Without it, truncation is tokentrim's per-turn sliding
  window, which shifts the prompt prefix on every call and invalidates the
  provider's KV cache. With it, the prompt is cut in whole turns down to 80% of the
  window, so the prefix stays stable for several turns.
- **`max_budget: 5.0`** is the only spend guard; there is no default.
- **`sanitize_secrets: "on"`** rather than the `"auto"` default, because `auto` only
  decides by whether the model looks remote. Be explicit for a hosted model. (Note
  that the terminal will reset this to `"auto"` — see
  [Known rough edges](#10-known-rough-edges).)
- **`auto_run_mode: allowlist`** is fail-closed: the read-only git commands run
  unattended, everything else stops.

Save as `~/.config/open-interpreter/profiles/hosted.yaml` and run:

```shell
interpreter --profile hosted.yaml
```

---

## 8. Worked example: local Ollama

This is a real, working profile from this machine
(`~/.config/open-interpreter/profiles/ollama.yaml`), verified end to end against a
running Ollama server.

```yaml
llm:
  model: "ollama_chat/qwen-thinking:latest"
  api_base: "http://localhost:11434"
  api_key: "ollama"          # unused by Ollama, keeps LiteLLM quiet
  supports_functions: true
  supports_vision: false
  context_window: 131072
  max_tokens: 8192
  temperature: 1.0
  retention_ratio: 0.8

toolbox:
  import_toolbox_api: true

offline: true
disable_telemetry: true
auto_run_mode: prompt        # confirm before running code; "denylist" runs all but the dangerous
version: 0.2.5
```

### Why `ollama_chat/` and not `ollama/`

LiteLLM routes the two prefixes to different Ollama HTTP endpoints, implemented in
different modules:

| Prefix | LiteLLM module | Ollama endpoint | Tool calling |
| --- | --- | --- | --- |
| `ollama/` | `litellm/llms/ollama/completion/` | `/api/generate` | none |
| `ollama_chat/` | `litellm/llms/ollama/chat/` | `/api/chat` | forwards `tools`, returns real `tool_calls` |

Open Interpreter's tool-calling path needs the model to return structured
`tool_calls`, which only `/api/chat` produces. With `ollama/`, the request never
carries a `tools` array, so the model can only emit prose and Open Interpreter falls
back to scraping markdown code fences — noticeably less reliable.

The trade-off: `providers.configure()` only special-cases models starting with
`ollama/`. It is that branch that appends `:latest` to a bare tag, checks
`/api/tags`, pulls a missing model, and probes `/api/show` for the context length.
With `ollama_chat/` none of that runs, so the tag must be written out in full
(`qwen-thinking:latest`, not `qwen-thinking`), the model must already be pulled, and
`context_window` must be set by hand.

### Why `supports_functions` must be asserted

When `supports_functions` is `None`, `Llm.load()` calls
`litellm.supports_function_calling(model)`. LiteLLM's registry has no entry for
locally-served models, so it returns `False`. Verified:

```
ollama_chat/qwen-thinking:latest -> False
ollama/qwen-thinking:latest      -> False
gpt-4.1-mini                     -> True
```

`False` makes Open Interpreter append `execution_instructions` to the system message
and parse markdown fences out of prose instead of issuing tool calls — a silent,
substantial capability downgrade for a model that in fact advertises `tools` in its
Ollama capability list. Setting `supports_functions: true` is the only way to
override a registry that does not know about your local model.

`supports_vision: false` is the mirror of this. The auto-probe would also return
`False`; stating it explicitly makes the outcome deliberate, and it tells the
toolbox to add the note steering the model to `toolbox.vision.query()` instead of
the `view_image` tool.

### Why the other values

- **`context_window: 131072`.** For any `ollama` or `ollama_chat` model, Open
  Interpreter sends `context_window` to the server as `num_ctx`, so the server's KV
  cache and the interpreter's truncation budget agree. Without it the server
  allocates its own much smaller default while the interpreter keeps trimming to a
  larger number, and requests silently overflow what was actually reserved. This
  model can do 262144; 131072 keeps the KV cache to a reasonable size.
- **`api_base: "http://localhost:11434"`.** Explicit rather than relying on
  `OLLAMA_HOST`, so the profile is self-contained.
- **`api_key: "ollama"`.** Ollama ignores it; it is a placeholder so LiteLLM does
  not complain about a missing key.
- **`temperature: 1.0`.** The `qwen-thinking` preset is built for temperature 1.0 /
  top_p 0.95. A non-thinking Qwen would want ~0.7. The weights are identical between
  the two presets; only the baked sampling parameters differ.
- **`offline: true` + `disable_telemetry: true`.** No update check, no usage stats,
  no hosted fallback. Fully local.
- **`auto_run_mode: prompt`.** Local models are cheap to re-run, so there is little
  reason to give up the confirmation gate.

Run it with:

```shell
interpreter --profile ollama.yaml
```

---

## 9. Resetting and migrating

- `interpreter --reset_profile default.yaml` replaces the user `default.yaml` with
  the bundled copy. If the existing file differs from any known historical version,
  it asks first, and the replaced file goes to the trash rather than being deleted.
- `interpreter --profiles` opens the profiles directory.
- `migrate_profile(old, new)` rewrites a flat pre-0.2.0 config into the nested
  layout, strips a `system_message` that matches any of eleven known historical
  prompts (or splits the custom tail off into `custom_instructions`), and appends a
  commented template plus the version footer.
- `determine_user_version()` reports `pre_0.2.0` if only the `Open Interpreter`
  config directory exists, `0.2.0` if the `Open Interpreter Terminal` one does, the
  `version:` value from `default.yaml` if that file exists, and `None` otherwise.

---

## 10. Known rough edges

These are real, reproduced behaviours of the code as it stands. Work around them;
do not assume they are intentional.

1. **The bundled `develop.yaml` and `default.yaml` have no `version:` line.** Every
   other bundled YAML profile (`fast.yaml`, `vision.yaml`) does. The consequence is
   that a fresh install hits the migration prompt on the very first run of the
   default profile, and in a non-interactive environment exits 1 before doing
   anything. Fix: add `version: 0.2.5` to your copy in the user profiles directory.

2. **For `develop.yaml`, the migration prompt never goes away.** Answering `y` runs
   the app-directory migration and applies the profile correctly, but only
   `default.yaml` gets its `version:` line written back. Answering `n` likewise only
   appends the line for `default.yaml`. So `develop.yaml` re-prompts on every single
   launch until you add the line by hand. Reproduced: two consecutive loads, two
   prompts.

3. **`--profile <bundled-name>` silently renames your file.** If you have your own
   `~/.config/open-interpreter/profiles/fast.yaml` and run `--profile fast`, the
   loader renames it to `fast_custom.yaml` and loads the *bundled* `fast.yaml`
   instead. Your settings are not deleted, but they are not used either, and the
   rename is permanent and unannounced. Only `default.yaml` and `develop.yaml` are
   exempt. Avoid naming a personal profile after any file in `defaults/`.

4. **`safe_mode` and `llm.sanitize_secrets` set in a profile are overwritten on
   every terminal run.** Their CLI flags carry non-`None` argparse defaults (`"off"`
   and `"auto"`), and `set_attributes` runs again after the profile is applied, so
   the default clobbers the profile value. Reproduced: a profile with
   `safe_mode: "auto"` and `llm.sanitize_secrets: "on"` ends up with `"off"` and
   `"auto"`. Every other setting flag defaults to `None` and behaves correctly.
   There is no profile-side workaround — a `start_script` also runs before the
   second `set_attributes` pass — so pass `--safe_mode auto` /
   `--sanitize_secrets on` on the command line. The Python API, which never calls
   `set_attributes`, is unaffected.

5. **The validation warning is wrong about what happens next.** It says the setting
   "will be ignored", but `apply_profile_to_object` calls `setattr` unconditionally,
   so a misspelled key really does get created on the object. It is inert, not
   ignored — and because it is inert, a typo like `auto_run_moed: all` is a silent
   security downgrade that only the warning line reveals.

6. **The `toolbox:` block is never validated.** `_validate_profile` checks only the
   `llm` and `computer` sub-mappings. Worse, the `computer` check is dead code: the
   loader renames `computer` → `toolbox` *before* calling `_validate_profile`, so
   the `computer` branch can never fire. Reproduced: a bogus key under either
   `toolbox:` or `computer:` produces no warning at all.

7. **A profile with both `computer:` and `toolbox:` blocks silently drops the
   `toolbox:` one.** The rename is `profile["toolbox"] = profile.pop("computer")`, a
   plain overwrite with no merge. Use one or the other, never both.

8. **`languages` only works under `computer:`.** Under `toolbox:` it reaches the
   `Toolbox.languages` property setter, which assigns the raw list of *strings* to
   `interpreter.terminal.languages`, replacing the Language objects and breaking
   execution with `AttributeError: 'str' object has no attribute 'name'`.

9. **`migrate_profile` does not actually rename old attributes.** It builds a
   `mapped_profile` dict from its `attribute_mapping` table (`model` → `llm.model`,
   `local` → `offline`, and ten more) and then never uses it — the reformatting loop
   below iterates the original `profile`. Reproduced: migrating
   `model: gpt-4 / temperature: 0.5 / local: true` produces a file with those exact
   flat keys intact, which then fail validation on load.

10. **`--reset_profile` with no argument resets nothing**, despite the help text
    saying it resets all default profiles. `reset_profile` short-circuits on
    `if specific_default_profile != "default.yaml": continue`, so only
    `--reset_profile default.yaml` has any effect — `--reset_profile develop.yaml`
    is also a no-op.

11. **The `wtf:` block is parsed and discarded.** Both `default.yaml` and
    `develop.yaml` document a `wtf: {model: …}` block for the `wtf` command, and
    `_validate_profile` exempts the key, but `apply_profile_to_object` explicitly
    `continue`s past it. Nothing on this branch reads it.

12. **`version` and `start_script` become attributes on the interpreter.** They are
    exempt from validation but not from the apply pass, so a loaded profile leaves
    `interpreter.version = "0.2.5"` and `interpreter.start_script = "<source>"`
    behind. Harmless today, but `interpreter.version` is easy to mistake for the
    package version.

13. **`default_profiles_names` includes `__pycache__`.** It is built by globbing
    `defaults/*` with no filter, so the compiled-bytecode directory is treated as a
    profile name. `--profile __pycache__` would hit the reserved-name path.

14. **Denylist mode is fail-open and the built-ins are thin.** Nine regexes covering
    bash and python only. No coverage for `curl | sh`, `subprocess`, package-manager
    removal, credential exfiltration, or any other language. If you run denylist
    mode, write your own rules; do not treat the built-ins as sufficient.
