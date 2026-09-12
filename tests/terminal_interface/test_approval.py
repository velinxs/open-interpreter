"""The approval gate: the one place a human stands between the model and the machine.

handle_confirmation answers a confirmation chunk. Every branch either runs
code, declines it, edits it, or permanently allowlists it, and the defaults
matter more than the happy path — a wrong fallback when nobody can answer
means code runs unattended.
"""

import time
from types import SimpleNamespace

import pytest

import interpreter.core.utils.execution_allowlist as allowlist_module
import interpreter.terminal_interface.approval as approval
from interpreter.core.core import OpenInterpreter
from interpreter.core.utils.prompt_choice import NoInteractiveInput
from interpreter.terminal_interface.approval import NO_APPROVER_NOTICE, handle_confirmation


@pytest.fixture(autouse=True)
def _isolate_allowlist_files(tmp_path, monkeypatch):
    """Point the allow/deny list files at the test's own directory.

    These default to the user's config directory. Without this, a developer
    who has ever answered "a" at a run prompt fails the suite, and the
    allowlist test writes a permanent rule into their real config.
    """
    monkeypatch.setattr(allowlist_module, "DEFAULT_ALLOWLIST_FILE", str(tmp_path / "allowlist.yaml"))
    monkeypatch.setattr(allowlist_module, "DEFAULT_DENYLIST_FILE", str(tmp_path / "denylist.yaml"))


@pytest.fixture
def interpreter(tmp_path):
    interp = OpenInterpreter()
    interp.auto_run_allowlist_file = allowlist_module.DEFAULT_ALLOWLIST_FILE
    interp.auto_run_denylist_file = allowlist_module.DEFAULT_DENYLIST_FILE
    interp.plain_text_display = True
    yield interp
    try:
        interp.terminal.terminate()
    except Exception:
        pass


@pytest.fixture
def answer(monkeypatch):
    """Script the y/n/e/a answer, and record the prompt and choices offered."""
    state = SimpleNamespace(response="n", asked=[])

    def _prompt_choice(prompt, choices):
        state.asked.append((prompt, choices))
        if isinstance(state.response, Exception):
            raise state.response
        return state.response

    monkeypatch.setattr(approval, "prompt_choice", _prompt_choice)
    monkeypatch.setattr(approval, "_prompt_or_skip", lambda prompt, choices, skipped="n": _prompt_choice(prompt, choices))
    return state


def _code_chunk(code="print(1)", language="python", **extra):
    return {
        "role": "computer",
        "type": "confirmation",
        "format": "execution",
        "content": {"type": "code", "format": language, "content": code},
        **extra,
    }


def _edit_chunk(**content_extra):
    content = {
        "type": "edit",
        "format": "write",
        "content": "new file body",
        "target": "/tmp/target.py",
        **content_extra,
    }
    return {"role": "computer", "type": "confirmation", "format": "edit", "content": content}


# --- auto-run ---------------------------------------------------------------


def test_auto_run_never_prompts(interpreter, answer):
    """With auto_run_mode "all" the gate is not consulted at all.

    Prompting anyway would hang every headless run, which is the whole reason
    the mode exists.
    """
    interpreter.auto_run_mode = "all"
    answer.response = AssertionError("should not have prompted")
    block, action = handle_confirmation(interpreter, _code_chunk(), None)
    assert action == "continue"
    assert answer.asked == []
    assert interpreter.messages == []


def test_allowlisted_commands_run_without_a_prompt(interpreter, answer):
    """A built-in allowlisted command in allowlist mode skips the prompt.

    That is what allowlist mode is for; prompting for `ls` anyway makes the
    mode indistinguishable from plain prompting.
    """
    interpreter.auto_run_mode = "allowlist"
    answer.response = AssertionError("should not have prompted")
    _, action = handle_confirmation(interpreter, _code_chunk(code="ls", language="bash"), None)
    assert action == "continue"
    assert answer.asked == []


def test_a_stripped_boilerplate_notice_is_shown_in_both_paths(interpreter, answer, capsys):
    """The "we removed some of your code" notice reaches the user whether or not they are asked.

    It is the only signal that what runs is not byte-for-byte what the model
    wrote. The auto-run branch prints it separately precisely because there is
    no prompt to attach it to.
    """
    chunk = _code_chunk()
    chunk["content"]["removed"] = "pip install boilerplate"

    interpreter.auto_run_mode = "all"
    handle_confirmation(interpreter, chunk, None)
    assert "pip install boilerplate" in capsys.readouterr().out

    interpreter.auto_run_mode = "prompt"
    answer.response = "n"
    handle_confirmation(interpreter, chunk, None)
    assert "pip install boilerplate" in capsys.readouterr().out


# --- answering the run prompt -----------------------------------------------


def test_yes_builds_a_block_carrying_the_code(interpreter, answer):
    """"y" hands back a CodeBlock primed with the language and the code.

    respond() renders into whatever block comes back. A block without the
    code shows an empty panel while the code runs underneath it.
    """
    answer.response = "y"
    block, action = handle_confirmation(interpreter, _code_chunk("print(42)"), None)
    assert action == "continue"
    assert block.language == "python"
    assert block.code == "print(42)"
    assert interpreter.messages == []


def test_highlighting_off_leaves_the_block_empty(interpreter, answer):
    """With highlight_active_line off, the code is not re-printed below the prompt.

    It was already shown in the streaming preview above the prompt; filling
    the block would print the same code twice.
    """
    interpreter.highlight_active_line = False
    answer.response = "y"
    block, _ = handle_confirmation(interpreter, _code_chunk("print(42)"), None)
    assert block.code == ""


def test_no_ends_the_turn_and_tells_the_model_why(interpreter, answer):
    """"n" stops the turn and leaves a note the model can read.

    Without the note the model sees its code produce no output and retries the
    identical block. The note is marked source="terminal" so %undo does not
    treat it as a user turn.
    """
    answer.response = "n"
    _, action = handle_confirmation(interpreter, _code_chunk(), None)
    assert action == "break"
    assert interpreter.messages[-1]["content"] == "[User declined to run this code.]"
    assert interpreter.messages[-1]["source"] == "terminal"


def test_no_answerer_declines_rather_than_editing_or_allowlisting(interpreter, answer):
    """With no terminal, the answer is "n" — never the last option offered.

    The choices end in "e" (open an editor) or "a" (permanently allowlist the
    command). Any "pick a default" fallback would do one of those unattended.
    """
    interpreter.auto_run_mode = "allowlist"
    answer.response = NoInteractiveInput("prompt", ("y", "n", "e", "a"))
    _, action = handle_confirmation(interpreter, _code_chunk("rm -rf /", "bash"), None)
    assert action == "break"
    assert interpreter.messages[-1]["content"] == NO_APPROVER_NOTICE


def test_the_no_approver_notice_tells_the_model_to_stop_retrying(interpreter):
    """The notice explicitly says not to retry, and how to enable auto-run.

    Without "do not retry" the model loops on the same rejected action until
    the context fills.
    """
    assert "Do not retry" in NO_APPROVER_NOTICE
    assert "-y" in NO_APPROVER_NOTICE


# --- allowlist mode ---------------------------------------------------------


def test_allowlist_mode_offers_the_extra_add_option(interpreter, answer):
    """Only allowlist mode offers "a"; the others must not.

    Offering "a" in plain prompt mode would let a keystroke create a permanent
    rule in a mode that has no allowlist.
    """
    interpreter.auto_run_mode = "allowlist"
    answer.response = "n"
    handle_confirmation(interpreter, _code_chunk("curl example.com", "bash"), None)
    assert answer.asked[-1][1] == ("y", "n", "e", "a")

    interpreter.auto_run_mode = "prompt"
    handle_confirmation(interpreter, _code_chunk(), None)
    assert answer.asked[-1][1] == ("y", "n", "e")


def test_add_to_allowlist_persists_an_exact_rule_and_runs_the_code(interpreter, answer, capsys):
    """"a" writes a rule and then runs this invocation too.

    Saying "always allow this" and then still declining it would be absurd,
    and the rule must be exact-match: a pattern rule from a single keystroke
    would silently approve commands the user never saw.
    """
    interpreter.auto_run_mode = "allowlist"
    answer.response = "a"
    block, action = handle_confirmation(interpreter, _code_chunk("git status", "bash"), None)

    assert action == "continue"
    assert block.code == "git status"
    assert "Added to allowlist" in capsys.readouterr().out

    answer.response = AssertionError("the persisted rule should have skipped the prompt")
    _, action = handle_confirmation(interpreter, _code_chunk("git status", "bash"), None)
    assert action == "continue"


# --- safe mode --------------------------------------------------------------


def test_safe_mode_auto_scans_without_asking(interpreter, answer, monkeypatch):
    """safe_mode "auto" runs the scanner on every block, with no extra prompt."""
    scanned = []
    monkeypatch.setattr(approval, "scan_code", lambda code, language, interp: scanned.append(code))
    interpreter.safe_mode = "auto"
    answer.response = "n"
    handle_confirmation(interpreter, _code_chunk("print(1)"), None)
    assert scanned == ["print(1)"]


@pytest.mark.parametrize("reply,scans", [("y", 1), ("n", 0)])
def test_safe_mode_ask_scans_only_on_request(interpreter, answer, monkeypatch, reply, scans):
    """safe_mode "ask" scans when the user says yes and not when they say no.

    Scanning is slow. Running it after a "no" would make declining the scan
    cost exactly as much as accepting it.
    """
    scanned = []
    monkeypatch.setattr(approval, "scan_code", lambda code, language, interp: scanned.append(code))
    interpreter.safe_mode = "ask"
    answer.response = reply
    handle_confirmation(interpreter, _code_chunk(), None)
    assert len(scanned) == scans


def test_safe_mode_off_never_scans(interpreter, answer, monkeypatch):
    """The default does no scanning and asks nothing about it."""
    monkeypatch.setattr(approval, "scan_code", lambda *a: pytest.fail("scanned with safe_mode off"))
    answer.response = "n"
    handle_confirmation(interpreter, _code_chunk(), None)
    assert len(answer.asked) == 1


# --- editing ----------------------------------------------------------------


def test_editing_replaces_the_code_and_tells_the_model_what_changed(interpreter, answer, monkeypatch, tmp_path):
    """"e" opens an editor, runs what comes back, and records the original.

    The message history must end up holding the edited code (that is what
    ran) plus a terminal-sourced note containing the original, or the model's
    next turn reasons about code that never executed.
    """
    interpreter.messages.append({"role": "assistant", "type": "code", "format": "python", "content": "print(1)"})

    def _fake_editor(command):
        path = command[-1]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("print(2)")

    monkeypatch.setenv("EDITOR", "fake-editor")
    monkeypatch.setattr(approval.subprocess, "call", _fake_editor)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    answer.response = "e"

    block, action = handle_confirmation(interpreter, _code_chunk("print(1)"), None)

    assert action == "continue"
    assert block.code == "print(2)"
    assert interpreter.messages[0]["content"] == "print(2)"
    notice = interpreter.messages[-1]
    assert "print(1)" in notice["content"]
    assert notice["source"] == "terminal"


def test_the_temp_file_gets_the_language_extension_and_is_cleaned_up(interpreter, answer, monkeypatch):
    """The scratch file is named for the language and deleted afterwards.

    The extension is what gives the editor syntax highlighting, and the file
    holds the code being reviewed — leaving copies in /tmp would scatter it.
    """
    interpreter.messages.append({"role": "assistant", "type": "code", "format": "bash", "content": "ls"})
    seen = {}

    def _fake_editor(command):
        seen["path"] = command[-1]

    monkeypatch.setenv("EDITOR", "fake-editor")
    monkeypatch.setattr(approval.subprocess, "call", _fake_editor)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    answer.response = "e"

    handle_confirmation(interpreter, _code_chunk("ls", "bash"), None)
    assert seen["path"].endswith(".sh")
    assert not approval.os.path.exists(seen["path"])


def test_no_editor_available_says_so_and_continues(interpreter, answer, monkeypatch, capsys):
    """With no editor on PATH the user is told which variables to set.

    Silently continuing unedited would run code the user was in the middle of
    changing.
    """
    interpreter.messages.append({"role": "assistant", "type": "code", "format": "python", "content": "print(1)"})
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(approval.platform, "system", lambda: "Linux")
    monkeypatch.setattr(approval.shutil, "which", lambda name: None)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    answer.response = "e"

    _, action = handle_confirmation(interpreter, _code_chunk(), None)
    assert action == "continue"
    assert "Could not find a suitable text editor" in capsys.readouterr().out


# --- edit-tool confirmations ------------------------------------------------


def test_edit_confirmations_offer_only_yes_or_no(interpreter, answer):
    """An edit is structured JSON, so there is nothing sensible to hand to an editor.

    Offering "e" here would drop the user into a temp file containing the
    tool's internal payload.
    """
    answer.response = "n"
    handle_confirmation(interpreter, _edit_chunk(), None)
    assert answer.asked[-1][1] == ("y", "n")


def test_declining_an_edit_ends_the_turn_with_its_own_notice(interpreter, answer):
    """Declining an edit says "edit", not "run this code".

    The model needs to know which of the two it was refused; the wording is
    the only thing distinguishing them.
    """
    answer.response = "n"
    _, action = handle_confirmation(interpreter, _edit_chunk(), None)
    assert action == "break"
    assert interpreter.messages[-1]["content"] == "[User declined to apply this edit.]"


def test_accepting_an_edit_returns_a_block_labelled_with_the_target(interpreter, answer):
    """"y" builds a block carrying the edit's language and target path.

    The target is what tells the user which file is about to change; a block
    without it shows an anonymous diff.
    """
    answer.response = "y"
    block, action = handle_confirmation(interpreter, _edit_chunk(), None)
    assert action == "continue"
    assert block.language == "write"
    assert block.target_path == "/tmp/target.py"


def test_an_unanswerable_edit_is_declined_with_the_no_approver_notice(interpreter, answer):
    """Edits get the same "nothing can approve this" treatment as code."""
    answer.response = NoInteractiveInput("prompt", ("y", "n"))
    _, action = handle_confirmation(interpreter, _edit_chunk(), None)
    assert action == "break"
    assert interpreter.messages[-1]["content"] == NO_APPROVER_NOTICE


def test_a_dry_run_preview_is_shown_before_the_edit_prompt(interpreter, answer, capsys):
    """The dry run is printed so the user approves a diff, not a promise.

    Approving an edit without seeing what it does is the same as auto-run.
    """
    answer.response = "n"
    handle_confirmation(interpreter, _edit_chunk(dry_run_output="--- a\n+++ b\n+added line"), None)
    output = capsys.readouterr().out
    assert "Dry run" in output
    assert "+added line" in output


def test_a_failed_dry_run_is_shown_as_plain_text(interpreter, answer, capsys):
    """A dry-run error is not syntax-highlighted as the target file's language.

    Highlighting an error message as Python mangles it; the user needs to read
    the failure, not admire it.
    """
    answer.response = "n"
    handle_confirmation(
        interpreter,
        _edit_chunk(dry_run_output="patch does not apply", dry_run_ok=False),
        None,
    )
    assert "patch does not apply" in capsys.readouterr().out


def test_the_active_block_is_torn_down_before_the_prompt(interpreter, answer):
    """A live Rich display is finalised and ended before input() is called.

    Otherwise the Live viewport is still on screen when the prompt prints, and
    the cursor lands inside the panel — the prompt appears on the wrong line
    and the answer is typed over the code.
    """
    interpreter.plain_text_display = False
    calls = []
    block = SimpleNamespace(
        finalize=lambda: calls.append("finalize"),
        end=lambda: calls.append("end"),
    )
    answer.response = "n"
    handle_confirmation(interpreter, _code_chunk(), block)
    assert calls == ["finalize", "end"]
