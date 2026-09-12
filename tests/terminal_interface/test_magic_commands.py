"""The `%` commands: the only controls the user has mid-conversation.

They mutate message history, settings, and files on disk with no confirmation
step. %undo in particular decides how far back a session rewinds, and getting
its boundary wrong either loses work or leaves a half-removed turn that the
model then has to make sense of.
"""

import json
import subprocess
from types import SimpleNamespace

import pytest

import interpreter.terminal_interface.magic_commands as magic
from interpreter.terminal_interface.magic_commands import (
    handle_auto_run,
    handle_last_usage,
    handle_load_message,
    handle_magic_command,
    handle_save_message,
    handle_undo,
    handle_verbose,
    install_and_import,
    markdown,
)


class FakeInterpreter:
    """Records everything the magic commands do, without a real session."""

    def __init__(self, messages=None):
        self.messages = messages if messages is not None else []
        self.displayed = []
        self.verbose = False
        self.debug = False
        self.auto_run_mode = "prompt"
        self.system_message = "you are a helpful assistant"
        self.conversation_filename = "a_saved_chat.json"
        self.llm = SimpleNamespace(model="gpt-4o", last_completion_usage=None)
        self.resets = 0
        self.renames = []
        self.shell_runs = []
        self.toolbox = SimpleNamespace(run=lambda *a, **kw: self.shell_runs.append((a, kw)))

    def display_message(self, message):
        self.displayed.append(message)

    def reset(self):
        self.resets += 1

    def rename_conversation_file_from_llm_title(self, use_full_transcript=False, manual_title=None):
        self.renames.append({"use_full_transcript": use_full_transcript, "manual_title": manual_title})


def _user(content, **extra):
    return {"role": "user", "type": "message", "content": content, **extra}


def _assistant(content, **extra):
    return {"role": "assistant", "type": "message", "content": content, **extra}


def _resume_alert(content="resumed"):
    return {
        "role": "user",
        "type": "message",
        "content": content,
        "source": "terminal",
        "format": "system_alert",
        "alert_kind": "conversation_resumed",
    }


# --- dispatch ---------------------------------------------------------------


def test_unknown_commands_show_the_help_instead_of_failing_silently():
    """A mistyped command lists what is available.

    Magic commands are not discoverable and not completed by the shell.
    Swallowing an unknown one would look identical to a command that ran and
    did nothing.
    """
    interpreter = FakeInterpreter()
    handle_magic_command(interpreter, "%nosuchthing")
    joined = "\n".join(interpreter.displayed)
    assert "Unknown command" in joined
    assert "%undo" in joined


def test_the_command_and_its_argument_are_split_on_the_first_space():
    """Everything after the command name is one argument, spaces included.

    Paths with spaces are normal. Splitting on every space would truncate
    `%markdown /tmp/my notes.md` to `/tmp/my`.
    """
    interpreter = FakeInterpreter([_user("hi")])
    handle_magic_command(interpreter, "%rename  my long title ")
    assert interpreter.renames == [{"use_full_transcript": False, "manual_title": "my long title"}]


def test_rename_with_no_title_asks_the_model_for_one():
    """A bare %rename generates a title from the whole transcript.

    The two modes write the same file; passing manual_title=None with
    use_full_transcript=False would rename it to nothing at all.
    """
    interpreter = FakeInterpreter([_user("hi")])
    handle_magic_command(interpreter, "%rename")
    assert interpreter.renames == [{"use_full_transcript": True, "manual_title": None}]


def test_double_percent_runs_a_shell_command():
    """`%%ls -la` goes to the system shell, not to the model.

    It is the escape hatch for "just run this". Routing it through the model
    would cost a round trip and could rewrite the command.
    """
    interpreter = FakeInterpreter()
    handle_magic_command(interpreter, "%%echo hello")
    (args, kwargs) = interpreter.shell_runs[0]
    assert args[1] == "echo hello"
    assert args[0] in ("bash", "cmd")
    assert kwargs["stream"] is False


def test_the_renamed_debug_command_redirects_to_verbose(capsys):
    """%debug still works and says it was renamed.

    It was the documented way to see the message list for years. Dropping it
    would strand anyone following an old tutorial.
    """
    interpreter = FakeInterpreter()
    handle_magic_command(interpreter, "%debug false")
    assert "renamed" in capsys.readouterr().out
    assert interpreter.verbose is False
    assert any("Exited verbose mode" in message for message in interpreter.displayed)


# --- %undo ------------------------------------------------------------------


def test_undo_rewinds_to_just_before_the_last_user_message():
    """%undo removes the last user turn and everything the model did in response.

    The point is to retry a prompt. Leaving the user message in place would
    make the next model call see the question twice.
    """
    interpreter = FakeInterpreter(
        [
            _user("first question"),
            _assistant("first answer"),
            _user("second question"),
            _assistant("second answer"),
        ]
    )
    handle_undo(interpreter, "")
    assert interpreter.messages == [_user("first question"), _assistant("first answer")]


def test_undo_on_an_empty_conversation_does_nothing():
    """Nothing to undo is a no-op, not an IndexError."""
    interpreter = FakeInterpreter()
    handle_undo(interpreter, "")
    assert interpreter.messages == []


def test_terminal_injected_messages_are_not_undo_boundaries():
    """"[User declined to run this code.]" is not a user turn.

    Those entries have role=user so the model reads them, but they were
    written by the terminal. Treating one as a boundary would make %undo
    remove only the notice and leave the code block it refers to.
    """
    declined = _user("[User declined to run this code.]", source="terminal")
    interpreter = FakeInterpreter(
        [
            _user("run something"),
            {"role": "assistant", "type": "code", "format": "python", "content": "rm -rf /"},
            declined,
        ]
    )
    handle_undo(interpreter, "")
    assert interpreter.messages == []


def test_a_trailing_resume_alert_survives_undo():
    """The resumed-session alert is re-attached after the rollback.

    It describes the state of the world (fresh kernel, reset cwd), not
    something the user said. Undoing it would leave the model believing its
    old Python variables still exist.
    """
    interpreter = FakeInterpreter([_user("older"), _assistant("reply"), _resume_alert()])
    handle_undo(interpreter, "")
    assert interpreter.messages == [_resume_alert()]


def test_stacked_resume_alerts_collapse_to_the_newest_one():
    """Repeated resumes leave one alert, not a pile of them.

    Each resume appends an alert and each undo re-attaches whatever it found.
    Without the collapse, a resume/undo cycle grows the history by one
    contradictory "just resumed" notice every time.
    """
    interpreter = FakeInterpreter([_user("older"), _resume_alert("first"), _resume_alert("second")])
    handle_undo(interpreter, "")
    assert interpreter.messages == [_resume_alert("second")]


def test_undo_previews_what_it_removed_without_previewing_terminal_noise():
    """The user is shown what disappeared, labelled by kind, but not the injected entries.

    %undo is destructive and irreversible. The preview is the only record of
    what was thrown away; terminal-injected entries are excluded because the
    user never wrote them.
    """
    interpreter = FakeInterpreter(
        [
            _user("delete the file"),
            {"role": "assistant", "type": "code", "format": "python", "content": "import os"},
            {"role": "computer", "type": "console", "format": "output", "content": "ok"},
            _user("[User declined to run this code.]", source="terminal"),
        ]
    )
    handle_undo(interpreter, "")
    joined = "\n".join(interpreter.displayed)
    assert "Removed user message" in joined
    assert "Removed assistant code block" in joined
    assert "Removed tool output" in joined
    assert "declined" not in joined


def test_undo_previews_are_flattened_to_one_line():
    """Multi-line content is collapsed before it goes into an inline code span.

    A newline inside Rich's inline code breaks out of the span, so a removed
    code block starting with "# comment" would render the rest as a heading.
    """
    interpreter = FakeInterpreter([_user("go"), {"role": "assistant", "type": "code", "content": "# one\nprint(1)"}])
    handle_undo(interpreter, "")
    previews = [m for m in interpreter.displayed if "Removed" in m]
    assert all("\n" not in preview for preview in previews)


# --- settings toggles -------------------------------------------------------


@pytest.mark.parametrize(
    "argument,expected",
    [("", "all"), ("true", "all"), ("all", "all"), ("false", "prompt"), ("prompt", "prompt"), ("allowlist", "allowlist"), ("denylist", "denylist")],
)
def test_auto_run_accepts_every_documented_spelling(argument, expected):
    """%auto_run maps each accepted word to a mode.

    This is how a user turns off confirmation mid-session. A word that falls
    through to the unknown branch leaves the previous mode in place while
    printing nothing that says so.
    """
    interpreter = FakeInterpreter()
    handle_auto_run(interpreter, argument)
    assert interpreter.auto_run_mode == expected


def test_auto_run_rejects_an_unrecognised_mode():
    """An unknown argument changes nothing and says so.

    Silently defaulting to "all" here would run code without asking, from a
    typo.
    """
    interpreter = FakeInterpreter()
    handle_auto_run(interpreter, "yolo")
    assert interpreter.auto_run_mode == "prompt"
    assert any("Unknown argument" in message for message in interpreter.displayed)


def test_verbose_dumps_the_message_list_but_truncates_inline_images(capsys):
    """%verbose prints the raw history with base64 image blobs shortened.

    The whole point is to read the history. A single inline image would flood
    the terminal with tens of thousands of characters and push everything
    else out of the scrollback.
    """
    interpreter = FakeInterpreter(
        [
            _user("look"),
            {"role": "user", "type": "image", "format": "base64", "content": "A" * 5000},
        ]
    )
    handle_verbose(interpreter, "")
    output = capsys.readouterr().out
    assert interpreter.verbose is True
    assert "A" * 200 not in output
    assert "..." in output


def test_verbose_leaves_image_paths_readable(capsys):
    """A path-format image is shown in full; there is nothing to truncate.

    Truncating it would hide the filename, which is the only useful part.
    """
    interpreter = FakeInterpreter([{"role": "user", "type": "image", "format": "path", "content": "/tmp/photo.png"}])
    handle_verbose(interpreter, "")
    assert "/tmp/photo.png" in capsys.readouterr().out


def test_reset_clears_the_session():
    """%reset delegates to interpreter.reset() and confirms it happened."""
    interpreter = FakeInterpreter([_user("hi")])
    handle_magic_command(interpreter, "%reset")
    assert interpreter.resets == 1


# --- saving and loading -----------------------------------------------------


def test_save_defaults_to_messages_json_in_the_working_directory(tmp_path, monkeypatch):
    """%save_message with no path writes ./messages.json.

    The documented default. Writing somewhere else would leave the user
    hunting for a file they were told was in the current directory.
    """
    monkeypatch.chdir(tmp_path)
    interpreter = FakeInterpreter([_user("hi")])
    handle_save_message(interpreter, "")
    assert json.loads((tmp_path / "messages.json").read_text()) == [_user("hi")]


def test_save_appends_a_json_extension(tmp_path, monkeypatch):
    """A path without .json gets one, so %load_message can find it again.

    Both commands apply the same rule; if only one did, a saved file would be
    unloadable under the name the user typed.
    """
    monkeypatch.chdir(tmp_path)
    interpreter = FakeInterpreter([_user("hi")])
    handle_save_message(interpreter, "backup")
    assert (tmp_path / "backup.json").exists()

    loaded = FakeInterpreter()
    handle_load_message(loaded, "backup")
    assert loaded.messages == [_user("hi")]


def test_loading_a_file_with_several_resume_alerts_keeps_only_the_last(tmp_path, monkeypatch):
    """Stale resume alerts in a saved file are dropped on load.

    Old saved conversations accumulated one alert per resume. Replaying all of
    them tells the model its environment was reset several times in a row,
    which is both false and confusing.
    """
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "old.json"
    path.write_text(json.dumps([_resume_alert("first"), _user("hi"), _resume_alert("second"), _resume_alert("third")]))

    interpreter = FakeInterpreter()
    handle_load_message(interpreter, "old.json")
    alerts = [m for m in interpreter.messages if m.get("alert_kind") == "conversation_resumed"]
    assert alerts == [_resume_alert("third")]
    assert _user("hi") in interpreter.messages


# --- exports ----------------------------------------------------------------


def test_markdown_export_refuses_an_empty_conversation(capsys, tmp_path, monkeypatch):
    """Exporting nothing prints a message instead of writing an empty file.

    An empty .md in Downloads looks like a failed export with no explanation.
    """
    monkeypatch.setattr(magic, "get_downloads_path", lambda: str(tmp_path))
    markdown(FakeInterpreter(), "")
    assert "No messages to export" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_markdown_export_defaults_to_downloads_named_after_the_conversation(tmp_path, monkeypatch):
    """With no path, the export lands in Downloads under the conversation's own name.

    The default has to be somewhere the user can find. Naming it after the
    conversation is what makes several exports distinguishable.
    """
    monkeypatch.setattr(magic, "get_downloads_path", lambda: str(tmp_path))
    interpreter = FakeInterpreter([_user("hi"), _assistant("hello")])
    markdown(interpreter, "")
    assert (tmp_path / "a_saved_chat.md").exists()


def test_markdown_final_drops_reasoning_blocks(tmp_path, monkeypatch):
    """%markdown_final exports answers and code only, not the model's thinking.

    Reasoning text is often longer than the answer and is not meant to be
    shared. The two commands differ by exactly this flag.
    """
    monkeypatch.setattr(magic, "get_downloads_path", lambda: str(tmp_path))
    interpreter = FakeInterpreter(
        [
            _user("hi"),
            _assistant("let me think", format="reasoning"),
            _assistant("the answer"),
        ]
    )
    magic.markdown_final(interpreter, str(tmp_path / "final.md"))
    text = (tmp_path / "final.md").read_text()
    assert "the answer" in text
    assert "let me think" not in text


# --- usage and tokens -------------------------------------------------------


def test_usage_explains_itself_when_the_provider_reported_nothing():
    """%usage with no recorded usage names the likely cause.

    Many OpenAI-compatible servers omit usage from a stream unless asked. A
    bare "no usage" would read as a bug in Open Interpreter.
    """
    interpreter = FakeInterpreter()
    handle_last_usage(interpreter, "")
    assert any("include_usage" in message for message in interpreter.displayed)


def test_usage_renders_the_recorded_numbers():
    """Recorded usage is formatted for display rather than dumped raw."""
    interpreter = FakeInterpreter()
    interpreter.llm.last_completion_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }
    handle_last_usage(interpreter, "")
    assert "100" in "\n".join(interpreter.displayed)


def test_token_count_reports_context_and_prompt_separately():
    """%tokens <prompt> breaks the estimate into context, prompt, and total.

    The number that matters before sending a long prompt is the total, and the
    split is what tells the user whether the history or the prompt is the
    expensive half.
    """
    interpreter = FakeInterpreter([_user("hello")])
    handle_magic_command(interpreter, "%tokens some extra prompt text")
    joined = "\n".join(interpreter.displayed)
    assert "Tokens sent with next request as context" in joined
    assert "Tokens used by this prompt" in joined
    assert "Total tokens for next request" in joined


def test_token_count_without_a_prompt_reports_only_the_context():
    """A bare %tokens does not invent a prompt line."""
    interpreter = FakeInterpreter([_user("hello")])
    handle_magic_command(interpreter, "%tokens")
    joined = "\n".join(interpreter.displayed)
    assert "Tokens sent with next request as context" in joined
    assert "Tokens used by this prompt" not in joined


# --- misc -------------------------------------------------------------------


def test_width_reports_both_detection_methods():
    """%width shows shutil's answer, the OS's answer, and the env overrides.

    It exists because COLUMNS/LINES silently override auto-detection and break
    resizing. Showing only one source would hide exactly the conflict the
    command is for.
    """
    interpreter = FakeInterpreter()
    handle_magic_command(interpreter, "%width")
    message = interpreter.displayed[0]
    assert "shutil.get_terminal_size()" in message
    assert "os.get_terminal_size()" in message
    assert "COLUMNS" in message


def test_install_and_import_returns_an_already_installed_module():
    """A package that is already importable is returned without touching pip."""
    assert install_and_import("json") is json


def test_a_failed_install_raises_unbound_local_instead_of_reporting_it(monkeypatch, capsys):
    """Characterisation bug: the failure path never assigns `module`.

    When pip and pip3 both fail, the function prints "Failed to install
    package" and returns — but the `finally` block then runs
    `globals()[package] = module`, and `module` was never bound. The user sees
    an UnboundLocalError traceback instead of the message written for them.
    The same hole swallows the success case of the pip3 retry.
    """

    def _fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "pip")

    monkeypatch.setattr(magic.subprocess, "check_call", _fail)
    with pytest.raises(UnboundLocalError):
        install_and_import("definitely_not_a_real_package_xyz")
    assert "Failed to install package" in capsys.readouterr().out
