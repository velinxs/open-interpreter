"""Small core utilities: recipient tagging, temporary files, and the safe-mode scanner.

Each is a few lines, and each sits under something that fails quietly. A
truncated recipient payload silently changes what the model is told; a
temporary file that is never cleaned up leaks the code being scanned into
/tmp; a scanner that raises takes the turn down over an optional feature.
"""

import os
from types import SimpleNamespace

import pytest

import interpreter.core.utils.scan_code as scan_code_module
from interpreter.core.utils.recipient_utils import format_to_recipient, parse_for_recipient
from interpreter.core.utils.scan_code import scan_code
from interpreter.core.utils.temporary_file import cleanup_temporary_file, create_temporary_file

# --- recipient tagging ------------------------------------------------------


def test_untagged_content_is_returned_with_no_recipient():
    """Ordinary output has no recipient and comes back unchanged.

    Almost every chunk takes this path. Mangling it would corrupt normal
    program output.
    """
    assert parse_for_recipient("just some output") == (None, "just some output")


def test_a_tagged_message_is_routed_to_its_recipient():
    """The wrapper marks output as meant for the model rather than the user.

    Toolbox helpers use it to pass instructions the terminal should not show.
    Losing the recipient prints machine-to-machine text into the transcript.
    """
    tagged = format_to_recipient("run the next step", "assistant")
    assert parse_for_recipient(tagged) == ("assistant", "run the next step")


def test_content_containing_a_colon_is_truncated():
    """Characterisation bug: the payload is split on ":" and only one field kept.

    parse_for_recipient does `parts[2].split(":")[1]`, so a message containing
    a colon — a URL, a timestamp, a dict repr, a Windows path — loses
    everything from the second colon onwards. Here "http://example.com" comes
    back as "http". Nothing in the tree currently sends colons through this
    path, which is why it has gone unnoticed.
    """
    tagged = format_to_recipient("http://example.com", "assistant")
    recipient, content = parse_for_recipient(tagged)
    assert recipient == "assistant"
    assert content == "http"


def test_a_truncated_wrapper_is_treated_as_plain_content():
    """Text that starts like a tag but never ends is not parsed as one.

    Streamed content arrives in fragments; parsing a half-written tag would
    route a partial message to the wrong place.
    """
    assert parse_for_recipient("@@@RECIPIENT:assistant@@@CONTENT:half") == (
        None,
        "@@@RECIPIENT:assistant@@@CONTENT:half",
    )


# --- temporary files --------------------------------------------------------


def test_a_temporary_file_gets_the_requested_extension_and_contents():
    """The extension matters: semgrep picks its rules from it.

    A .txt file gets scanned with no language rules at all, so safe mode would
    report "no issues" for anything.
    """
    path = create_temporary_file("print(1)", "py")
    try:
        assert path.endswith(".py")
        assert open(path).read() == "print(1)"
    finally:
        os.remove(path)


def test_a_temporary_file_without_an_extension_is_allowed():
    """Languages with no file extension still get a file."""
    path = create_temporary_file("echo hi")
    try:
        assert os.path.isfile(path)
    finally:
        os.remove(path)


def test_cleanup_removes_the_file():
    """The scratch file holds the code being scanned and must not survive the scan."""
    path = create_temporary_file("secret = 'value'", "py")
    cleanup_temporary_file(path)
    assert not os.path.exists(path)


def test_cleaning_up_a_missing_file_reports_but_does_not_raise(capsys):
    """A double cleanup is noisy, not fatal.

    This runs in the scanner's tail; raising would turn an optional safety
    feature into a crash.
    """
    cleanup_temporary_file("/definitely/not/a/file")
    assert "Could not clean up temporary file" in capsys.readouterr().out


def test_creating_an_unwritable_file_reports_none(capsys, monkeypatch):
    """A failure to create the scratch file returns None instead of raising.

    Same reason: the caller is an optional feature, not the turn itself.
    """

    def _boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr("tempfile.NamedTemporaryFile", _boom)
    assert create_temporary_file("x", "py") is None
    assert "Could not create temporary file" in capsys.readouterr().out


# --- safe-mode scanning -----------------------------------------------------


@pytest.fixture
def interpreter(monkeypatch):
    """An interpreter stub whose language lookup returns a real-looking language."""
    language = SimpleNamespace(name="Python", file_extension="py")
    return SimpleNamespace(
        verbose=False,
        safe_mode="auto",
        terminal=SimpleNamespace(get_language=lambda name: language),
    )


def test_a_clean_scan_reports_no_issues_and_names_the_language(interpreter, monkeypatch, capsys):
    """A zero exit code means the code passed, and the message says what was scanned.

    In "auto" mode nothing else confirms the scan ran; without the line the
    user cannot tell scanning from not scanning.
    """
    monkeypatch.setattr(
        scan_code_module.subprocess,
        "run",
        lambda command, shell=False: SimpleNamespace(returncode=0),
    )
    scan_code("print(1)", "python", interpreter)
    output = capsys.readouterr().out
    assert "No issues were found in this Python code" in output
    assert "Code Scanner:" in output


def test_a_failing_scan_says_nothing_rather_than_claiming_the_code_is_clean(interpreter, monkeypatch, capsys):
    """A non-zero exit code (semgrep found something) must not print "no issues".

    semgrep's own findings go to the terminal. Adding a reassuring line under
    them would contradict what the user just read.
    """
    monkeypatch.setattr(
        scan_code_module.subprocess,
        "run",
        lambda command, shell=False: SimpleNamespace(returncode=1),
    )
    scan_code("import os; os.system('rm -rf /')", "python", interpreter)
    assert "No issues were found" not in capsys.readouterr().out


def test_a_missing_semgrep_is_reported_and_does_not_stop_the_turn(interpreter, monkeypatch, capsys):
    """Without semgrep installed the scan explains itself instead of raising.

    Safe mode is opt-in and semgrep is an optional dependency; an exception
    here would make --safe unusable for everyone who has not installed it.
    """

    def _boom(*args, **kwargs):
        raise FileNotFoundError("semgrep")

    monkeypatch.setattr(scan_code_module.subprocess, "run", _boom)
    scan_code("print(1)", "python", interpreter)
    assert "Have you installed 'semgrep'?" in capsys.readouterr().out


def test_the_scanned_file_is_deleted_afterwards(interpreter, monkeypatch):
    """The temporary copy of the code does not outlive the scan.

    It contains whatever the model was about to run, including any secrets in
    it, and /tmp is world-readable on most systems.
    """
    seen = {}

    def _run(command, shell=False):
        seen["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scan_code_module.subprocess, "run", _run)
    scan_code("print(1)", "python", interpreter)

    file_name = seen["command"].split()[-1]
    directory = seen["command"].split()[1]
    assert not os.path.exists(os.path.join(directory, file_name))
