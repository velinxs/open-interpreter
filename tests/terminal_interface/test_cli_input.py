"""Terminal escape bytes that arrive on the same line as typed input.

A mouse movement or a device-status reply lands in stdin as though typed, so
whatever cli_input returns has to have them removed -- otherwise they end up
in the message sent to the model, and in the saved conversation.

The prompt_choice half of this behaviour is covered in
tests/core/test_prompt_choice.py, next to the module it belongs to.
"""

from interpreter.terminal_interface.utils.cli_input import cli_input


def test_cli_input_strips_mouse_reports(monkeypatch):
    """Mouse-report bytes arriving mid-prompt never reach the caller."""
    monkeypatch.setattr("builtins.input", lambda _prompt="": "hi\x1b[<35;25;57M\x1b[61;4R")
    assert cli_input("> ") == "hi"


def test_cli_input_mouse_only_becomes_empty(monkeypatch):
    """A prompt holding only leaked reports reads as blank (ignored downstream)."""
    monkeypatch.setattr("builtins.input", lambda _prompt="": "\x1b[<35;25;57M\x1b[61;4R")
    assert cli_input("> ") == ""
