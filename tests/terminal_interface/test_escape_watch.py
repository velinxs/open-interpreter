"""Escape interrupts a running command without disturbing the terminal.

The watcher puts stdin in cbreak while a command runs, so the two things that
matter are that it fires on Escape and that it always hands the terminal back
in the state it found it, including when the body raises.
"""

import os
import pty
import sys
import termios
import threading
import time

import pytest

from interpreter.terminal_interface.escape_watch import watch_for_escape


@pytest.fixture
def fake_terminal(monkeypatch):
    """A real pty, so termios operates on something that behaves like a tty."""
    controller, follower = pty.openpty()

    class _Stdin:
        def fileno(self):
            return follower

        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", _Stdin())
    yield controller, follower
    os.close(controller)
    os.close(follower)


def test_escape_calls_the_handler(fake_terminal):
    controller, _ = fake_terminal
    fired = threading.Event()

    with watch_for_escape(fired.set):
        os.write(controller, b"\x1b")
        assert fired.wait(2), "Escape did not reach the handler"


def test_other_keys_are_ignored(fake_terminal):
    controller, _ = fake_terminal
    fired = threading.Event()

    with watch_for_escape(fired.set):
        os.write(controller, b"hello\n")
        time.sleep(0.4)

    assert not fired.is_set(), "ordinary typing must not interrupt the command"


def test_terminal_settings_are_restored_even_when_the_body_raises(fake_terminal):
    _, follower = fake_terminal
    before = termios.tcgetattr(follower)

    with pytest.raises(RuntimeError):
        with watch_for_escape(lambda: None):
            assert termios.tcgetattr(follower) != before, "cbreak was never entered"
            raise RuntimeError("boom")

    assert termios.tcgetattr(follower) == before


def test_no_terminal_is_a_no_op(monkeypatch):
    """A server or a pipe has no keys to read; the block must still run."""

    class _NotATty:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", _NotATty())
    ran = []

    with watch_for_escape(lambda: pytest.fail("must not fire")):
        ran.append(True)

    assert ran == [True]
