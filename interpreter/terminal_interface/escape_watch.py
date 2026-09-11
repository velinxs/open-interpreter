"""Watching for the Escape key while code is running.

Ctrl-C is a blunt instrument here: it raises in whatever the main thread is
doing, unwinds the turn, and leaves the user unsure whether the command was
killed or merely abandoned. Escape is the gentler verb people expect from a
terminal UI: stop this command, keep what it printed, stay in the session.

The watcher only makes sense while a command is executing. During that window
nothing else is reading the terminal (the executing shell has its own stdin
pipe, not ours), so it is safe to put the terminal in cbreak and read keys. The
moment the window closes the previous terminal settings go back, including when
the body raises, because a terminal left in cbreak is far worse than a missed
keystroke.
"""

import contextlib
import os
import sys
import threading

# Watchers currently holding the terminal in cbreak. Anything that needs to read
# a line from the user (every prompt goes through prompt_choice) suspends them
# first, because cbreak delivers characters one at a time without echo and
# input() in that mode reads as gibberish.
_active = []

# Escape, and the Ctrl-] that terminals without a usable Escape can send instead.
_ESCAPE_KEYS = frozenset({"\x1b", "\x1d"})


def _stdin_is_a_terminal():
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False  # a closed or exotic stdin is not one we can watch


@contextlib.contextmanager
def paused():
    """Hand the terminal back for as long as this block runs.

    Used by the prompt layer, so a question asked in the middle of an
    interruptible command still reads a normal line.
    """
    watchers = list(_active)
    for watcher in watchers:
        watcher["suspend"]()
    try:
        yield
    finally:
        for watcher in watchers:
            watcher["resume"]()


@contextlib.contextmanager
def watch_for_escape(on_escape, enabled=True):
    """Call `on_escape()` if Escape is pressed inside this block.

    Falls through to a plain no-op context when there is no terminal to read
    (a server, a pipe, a test), so callers need no branch of their own.
    """
    if not enabled or not _stdin_is_a_terminal() or os.name == "nt":
        yield
        return

    try:
        import termios
        import tty
    except ImportError:  # a POSIX without termios is not a terminal we can watch
        yield
        return

    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except Exception:
        yield
        return

    stop = threading.Event()

    def read_keys():
        import select

        while not stop.is_set():
            try:
                # A short timeout rather than a blocking read, so the thread
                # notices `stop` promptly once the command finishes.
                if suspended.is_set():
                    stop.wait(0.1)  # someone else owns the terminal right now
                    continue
                if not select.select([fd], [], [], 0.15)[0]:
                    continue
                key = os.read(fd, 1).decode("utf-8", errors="ignore")
            except Exception:
                return  # stdin went away mid-command; nothing left to watch
            if key in _ESCAPE_KEYS:
                stop.set()
                try:
                    on_escape()
                except Exception:
                    pass  # the interrupt is best-effort; never take the turn down with it
                return

    suspended = threading.Event()

    def suspend():
        suspended.set()
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except Exception:
            pass

    def resume():
        try:
            tty.setcbreak(fd)
        except Exception:
            return
        suspended.clear()

    handle = {"suspend": suspend, "resume": resume}
    reader = threading.Thread(target=read_keys, daemon=True, name="oi-escape-watch")
    try:
        tty.setcbreak(fd)
        _active.append(handle)
        reader.start()
        yield
    finally:
        stop.set()
        if handle in _active:
            _active.remove(handle)
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except Exception:
            pass
        reader.join(timeout=0.5)
