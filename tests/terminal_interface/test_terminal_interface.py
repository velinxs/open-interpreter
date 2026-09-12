"""The render loop: what the user actually sees while a turn runs.

terminal_interface() consumes the chunk stream from chat() and decides what to
print, when to start and end a block, and when to stop the turn. It is a view,
but it also writes back into interpreter.messages, so a mistake here can
corrupt the conversation as well as the display. These tests drive it in plain
text mode, where the output is exactly the string under test rather than a
Rich viewport.
"""

import pytest

import interpreter.terminal_interface.terminal_interface as ti
from interpreter.core.core import OpenInterpreter
from interpreter.terminal_interface.terminal_interface import terminal_interface


@pytest.fixture
def interpreter(monkeypatch):
    """A real interpreter whose chat() replays a scripted chunk list."""
    interp = OpenInterpreter()
    interp.plain_text_display = True
    interp.auto_run = True
    interp.offline = True
    interp.disable_telemetry = True

    script = {"chunks": []}

    def _chat(message=None, display=False, stream=False):
        interp.last_message = message
        return iter(script["chunks"])

    monkeypatch.setattr(interp, "chat", _chat)
    interp.script = script
    yield interp
    try:
        interp.terminal.terminate()
    except Exception:
        pass


def _run(interpreter, message="go"):
    """Drive one non-interactive turn and return the chunks it yielded."""
    return list(terminal_interface(interpreter, message))


def _message_chunks(text, **extra):
    return [
        {"role": "assistant", "type": "message", "start": True, **extra},
        {"role": "assistant", "type": "message", "content": text, **extra},
        {"role": "assistant", "type": "message", "end": True, **extra},
    ]


# --- the loop itself --------------------------------------------------------


def test_every_chunk_is_yielded_to_the_caller(interpreter):
    """terminal_interface is a pass-through generator as well as a renderer.

    The server and the Python API consume the same stream. Swallowing a chunk
    here would make the display and the API disagree about what happened.
    """
    interpreter.script["chunks"] = _message_chunks("hello")
    assert _run(interpreter) == interpreter.script["chunks"]


def test_a_supplied_message_runs_exactly_one_turn(interpreter, capsys):
    """Passing a message means one turn, then return — no input prompt.

    This is the API and `interpreter "do the thing"` path. Looping would block
    on input() with nothing to read.
    """
    interpreter.script["chunks"] = _message_chunks("done")
    _run(interpreter)
    assert interpreter.last_message == "go"
    assert "> " not in capsys.readouterr().out


def test_an_interactive_turn_reads_from_the_prompt(interpreter, monkeypatch):
    """With no message, the loop reads one line and sends it.

    A second read raises EOFError here, which the loop turns into
    KeyboardInterrupt — the documented way the session ends.
    """
    lines = iter(["what is 21*2"])

    def _input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", _input)
    interpreter.script["chunks"] = _message_chunks("42")

    with pytest.raises(KeyboardInterrupt):
        list(terminal_interface(interpreter, None))
    assert interpreter.last_message == "what is 21*2"


def test_ctrl_d_exits_rather_than_looping_forever(interpreter, monkeypatch, capsys):
    """EOF on an empty prompt ends the session with a message.

    Without the EOFError branch, a closed stdin spins the loop reading
    nothing, forever.
    """
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError))
    with pytest.raises(KeyboardInterrupt):
        list(terminal_interface(interpreter, None))
    assert "Exiting" in capsys.readouterr().out


def test_a_preseeded_message_is_consumed_instead_of_prompting(interpreter, monkeypatch):
    """The "i {command}" entry leaves one user message in history; the loop picks it up.

    start_terminal_interface queues it before chat() runs. Prompting anyway
    would ask the user to retype what they already typed on the command line.
    """
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError))
    interpreter.messages = [{"role": "user", "type": "message", "content": "I fix the build"}]
    interpreter.script["chunks"] = _message_chunks("ok")

    with pytest.raises(KeyboardInterrupt):
        list(terminal_interface(interpreter, None))
    assert interpreter.last_message == "I fix the build"
    assert interpreter.messages == []


def test_chunks_addressed_to_the_assistant_are_yielded_but_not_printed(interpreter, capsys):
    """A chunk with recipient != "user" is passed on without being displayed.

    It is context meant for the model (tool plumbing, internal notices).
    Printing it would leak machine-to-machine traffic into the transcript.
    """
    interpreter.script["chunks"] = [
        {"role": "computer", "type": "console", "format": "output", "content": "SECRET", "recipient": "assistant"}
    ]
    yielded = _run(interpreter)
    assert len(yielded) == 1
    assert "SECRET" not in capsys.readouterr().out


# --- plain text rendering ---------------------------------------------------


def test_code_and_output_are_fenced_with_their_language(interpreter, capsys):
    """Plain text mode wraps code and console blocks in ``` fences.

    The output is being piped somewhere. The fences are the only structure a
    consumer has to tell code from prose from output.
    """
    interpreter.script["chunks"] = [
        {"role": "assistant", "type": "code", "format": "python", "start": True},
        {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"},
        {"role": "assistant", "type": "code", "format": "python", "end": True},
        {"role": "computer", "type": "console", "format": "output", "start": True},
        {"role": "computer", "type": "console", "format": "output", "content": "1"},
        {"role": "computer", "type": "console", "format": "output", "end": True},
    ]
    capsys.readouterr()
    _run(interpreter)
    text = capsys.readouterr().out
    assert "```python" in text
    assert "print(1)" in text
    assert "1" in text


def test_active_line_markers_are_not_printed(interpreter, capsys):
    """Line-highlight markers are UI state and never reach plain text output.

    They are integers. Printing them splices stray numbers into the program's
    output, which is often being parsed by whatever consumes stdin mode.
    """
    interpreter.script["chunks"] = [
        {"role": "computer", "type": "console", "format": "active_line", "content": 3},
        {"role": "computer", "type": "console", "format": "output", "content": "real output"},
    ]
    _run(interpreter)
    text = capsys.readouterr().out
    assert "real output" in text
    assert "3" not in text


def test_reasoning_is_delimited_so_it_can_be_told_from_the_answer(interpreter, capsys):
    """Thinking is bracketed by [Thinking] / [/Thinking] in plain text.

    Without the delimiters a consumer cannot separate the model's scratch work
    from its answer, and reasoning is often the longer half.
    """
    interpreter.script["chunks"] = [
        {"role": "assistant", "type": "message", "format": "reasoning", "start": True},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "hmm"},
        {"role": "assistant", "type": "message", "format": "reasoning", "end": True},
        *_message_chunks("the answer"),
    ]
    _run(interpreter)
    text = capsys.readouterr().out
    assert "[Thinking]" in text
    assert "[/Thinking]" in text
    assert text.index("[/Thinking]") < text.index("the answer")


def test_replace_chunks_are_not_printed_twice(interpreter, capsys):
    """A replace chunk only updates the stored message; its text was already streamed.

    It carries the whole message again. Printing it would duplicate every
    reasoning block in piped output.
    """
    interpreter.script["chunks"] = [
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "streamed"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "streamed", "replace": True},
        {"role": "assistant", "type": "message", "content": "body"},
        {"role": "assistant", "type": "message", "content": "body", "replace": True},
    ]
    _run(interpreter)
    text = capsys.readouterr().out
    assert text.count("streamed") == 1
    assert text.count("body") == 1


# --- confirmation handling --------------------------------------------------


def test_a_declined_confirmation_ends_the_turn(interpreter, monkeypatch, capsys):
    """When handle_confirmation says "break", the remaining chunks are dropped.

    The user said no. Continuing to render the output of code that never ran
    would show them results from nowhere.
    """
    monkeypatch.setattr(ti, "handle_confirmation", lambda interp, chunk, block: (block, "break"))
    interpreter.script["chunks"] = [
        {"role": "computer", "type": "confirmation", "format": "execution", "content": {}},
        {"role": "computer", "type": "console", "format": "output", "content": "SHOULD NOT APPEAR"},
    ]
    _run(interpreter)
    assert "SHOULD NOT APPEAR" not in capsys.readouterr().out


def test_an_approved_confirmation_continues_without_rendering_the_chunk(interpreter, monkeypatch, capsys):
    """An approved confirmation is consumed, not printed as content.

    Its content is a code payload the user has already seen in the streaming
    preview above the prompt.
    """
    monkeypatch.setattr(ti, "handle_confirmation", lambda interp, chunk, block: (block, "continue"))
    interpreter.script["chunks"] = [
        {
            "role": "computer",
            "type": "confirmation",
            "format": "execution",
            "content": {"type": "code", "format": "python", "content": "print('payload')"},
        },
    ]
    _run(interpreter)
    assert "payload" not in capsys.readouterr().out


# --- stopping -------------------------------------------------------------


def test_escape_stops_the_stream_and_keeps_what_was_printed(interpreter, monkeypatch, capsys):
    """Pressing Escape ends the turn and says the output above was kept.

    That distinction is the whole reason Escape exists alongside Ctrl-C: the
    user needs to know the printed output is real and the session is intact.
    """
    captured = {}

    class _Watch:
        def __init__(self, on_escape, enabled=True):
            captured["on_escape"] = on_escape

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(ti, "watch_for_escape", _Watch)

    def _chunks():
        yield {"role": "assistant", "type": "message", "content": "before"}
        captured["on_escape"]()
        yield {"role": "assistant", "type": "message", "content": "after"}

    interpreter.script["chunks"] = _chunks()
    _run(interpreter)
    text = capsys.readouterr().out
    assert "before" in text
    assert "after" not in text
    assert "Stopped. Everything above was kept." in text


def test_ctrl_c_during_a_supplied_message_ends_the_turn(interpreter):
    """KeyboardInterrupt in non-interactive mode returns instead of re-prompting.

    There is no prompt to return to, so continuing the loop would call input()
    on a stdin nobody is writing to.
    """

    def _chunks():
        yield {"role": "assistant", "type": "message", "content": "partial"}
        raise KeyboardInterrupt

    interpreter.script["chunks"] = _chunks()
    assert [c["content"] for c in _run(interpreter)] == ["partial"]


def test_declining_a_provider_retry_exits_and_removes_the_unanswered_message(interpreter, monkeypatch, capsys):
    """Answering "n" at the API retry prompt exits, dropping the message that never got a reply.

    Leaving it in history means the next session resends a message the user
    has already given up on, and pays for it.
    """
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")
    interpreter.messages = [{"role": "user", "type": "message", "content": "hello"}]
    interpreter._stopped_retrying = True
    interpreter.script["chunks"] = []

    with pytest.raises(SystemExit) as excinfo:
        list(terminal_interface(interpreter, None))
    assert excinfo.value.code == 1
    assert interpreter.messages == []
    assert "Stopped retrying" in capsys.readouterr().out


def test_an_unexpected_error_propagates(interpreter):
    """Rendering errors are not swallowed; the turn fails loudly.

    A caught-and-ignored exception here would leave the user staring at a
    half-drawn block with no idea anything went wrong.
    """

    def _chunks():
        yield {"role": "assistant", "type": "message", "content": "x"}
        raise ValueError("render broke")

    interpreter.script["chunks"] = _chunks()
    with pytest.raises(ValueError):
        _run(interpreter)


# --- writing back to history ------------------------------------------------


def test_stale_terminal_size_env_vars_are_cleared(interpreter, monkeypatch):
    """COLUMNS and LINES are removed so width is re-detected after a resize.

    shutil and Rich prefer the env vars. A stale COLUMNS from a parent process
    pins every panel to the wrong width for the whole session.
    """
    monkeypatch.setenv("COLUMNS", "40")
    monkeypatch.setenv("LINES", "10")
    interpreter.script["chunks"] = []
    _run(interpreter)
    assert "COLUMNS" not in ti.os.environ
    assert "LINES" not in ti.os.environ
