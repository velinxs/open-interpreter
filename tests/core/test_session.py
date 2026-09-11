"""core.session.drive: one headless turn, approvals decided by a callback.

The server and the channels share this seam. Before it, the server kept its
own copy of the approval state machine and the CLI another.
"""

from interpreter.core import session
from interpreter.core.session import PAUSE, RUN, SKIP, drive

CODE_REPLY = "Sure.\n```python\nprint(6 * 7)\n```"


def _console(chunks):
    return "".join(
        c.get("content", "") for c in chunks if c.get("type") == "console" and isinstance(c.get("content"), str)
    )


def test_run_executes_pending_code_and_finishes_the_turn(offline_interpreter):
    fake = offline_interpreter.script([CODE_REPLY, "It printed 42."])
    offline_interpreter.auto_run = False

    chunks = list(drive(offline_interpreter, "what is 6*7", approve=lambda chunk: RUN))

    assert "42" in _console(chunks)
    assert offline_interpreter.messages[-1]["content"] == "It printed 42."
    assert len(fake.calls) == 2


def test_skip_records_the_decline_and_does_not_call_the_model_again(offline_interpreter):
    fake = offline_interpreter.script([CODE_REPLY])
    offline_interpreter.auto_run = False

    chunks = list(drive(offline_interpreter, "what is 6*7", approve=lambda chunk: SKIP))

    assert "42" not in _console(chunks)
    assert chunks[-1]["type"] == "confirmation"
    assert offline_interpreter.messages[-1]["content"] == session.DECLINED_NOTICE
    assert len(fake.calls) == 1


def test_pause_leaves_the_code_pending_and_a_later_run_executes_it(offline_interpreter):
    fake = offline_interpreter.script([CODE_REPLY, "It printed 42."])
    offline_interpreter.auto_run = False

    first = list(drive(offline_interpreter, "what is 6*7", approve=lambda chunk: PAUSE))
    assert first[-1]["type"] == "confirmation"
    assert offline_interpreter.messages[-1]["type"] == "code"
    assert len(fake.calls) == 1

    second = list(drive(offline_interpreter, None, approve=lambda chunk: RUN))
    assert "42" in _console(second)
    assert offline_interpreter.messages[-1]["content"] == "It printed 42."
    assert len(fake.calls) == 2


def test_auto_run_never_asks(offline_interpreter):
    offline_interpreter.script([CODE_REPLY, "Done."])
    offline_interpreter.auto_run = True
    asked = []

    chunks = list(drive(offline_interpreter, "go", approve=lambda chunk: asked.append(chunk) or SKIP))

    assert asked == []
    assert "42" in _console(chunks)
