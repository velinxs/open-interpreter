"""Shell output must reach the reader while the command is still running.

bash and cmd used to have their output held back until the command exited, to
reduce flicker in the live display. The cost was that a long command showed
nothing at all: no progress, no partial results, and no way to tell a slow
command from a wedged one. A command that prints nothing for two minutes is
also killed by the idle timeout, so the user saw a freeze and then a kill.
"""

import time

import pytest

from interpreter.core.core import OpenInterpreter


@pytest.fixture
def interpreter():
    interp = OpenInterpreter()
    interp.offline = True
    interp.disable_telemetry = True
    yield interp
    interp.terminal.terminate()


def test_output_arrives_while_the_command_is_still_running(interpreter):
    """Three lines a second apart must not arrive together at the end."""
    code = 'for n in 1 2 3; do echo "tick $n"; sleep 0.4; done'

    start = time.monotonic()
    arrivals = [
        time.monotonic() - start
        for chunk in interpreter.terminal.run("bash", code, stream=True, display=False)
        if chunk.get("type") == "console" and "tick" in str(chunk.get("content", ""))
    ]

    assert arrivals, "no console output was produced"
    assert arrivals[0] < 0.9, f"first line took {arrivals[0]:.2f}s; output is being withheld"


def test_no_placeholder_note_replaces_the_output(interpreter):
    """The 'output will be shown after completion' placeholder is gone."""
    chunks = list(interpreter.terminal.run("bash", "echo hi", stream=True, display=False))
    text = "".join(str(c.get("content", "")) for c in chunks)

    assert "hi" in text
    assert "after completion" not in text
