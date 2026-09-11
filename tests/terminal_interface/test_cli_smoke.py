"""The CLI must start, take a message on stdin, run code, and print the result."""

import os
import subprocess
import sys
from pathlib import Path

PROFILE = Path(__file__).resolve().parents[1] / "support" / "fake_profile.py"


def test_cli_stdin_mode_runs_a_turn(tmp_path):
    """interpreter --stdin reads one line, drives a turn, and prints console output.

    Runs the real entry point in a subprocess with a profile that fakes the
    model, so argument parsing, profile loading, rendering, and execution are
    all exercised end to end without a provider.
    """
    code = (
        "import sys;"
        "sys.argv=['interpreter','--stdin','-y','--disable_telemetry','--plain','--profile',sys.argv[1]];"
        "from interpreter.terminal_interface.start_terminal_interface import main;"
        "main()"
    )
    env = dict(os.environ, HOME=str(tmp_path), TERM="dumb", NO_COLOR="1")
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(PROFILE)],
        input="what is 21*2\n",
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "42" in result.stdout, result.stdout
