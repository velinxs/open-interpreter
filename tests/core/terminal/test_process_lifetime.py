"""Language runtimes must not outlive the interpreter's process.

Terminal.stop() only interrupts running code; only Terminal.terminate() kills
the Jupyter kernel and shell subprocesses, and nothing called it at exit, so
every CLI session and every test left an ipykernel_launcher behind.
"""

import json
import subprocess
import sys
import time

import psutil

# Writes the spawned child PIDs to the path in argv[1]. Output goes to files,
# not pipes: a leaked child would hold a pipe open and hang the test instead
# of failing it.
SCRIPT = """
import json, sys
import psutil
from interpreter import OpenInterpreter
i = OpenInterpreter()
i.offline = True
i.disable_telemetry = True
i.terminal.run("python", "print(1)")
i.terminal.run("shell", "echo 1")
children = [p.pid for p in psutil.Process().children(recursive=True)]
open(sys.argv[1], "w").write(json.dumps(children))
sys.exit(0)
"""


def _alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_runtimes_are_terminated_at_interpreter_exit(tmp_path):
    """After a normal exit, every child the terminal spawned is gone within five seconds."""
    out = tmp_path / "children.json"
    log = tmp_path / "log.txt"
    with open(log, "w") as logfile:
        r = subprocess.run(
            [sys.executable, "-c", SCRIPT, str(out)],
            stdin=subprocess.DEVNULL,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    assert r.returncode == 0, log.read_text()
    children = json.loads(out.read_text())
    assert children, "the script should have spawned a kernel and a shell"

    deadline = time.time() + 5
    while time.time() < deadline and any(_alive(pid) for pid in children):
        time.sleep(0.2)
    survivors = [pid for pid in children if _alive(pid)]
    for pid in survivors:  # never leave them behind even when the test fails
        psutil.Process(pid).kill()
    assert not survivors, f"processes survived interpreter exit: {survivors}"


def test_a_spawned_shell_resolves_python_to_the_one_open_interpreter_runs_under(monkeypatch):
    """A shell we spawn must reach our own Python, even from a stripped PATH.

    Open Interpreter is normally launched as ~/somevenv/bin/interpreter without
    that venv activated. A bare `python3` in a command then resolved to the
    system Python, where `import interpreter` fails, and the model concluded
    Open Interpreter was not installed on the machine it was running on.
    """
    import os

    from interpreter.core.terminal.languages.bash import Bash

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    shell = Bash()
    try:
        chunks = list(shell.run("command -v python3"))
    finally:
        shell.terminate()

    resolved = "".join(c.get("content", "") for c in chunks if c.get("format") == "output").strip()
    assert resolved.startswith(os.path.dirname(sys.executable)), (
        f"spawned shell resolved python3 to {resolved!r}, "
        f"not to our own {os.path.dirname(sys.executable)!r}"
    )
