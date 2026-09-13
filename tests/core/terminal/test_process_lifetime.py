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
    """`python3` in a command must be our Python, even from a stripped PATH.

    Open Interpreter is normally launched as ~/oi-venv/bin/interpreter without
    that venv activated, so a bare `python3` resolved to the system Python where
    `import interpreter` fails, and the model concluded Open Interpreter was not
    installed on the machine it was itself running on.
    """
    from interpreter.core.terminal.languages.bash import Bash

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    shell = Bash()
    try:
        # sys.prefix, not sys.executable: Python reports the path it was invoked
        # as, which is the shim link. The prefix is what says "the same venv".
        chunks = list(shell.run('python3 -c "import sys; print(sys.prefix)"'))
    finally:
        shell.terminate()

    resolved = "".join(c.get("content", "") for c in chunks if c.get("format") == "output").strip()
    assert resolved.splitlines()[0] == sys.prefix, (
        f"spawned shell ran a Python from {resolved!r}, not our own env {sys.prefix!r}"
    )


def test_the_python_shim_does_not_shadow_the_users_other_commands(monkeypatch):
    """Only python links go on PATH, never the whole virtualenv bin directory.

    A venv's bin holds a console script for every dependency that ships one —
    `jupyter`, `httpx`, `litellm`, and single-letter ones like `i`. Putting that
    directory ahead of the user's PATH would silently replace their commands
    with ours, so the fix prepends a directory holding nothing but the Python.
    """
    import os

    from interpreter.core.terminal.languages.subprocess_language import _python_shim_dir

    shim = _python_shim_dir()
    if shim is None:  # platform could not make the links; PATH is left alone
        return

    entries = set(os.listdir(shim))
    assert entries <= {"python", "python3", f"python{sys.version_info.major}.{sys.version_info.minor}"}, (
        f"the shim directory exposes more than python: {sorted(entries)}"
    )
    venv_bin = os.path.dirname(sys.executable)
    assert shim != venv_bin, "the shim must be its own directory, not the virtualenv's bin"


def test_python_path_points_spawned_commands_at_another_interpreter(tmp_path, monkeypatch):
    """`python_path` overrides which Python a shell command's `python3` reaches.

    Without it the only choice was the Python Open Interpreter runs under, which
    is wrong for anyone whose agents should work inside a project's own venv.
    """
    import os
    import subprocess as sp

    from interpreter.core.terminal.languages.bash import Bash

    venv = tmp_path / "project-venv"
    sp.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    shell = Bash()
    shell.python_path = str(venv)  # a directory, not a binary: both are accepted
    try:
        chunks = list(shell.run('python3 -c "import sys; print(sys.prefix)"'))
    finally:
        shell.terminate()

    resolved = "".join(c.get("content", "") for c in chunks if c.get("format") == "output").strip()
    assert resolved.splitlines()[0] == str(venv), (
        f"python_path was ignored: spawned shell used {resolved!r}, not {str(venv)!r}"
    )
    assert os.path.isdir(venv)


def test_python_path_is_a_profile_setting(tmp_path):
    """A profile may set python_path, so it must exist on the interpreter.

    Profiles now skip keys the interpreter does not have, so an attribute that is
    never initialised would make the setting silently do nothing.
    """
    from interpreter import OpenInterpreter
    from interpreter.terminal_interface.profiles.profiles import apply_profile_to_object

    oi = OpenInterpreter()
    try:
        assert hasattr(oi, "python_path"), "python_path must exist or profiles will skip it"
        assert oi.python_path is None, "the default must mean 'the Python we run under'"
        apply_profile_to_object(oi, {"python_path": str(tmp_path)})
        assert oi.python_path == str(tmp_path)
    finally:
        oi.toolbox.terminate()


def test_a_python_block_that_blocks_forever_is_interrupted(monkeypatch):
    """A Python block with no output is cut off, the way a shell command is.

    Bash has had an idle timeout since a `find /` looked hung, but Python never
    did, so a call that blocks forever — a request with no timeout, a prompt
    nobody can answer — wedged the session until the user found Ctrl-C.
    """
    monkeypatch.setenv("INTERPRETER_COMMAND_IDLE_TIMEOUT", "3")

    from interpreter import OpenInterpreter

    oi = OpenInterpreter()
    try:
        started = time.monotonic()
        chunks = list(oi.toolbox.run("python", "import time\ntime.sleep(120)"))
    finally:
        oi.toolbox.terminate()

    elapsed = time.monotonic() - started
    text = "".join(c.get("content", "") for c in chunks if c.get("format") == "output")
    assert elapsed < 60, f"the block ran for {elapsed:.0f}s; the idle timeout did not fire"
    assert "Interrupted: no output for" in text, text[:200]


def test_an_interrupted_block_does_not_leak_its_output_into_the_next_one(monkeypatch):
    """The command after an interrupted one gets its own output, not the last one's.

    An interrupted execution keeps emitting its tail — traceback, REPL state,
    final idle status — after the listener has gone. Those messages were read by
    the *next* block's listener, so the next command appeared under the previous
    command's output and looked broken. This is the "one command fails and then
    the next one hangs" report.
    """
    monkeypatch.setenv("INTERPRETER_COMMAND_IDLE_TIMEOUT", "3")

    from interpreter import OpenInterpreter

    oi = OpenInterpreter()
    try:
        list(oi.toolbox.run("python", "import time\ntime.sleep(120)"))
        chunks = list(oi.toolbox.run("python", "print('SECOND')"))
    finally:
        oi.toolbox.terminate()

    text = "".join(c.get("content", "") for c in chunks if c.get("format") == "output")
    assert "SECOND" in text, f"the second block produced no output of its own: {text[:200]!r}"
    assert "Interrupted: no output for" not in text, (
        f"the interrupted block's message leaked into the next command: {text[:200]!r}"
    )


def test_a_silent_python_block_reports_that_it_is_still_running(monkeypatch):
    """Long silence is reported, so a slow command is not mistaken for a hung one.

    A block that runs for two minutes without printing looks exactly like a hang
    from the terminal. The notices say how long it has been quiet and when it
    will be interrupted, which is the difference between waiting and killing it.
    """
    monkeypatch.setenv("INTERPRETER_COMMAND_IDLE_TIMEOUT", "8")

    from interpreter import OpenInterpreter

    oi = OpenInterpreter()
    try:
        chunks = list(oi.toolbox.run("python", "import time\ntime.sleep(60)"))
    finally:
        oi.toolbox.terminate()

    notices = [c for c in chunks if c.get("type") == "notice"]
    assert notices, "a silent block produced no progress notices at all"
    assert any("Still running" in c.get("content", "") for c in notices)
    assert any("interrupt at 8s" in c.get("content", "") for c in notices), (
        "the notice must say when the block will be cut off"
    )

