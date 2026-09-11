"""Importing the package must be cheap and must not run anything.

Before this, `import interpreter` constructed OpenInterpreter, and with
`--os` on argv it made an HTTP call to PyPI and started the computer-use
loop before any CLI code ran.
"""

import subprocess
import sys


def _run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)


def test_import_does_not_construct_the_singleton():
    r = _run("import sys, interpreter; assert 'interpreter.core.core' not in sys.modules; print('ok')")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_singleton_is_built_on_first_access():
    r = _run("from interpreter import interpreter; print(type(interpreter).__name__)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "OpenInterpreter"


def test_public_names_still_importable():
    r = _run("from interpreter import OpenInterpreter, AsyncInterpreter, BaseLanguage, toolbox, ai2; print('ok')")
    assert r.returncode == 0, r.stderr


def test_os_flag_is_ignored_by_import():
    r = _run("import sys; sys.argv=['x','--os']; import interpreter; print('ok')")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
