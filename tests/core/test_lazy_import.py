"""Deferred imports of optional packages, and what happens when one cannot load.

Twelve toolbox modules reach their optional dependency through lazy_import, so
this placeholder sits between the interpreter and every package that may not
be installed or may refuse to import on this machine (pynput without a
display, for instance). Two things must hold: merely looking at the
placeholder — which is what a traceback printer does — may never execute the
import, and a failed import must keep failing with the original error instead
of leaving a half-built module in sys.modules.
"""

import sys
import types

import pytest

from interpreter.core.utils.lazy_import import lazy_import


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    """Write single-file modules into an isolated directory on sys.path.

    Returns a factory; every module it creates is purged from sys.modules
    afterwards so a failed import in one test cannot leak into the next.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    created = []

    def _write(name, code):
        (tmp_path / f"{name}.py").write_text(code)
        created.append(name)
        return name

    yield _write

    for name in created:
        sys.modules.pop(name, None)


def test_a_missing_optional_module_is_none_rather_than_an_error():
    """A package that is not installed comes back as None.

    This is the entire point of the `optional` default: the toolbox modules
    check the result for None and disable the feature. Raising here would
    turn an absent optional dependency into a startup crash.
    """
    assert lazy_import("oi_no_such_module_xyz_123") is None


def test_introspecting_a_broken_module_never_executes_it(fake_module):
    """Reading __file__ and __name__ does not trigger the real import.

    Traceback rendering calls inspect.getmodule, which touches exactly these
    attributes on every module in sys.modules. If that executed the import,
    a module that cannot load (pynput with no display) would raise from
    inside the error printer and bury the error actually being reported.
    """
    fake_module("oi_broken_mod", "raise ImportError('no display here')\n")
    module = lazy_import("oi_broken_mod")

    assert module.__file__.endswith("oi_broken_mod.py")
    assert module.__name__ == "oi_broken_mod"
    # The placeholder is still what sys.modules holds, not a poisoned
    # half-imported module left behind by a failed exec_module.
    assert sys.modules["oi_broken_mod"] is module


def test_real_use_raises_the_original_import_error_every_time(fake_module):
    """Touching a real attribute raises the module's own error, and keeps raising it.

    The caller has to learn why the optional package is unusable, so the
    first real use must surface the real exception rather than an AttributeError.
    Re-executing a module that already failed can half-apply its side effects,
    so the failure is cached and replayed instead.
    """
    fake_module("oi_broken_use", "raise ImportError('no display here')\n")
    module = lazy_import("oi_broken_use")

    with pytest.raises(ImportError, match="no display"):
        module.anything
    with pytest.raises(ImportError, match="no display"):
        module.anything
    assert sys.modules["oi_broken_use"] is module


def test_a_working_module_loads_on_use_and_replaces_the_placeholder(fake_module):
    """A healthy module is indistinguishable from a normal import once used.

    Anything else that imports the same package must get the real module, not
    the placeholder, so the loaded module has to be swapped into sys.modules.
    """
    fake_module("oi_good_mod", "VALUE = 42\n")
    module = lazy_import("oi_good_mod")

    assert module.VALUE == 42
    real = sys.modules["oi_good_mod"]
    assert real is not module
    assert type(real) is types.ModuleType
    assert real.VALUE == 42


def test_an_already_imported_module_is_returned_as_is():
    """A package that is already in sys.modules is handed back directly.

    Wrapping it again would replace a live module object other code already
    holds references to.
    """
    assert lazy_import("json") is sys.modules["json"]
