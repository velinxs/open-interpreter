"""Action modules: capabilities that cost nothing until they are used.

Every toolbox method is listed in the system message, so it is paid for on
every request whether the session touches it or not. That is the right trade
for a handful of general capabilities and the wrong one for a long tail of
specific ones. An action is an ordinary Python file whose name and one-line
summary are all the model sees until it asks for more.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from interpreter.core.toolbox.actions.actions import ActionError, Actions


@pytest.fixture
def actions(tmp_path):
    """An Actions bound to an empty directory of its own."""
    instance = Actions(SimpleNamespace())
    instance.path = tmp_path / "actions"
    instance.path.mkdir()
    return instance


def _write(actions, name, body):
    (actions.path / f"{name}.py").write_text(body, encoding="utf-8")


def test_listing_reports_each_action_and_its_first_docstring_line(actions):
    """The model is shown names and summaries, which is all discovery needs."""
    _write(actions, "deploy", '"""Ship the current branch to staging.\n\nMore detail."""\n\ndef run():\n    pass\n')

    assert actions.list() == [{"name": "deploy", "summary": "Ship the current branch to staging."}]


def test_listing_does_not_execute_the_actions(actions, tmp_path):
    """Asking what is available must not run every file on disk.

    The summary is parsed out of the source rather than imported, so a
    directory of fifty actions costs fifty docstring reads and no side effects.
    """
    marker = tmp_path / "ran"
    _write(actions, "sneaky", f'"""Looks innocent."""\n\nPath = __import__("pathlib").Path\nPath(r"{marker}").write_text("x")\n')

    actions.list()

    assert not marker.exists()


def test_an_action_with_no_docstring_still_lists(actions):
    """A missing summary is reported, not a crash — the file is still usable."""
    _write(actions, "bare", "def run():\n    pass\n")

    assert actions.list() == [{"name": "bare", "summary": "(no description)"}]


def test_underscored_files_are_not_actions(actions):
    """A leading underscore marks a helper the model should not be offered."""
    _write(actions, "_helper", '"""Shared bits."""\n')
    _write(actions, "real", '"""A real one."""\n')

    assert [a["name"] for a in actions.list()] == ["real"]


def test_loading_returns_the_module_and_its_functions_work(actions):
    """load() hands back the module so the model can call into it."""
    _write(actions, "adder", '"""Add two numbers."""\n\ndef run(a, b):\n    return a + b\n')

    module = actions.load("adder")

    assert module.run(2, 3) == 5


def test_showing_an_action_returns_its_source(actions):
    """The model (and the user) can read an action before running it."""
    source = '"""Read me first."""\n\ndef run():\n    return 1\n'
    _write(actions, "readable", source)

    assert actions.show("readable") == source


def test_an_action_that_acts_at_import_time_is_refused(actions, tmp_path):
    """Loading must define things and do nothing else.

    The approval model is that the code block the user approved is the code
    that runs. An action doing work at import time would act on the load()
    line, before the user had seen any of it — so it is refused, and the
    error says how to fix it.
    """
    marker = tmp_path / "escaped"
    _write(
        actions,
        "naughty",
        f'"""Acts immediately."""\n\nPath = __import__("pathlib").Path\nPath(r"{marker}").write_text("x")\n\ndef run():\n    pass\n',
    )

    with pytest.raises(ActionError, match="import time"):
        actions.load("naughty")

    assert not marker.exists(), "the refused action still ran"


@pytest.mark.parametrize(
    "body",
    [
        '"""Imports are fine."""\nimport os\n\ndef run():\n    pass\n',
        '"""Constants are fine."""\nTIMEOUT = 30\n\ndef run():\n    pass\n',
        '"""Classes are fine."""\n\nclass Thing:\n    pass\n',
        '"""Annotated assignment is fine."""\nX: int = 1\n\ndef run():\n    pass\n',
    ],
)
def test_definitions_and_imports_are_allowed_at_module_level(actions, body):
    """Only statements that *do* something are refused, not ordinary structure."""
    _write(actions, "fine", body)

    actions.load("fine")  # must not raise


def test_editing_an_action_takes_effect_without_a_restart(actions):
    """The module is keyed by content, so a changed file is reloaded.

    A kernel lives for the whole session. Caching by name alone would mean a
    fixed action kept running its old code until the user restarted.
    """
    _write(actions, "changing", '"""v1."""\n\ndef run():\n    return "before"\n')
    assert actions.load("changing").run() == "before"

    _write(actions, "changing", '"""v2."""\n\ndef run():\n    return "after"\n')
    assert actions.load("changing").run() == "after"


def test_a_missing_action_names_what_is_available(actions):
    """The error tells the model what it could have called instead."""
    _write(actions, "real", '"""Exists."""\n')

    with pytest.raises(ActionError, match="real"):
        actions.load("nope")


def test_a_name_cannot_escape_the_actions_directory(actions, tmp_path):
    """A traversing name is rejected rather than loading arbitrary files.

    The name reaches this from model-written code, so it is untrusted input.
    """
    outside = tmp_path / "outside.py"
    outside.write_text('"""Not an action."""\n')

    with pytest.raises(ActionError):
        actions.load("../outside")


def test_a_syntax_error_is_reported_against_the_action(actions):
    """A broken file names itself instead of raising from deep in importlib."""
    _write(actions, "broken", '"""Broken."""\n\ndef run(\n')

    with pytest.raises(ActionError, match="broken"):
        actions.load("broken")


def test_a_missing_directory_lists_nothing(actions):
    """No actions directory is an empty list, not an error."""
    actions.path = Path(tempfile.mkdtemp()) / "does-not-exist"

    assert actions.list() == []


def test_create_writes_an_action_that_can_then_be_loaded(actions):
    """The round trip works: write it, list it, load it, call it."""
    path = actions.create("greet", '"""Say hello to someone."""\n\ndef run(name):\n    return f"hi {name}"\n')

    assert Path(path).exists()
    assert {"name": "greet", "summary": "Say hello to someone."} in actions.list()
    assert actions.load("greet").run("you") == "hi you"


def test_create_refuses_an_action_that_would_act_at_import(actions):
    """The load-time rule is enforced at write time, so a bad file cannot exist.

    Refusing only on load would leave a file that lists fine and fails later;
    refusing on write means every action on disk is one that can be loaded.
    """
    with pytest.raises(ActionError, match="define things"):
        actions.create("bad", '"""Acts now."""\n\nprint("side effect")\n')

    assert actions.list() == []


def test_create_requires_a_docstring(actions):
    """Without one there is nothing to show in the listing."""
    with pytest.raises(ActionError, match="docstring"):
        actions.create("undocumented", "def run():\n    pass\n")


def test_create_rejects_a_name_that_is_not_an_identifier(actions):
    """The name becomes a filename and a module name, so it must be usable as both."""
    for bad in ["has space", "has-dash", "_private", "1leading"]:
        with pytest.raises(ActionError, match="usable action name"):
            actions.create(bad, '"""Fine."""\n')


def test_create_will_not_silently_replace_an_existing_action(actions):
    """Overwriting is possible but never accidental."""
    actions.create("keep", '"""First."""\n\ndef run():\n    return 1\n')

    with pytest.raises(ActionError, match="already exists"):
        actions.create("keep", '"""Second."""\n\ndef run():\n    return 2\n')

    actions.create("keep", '"""Second."""\n\ndef run():\n    return 2\n', overwrite=True)
    assert actions.load("keep").run() == 2
