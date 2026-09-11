import pytest

import interpreter.core.utils.execution_allowlist as allowlist_module
from interpreter.core.core import OpenInterpreter
from interpreter.core.utils.execution_allowlist import (
    is_execution_allowlisted,
    normalize_auto_run_mode,
    persist_allowlist_rule,
    should_require_execution_confirmation,
    should_require_execution_confirmation_for_code,
)


@pytest.fixture(autouse=True)
def _isolate_allowlist_files(tmp_path, monkeypatch):
    """Point the allow/deny list files at the test's own directory.

    These default to the user's config directory, so without this a developer
    who has ever answered "a" at a run prompt fails the suite, and a test that
    persists a rule writes into their real allowlist.
    """
    monkeypatch.setattr(allowlist_module, "DEFAULT_ALLOWLIST_FILE", str(tmp_path / "allowlist.yaml"))
    monkeypatch.setattr(allowlist_module, "DEFAULT_DENYLIST_FILE", str(tmp_path / "denylist.yaml"))


def _interpreter(**kwargs):
    interpreter = OpenInterpreter()
    interpreter.auto_run_allowlist_file = allowlist_module.DEFAULT_ALLOWLIST_FILE
    interpreter.auto_run_denylist_file = allowlist_module.DEFAULT_DENYLIST_FILE
    for key, value in kwargs.items():
        setattr(interpreter, key, value)
    return interpreter


def test_normalize_auto_run_mode():
    assert normalize_auto_run_mode(False) == "prompt"
    assert normalize_auto_run_mode(True) == "all"
    assert normalize_auto_run_mode("allowlist") == "allowlist"
    assert normalize_auto_run_mode("all") == "all"


def test_auto_run_bool_property_backcompat():
    interpreter = OpenInterpreter()
    assert interpreter.auto_run is False
    assert interpreter.auto_run_mode == "prompt"

    interpreter.auto_run = True
    assert interpreter.auto_run is True
    assert interpreter.auto_run_mode == "all"

    interpreter.auto_run = "allowlist"
    assert interpreter.auto_run is False
    assert interpreter.auto_run_mode == "allowlist"


def test_builtin_allowlist_exact_matches():
    interpreter = _interpreter(auto_run_mode="allowlist")

    assert is_execution_allowlisted(interpreter, "bash", "ls")
    assert is_execution_allowlisted(interpreter, "shell", "ls")
    assert is_execution_allowlisted(interpreter, "cmd", "dir")
    assert is_execution_allowlisted(interpreter, "python", "help(os)")

    assert not is_execution_allowlisted(interpreter, "bash", "ls -la")
    assert not is_execution_allowlisted(interpreter, "bash", "ls; rm -rf /")
    assert not is_execution_allowlisted(interpreter, "python", "help(json)")
    assert not is_execution_allowlisted(interpreter, "python", "print(1)")


def test_allowlist_disabled_outside_allowlist_mode():
    interpreter = _interpreter(auto_run_mode="prompt")
    assert not is_execution_allowlisted(interpreter, "bash", "ls")


def test_should_require_execution_confirmation_for_code():
    interpreter = _interpreter(auto_run_mode="all")
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "rm -rf /") is False

    interpreter.auto_run_mode = "allowlist"
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "ls") is False
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "pwd") is True

    interpreter.auto_run_mode = "prompt"
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "ls") is True


def test_should_require_execution_confirmation_chunk():
    interpreter = _interpreter(auto_run_mode="allowlist")

    allowlisted = {
        "type": "confirmation",
        "format": "execution",
        "content": {"type": "code", "format": "bash", "content": "ls"},
    }
    blocked = {
        "type": "confirmation",
        "format": "execution",
        "content": {"type": "code", "format": "bash", "content": "pwd"},
    }
    edit = {
        "type": "confirmation",
        "format": "edit",
        "content": {"format": "python", "content": "x = 1", "target": "a.py"},
    }

    assert should_require_execution_confirmation(interpreter, allowlisted) is False
    assert should_require_execution_confirmation(interpreter, blocked) is True
    assert should_require_execution_confirmation(interpreter, edit) is True


def test_edit_confirmation_follows_auto_run_all():
    """
    auto_run_mode "all" is the yolo setting: it covers edits too.

    Edits used to be confirmed unconditionally, which left a headless run
    (server mode, no TTY) with a confirmation nobody could answer. Every mode
    other than "all" still confirms them.
    """
    edit = {
        "type": "confirmation",
        "format": "edit",
        "content": {"format": "python", "content": "x = 1", "target": "a.py"},
    }

    assert should_require_execution_confirmation(_interpreter(auto_run_mode="all"), edit) is False

    for mode in ("allowlist", "prompt"):
        interpreter = _interpreter(auto_run_mode=mode)
        assert should_require_execution_confirmation(interpreter, edit) is True


def test_denylist_mode_runs_everything_except_matched_rules():
    """Denylist inverts the allowlist: unmatched code runs, matched code confirms."""
    interpreter = _interpreter(auto_run_mode="denylist")

    # Ordinary work is not interrupted.
    for code in ("ls -la", "pip install requests", "cat /etc/hostname"):
        assert should_require_execution_confirmation_for_code(interpreter, "bash", code) is False

    # Built-in destructive rules still stop for a human.
    for code in (
        "rm -rf /",
        "rm -rf ~",
        "sudo rm -fr /home/{user}",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
    ):
        assert should_require_execution_confirmation_for_code(interpreter, "bash", code) is True


def test_denylist_matches_inside_a_larger_script():
    """A dangerous line buried in a longer block must still be caught."""
    interpreter = _interpreter(auto_run_mode="denylist")
    script = "echo starting\ncd /tmp\nrm -rf /var/important\necho done"
    assert should_require_execution_confirmation_for_code(interpreter, "bash", script) is True


def test_denylist_python_rules():
    interpreter = _interpreter(auto_run_mode="denylist")
    assert should_require_execution_confirmation_for_code(interpreter, "python", "print(1)") is False
    assert (
        should_require_execution_confirmation_for_code(interpreter, "python", "import shutil; shutil.rmtree('/data')")
        is True
    )


def test_denylist_file_rules(tmp_path):
    denylist_file = tmp_path / "denylist.yaml"
    denylist_file.write_text("rules:\n  - language: bash\n    match: regex\n    pattern: 'terraform\\s+destroy'\n")
    interpreter = _interpreter(
        auto_run_mode="denylist",
        auto_run_denylist_file=str(denylist_file),
    )
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "terraform plan") is False
    assert (
        should_require_execution_confirmation_for_code(interpreter, "bash", "terraform destroy -auto-approve") is True
    )


def test_denylist_is_ignored_in_yolo_mode():
    """auto_run "all" means all - the denylist does not apply."""
    interpreter = _interpreter(auto_run_mode="all")
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "rm -rf /") is False


def test_invalid_regex_rule_is_rejected():
    interpreter = _interpreter(
        auto_run_mode="denylist",
        auto_run_denylist_rules=[{"language": "bash", "match": "regex", "pattern": "("}],
    )
    with pytest.raises(ValueError, match="Invalid regex"):
        should_require_execution_confirmation_for_code(interpreter, "bash", "ls")


def test_persist_allowlist_rule(tmp_path):
    allowlist_file = tmp_path / "allowlist.yaml"
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_file=str(allowlist_file),
    )

    rule, added = persist_allowlist_rule(interpreter, "bash", "pwd")
    assert added is True
    assert rule["pattern"] == "pwd"

    assert is_execution_allowlisted(interpreter, "bash", "pwd")

    _, added_again = persist_allowlist_rule(interpreter, "bash", "pwd")
    assert added_again is False

    assert allowlist_file.is_file()


def test_profile_rule_merge():
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_rules=[
            {"language": "bash", "match": "exact", "pattern": "pwd"},
        ],
    )
    assert is_execution_allowlisted(interpreter, "bash", "pwd")
    assert is_execution_allowlisted(interpreter, "bash", "ls")
