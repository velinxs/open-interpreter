"""What a session says about itself before the first prompt.

Starting the interpreter printed nothing in the common case: the one-time
welcome is only reached from a branch in validate_llm_settings, and the
approval notice was suppressed whenever `offline` was set -- so a local-model
session showed least, despite being the configuration where "which model is
this and will it run code without asking" is hardest to guess.
"""

from types import SimpleNamespace

import pytest

from interpreter.terminal_interface.startup_banner import print_startup_banner


def _session(**overrides):
    settings = dict(
        llm=SimpleNamespace(model="ollama_chat/qwen3"),
        auto_run_mode="prompt",
        plain_text_display=False,
        messages=[],
    )
    settings.update(overrides)
    return SimpleNamespace(**settings)


def test_the_banner_names_the_model_and_whether_it_asks(capsys):
    """The two facts that change what happens next are both on screen."""
    print_startup_banner(_session())

    shown = capsys.readouterr().out
    assert "ollama_chat/qwen3" in shown
    assert "asks before running code" in shown


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("all", "runs code without asking"),
        ("prompt", "asks before running code"),
        ("allowlist", "asks unless allowlisted"),
        ("denylist", "runs unless denylisted"),
    ],
)
def test_each_approval_mode_says_what_it_does(mode, expected, capsys):
    """Read from the user's side: will this run something without asking me.

    `-y` is the one worth being unambiguous about -- it is the mode where the
    machine acts without a further keystroke.
    """
    print_startup_banner(_session(auto_run_mode=mode))

    assert expected in capsys.readouterr().out


def test_the_hints_are_ones_that_actually_work(capsys):
    """/exit leaves and Escape interrupts; Ctrl-C does not leave.

    The old notice said "Press CTRL-C to exit", which cancels the turn and
    returns to the prompt instead -- and people who believed it killed the
    process, orphaning its Jupyter kernel.
    """
    print_startup_banner(_session())

    shown = capsys.readouterr().out
    assert "/exit" in shown
    assert "%help" in shown
    assert "CTRL-C" not in shown.upper()


def test_a_local_session_is_not_the_one_told_least(capsys):
    """An offline session still gets the banner.

    The approval notice was gated on `not interpreter.offline`, so a local
    model -- where the user is least able to infer what is configured -- saw
    nothing at all.
    """
    print_startup_banner(_session(offline=True))

    assert "ollama_chat/qwen3" in capsys.readouterr().out


def test_piped_output_gets_no_box_drawing(capsys):
    """--stdin output is somebody's data; a panel would be corruption in it."""
    print_startup_banner(_session(plain_text_display=True))

    assert capsys.readouterr().out == ""


def test_the_one_shot_entry_stays_quiet(capsys):
    """`i {command}` prints one answer; a banner would be most of the output."""
    print_startup_banner(_session(messages=[{"role": "user", "content": "hi"}]))

    assert capsys.readouterr().out == ""


def test_a_session_with_no_model_still_renders(capsys):
    """Nothing configured yet is exactly when someone needs to be told."""
    print_startup_banner(_session(llm=SimpleNamespace(model=None)))

    assert "no model set" in capsys.readouterr().out
