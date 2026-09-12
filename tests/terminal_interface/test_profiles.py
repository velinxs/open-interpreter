"""Loading a profile: the file that decides which model runs and whether code runs unattended.

profiles.py reads YAML/JSON/Python out of the user's config directory and
setattrs the result straight onto the interpreter. A key that silently does
not apply is the failure mode that matters — the run looks fine and behaves
nothing like the file says — so most of these tests are about what happens to
a setting that is wrong, renamed, or nested.
"""

import json
import os

import pytest
import yaml

from interpreter.core.core import OpenInterpreter
from interpreter.terminal_interface.profiles import profiles


@pytest.fixture(autouse=True)
def _isolate_profile_dir(tmp_path, monkeypatch):
    """Redirect every profile path into the test's own directory.

    profile_dir defaults to ~/.config/open-interpreter/profiles, which on a
    developer machine holds their real profiles. profile() renames files there
    and reset_profile() sends them to the trash, so an unredirected test is
    destructive.
    """
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    monkeypatch.setattr(profiles, "profile_dir", str(profile_dir))
    monkeypatch.setattr(profiles, "oi_dir", str(tmp_path))
    monkeypatch.setattr(profiles, "user_default_profile_path", str(profile_dir / "default.yaml"))
    return profile_dir


@pytest.fixture
def interpreter():
    interp = OpenInterpreter()
    yield interp
    try:
        interp.terminal.terminate()
    except Exception:
        pass


CURRENT = {"version": profiles.OI_VERSION}


def _no_tty(prompt, choices):
    """Stand in for prompt_choice on a machine with no terminal to ask."""
    raise profiles.NoInteractiveInput(prompt, choices)


# --- reading the file -------------------------------------------------------


def test_yaml_json_and_python_profiles_all_load(_isolate_profile_dir):
    """All three documented profile formats are read, and dispatch is by extension.

    A format that silently fell through would return None and take the
    "regenerate the default profile" path, quietly replacing the user's
    settings with stock ones.
    """
    (_isolate_profile_dir / "a.yaml").write_text(yaml.safe_dump({"version": profiles.OI_VERSION, "max_output": 11}))
    (_isolate_profile_dir / "b.json").write_text(json.dumps({"version": profiles.OI_VERSION, "max_output": 22}))
    (_isolate_profile_dir / "c.py").write_text("interpreter.max_output = 33\n")

    assert profiles.get_profile("a.yaml", "")["max_output"] == 11
    assert profiles.get_profile("b.json", "")["max_output"] == 22
    assert "interpreter.max_output = 33" in profiles.get_profile("c.py", "")["start_script"]


def test_python_profiles_are_always_treated_as_current(_isolate_profile_dir):
    """A .py profile never triggers the migration prompt.

    There is no version key to write into a script, so stamping it with the
    current version is the only way to stop apply_profile asking to migrate a
    file it cannot migrate.
    """
    (_isolate_profile_dir / "script.py").write_text("interpreter.auto_run = True\n")
    assert profiles.get_profile("script.py", "")["version"] == profiles.OI_VERSION


def test_python_profiles_lose_their_own_interpreter_construction(_isolate_profile_dir):
    """`from interpreter import interpreter` and `interpreter = OpenInterpreter()` are stripped.

    The script runs with `interpreter` already bound to the live instance. If
    the import or the constructor survived, the profile would configure a
    second, throwaway interpreter and none of its settings would reach the
    session — with no error anywhere.
    """
    (_isolate_profile_dir / "p.py").write_text(
        "from interpreter import interpreter\n"
        "from interpreter import OpenInterpreter\n"
        "interpreter = OpenInterpreter()\n"
        "interpreter.max_output = 4321\n"
    )
    script = profiles.get_profile("p.py", "")["start_script"]
    assert "OpenInterpreter()" not in script
    assert "from interpreter import interpreter" not in script
    assert "interpreter.max_output = 4321" in script


def test_other_imports_from_the_interpreter_package_are_kept(_isolate_profile_dir):
    """Only the `interpreter` name is removed from `from interpreter import ...`.

    Profiles legitimately import OpenInterpreter or helpers. Dropping the whole
    import statement would break the script with a NameError at exec time.
    """
    (_isolate_profile_dir / "p.py").write_text("from interpreter import OpenInterpreter\nx = 1\n")
    script = profiles.get_profile("p.py", "")["start_script"]
    assert "from interpreter import OpenInterpreter" in script


def test_a_start_script_runs_against_the_live_interpreter(interpreter):
    """apply_profile execs start_script with `interpreter` bound to the real instance.

    This is the entire mechanism behind .py profiles. If the exec scope were
    wrong, every Python profile would be a no-op.
    """
    profiles.apply_profile(interpreter, {**CURRENT, "start_script": "interpreter.max_output = 7777"}, "/tmp/x.py")
    assert interpreter.max_output == 7777


# --- name resolution --------------------------------------------------------


def test_the_extension_may_be_omitted_for_a_bundled_profile(interpreter, monkeypatch):
    """`--profile local` resolves to the bundled local.py.

    The documented shortcuts are extensionless. Without the lookup the CLI
    would try to open a file called "local" and fail.
    """
    loaded = {}

    def _default_profile(name):
        loaded["name"] = name
        return {**CURRENT}

    monkeypatch.setattr(profiles, "get_default_profile", _default_profile)
    profiles.profile(interpreter, "local")
    assert loaded["name"] == "local.py"


def test_a_user_file_shadowing_a_bundled_name_is_renamed_out_of_the_way(interpreter, _isolate_profile_dir, monkeypatch):
    """A user profile at a reserved name is moved to {name}_custom and the bundled one wins.

    Bundled profiles are upgraded with each release. A stale user copy under
    the same name would pin an old model and old settings forever, so it is
    preserved under a new name rather than loaded or deleted.
    """
    reserved = "fast.yaml"
    assert reserved in profiles.default_profiles_names
    shadow = _isolate_profile_dir / reserved
    shadow.write_text("max_output: 1\n")
    monkeypatch.setattr(profiles, "get_default_profile", lambda name: {**CURRENT})

    profiles.profile(interpreter, reserved)

    base, extension = os.path.splitext(reserved)
    assert not shadow.exists()
    assert (_isolate_profile_dir / f"{base}_custom{extension}").exists()


@pytest.mark.parametrize("name", ["default.yaml", "develop.yaml"])
def test_default_and_develop_profiles_are_never_renamed(interpreter, _isolate_profile_dir, name):
    """The user's own default.yaml and develop.yaml are loaded, not shunted aside.

    These two are meant to be edited. Renaming them would discard the user's
    entire configuration on the next launch.
    """
    (_isolate_profile_dir / name).write_text(yaml.safe_dump({**CURRENT, "max_output": 555}))
    profiles.profile(interpreter, name)
    assert (_isolate_profile_dir / name).exists()
    assert interpreter.max_output == 555


def test_a_missing_develop_profile_is_installed_and_loads_without_prompting(interpreter, _isolate_profile_dir, monkeypatch):
    """A fresh install copies the bundled develop.yaml in and applies it straight away.

    develop.yaml is this fork's default profile, so a fresh checkout has no
    copy and every launch would otherwise die on a missing file. The second
    half matters just as much: the bundled file must carry a current
    `version:` line, or apply_profile takes the migration branch on every
    launch — a prompt nothing can silence interactively, and a hard exit in a
    script.
    """

    def _must_not_prompt(*args, **kwargs):
        raise AssertionError("loading the bundled develop profile asked to migrate it")

    monkeypatch.setattr(profiles, "prompt_choice", _must_not_prompt)
    profiles.profile(interpreter, "develop.yaml")
    assert (_isolate_profile_dir / "develop.yaml").exists()
    assert interpreter.llm.model == "gpt-4.1-mini"


def test_a_missing_named_profile_raises_rather_than_falling_back(interpreter):
    """`--profile typo.yaml` fails loudly.

    Silently starting with stock settings would look like the profile applied
    and did nothing, which is much harder to diagnose than a missing file.
    """
    with pytest.raises(Exception):
        profiles.profile(interpreter, "definitely-missing.yaml")


# --- applying it ------------------------------------------------------------


def test_nested_sections_are_applied_to_the_nested_objects(interpreter):
    """A profile's `llm:` block configures interpreter.llm, not the interpreter.

    Flattening it would set interpreter.model, which nothing reads, and leave
    the real model at its default.
    """
    profiles.apply_profile(interpreter, {**CURRENT, "llm": {"model": "gpt-4.1", "temperature": 0.3}}, "/tmp/x.yaml")
    assert interpreter.llm.model == "gpt-4.1"
    assert interpreter.llm.temperature == 0.3


def test_the_legacy_computer_section_still_configures_the_toolbox(interpreter):
    """Profiles written before the rename use `computer:`; it maps to toolbox.

    Breaking this makes every pre-rename profile's tool settings vanish
    without a warning.
    """
    profiles.apply_profile(interpreter, {**CURRENT, "computer": {"import_computer_api": True}}, "/tmp/x.yaml")
    assert interpreter.toolbox.import_computer_api is True


def test_a_language_allowlist_filters_the_terminal_instead_of_replacing_it(interpreter):
    """`computer.languages` keeps only the named languages, matched case-insensitively.

    This is a security control — it is how a user stops the model running
    shell. A failed match would leave every language enabled while appearing
    to restrict them.
    """
    profiles.apply_profile(interpreter, {**CURRENT, "computer": {"languages": ["PYTHON"]}}, "/tmp/x.yaml")
    names = [language.name.lower() for language in interpreter.terminal.languages]
    assert names == ["python"]


def test_max_output_is_rehomed_from_the_llm_section(interpreter):
    """`llm.max_output` lands on the interpreter, where the attribute actually lives.

    Profiles have always written it under llm. Without the remap it would be
    set on the Llm object, read by nothing, and code output would keep being
    truncated at the default.
    """
    profiles.apply_profile(interpreter, {**CURRENT, "llm": {"max_output": 1234}}, "/tmp/x.yaml")
    assert interpreter.max_output == 1234


def test_unknown_keys_are_reported_rather_than_silently_dropped(interpreter, capsys):
    """A key that matches no attribute produces a warning naming it.

    setattr would happily create interpreter.temprature. The warning is the
    only signal that a typo'd setting is doing nothing.
    """
    profiles.apply_profile(
        interpreter,
        {**CURRENT, "not_a_real_setting": 1, "llm": {"temprature": 0.5}},
        "/tmp/x.yaml",
    )
    output = capsys.readouterr().out
    assert "not_a_real_setting" in output
    assert "temprature" in output


def test_the_wtf_section_is_left_alone(interpreter):
    """`wtf:` is consumed by a different entry point and must not be applied here.

    Recursing into it would try to setattr onto interpreter.wtf, which does
    not exist.
    """
    profiles.apply_profile(interpreter, {**CURRENT, "wtf": {"model": "gpt-4.1"}}, "/tmp/x.yaml")


def test_an_old_profile_version_asks_before_migrating(interpreter, monkeypatch, capsys):
    """A version mismatch prompts, and declining loads nothing.

    Declining deliberately returns without applying the profile: guessing
    would silently drop every setting the user configured, including the model
    and auto_run, and the run would look fine while behaving nothing like the
    file.
    """
    monkeypatch.setattr(profiles, "prompt_choice", lambda *a, **kw: "n")
    interpreter.max_output = 999
    profiles.apply_profile(interpreter, {"version": "0.0.1", "max_output": 42}, "/tmp/old.yaml")
    assert interpreter.max_output == 999
    assert "Skipping loading profile" in capsys.readouterr().out


def test_declining_migration_stamps_the_version_so_it_stops_asking(interpreter, monkeypatch, tmp_path):
    """A declined default.yaml gets a version line appended.

    Otherwise the same question is asked on every single launch, forever.
    """
    monkeypatch.setattr(profiles, "prompt_choice", lambda *a, **kw: "n")
    path = tmp_path / "default.yaml"
    path.write_text("max_output: 42\n")
    profiles.apply_profile(interpreter, {"version": "0.0.1"}, str(path))
    assert f"version: {profiles.OI_VERSION}" in path.read_text()


def test_a_non_interactive_terminal_refuses_to_guess_at_migration(interpreter, monkeypatch, capsys):
    """With no tty, an old profile stops the run and explains the fix.

    Both answers are destructive in a script: migrating rewrites the user's
    file unattended, declining silently ignores every setting. Exiting with
    instructions is the only safe option.
    """

    monkeypatch.setattr(profiles, "prompt_choice", _no_tty)
    with pytest.raises(SystemExit):
        profiles.apply_profile(interpreter, {"version": "0.0.1"}, "/tmp/old.yaml")
    assert "Cannot start" in capsys.readouterr().out


# --- writing back -----------------------------------------------------------


def test_write_key_to_profile_inserts_above_the_version_line(_isolate_profile_dir):
    """A new key is written before the trailing `version:` line, which stays last.

    The version line is what tells the loader the file is current. Appending
    after it is harmless for YAML but the insertion point is what keeps the
    file readable, and losing the version line would re-trigger the migration
    prompt.
    """
    path = _isolate_profile_dir / "default.yaml"
    path.write_text("llm:\n  model: gpt-4.1\n\nversion: 0.2.5  # Profile version (do not modify)\n")

    profiles.write_key_to_profile("contribute_conversation", True)

    text = path.read_text()
    assert "contribute_conversation: True" in text
    assert text.index("contribute_conversation") < text.index("version:")


def test_write_key_to_profile_does_not_duplicate_an_existing_key(_isolate_profile_dir):
    """Writing the same key twice leaves one copy.

    A duplicated YAML key silently wins or loses depending on order, and the
    file grows a line every time the user answers the contribution prompt.
    """
    path = _isolate_profile_dir / "default.yaml"
    path.write_text("contribute_conversation: True\n\nversion: 0.2.5\n")
    profiles.write_key_to_profile("contribute_conversation", True)
    assert path.read_text().count("contribute_conversation") == 1


def test_write_key_to_profile_is_silent_when_there_is_no_profile(_isolate_profile_dir):
    """A missing default.yaml is not an error worth crashing the session over.

    This is called from the contribution prompt on exit; raising there would
    replace a clean quit with a traceback.
    """
    profiles.write_key_to_profile("contribute_conversation", True)


def test_reset_profile_rejects_a_name_that_is_not_a_bundled_profile():
    """`--reset_profile made_up.yaml` fails instead of doing nothing.

    Resetting is destructive, so a name the code does not recognise must not
    be interpreted as "reset whatever is closest".
    """
    with pytest.raises(ValueError):
        profiles.reset_profile("made_up.yaml")


def test_reset_profile_only_ever_touches_default_yaml(_isolate_profile_dir, monkeypatch):
    """Characterisation: every profile except default.yaml is skipped, including "reset all".

    The loop body starts `if specific_default_profile != "default.yaml":
    continue`, so reset_profile(None) — the documented "reset all default
    profiles" behaviour — resets nothing at all. Combined with the CLI never
    passing None (see test_bare_reset_profile_does_not_reset_anything), the
    "reset all" feature is unreachable. This pins today's behaviour.
    """
    profiles.reset_profile(None)
    assert list(_isolate_profile_dir.iterdir()) == []

    profiles.reset_profile("default.yaml")
    assert (_isolate_profile_dir / "default.yaml").exists()


def test_resetting_an_existing_default_asks_before_trashing_it(_isolate_profile_dir, monkeypatch, capsys):
    """A customised default.yaml is not overwritten without a yes.

    It is the file the user edits. Declining must leave it byte-for-byte
    intact, and saying yes routes the old copy to the trash rather than
    deleting it.
    """
    path = _isolate_profile_dir / "default.yaml"
    path.write_text("my: custom settings\n")
    monkeypatch.setattr(profiles, "prompt_choice", lambda *a, **kw: "n")

    profiles.reset_profile("default.yaml")

    assert path.read_text() == "my: custom settings\n"
    assert "was not reset" in capsys.readouterr().out


def test_resetting_without_a_terminal_leaves_the_file_alone(_isolate_profile_dir, monkeypatch, capsys):
    """No tty means "no": keeping the existing file is the non-destructive answer."""
    path = _isolate_profile_dir / "default.yaml"
    path.write_text("my: custom settings\n")

    monkeypatch.setattr(profiles, "prompt_choice", _no_tty)
    profiles.reset_profile("default.yaml")

    assert path.read_text() == "my: custom settings\n"
    assert "no terminal to ask" in capsys.readouterr().out
