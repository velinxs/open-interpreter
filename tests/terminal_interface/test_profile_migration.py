"""Upgrading an old profile and app directory.

This runs once, against the user's real configuration, and it is the only
chance to carry their settings forward. Anything it drops is gone silently:
the session starts, looks healthy, and uses defaults for everything the
profile was supposed to set.
"""

import os

import pytest
import yaml

import interpreter.terminal_interface.profiles.migrate as migrate


@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    """Point every directory this module reads or writes at tmp_path.

    determine_user_version() inspects the real ~/.config/open-interpreter and
    the two historical platformdirs locations, and migrate_user_app_directory()
    copies into the real one.
    """
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    monkeypatch.setattr(migrate, "oi_dir", str(new_dir))
    monkeypatch.setattr(migrate, "profile_dir", str(new_dir / "profiles"))
    monkeypatch.setattr(
        migrate.platformdirs,
        "user_config_dir",
        lambda name: str(tmp_path / name.replace(" ", "_")),
    )
    return new_dir


def _old_profile(tmp_path, name="config.yaml", **settings):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(settings))
    return path


# --- migrate_profile --------------------------------------------------------


def test_a_migrated_profile_keeps_a_version_marker_and_the_documentation_comments(tmp_path):
    """The rewritten file carries the current version and the commented settings reference.

    Without the version line the migration prompt reappears on every launch,
    which is the exact loop the migration exists to end.
    """
    old = _old_profile(tmp_path, model="gpt-4", temperature=0.5)
    new = tmp_path / "migrated.yaml"

    migrate.migrate_profile(str(old), str(new))
    text = new.read_text()

    assert f"version: {migrate.OI_VERSION}" in text
    assert "OPEN INTERPRETER PROFILE" in text
    assert "All options: https://docs.openinterpreter.com/settings" in text


def test_old_attribute_names_are_not_actually_renamed(tmp_path):
    """Characterisation bug: the renaming table is computed and then thrown away.

    migrate_profile builds `mapped_profile` from attribute_mapping (model ->
    llm.model, api_key -> llm.api_key, local -> offline, ...) and then never
    uses it: the reformatting loop directly below iterates `profile`, the
    original dict, so the mapped names are discarded. The migrated file keeps
    the flat 0.1.x keys, which the current loader does not read.

    The effect on a real migration is that the model, API key, API base,
    temperature, context window, max tokens and offline flag all silently
    revert to defaults, and apply_profile prints "this attribute doesn't exist
    on the Interpreter class" warnings for each. This pins what it does today.
    """
    old = _old_profile(tmp_path, model="gpt-4", api_key="sk-secret", local=True)
    new = tmp_path / "migrated.yaml"

    migrate.migrate_profile(str(old), str(new))
    migrated = yaml.safe_load(new.read_text())

    assert migrated["model"] == "gpt-4"
    assert "llm" not in migrated
    assert migrated["local"] is True
    assert "offline" not in migrated


def test_a_stock_old_system_message_is_dropped(tmp_path):
    """A profile holding one of the shipped system messages loses it entirely.

    The default system message has been improved many times since. Carrying an
    old copy forward would pin the user to a prompt from 2023 while looking
    like a deliberate customisation.
    """
    old = _old_profile(tmp_path, system_message=_stock_message(), auto_run=True)
    new = tmp_path / "migrated.yaml"

    migrate.migrate_profile(str(old), str(new))
    migrated = yaml.safe_load(new.read_text())

    assert "system_message" not in migrated
    assert migrated["auto_run"] is True


def test_a_profile_left_empty_by_migration_is_written_as_invalid_yaml(tmp_path):
    """Characterisation bug: an emptied profile becomes an unparseable file.

    When the only setting was a stock system message, the profile dict ends up
    empty and yaml.dump writes a literal "{}". The comment wrapper then appends
    a "version:" key after it, so the file is a flow mapping followed by a
    block mapping and yaml.safe_load raises. The author saw this coming — there
    is a line `comment_wrapper.replace("\n{}\n", "\n")` meant to strip it — but
    the result is never assigned, so it does nothing.

    The damage is downstream: get_profile() raises, and profile() reacts by
    resetting default.yaml (losing the migration) or re-raising for any other
    profile name (the CLI will not start).
    """
    old = _old_profile(tmp_path, system_message=_stock_message())
    new = tmp_path / "migrated.yaml"

    migrate.migrate_profile(str(old), str(new))

    assert "\n{}\n" in new.read_text()
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(new.read_text())


def test_a_customised_system_message_keeps_only_the_custom_tail(tmp_path):
    """Text appended to a stock system message becomes custom_instructions.

    This is how people customised before custom_instructions existed. Dropping
    the whole thing would lose their additions; keeping the whole thing would
    pin the stale prefix.
    """
    old = _old_profile(tmp_path, system_message=_stock_message() + "\n\nAlways answer in French.")
    new = tmp_path / "migrated.yaml"

    migrate.migrate_profile(str(old), str(new))
    migrated = yaml.safe_load(new.read_text())

    assert "system_message" not in migrated
    assert migrated["custom_instructions"] == "Always answer in French."


def test_an_unrecognised_system_message_is_left_alone(tmp_path):
    """A system message nobody shipped is the user's own and survives untouched."""
    old = _old_profile(tmp_path, system_message="You are a pirate.")
    new = tmp_path / "migrated.yaml"

    migrate.migrate_profile(str(old), str(new))
    assert yaml.safe_load(new.read_text())["system_message"] == "You are a pirate."


def test_old_version_lines_are_not_duplicated(tmp_path):
    """A profile that already had a version ends up with exactly one.

    Two version keys in one YAML file is ambiguous, and the loader reads
    whichever wins — which may be the old one, re-triggering the migration.
    """
    old = _old_profile(tmp_path, version="0.2.0", model="gpt-4")
    new = tmp_path / "migrated.yaml"

    migrate.migrate_profile(str(old), str(new))
    version_lines = [line for line in new.read_text().splitlines() if line.startswith("version:")]
    assert version_lines == [f"version: {migrate.OI_VERSION}  # Profile version (do not modify)"]


# --- determine_user_version -------------------------------------------------


def test_a_current_install_reports_its_profile_version(tmp_path, _isolate_dirs):
    """A default.yaml with a version key is the authoritative answer.

    Getting this wrong re-runs a migration against an already-migrated
    directory.
    """
    profiles = _isolate_dirs / "profiles"
    profiles.mkdir()
    (profiles / "default.yaml").write_text("version: 0.2.5\n")
    assert migrate.determine_user_version() == "0.2.5"


def test_a_fresh_machine_reports_no_version(tmp_path):
    """Nothing installed anywhere means None, which means "nothing to migrate"."""
    assert migrate.determine_user_version() is None


def test_the_two_historical_directories_are_recognised(tmp_path):
    """Each legacy app directory maps to the version that created it.

    The two layouts differ, so migrating one as if it were the other copies
    files to the wrong places.
    """
    (tmp_path / "Open_Interpreter").mkdir()
    assert migrate.determine_user_version() == "pre_0.2.0"

    (tmp_path / "Open_Interpreter_Terminal").mkdir()
    assert migrate.determine_user_version() == "0.2.0"


# --- migrate_app_directory --------------------------------------------------


def test_profiles_and_conversations_are_carried_across(tmp_path, _isolate_dirs):
    """Both directories move, with yaml profiles rewritten and everything else copied.

    Conversations are the user's history; a migration that loses them is
    indistinguishable from deleting them.
    """
    old_dir = tmp_path / "old"
    (old_dir / "profiles").mkdir(parents=True)
    (old_dir / "profiles" / "mine.yaml").write_text(yaml.safe_dump({"model": "gpt-4"}))
    (old_dir / "profiles" / "custom.py").write_text("interpreter.auto_run = True\n")
    (old_dir / "conversations").mkdir()
    (old_dir / "conversations" / "chat.json").write_text("[]")

    migrate.migrate_app_directory(str(old_dir), str(_isolate_dirs), str(_isolate_dirs / "profiles"))

    assert (_isolate_dirs / "profiles" / "mine.yaml").exists()
    assert (_isolate_dirs / "profiles" / "custom.py").read_text() == "interpreter.auto_run = True\n"
    assert (_isolate_dirs / "conversations" / "chat.json").exists()
    assert f"version: {migrate.OI_VERSION}" in (_isolate_dirs / "profiles" / "mine.yaml").read_text()


def test_a_legacy_config_yaml_becomes_the_default_profile(tmp_path, _isolate_dirs):
    """The old single config.yaml is migrated into profiles/default.yaml.

    That file held every setting the user had. Not migrating it resets them
    all.
    """
    old_dir = tmp_path / "old"
    (old_dir / "profiles").mkdir(parents=True)
    (old_dir / "config.yaml").write_text(yaml.safe_dump({"model": "gpt-4", "auto_run": True}))

    migrate.migrate_app_directory(str(old_dir), str(_isolate_dirs), str(_isolate_dirs / "profiles"))

    default = _isolate_dirs / "profiles" / "default.yaml"
    assert default.exists()
    assert yaml.safe_load(default.read_text())["auto_run"] is True


def test_a_pre_020_directory_without_a_profiles_folder_crashes(tmp_path, _isolate_dirs):
    """Characterisation bug: config.yaml with no profiles/ folder fails to migrate.

    The destination profiles directory is only created inside
    `if os.path.exists(profiles_old_path)`, but the config.yaml migration below
    it writes into that directory unconditionally. The pre-0.2.0 layout is
    exactly this shape — a bare config.yaml and no profiles directory — so the
    older of the two migrations this module exists for is the one that dies
    with FileNotFoundError, taking the launch with it.
    """
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    (old_dir / "config.yaml").write_text(yaml.safe_dump({"model": "gpt-4"}))

    with pytest.raises(FileNotFoundError):
        migrate.migrate_app_directory(str(old_dir), str(_isolate_dirs), str(_isolate_dirs / "profiles"))


def test_profiles_without_a_version_get_one_stamped_on(tmp_path, _isolate_dirs):
    """Every yaml in the migrated directory ends up versioned.

    A single unversioned file is enough to re-trigger the migration prompt on
    every launch, which is why this pass exists at all.
    """
    old_dir = tmp_path / "old"
    (old_dir / "profiles").mkdir(parents=True)
    (old_dir / "profiles" / "plain.yaml").write_text("auto_run: true\n")

    migrate.migrate_app_directory(str(old_dir), str(_isolate_dirs), str(_isolate_dirs / "profiles"))

    assert "version:" in (_isolate_dirs / "profiles" / "plain.yaml").read_text()


def test_nothing_to_migrate_is_a_no_op(tmp_path, _isolate_dirs):
    """migrate_user_app_directory does nothing when no legacy directory exists.

    It is called from the migration prompt, which a user can reach on a clean
    install; it must not create or copy anything there.
    """
    migrate.migrate_user_app_directory()
    assert os.listdir(_isolate_dirs) == []


def _stock_message():
    """One of the shipped historical system messages, taken verbatim from the module."""
    return (
        "You are Open Interpreter, a world-class programmer that can complete any goal by executing code.\n"
        "First, write a plan. **Always recap the plan between each code block** (you have extreme short-term "
        "memory loss, so you need to recap the plan between each message block to retain it).\n"
        "When you execute code, it will be executed **on the user's machine**. The user has given you **full "
        "and complete permission** to execute any code necessary to complete the task. Execute the code.\n"
        "If you want to send data between programming languages, save the data to a txt or json.\n"
        "You can access the internet. Run **any code** to achieve the goal, and if at first you don't succeed, "
        "try again and again.\n"
        "You can install new packages.\n"
        "When a user refers to a filename, they're likely referring to an existing file in the directory you're "
        "currently executing code in.\n"
        "Write messages to the user in Markdown.\n"
        "In general, try to **make plans** with as few steps as possible. As for actually executing code to "
        "carry out that plan, for *stateful* languages (like python, javascript, shell, but NOT for html which "
        "starts from 0 every time) **it's critical not to try to do everything in one code block.** You should "
        "try something, print information about it, then continue from there in tiny, informed steps. You will "
        "never get it on the first try, and attempting it in one go will often lead to errors you can't see.\n"
        "You are capable of **any** task."
    )
