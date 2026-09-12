import ast
import difflib
import glob
import json
import os
import platform
import shutil
import string
import subprocess
import sys
import time

import platformdirs
import requests
import send2trash
import yaml

from ...core.utils.prompt_choice import NoInteractiveInput, prompt_choice
from ..utils.oi_dir import oi_dir
from .historical_profiles import historical_profiles
from .migrate import (
    determine_user_version,
    migrate_app_directory,
    migrate_profile,
    migrate_user_app_directory,
)
from .paths import (
    OI_VERSION,
    default_profiles_names,
    default_profiles_paths,
    here,
    oi_default_profiles_path,
    profile_dir,
    user_default_profile_path,
)

__all__ = [
    "OI_VERSION",
    "apply_profile",
    "determine_user_version",
    "migrate_app_directory",
    "migrate_profile",
    "migrate_user_app_directory",
    "profile",
    "profile_dir",
    "reset_profile",
]


def profile(interpreter, filename_or_url):
    # See if they're doing shorthand for a default profile
    filename_without_extension = os.path.splitext(filename_or_url)[0]
    for profile in default_profiles_names:
        if filename_without_extension == os.path.splitext(profile)[0]:
            filename_or_url = profile
            break

    profile_path = os.path.join(profile_dir, filename_or_url)
    profile = None

    # If they have a profile at a reserved profile name, rename it to {name}_custom.
    # Don't do this for default.yaml or develop.yaml (develop: classic/develop only).
    if (
        filename_or_url not in ["default", "default.yaml", "develop", "develop.yaml"]
        and filename_or_url in default_profiles_names
    ):
        if os.path.isfile(profile_path):
            base, extension = os.path.splitext(profile_path)
            os.rename(profile_path, f"{base}_custom{extension}")
        profile = get_default_profile(filename_or_url)

    if profile == None:
        try:
            profile = get_profile(filename_or_url, profile_path)
        except:
            if filename_or_url in ["default", "default.yaml"]:
                # Literally this just happens to default.yaml
                reset_profile(filename_or_url)
                profile = get_profile(filename_or_url, profile_path)
            elif filename_or_url in ["develop", "develop.yaml"]:
                ensure_develop_profile()
                profile = get_profile(filename_or_url, profile_path)
            else:
                raise

    return apply_profile(interpreter, profile, profile_path)


def get_profile(filename_or_url, profile_path):
    # i.com/ is a shortcut for openinterpreter.com/profiles/
    shortcuts = ["i.com/", "www.i.com/", "https://i.com/", "http://i.com/"]
    for shortcut in shortcuts:
        if filename_or_url.startswith(shortcut):
            filename_or_url = filename_or_url.replace(shortcut, "https://openinterpreter.com/profiles/")
            if "." not in filename_or_url.split("/")[-1]:
                extensions = [".json", ".py", ".yaml"]
                for ext in extensions:
                    try:
                        response = requests.get(filename_or_url + ext)
                        response.raise_for_status()
                        filename_or_url += ext
                        break
                    except requests.exceptions.HTTPError:
                        continue
            break

    profile_path = os.path.join(profile_dir, filename_or_url)
    extension = os.path.splitext(filename_or_url)[-1]

    # Try local
    if os.path.exists(profile_path):
        with open(profile_path, encoding="utf-8") as file:
            if extension == ".py":
                python_script = file.read()

                # Remove `from interpreter import interpreter` and `interpreter = OpenInterpreter()`, because we handle that before the script
                tree = ast.parse(python_script)
                tree = RemoveInterpreter().visit(tree)
                python_script = ast.unparse(tree)

                return {
                    "start_script": python_script,
                    "version": OI_VERSION,
                }  # Python scripts are always the latest version
            elif extension == ".json":
                return json.load(file)
            else:
                return yaml.safe_load(file)

    # Try URL
    response = requests.get(filename_or_url)
    response.raise_for_status()
    if extension == ".py":
        return {"start_script": response.text, "version": OI_VERSION}
    elif extension == ".json":
        return json.loads(response.text)
    elif extension == ".yaml":
        return yaml.safe_load(response.text)

    raise Exception(f"Profile '{filename_or_url}' not found.")


class RemoveInterpreter(ast.NodeTransformer):
    """Remove `from interpreter import interpreter` and `interpreter = OpenInterpreter()`"""

    def visit_ImportFrom(self, node):
        if node.module == "interpreter":
            for alias in node.names:
                if alias.name == "interpreter":
                    return None
        return node

    def visit_Assign(self, node):
        if (
            isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "interpreter"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "OpenInterpreter"
        ):
            return None  # None will remove the node from the AST
        return node  # return node otherwise to keep it in the AST


def _known_attributes(obj):
    """Public, non-callable attribute names on obj, offered as difflib candidates for a typo'd key.

    Methods are excluded by checking each attribute's current value, not by
    name: dir(obj) includes bound methods (run, load, exec...), and a typo
    that happens to be close to one would nominate it as "did you mean" —
    walking the user straight into overwriting a method (see
    _is_unsettable).
    """
    names = []
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if callable(value):
            continue
        names.append(attr)
    return names


def _is_unsettable(obj, key):
    """True if `key` must not be setattr'd onto obj: missing, or currently a callable.

    hasattr alone lets a profile key that matches a method name (e.g. `run`,
    `load`, `completions`) overwrite that method — the attribute exists, so
    the old guard let it through. Treat that the same as an unknown key:
    report and skip, rather than silently replacing behaviour with data.
    """
    if not hasattr(obj, key):
        return True
    return callable(getattr(obj, key))


def _unknown_key_warning(displayed_key, key, known_attributes, class_name):
    """One-line warning for a profile key with no matching attribute.

    `displayed_key` is what the user wrote (e.g. "llm.temprature"), `key` is
    the bare attribute name difflib matches against (e.g. "temprature").
    get_close_matches names the probable intent (e.g. auto_run_mode for
    auto_run_moed) so a typo is a one-glance fix. With no close match, only
    the unknown-key fact is reported — guessing a lookalike would be worse
    than saying nothing. The setting is actually skipped now (see
    apply_profile_to_object), so this warning is no longer a false alarm.
    """
    match = difflib.get_close_matches(key, known_attributes, n=1)
    if match:
        return (
            f"Profile has '{displayed_key}' but this attribute doesn't exist on the {class_name} class. "
            f"Did you mean '{match[0]}'? This setting was skipped."
        )
    return (
        f"Profile has '{displayed_key}' but this attribute doesn't exist on the {class_name} class. "
        "This setting was skipped."
    )


def _validate_profile(interpreter, profile):
    """Validate profile and warn about unknown attributes that will be skipped."""
    warnings = []

    # Nested dicts to validate: (profile_key, obj_attribute, obj_class_name, skip_keys)
    # This used to list "computer" here, which was dead: apply_profile renames
    # "computer" to "toolbox" in the profile dict before _validate_profile ever
    # runs (see the "Map 'computer' key to 'toolbox'" comment below), so
    # `"computer" in profile` was always False and a typo'd toolbox key never
    # warned. Validate the post-rename key it actually arrives as.
    nested_dicts = [
        ("llm", interpreter.llm, "Llm", set()),
        ("toolbox", interpreter.toolbox, "Toolbox", {"languages"}),
    ]

    for profile_key, obj, class_name, skip_keys in nested_dicts:
        if profile_key in profile and isinstance(profile[profile_key], dict):
            known = _known_attributes(obj)
            for key in profile[profile_key]:
                if key.startswith("_") or key in skip_keys:
                    continue
                if _is_unsettable(obj, key):
                    warnings.append(_unknown_key_warning(f"{profile_key}.{key}", key, known, class_name))

    # Check for invalid top-level attributes
    # Skip known nested dicts and special keys. "version" and "start_script" are
    # metadata the loader consumes elsewhere (see apply_profile) and are never
    # attributes on the interpreter, so they must never be reported as unknown.
    skip_keys = {"llm", "computer", "toolbox", "wtf", "version", "start_script"}
    known = _known_attributes(interpreter)
    for key in profile:
        if key in skip_keys:
            continue
        if key.startswith("_"):
            continue  # Skip private attributes
        if isinstance(profile[key], dict):
            # llm and toolbox are the only nested sections that exist (handled
            # above); anything else here is a typo'd or invented section name.
            # Without this check it fell straight through to
            # apply_profile_to_object's getattr, which raised instead of
            # warning.
            if _is_unsettable(interpreter, key):
                warnings.append(_unknown_key_warning(key, key, known, "Interpreter"))
            continue
        if _is_unsettable(interpreter, key):
            warnings.append(_unknown_key_warning(key, key, known, "Interpreter"))

    if warnings:
        print("\n⚠️  Profile validation warnings:")
        for warning in warnings:
            print(f"   {warning}")
        print()


def apply_profile(interpreter, profile, profile_path):
    if "start_script" in profile:
        scope = {"interpreter": interpreter}
        exec(profile["start_script"], scope, scope)

    if (
        "version" not in profile or profile["version"] != OI_VERSION
    ):  # Remember to update this version number at the top of the file ^
        print("")
        print(
            "We have updated our profile file format. Would you like to migrate your profile file to the new format? No data will be lost."
        )
        print("")
        try:
            message = prompt_choice("(y/n) ", ("y", "n"))
        except NoInteractiveInput:
            # Declining here does not just skip the migration, it returns without
            # loading the profile at all. Guessing "no" would silently drop every
            # setting the user configured — including the model and auto_run — and
            # the run would look fine until it behaved nothing like the profile
            # says. There is no safe assumption, so stop and say what to do.
            print("")
            print(
                f"Cannot start: {profile_path} uses an older profile format and\n"
                f"migrating it needs a yes/no answer, but there is no interactive\n"
                f"terminal to ask.\n\n"
                f"Either run `interpreter` once in a terminal to migrate it, or add\n"
                f"this line to the profile to keep the current format and skip the\n"
                f"prompt:\n\n"
                f"    version: {OI_VERSION}  # Profile version (do not modify)\n"
            )
            sys.exit(1)
        if message == "y":
            migrate_user_app_directory()
            print("Migration complete.")
            print("")
            if profile_path.endswith("default.yaml"):
                with open(profile_path) as file:
                    text = file.read()
                text = text.replace("version: " + str(profile["version"]), f"version: {OI_VERSION}")

                try:
                    if profile["llm"]["model"] == "gpt-4":
                        text = text.replace("gpt-4", "gpt-4.1")
                        profile["llm"]["model"] = "gpt-4.1"
                    elif profile["llm"]["model"] == "gpt-4-turbo-preview":
                        text = text.replace("gpt-4-turbo-preview", "gpt-4.1")
                        profile["llm"]["model"] = "gpt-4.1"
                except:
                    raise
                    pass  # fine

                with open(profile_path, "w") as file:
                    file.write(text)
        else:
            print("Skipping loading profile...")
            print("")
            # If the migration is skipped, add the version number to the end of the file
            if profile_path.endswith("default.yaml"):
                with open(profile_path, "a") as file:
                    file.write(f"\nversion: {OI_VERSION}  # Profile version (do not modify)")
            return interpreter

    if "system_message" in profile:
        interpreter.display_message(
            "\n**FYI:** A `system_message` was found in your profile.\n\nBecause we frequently improve our default system message, we highly recommend removing the `system_message` parameter in your profile (which overrides the default system message) or simply resetting your profile.\n\n**To reset your profile, run `interpreter --reset_profile`.**\n"
        )
        time.sleep(2)
        interpreter.display_message("---")

    if "computer" in profile and "languages" in profile["computer"]:
        # this is handled specially
        interpreter.terminal.languages = [
            i
            for i in interpreter.terminal.languages
            if i.name.lower() in [l.lower() for l in profile["computer"]["languages"]]
        ]
        del profile["computer"]["languages"]

    # Map "computer" key to "toolbox" for backward compatibility with profile files
    if "computer" in profile:
        profile["toolbox"] = profile.pop("computer")

    # Map llm.max_output to interpreter.max_output (max_output is on interpreter, not llm)
    if "llm" in profile and isinstance(profile["llm"], dict) and "max_output" in profile["llm"]:
        interpreter.max_output = profile["llm"]["max_output"]
        del profile["llm"]["max_output"]

    # Map deprecated llm.truncation_step to llm.retention_ratio. The old knob dropped a
    # fixed token chunk; the new one keeps a fraction of the context window. There is no
    # exact conversion, so a profile that asked for truncation is mapped to the standard
    # retention_ratio default (0.8) rather than being silently ignored — keeping the
    # cache-aware trimming the user opted into.
    if "llm" in profile and isinstance(profile["llm"], dict) and "truncation_step" in profile["llm"]:
        del profile["llm"]["truncation_step"]
        profile["llm"].setdefault("retention_ratio", 0.8)

    # "version" and "start_script" are metadata the loader has already consumed
    # (the version check above, the exec at the top of this function) — neither
    # is an attribute on the interpreter. Strip them before validating/applying
    # so they are never reported as unknown and never setattr'd as dead
    # attributes nothing reads.
    profile.pop("version", None)
    profile.pop("start_script", None)

    # Validate profile for common mistakes
    _validate_profile(interpreter, profile)

    apply_profile_to_object(interpreter, profile)

    return interpreter


def apply_profile_to_object(obj, profile):
    for key, value in profile.items():
        # Map "computer" key to "toolbox" for backward compatibility
        if key == "computer":
            key = "toolbox"
        if isinstance(value, dict):
            if key == "wtf":  # The wtf command has a special part of the profile, not used here
                continue
            # An unknown or invented section name (e.g. a typo'd top-level
            # key) has no matching attribute to recurse into — getattr would
            # raise AttributeError and abort startup with a bare traceback.
            # The matching warning is emitted by _validate_profile before
            # this runs; here we only need to not crash.
            if _is_unsettable(obj, key):
                continue
            apply_profile_to_object(getattr(obj, key), value)
        else:
            # A key with no matching attribute is a typo or a setting that
            # never existed. A key whose current value is callable matches a
            # method name (see _is_unsettable) — setattr would silently
            # replace behaviour with data. Either way setattr would happily
            # apply it, and for a restriction (e.g. auto_run_mode) that is a
            # silent security downgrade, not just a cosmetic no-op. The
            # matching warning is emitted by _validate_profile before this
            # runs; here we only need to not create/overwrite the attribute.
            if _is_unsettable(obj, key):
                continue
            setattr(obj, key, value)


def open_storage_dir(directory):
    dir = os.path.join(oi_dir, directory)

    print(f"Opening {directory} directory ({dir})...")

    if platform.system() == "Windows":
        os.startfile(dir)
    else:
        try:
            # Try using xdg-open on non-Windows platforms
            subprocess.call(["xdg-open", dir])
        except FileNotFoundError:
            # Fallback to using 'open' on macOS if 'xdg-open' is not available
            subprocess.call(["open", dir])
    return


def ensure_develop_profile():
    """
    Copy bundled develop.yaml into the user profile dir if missing.

    classic/develop only — do not merge to main / classic/main.
    """
    bundled = os.path.join(oi_default_profiles_path, "develop.yaml")
    if not os.path.exists(bundled):
        raise FileNotFoundError("Bundled develop.yaml not found. This profile is for classic/develop only.")

    target_file = os.path.join(profile_dir, "develop.yaml")

    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)

    if not os.path.exists(target_file):
        shutil.copy(bundled, target_file)


def reset_profile(specific_default_profile=None):
    if specific_default_profile and specific_default_profile not in default_profiles_names:
        raise ValueError(f"The specific default profile '{specific_default_profile}' is not a default profile.")

    # Check version, before making the profile directory
    current_version = determine_user_version()

    for default_yaml_file in default_profiles_paths:
        filename = os.path.basename(default_yaml_file)

        if specific_default_profile and filename != specific_default_profile:
            continue

        # Python profiles are read from the package at load time, so there is
        # no user copy to reset; only the YAML ones exist as files. This used to
        # read `if specific_default_profile != "default.yaml": continue`, which
        # skipped every file whenever the flag was used as documented, and
        # skipped the named file whenever it was anything but default.yaml.
        if not filename.endswith((".yaml", ".yml")):
            continue

        target_file = os.path.join(profile_dir, filename)

        # Variable to see if we should display the 'reset' print statement or not
        create_oi_directory = False

        # Make the profile directory if it does not exist
        if not os.path.exists(profile_dir):
            if not os.path.exists(oi_dir):
                create_oi_directory = True

            os.makedirs(profile_dir)

        if not os.path.exists(target_file):
            shutil.copy(default_yaml_file, target_file)
            if current_version is None:
                # If there is no version, add it to the default yaml
                with open(target_file, "a") as file:
                    file.write(f"\nversion: {OI_VERSION}  # Profile version (do not modify)")
            if not create_oi_directory:
                print(f"{filename} has been reset.")
        else:
            with open(target_file) as file:
                current_profile = file.read()
            if current_profile not in historical_profiles:
                try:
                    user_input = prompt_choice(
                        f"Would you like to reset/update {filename}? (y/n) ",
                        ("y", "n"),
                    )
                except NoInteractiveInput:
                    # Leaving the file alone is the status quo, so this one can
                    # be skipped without losing anything.
                    print(
                        f"Leaving {filename} as it is (no terminal to ask). "
                        f"Run `interpreter --reset_profile` to reset it."
                    )
                    user_input = "n"
                if user_input == "y":
                    send2trash.send2trash(target_file)  # This way, people can recover it from the trash
                    shutil.copy(default_yaml_file, target_file)
                    print(f"{filename} has been reset.")
                else:
                    print(f"{filename} was not reset.")
            else:
                shutil.copy(default_yaml_file, target_file)
                print(f"{filename} has been reset.")


def get_default_profile(specific_default_profile):
    for default_yaml_file in default_profiles_paths:
        filename = os.path.basename(default_yaml_file)

        if specific_default_profile and filename != specific_default_profile:
            continue

        profile_path = os.path.join(oi_default_profiles_path, filename)
        extension = os.path.splitext(filename)[-1]

        with open(profile_path, encoding="utf-8") as file:
            if extension == ".py":
                python_script = file.read()

                # Remove `from interpreter import interpreter` and `interpreter = OpenInterpreter()`, because we handle that before the script
                tree = ast.parse(python_script)
                tree = RemoveInterpreter().visit(tree)
                python_script = ast.unparse(tree)

                return {
                    "start_script": python_script,
                    "version": OI_VERSION,
                }  # Python scripts are always the latest version
            elif extension == ".json":
                return json.load(file)
            else:
                return yaml.safe_load(file)


def write_key_to_profile(key, value):
    try:
        with open(user_default_profile_path) as file:
            lines = file.readlines()

        version_line_index = None
        new_lines = []
        for index, line in enumerate(lines):
            if line.strip().startswith("version:"):
                version_line_index = index
                break
            new_lines.append(line)

        # Insert the new key-value pair before the version line
        if version_line_index is not None:
            if f"{key}: {value}\n" not in new_lines:
                new_lines.append(f"{key}: {value}\n\n")  # Adding a newline for separation
            # Append the version line and all subsequent lines
            new_lines.extend(lines[version_line_index:])

        with open(user_default_profile_path, "w") as file:
            file.writelines(new_lines)
    except Exception:
        pass  # Fail silently
