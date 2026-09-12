"""The CLI flag table itself: a row that points at a nonexistent attribute is a silent no-op.

build_arguments() is data, and set_attributes() applies it reflectively with
setattr. Nothing validates that a row's target attribute exists, so a typo in
`attr_name` creates a flag that parses, prints no error, and does nothing —
or, worse, invents a new attribute on the interpreter that nothing reads.
These tests check the table against the real objects.
"""

import argparse

import pytest

from interpreter.core.core import OpenInterpreter
from interpreter.terminal_interface.arguments import (
    DEPRECATED_FLAGS,
    build_arguments,
    get_argument_dictionary,
    set_attributes,
)


@pytest.fixture
def interpreter():
    interp = OpenInterpreter()
    yield interp
    try:
        interp.terminal.terminate()
    except Exception:
        pass


def test_every_attribute_row_targets_an_attribute_that_exists(interpreter):
    """Each flag with an `attribute` row writes to a real attribute of a real object.

    setattr never fails, so a renamed or misspelled target turns the flag into
    a no-op that still parses and still prints "Setting attribute ..." under
    --verbose. This is the only check that the table and the classes agree.
    """
    missing = []
    for argument in build_arguments(interpreter):
        attribute = argument.get("attribute")
        if not attribute:
            continue
        if not hasattr(attribute["object"], attribute["attr_name"]):
            missing.append(f"--{argument['name']} -> {attribute['attr_name']}")
    assert missing == []


def test_flag_names_and_nicknames_are_unique(interpreter):
    """No two rows claim the same long or short flag.

    argparse raises at parser construction on a duplicate, so a collision
    breaks every invocation of the CLI, including --help.
    """
    names = [argument["name"] for argument in build_arguments(interpreter)]
    nicknames = [a["nickname"] for a in build_arguments(interpreter) if a.get("nickname")]
    assert len(names) == len(set(names))
    assert len(nicknames) == len(set(nicknames))


def test_the_table_builds_a_parser_without_argparse_complaining(interpreter):
    """The whole table survives being turned into a real argparse parser.

    This is what start_terminal_interface does with it, and it is where a bad
    row (unknown action, choices that exclude the default) blows up — at
    startup, for everyone, whatever flags they passed.
    """
    parser = argparse.ArgumentParser()
    for argument in build_arguments(interpreter):
        flags = [f"--{argument['name']}"]
        if argument.get("nickname"):
            flags.insert(0, f"-{argument['nickname']}")
        if argument["type"] == bool:
            parser.add_argument(
                *flags,
                dest=argument["name"],
                action=argument.get("action", "store_true"),
                default=argument.get("default"),
            )
        else:
            parser.add_argument(
                *flags,
                dest=argument["name"],
                type=argument["type"],
                choices=argument.get("choices"),
                default=argument.get("default"),
                nargs=argument.get("nargs"),
            )
    args = parser.parse_args([])
    assert args.profile is not None


def test_choices_include_their_own_default(interpreter):
    """A default outside its own `choices` makes the flag unusable.

    argparse only validates values the user types, so such a row parses fine
    until someone passes the flag explicitly and gets told the documented
    default is invalid.
    """
    for argument in build_arguments(interpreter):
        choices = argument.get("choices")
        default = argument.get("default")
        if choices and default is not None:
            assert default in choices, argument["name"]


def test_set_attributes_ignores_unset_flags(interpreter):
    """None means "the user did not pass this", and nothing is written.

    set_attributes runs a second time after the profile loads. If it wrote
    None for every omitted flag, that second pass would erase the entire
    profile.
    """
    arguments = build_arguments(interpreter)
    interpreter.llm.model = "from-profile"
    set_attributes(argparse.Namespace(model=None, verbose=False), arguments)
    assert interpreter.llm.model == "from-profile"


def test_set_attributes_writes_through_to_the_nested_llm_object(interpreter):
    """Rows can target interpreter.llm, not just the interpreter.

    Half the table configures the model. If set_attributes only looked at the
    interpreter, every model flag would land on the wrong object and be
    ignored.
    """
    arguments = build_arguments(interpreter)
    set_attributes(
        argparse.Namespace(model="gpt-4o", max_output=1234, verbose=False),
        arguments,
    )
    assert interpreter.llm.model == "gpt-4o"
    assert interpreter.max_output == 1234


def test_set_attributes_skips_flags_with_no_attribute_row(interpreter):
    """Action flags (--profiles, --version, --fast) are handled by the entry point, not by setattr.

    They have no `attribute` row on purpose. Writing them onto the interpreter
    would create junk attributes such as interpreter.version.
    """
    arguments = build_arguments(interpreter)
    set_attributes(argparse.Namespace(version=True, fast=True, verbose=False), arguments)
    assert not hasattr(interpreter, "version")
    assert not hasattr(interpreter, "fast")


def test_get_argument_dictionary_returns_empty_for_an_unknown_flag(interpreter):
    """An unknown name yields {} rather than raising.

    set_attributes walks every parsed value, including ones argparse added
    itself. Raising here would make any such value fatal.
    """
    arguments = build_arguments(interpreter)
    assert get_argument_dictionary(arguments, "profile")["name"] == "profile"
    assert get_argument_dictionary(arguments, "no_such_flag") == {}


def test_deprecated_flags_map_to_flags_that_still_exist(interpreter):
    """Every rename points at a flag the table actually defines.

    start_terminal_interface rewrites argv using this map. A stale target
    would turn a deprecated flag into an "Unrecognized argument" exit, which
    is worse than the deprecation it was meant to soften.
    """
    names = {f"--{argument['name']}" for argument in build_arguments(interpreter)}
    for old, new in DEPRECATED_FLAGS.items():
        assert new in names
        assert old not in names
