"""The small terminal_interface helpers.

None of these is interesting on its own, but each one sits on a path where a
wrong answer is silent: an image that is never attached, a lexer that makes a
diff unreadable, an export written somewhere the user cannot find.
"""

import os

import pytest

from interpreter.terminal_interface.utils.count_tokens import (
    count_messages_tokens,
    count_tokens,
    token_cost,
)
from interpreter.terminal_interface.utils.find_image_path import find_image_path
from interpreter.terminal_interface.utils.local_storage_path import get_storage_path
from interpreter.terminal_interface.utils.target_file_lexer import (
    syntax_lang_for_dry_run,
    syntax_lang_for_target_path,
)

# --- find_image_path --------------------------------------------------------


def test_only_paths_that_exist_are_returned(tmp_path):
    """A path-shaped string that is not a file on disk is not an image.

    Users talk about filenames constantly ("update /src/logo.png"). Treating
    every mention as an attachment would prompt for an upload of a file that
    does not exist.
    """
    real = tmp_path / "real.png"
    real.write_bytes(b"x")
    found = find_image_path(f"compare {real} with {tmp_path / 'missing.png'}")
    assert found == [str(real)]


@pytest.mark.parametrize("extension", ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"])
def test_every_supported_extension_is_recognised(tmp_path, extension):
    """Case variants count too; people drag files in from anywhere.

    A missed extension means the image is silently sent as a filename string
    the model cannot open.
    """
    path = tmp_path / f"shot.{extension}"
    path.write_bytes(b"x")
    assert find_image_path(str(path)) == [str(path)]


def test_a_message_with_no_image_returns_an_empty_list():
    """No paths means no prompt, not None.

    The caller does `if image_paths:` and then iterates; returning None would
    be an equally falsy but uniterable answer if anything ever changed.
    """
    assert find_image_path("what is 21*2") == []


def test_the_same_path_mentioned_twice_is_returned_once(tmp_path):
    """Duplicates are collapsed so the upload prompt lists each file once."""
    path = tmp_path / "a.png"
    path.write_bytes(b"x")
    assert find_image_path(f"{path} and again {path}") == [str(path)]


def test_multiple_images_are_returned_in_first_appearance_order(tmp_path):
    """Regression: the docstring promised insertion order, but `list(set(...))` used to discard it.

    The caller sends image_paths[0] to chat() and appends the rest, so which
    picture is "first" for the model depends on this order. Dedup now goes
    through dict.fromkeys, which preserves first-appearance order, so the
    result matches the order the paths appear in the message.
    """
    paths = []
    for index in range(4):
        path = tmp_path / f"img{index}.png"
        path.write_bytes(b"x")
        paths.append(str(path))
    assert find_image_path(" ".join(paths)) == paths


# --- target_file_lexer ------------------------------------------------------


@pytest.mark.parametrize(
    "target,expected",
    [
        ("a.py", "python"),
        ("a.TS", "typescript"),
        ("a.yml", "yaml"),
        ("a.hpp", "cpp"),
        ("Makefile", "text"),
        ("a.unknownext", "text"),
    ],
)
def test_lexers_are_chosen_by_extension_case_insensitively(target, expected):
    """The highlighter is picked from the file's suffix, lowercased, with a text fallback.

    An unknown extension must fall back rather than raise: Pygments rejects a
    lexer name it does not know, and that would turn a working edit preview
    into a traceback.
    """
    assert syntax_lang_for_target_path(target) == expected


def test_a_patch_is_highlighted_as_a_diff_whatever_it_patches():
    """Patch edits show as diff, not as the target file's language.

    A unified diff highlighted as Python renders the +/- markers as syntax
    errors, which is exactly the part the user is reading.
    """
    assert syntax_lang_for_dry_run("main.py", "patch") == "diff"
    assert syntax_lang_for_dry_run("main.py", "write") == "python"


# --- count_tokens -----------------------------------------------------------


def test_token_counting_strips_a_provider_prefix():
    """"openai/gpt-4o" is counted with gpt-4o's tokenizer.

    tiktoken does not know prefixed names. Without the strip every prefixed
    model would fall into the exception handler and report zero tokens, making
    %tokens useless for anyone not on a bare OpenAI model.
    """
    assert count_tokens("hello world", model="openai/gpt-4o") == count_tokens("hello world", model="gpt-4o")
    assert count_tokens("hello world", model="gpt-4o") > 0


def test_an_unknown_model_falls_back_instead_of_failing():
    """An unrecognised model is counted with the gpt-4 tokenizer and says so.

    The count is an estimate either way. Raising would break %tokens for every
    non-OpenAI model.
    """
    assert count_tokens("hello world", model="some-local-model") > 0


def test_token_counting_never_raises():
    """Every failure path returns 0, because this is a display-only feature.

    It runs inside %tokens and the cost display; an exception there would take
    down the turn over a number nobody depends on.
    """
    assert token_cost(100, model="not-a-real-model") == 0
    assert count_messages_tokens([{"message": None}], model="gpt-4") == (0, 0)


def test_message_token_counting_adds_code_and_output():
    """A message's code and output count towards the total, not just its text.

    Code output is often the largest part of a conversation. Counting only the
    prose would under-report the context by an order of magnitude.
    """
    text_only, _ = count_messages_tokens([{"message": "hello"}], model="gpt-4")
    with_code, _ = count_messages_tokens(
        [{"message": "hello", "code": "print('x')", "output": "x"}],
        model="gpt-4",
    )
    assert with_code > text_only


def test_plain_strings_are_counted_too():
    """A bare string in the message list is counted rather than skipped.

    %tokens passes the prompt through as a plain string.
    """
    tokens, _ = count_messages_tokens(["some prompt text"], model="gpt-4")
    assert tokens > 0


# --- local_storage_path -----------------------------------------------------


def test_storage_paths_are_built_under_one_config_directory():
    """Every subdirectory hangs off the same platform config dir.

    Conversations, profiles, and models all resolve through here. A
    subdirectory that escaped would scatter user data across the filesystem.
    """
    base = get_storage_path()
    assert get_storage_path("conversations") == os.path.join(base, "conversations")
    assert base.endswith("open-interpreter")
