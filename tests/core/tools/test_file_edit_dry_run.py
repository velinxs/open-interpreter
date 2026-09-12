"""The edit preview the user approves, and the dispatch that runs the real thing.

dry_run_edit is what the terminal shows above "Would you like to apply this
edit? (y/n)". Two things have to hold: the preview must reflect what will
actually happen, and running it must not change the file — otherwise the user
is approving an edit that has already been made.
"""

import shutil

import pytest

from interpreter.core.tools.file_edit import EDIT_LANGUAGES, dry_run_edit, run_edit

requires_sed = pytest.mark.skipif(shutil.which("sed") is None, reason="sed not installed")
requires_jq = pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
requires_patch = pytest.mark.skipif(shutil.which("patch") is None, reason="patch not installed")


@pytest.fixture
def text_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    return path


@pytest.fixture
def json_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"name": "old", "count": 1}\n', encoding="utf-8")
    return path


# --- languages with no preview ----------------------------------------------


@pytest.mark.parametrize("language", ["write", "poke"])
def test_languages_without_a_preview_return_none(language, tmp_path):
    """write and poke have nothing meaningful to preview, and say so with None.

    None means "no dry run available"; the caller renders nothing rather than
    an empty panel. write's preview would just be the code the user can
    already see, and poke edits binaries.
    """
    assert dry_run_edit(language, "anything", str(tmp_path / "new.txt")) is None


def test_an_unknown_language_has_no_preview(tmp_path):
    """A language dry_run_edit does not handle returns None rather than raising.

    The preview is optional; a missing one must not block the edit prompt.
    """
    assert dry_run_edit("klingon", "x", str(tmp_path / "f.txt")) is None


# --- sed --------------------------------------------------------------------


@requires_sed
def test_a_sed_preview_shows_the_result_without_touching_the_file(text_file):
    """The preview is the transformed text, and the file on disk is unchanged.

    This is the whole contract. If the dry run wrote, the user would be asked
    to approve an edit that had already happened.
    """
    before = text_file.read_text()
    preview = dry_run_edit("sed", "s/beta/BETA/", str(text_file))
    assert preview["ok"] is True
    assert "BETA" in preview["output"]
    assert text_file.read_text() == before


@requires_sed
def test_a_failing_sed_script_previews_as_not_ok(text_file):
    """A bad script reports ok=False and carries the tool's own error text.

    ok=False is what makes the terminal render the output as plain text rather
    than syntax-highlighted source, and the message is what lets the model fix
    its script.
    """
    preview = dry_run_edit("sed", "s/unterminated", str(text_file))
    assert preview["ok"] is False
    assert preview["output"]


@requires_sed
def test_an_empty_sed_script_is_rejected_before_the_tool_runs(text_file):
    """An empty program raises rather than previewing a no-op.

    sed with no commands succeeds and prints the file unchanged, which would
    show the user a preview that looks like a valid edit doing nothing.
    """
    with pytest.raises(ValueError):
        dry_run_edit("sed", "   ", str(text_file))


@requires_sed
def test_previewing_an_edit_to_a_missing_file_fails_loudly(tmp_path):
    """A target that does not exist raises, rather than previewing against nothing."""
    with pytest.raises(FileNotFoundError):
        dry_run_edit("sed", "s/a/b/", str(tmp_path / "nope.txt"))


@requires_sed
def test_previewing_a_relative_target_is_refused(text_file):
    """A relative target is rejected the same way it is for the real edit.

    Resolving it against the process cwd would preview a different file from
    the one the model named.
    """
    with pytest.raises(ValueError):
        dry_run_edit("sed", "s/a/b/", "notes.txt")


# --- jq ---------------------------------------------------------------------


@requires_jq
def test_a_jq_preview_shows_the_transformed_json_and_leaves_the_file(json_file):
    """The preview is the new document; the original stays on disk."""
    before = json_file.read_text()
    preview = dry_run_edit("jq", '.name = "new"', str(json_file))
    assert preview["ok"] is True
    assert '"new"' in preview["output"]
    assert json_file.read_text() == before


@requires_jq
def test_an_invalid_jq_filter_previews_as_not_ok(json_file):
    """A syntax error in the filter comes back as a failed preview, not an exception."""
    preview = dry_run_edit("jq", ".name =", str(json_file))
    assert preview["ok"] is False


# --- patch ------------------------------------------------------------------


@requires_patch
def test_a_patch_preview_includes_the_diff_and_does_not_apply_it(text_file):
    """GNU patch --dry-run prints only "checking file"; the diff is appended.

    Without the append the user is asked to approve a change described as
    "checking file notes.txt" — no indication of what it does.
    """
    diff = f"--- {text_file.name}\n+++ {text_file.name}\n@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETA\n gamma\n"
    before = text_file.read_text()
    preview = dry_run_edit("patch", diff, str(text_file))

    assert preview["ok"] is True
    assert "@@" in preview["output"]
    assert "+BETA" in preview["output"]
    assert text_file.read_text() == before


@requires_patch
def test_a_patch_that_does_not_apply_previews_as_not_ok(text_file):
    """A diff whose context does not match reports failure before anything is written.

    Applying a mismatched patch is how .rej files and half-edited source
    appear; catching it in the preview keeps the file untouched.
    """
    diff = f"--- {text_file.name}\n+++ {text_file.name}\n@@ -1,3 +1,3 @@\n nothing\n-like\n+this\n file\n"
    preview = dry_run_edit("patch", diff, str(text_file))
    assert preview["ok"] is False
    assert text_file.read_text() == "alpha\nbeta\ngamma\n"


@requires_patch
def test_an_empty_patch_body_is_rejected(text_file):
    """An empty diff raises instead of reporting a successful no-op edit."""
    with pytest.raises(ValueError):
        dry_run_edit("patch", "\n  \n", str(text_file))


@requires_patch
def test_patch_line_endings_are_normalised_before_preview(text_file):
    """A CRLF diff is previewed successfully.

    Models emit \\r\\n on Windows and in copied text. GNU patch treats the
    carriage returns as part of the context and rejects the hunk, so the
    normalisation is what stops a valid edit looking broken.
    """
    diff = (
        f"--- {text_file.name}\r\n+++ {text_file.name}\r\n@@ -1,3 +1,3 @@\r\n alpha\r\n-beta\r\n+BETA\r\n gamma\r\n"
    )
    preview = dry_run_edit("patch", diff, str(text_file))
    assert preview["ok"] is True


# --- dispatch ---------------------------------------------------------------


def test_run_edit_rejects_an_unknown_language_and_lists_the_real_ones(tmp_path):
    """An unsupported language names the alternatives.

    The model picks the language itself. An error that does not list the
    options leaves it guessing, usually at the same wrong name again.
    """
    with pytest.raises(ValueError) as excinfo:
        run_edit("perl", "s/a/b/", str(tmp_path / "f.txt"))
    message = str(excinfo.value)
    for language in EDIT_LANGUAGES:
        assert language in message


def test_run_edit_requires_string_code(tmp_path):
    """Non-string code is refused before any tool runs.

    A dict or list here would be stringified into the temp script and run as
    whatever that repr happens to mean.
    """
    with pytest.raises(ValueError):
        run_edit("write", {"not": "a string"}, str(tmp_path / "f.txt"))


def test_run_edit_is_case_and_whitespace_insensitive_about_the_language(tmp_path):
    """" WRITE " is the write language.

    The value comes straight from model-generated JSON, which is not reliably
    lowercase or trimmed.
    """
    target = tmp_path / "made.txt"
    run_edit("  WRITE  ", "body", str(target))
    assert target.read_text() == "body"


def test_write_creates_missing_parent_directories(tmp_path):
    """A new file in a directory that does not exist yet still gets written.

    Failing here would force the model into a separate mkdir round trip for
    something it plainly meant to do.
    """
    target = tmp_path / "deep" / "nested" / "file.txt"
    result = run_edit("write", "hello", str(target))
    assert target.read_text() == "hello"
    assert "5 bytes" in result


def test_write_refuses_to_clobber_an_existing_file(tmp_path):
    """write only creates; overwriting needs a real edit language.

    This is the guard against a model "rewriting" a file it never read and
    silently discarding everything that was in it.
    """
    target = tmp_path / "existing.txt"
    target.write_text("precious")
    with pytest.raises(FileExistsError) as excinfo:
        run_edit("write", "replacement", str(target))
    assert target.read_text() == "precious"
    assert "already exists" in str(excinfo.value)
