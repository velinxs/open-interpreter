"""Compacting console output before it is charged to the context.

Terminal programs pad their output for a human watching: colour escapes,
progress bars redrawn on the same line with carriage returns, the same
warning once per item. None of it tells the model anything, and all of it
is paid for on every later request, so it is squeezed out first.
"""

from interpreter.core.utils.compact_output import compact_console_output


def test_ansi_colour_and_cursor_codes_are_removed():
    """pip, npm and friends colour their output even when not a terminal."""
    coloured = "\x1b[32mSuccessfully installed\x1b[0m scipy-1.14.0\n"
    assert compact_console_output(coloured) == "Successfully installed scipy-1.14.0\n"


def test_progress_bar_frames_collapse_to_the_last_one():
    """A bar redrawn with \\r is one line to a human and 100 lines to a model."""
    frames = "".join(f"\rDownloading {p}%" for p in range(0, 101, 10)) + "\n"
    assert compact_console_output(frames) == "Downloading 100%\n"


def test_runs_of_identical_lines_are_folded_with_a_count():
    """A warning repeated per item says the same thing however many times it runs."""
    out = compact_console_output("warn: deprecated\n" * 40 + "done\n")
    assert out == "warn: deprecated\n... previous line repeated 39 more times ...\ndone\n"


def test_short_runs_are_left_alone():
    """Folding two identical lines would cost more characters than it saves."""
    text = "a\na\nb\n"
    assert compact_console_output(text) == text


def test_ordinary_output_is_returned_unchanged():
    text = "Frequency  Level\n30.000  -64.00\n32.432  -64.50\n"
    assert compact_console_output(text) == text


def test_empty_output_stays_empty():
    """Empty output is meaningful (the command ran and said nothing)."""
    assert compact_console_output("") == ""
