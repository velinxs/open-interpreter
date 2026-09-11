"""Squeezing the padding out of console output before the model is charged for it.

Terminal programs format for a human watching live: colour escapes, progress
bars redrawn in place with carriage returns, one identical warning per item.
The model sees none of that as information, but pays for it in every later
request, so it is removed before the output is stored and truncated.
"""

import re

# CSI/OSC escape sequences: colours, cursor moves, titles.
_ANSI = re.compile(r"\x1b(?:\[[0-9;?]*[A-Za-z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")

# Folding fewer than this many identical lines would cost more than it saves.
_MIN_REPEATS_TO_FOLD = 3


def _last_frame(line):
    """A line redrawn in place keeps only what was on screen at the end."""
    if "\r" not in line:
        return line
    frames = [frame for frame in line.split("\r") if frame]
    return frames[-1] if frames else ""


def compact_console_output(text):
    """Return `text` with escape codes, redrawn frames and repeat runs removed."""
    if not text:
        return text

    text = _ANSI.sub("", text)
    if "\r" in text:
        text = "\n".join(_last_frame(line) for line in text.split("\n"))

    if "\n" not in text:
        return text

    lines = text.split("\n")
    out = []
    index = 0
    while index < len(lines):
        line = lines[index]
        run = 1
        while index + run < len(lines) and lines[index + run] == line:
            run += 1
        out.append(line)
        if run >= _MIN_REPEATS_TO_FOLD:
            out.append(f"... previous line repeated {run - 1} more times ...")
        else:
            out.extend([line] * (run - 1))
        index += run
    return "\n".join(out)
