import os
import re


def preprocess_shell(code):
    """
    Add active line markers, end-of-execution marker (echo works in bash and cmd).
    """
    if (
        not has_multiline_commands(code)
        and not spans_lines_inside_a_quote(code)
        and os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower() == "true"
    ):
        code = add_active_line_prints(code)

    code += '\necho "##end_of_execution##"'

    return code


def add_active_line_prints(code):
    lines = code.split("\n")
    for index, line in enumerate(lines):
        # Skip lines that execute nothing. Marking them is meaningless, and the
        # marker is an `echo`, so one sitting after the user's last real command
        # becomes the command that sets `$?`. Code ending in a newline splits to
        # a trailing empty line, which is how a failing command's exit status
        # used to be reported as 0. The index still counts skipped lines so the
        # numbers keep matching the user's source.
        if _runs_nothing(line):
            continue
        lines[index] = f'echo "##active_line{index + 1}##"\n{line}'
    return "\n".join(lines)


def _runs_nothing(line):
    """True for blank and comment-only lines.

    ``#`` is the comment character in bash; cmd uses ``rem``/``::`` instead, but
    skipping a ``#`` line there only costs a highlight for a line that would
    have failed anyway.
    """
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def spans_lines_inside_a_quote(script_text):
    """True when a quoted string or heredoc runs past the end of a line.

    Active-line markers are inserted between lines, so a string that spans
    lines gets `echo "##active_line2##"` pasted into the middle of it. A command
    like

        curl ... | python3 -c "
        import sys
        print(sys.stdin.read())
        "

    then feeds `echo` to python3, which reports `NameError: name 'echo' is not
    defined` from a line the user never wrote. The continuation check below
    only looks at how a line *ends*, so it never saw this.

    Bash quoting, to the depth that matters here: a backslash escapes the next
    character outside quotes and inside double quotes, but is literal inside
    single quotes; `#` starts a comment only outside quotes.
    """
    quote = None  # None, "'" or '"'
    heredoc_delimiter = None

    for line in script_text.splitlines():
        if heredoc_delimiter is not None:
            if line.strip() == heredoc_delimiter:
                heredoc_delimiter = None
            continue

        index = 0
        while index < len(line):
            char = line[index]
            if quote == "'":
                if char == "'":
                    quote = None
            elif quote == '"':
                if char == "\\":
                    index += 1  # escaped character, whatever it is
                elif char == '"':
                    quote = None
            else:
                if char == "\\":
                    index += 1
                elif char in "'\"":
                    quote = char
                elif char == "#":
                    break  # comment: the rest of the line is not code
                elif char == "<" and line[index : index + 2] == "<<":
                    rest = line[index + 2 :].lstrip("-").strip()
                    delimiter = rest.split()[0] if rest.split() else ""
                    if delimiter and not delimiter.startswith("<"):
                        heredoc_delimiter = delimiter.strip("\"'")
                        break
            index += 1

        if quote is not None or heredoc_delimiter is not None:
            return True

    return False


def has_multiline_commands(script_text):
    continuation_patterns = [
        r"\\$",
        r"\|$",
        r"&&\s*$",
        r"\|\|\s*$",
        r"<\($",
        r"\($",
        r"{\s*$",
        r"\bif\b",
        r"\bwhile\b",
        r"\bfor\b",
        r"do\s*$",
        r"then\s*$",
    ]

    for line in script_text.splitlines():
        if any(re.search(pattern, line.rstrip()) for pattern in continuation_patterns):
            return True

    return False
