"""The active-line instrumentation must not change what the model's Python does.

Every block the model writes is rewritten before the kernel sees it: markers are
injected so the terminal can highlight the running line. The rewrite is only
allowed to add prints. These tests run the original and the instrumented code
and compare what each one actually did, because a text assertion cannot tell the
difference between "marked up" and "quietly corrupted".
"""

import contextlib
import io
import re

import pytest

from interpreter.core.terminal.languages.python_preprocess import preprocess_python


@pytest.fixture(autouse=True)
def active_line_detection_on(monkeypatch):
    """Instrumentation is opt-out; another test may have turned it off globally."""
    monkeypatch.setenv("INTERPRETER_ACTIVE_LINE_DETECTION", "true")


def run(code):
    """Execute `code` and return its stdout with any active-line markers removed."""
    namespace = {"__name__": "__main__"}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "<cell>", "exec"), namespace)
    return re.sub(r"##active_line\d+##\n", "", buffer.getvalue())


def markers(code):
    return [int(n) for n in re.findall(r"##active_line(\d+)##", code)]


@pytest.mark.parametrize(
    "code",
    [
        # A YAML/Markdown/SQL document held in a string: the `#` line is data.
        # The one-line docstring above it is what desynchronised the scanner.
        'def write_config():\n'
        '    """Write config."""\n'
        '    cfg = """\n'
        "# database\n"
        "host: x\n"
        '"""\n'
        "    return cfg\n"
        "\n\n"
        "print(repr(write_config()))",
        # A one-line triple-quoted string used to desynchronise the scanner, so
        # the *next* string's `#` line was rewritten to `pass`.
        'x = """one"""\n# a real comment\ny = """p\n# q\nr"""\nprint(repr(y))',
        # Mixed quote styles: `"""` appears inside a `'''` string.
        "x = '''a\n\"\"\"\n# keep me\nb'''\nprint(repr(x))",
        # A blank line inside a string is part of the string.
        's = """a\n\nb"""\nprint(repr(s))',
    ],
)
def test_text_inside_a_string_literal_is_left_alone(code):
    """Lines that only look like comments belong to the string that contains them.

    The scanner that decided "this line is a comment, replace it with pass" was
    tracking triple quotes by toggling a flag, which a one-line docstring or a
    nested quote style desynchronises. It rewrote the model's data.
    """
    assert run(preprocess_python(code)) == run(code)


def test_function_and_class_docstrings_survive():
    """A marker printed before a docstring demotes it to a dead expression.

    Everything that reads __doc__ — help(), the prompt's own advice to inspect
    objects, skill saving — then sees None for every function the model defines.
    """
    code = 'def f():\n    """the doc"""\n    return 1\n\n\nclass K:\n    """kdoc"""\n\n\nprint(f.__doc__, K.__doc__)'
    assert run(preprocess_python(code)) == "the doc kdoc\n"


def test_future_import_stays_the_first_statement():
    """`from __future__ import ...` is a syntax error anywhere but the top.

    Instrumented code must still compile; a marker printed ahead of the import
    made the whole block unrunnable.
    """
    code = 'from __future__ import annotations\n\n\ndef f(x: undefined_annotation) -> None:\n    print("ok")\n\n\nf(1)'
    compile(preprocess_python(code), "<cell>", "exec")  # raises if the import moved
    assert run(preprocess_python(code)) == run(code)


@pytest.mark.parametrize(
    "code",
    [
        "f = lambda x=1: x\nprint(f())",
        "a = 1 if True else 2\nprint(a)",
        "double = lambda n: n * 2\nprint([double(n) if n else 0 for n in (1, 2)])",
    ],
)
def test_lambdas_and_ternaries_are_instrumented(code):
    """A lambda body is an expression, not a block of statements.

    Treating it as one raised inside the transformer; the caller swallowed that
    and ran the block unmarked, so any block containing a lambda or a ternary
    silently lost highlighting.
    """
    processed = preprocess_python(code)
    assert markers(processed), processed
    assert run(processed) == run(code)
