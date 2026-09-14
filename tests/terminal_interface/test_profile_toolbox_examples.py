"""Every `toolbox.*` call written into a default profile must be callable.

A profile's prompt is an instruction, not documentation: a model shown
`toolbox.calendar.create_event(title=..., start_date=..., notes=...)` copies it
verbatim, and the missing required `end_date` becomes a TypeError the model had
no way to foresee. Binding each example against the live signature catches that
drift the moment the toolbox changes, instead of in someone's session.
"""

import ast
import inspect
import pathlib
import re
from unittest import mock

import pytest

from interpreter.core.toolbox.toolbox import Toolbox

PROFILES = pathlib.Path(__file__).resolve().parents[2] / "interpreter/terminal_interface/profiles/defaults"

# `ai2` is left out on purpose: reading it builds a LiteLLM client, and no
# profile calls it. Everything else is a plain attribute of Toolbox.
MODULES = (
    "ai",
    "browser",
    "calendar",
    "clipboard",
    "contacts",
    "display",
    "docs",
    "files",
    "keyboard",
    "mail",
    "mouse",
    "os",
    "skills",
    "sms",
    "vision",
    "web",
)
_CALL = re.compile(r"toolbox\.(" + "|".join(MODULES) + r")\.([A-Za-z_]\w*)\(")


def _call_source(text, start):
    """The full `toolbox.x.y(...)` expression starting at `start`, or None."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _examples():
    """(path, line, module, method, ast.Call) for every toolbox call in a profile."""
    for path in sorted(PROFILES.glob("*.py")):
        text = path.read_text()
        for match in _CALL.finditer(text):
            source = _call_source(text, match.start())
            if source is None:
                continue
            try:
                node = ast.parse(source, mode="eval").body
            except SyntaxError:
                continue  # prose, or a call broken across an escaped string
            if isinstance(node, ast.Call):
                yield path.name, text[: match.start()].count("\n") + 1, match[1], match[2], node


def test_there_are_examples_to_check():
    """A silent regex change would otherwise turn this file into a no-op."""
    assert len(list(_examples())) > 40


@pytest.mark.parametrize(
    "profile,line,module,method,call",
    [pytest.param(*e, id=f"{e[0]}:{e[1]}:{e[2]}.{e[3]}") for e in _examples()],
)
def test_profile_example_matches_the_real_signature(profile, line, module, method, call):
    """Each example binds to the method it names, with the arguments it passes."""
    toolbox = Toolbox(mock.Mock())
    target = getattr(getattr(toolbox, module), method, None)
    assert target is not None, f"{profile}:{line} calls toolbox.{module}.{method}, which does not exist"

    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
        pytest.skip("unpacked arguments say nothing about arity")

    # Only the shape matters, so every argument is a placeholder.
    args = [None] * len(call.args)
    kwargs = {kw.arg: None for kw in call.keywords}
    try:
        inspect.signature(target).bind(*args, **kwargs)
    except TypeError as error:
        pytest.fail(f"{profile}:{line}: toolbox.{module}.{method} — {error}")
