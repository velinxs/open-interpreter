"""Streaming code-fence parser for text (non-tool-calling) models.

Providers stream in arbitrary token boundaries: "```py" + "thon\\n", "\\n```"
glued to the last line of code, a lone "`" from inline code. The parser must
produce the same message and code chunks whatever the boundaries are.
"""

import pytest

from interpreter.core.llm.run_text_llm import stream_to_lmc


def _collect(pieces, default_language="python"):
    out = {"message": "", "code": "", "formats": set()}
    for chunk in stream_to_lmc(iter(pieces), default_language):
        out[chunk["type"]] += chunk["content"]
        if chunk["type"] == "code":
            out["formats"].add(chunk["format"])
    return out


@pytest.mark.parametrize(
    "pieces",
    [
        ["Sure.\n```python\nprint(21 * 2)\n```"],
        ["Sure.\n```py", "thon\nprint(", "21 * 2)\n```"],
        ["Sure.\n", "``", "`python", "\n", "print(21 * 2)", "\n``", "`"],
        ["Sure.\n```", "python\n", "print(21 * 2)\n", "```\nAnything after is ignored"],
    ],
)
def test_language_tag_and_fences_survive_any_chunking(pieces):
    """The language line and the fences are recognised however the stream is split."""
    got = _collect(pieces)
    assert got["message"] == "Sure.\n"
    assert got["code"] == "print(21 * 2)\n"
    assert got["formats"] == {"python"}


def test_code_may_contain_the_language_name():
    """`print("python")` must reach the kernel intact; the old parser stripped the word."""
    got = _collect(["```python\n", 'print("python")\n', "```"])
    assert got["code"] == 'print("python")\n'


def test_inline_backticks_in_prose_are_kept():
    """Single backticks are ordinary message text, even when split across chunks."""
    got = _collect(["use `", "ls` to", " list files"])
    assert got["message"] == "use `ls` to list files"
    assert got["code"] == ""


def test_empty_language_defaults():
    """A bare ``` fence uses the caller's default language."""
    got = _collect(["```\nls -la\n```"], default_language="shell")
    assert got["code"] == "ls -la\n"
    assert got["formats"] == {"shell"}


def test_language_is_normalised_to_letters():
    """Hallucinated tags such as ' Python 3' become 'Python'; spaces and digits go."""
    got = _collect(["``` Python 3\nprint(1)\n```"])
    assert got["formats"] == {"Python"}


def test_stream_stops_after_closing_fence():
    """Nothing after the first closing fence is emitted; the caller executes the code first."""
    got = _collect(["```python\nprint(1)\n```\nNow more prose\n```python\nprint(2)\n```"])
    assert got["code"] == "print(1)\n"
    assert got["message"] == ""
