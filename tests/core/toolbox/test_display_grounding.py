"""Finding something on screen without shipping the screen anywhere.

Locating an icon or a string used to POST a screenshot of the user's desktop to
a hosted endpoint whenever `offline` was False -- which is the default -- and
fall back to a multi-gigabyte local stack only if that failed. Whatever was on
screen went with it, unresized in two of the three call sites, with no prompt.

The model in the session can already read images, so it is asked instead. These
tests cover the parsing that turns its answer into coordinates, and the
dispatch that must never reach for the network again.
"""

from types import SimpleNamespace

import pytest
from PIL import Image

from interpreter.core.toolbox.display.display import Display
from interpreter.core.toolbox.display.ground import GroundingUnavailable, _coordinates_from


def _toolbox(supports_vision=True):
    llm = SimpleNamespace(
        model="ollama_chat/qwen3", api_base="http://127.0.0.1:11435",
        api_key=None, supports_vision=supports_vision,
    )
    return SimpleNamespace(
        interpreter=SimpleNamespace(llm=llm), debug=False, offline=True
    )


# --- turning a reply into coordinates ---------------------------------------


def test_coordinates_are_normalized_against_the_image():
    """mouse.move multiplies by display width/height, so 0-1 is the contract."""
    assert _coordinates_from('{"x": 400, "y": 250}', 800, 500) == (0.5, 0.5)


def test_the_last_object_wins():
    """A thinking model reasons out loud before answering.

    It routinely shows a candidate coordinate mid-thought and corrects it, so
    reading the first JSON object would click where it changed its mind about.
    """
    reply = 'Maybe {"x": 10, "y": 10}? No — the button is lower. {"x": 400, "y": 250}'
    assert _coordinates_from(reply, 800, 500) == (0.5, 0.5)


def test_not_found_is_none_rather_than_a_guess():
    """An absent element must report absence, not a plausible-looking point."""
    assert _coordinates_from('{"found": false}', 800, 500) is None


def test_coordinates_outside_the_image_are_rejected():
    """Off-screen numbers are a misread, and clicking them hits something else."""
    assert _coordinates_from('{"x": 4000, "y": 250}', 800, 500) is None
    assert _coordinates_from('{"x": -5, "y": 250}', 800, 500) is None


def test_unparseable_replies_do_not_raise():
    """A model that ignored the format yields nothing, not a traceback."""
    assert _coordinates_from("I can see a button near the middle.", 800, 500) is None
    assert _coordinates_from("", 800, 500) is None
    assert _coordinates_from('{"x": "left", "y": 250}', 800, 500) is None


# --- dispatch ---------------------------------------------------------------


def test_an_icon_lookup_returns_a_bare_tuple(monkeypatch):
    """mouse.move unpacks `x, y = coordinates[0]` on the icon path."""
    monkeypatch.setattr(
        "interpreter.core.toolbox.display.display.locate",
        lambda llm, screenshot, description: (0.5, 0.5),
    )
    found = Display(_toolbox()).find("the Delete button", screenshot=Image.new("RGB", (800, 500)))

    assert found == [(0.5, 0.5)]


def test_a_text_lookup_returns_the_dict_shape(monkeypatch):
    """The text path reads item["coordinates"] and item["similarity"]."""
    monkeypatch.setattr(
        "interpreter.core.toolbox.display.display.locate",
        lambda llm, screenshot, description: (0.25, 0.75),
    )
    found = Display(_toolbox()).find('"Submit"', screenshot=Image.new("RGB", (800, 500)))

    assert found == [{"coordinates": (0.25, 0.75), "text": "Submit", "similarity": 1}]


def test_something_not_on_screen_is_an_empty_list(monkeypatch):
    """Empty means "not there" -- mouse.move says so instead of clicking."""
    monkeypatch.setattr(
        "interpreter.core.toolbox.display.display.locate",
        lambda llm, screenshot, description: None,
    )
    display = Display(_toolbox())
    screenshot = Image.new("RGB", (800, 500))

    assert display.find("a button that is not there", screenshot=screenshot) == []
    assert display.find('"absent"', screenshot=screenshot) == []


def test_without_vision_it_says_so_instead_of_uploading(monkeypatch):
    """No vision and no local locator is an error, never a network call.

    This is the regression that matters: the old code answered this case by
    POSTing the screenshot to a hosted endpoint.
    """
    monkeypatch.setattr(
        "interpreter.core.toolbox.display.display.locate",
        lambda *a, **k: pytest.fail("locate() must not run without vision"),
    )
    display = Display(_toolbox(supports_vision=False))

    with pytest.raises((GroundingUnavailable, ImportError, Exception)) as error:
        display.find("the Delete button", screenshot=Image.new("RGB", (800, 500)))
    assert "requests" not in str(type(error.value)).lower()


def test_the_display_module_no_longer_posts_anything():
    """No upload path survives anywhere in the module.

    Belt and braces: the three call sites are gone, and this fails if one is
    reintroduced under any condition.
    """
    from pathlib import Path

    import interpreter.core.toolbox.display.display as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "requests.post" not in source
    assert "/point/" not in source
    assert "api_base" not in source
