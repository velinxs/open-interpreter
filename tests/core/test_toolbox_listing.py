"""The toolbox catalogue injected into the system prompt.

It is re-sent with every request, so it carries the one thing a model cannot
recover on its own: the names of the callables. Signatures, parameters and
return shapes come from help() at the moment of use, where the docstring is
live and cannot go stale.

Every entry is generated from the live signature. Anything hand-derived drifts
from the code, and a model that follows a wrong listing fails through no fault
of its own.
"""

import tiktoken

TOOLBOX_LISTING_TOKEN_BUDGET = 600
_enc = tiktoken.get_encoding("cl100k_base")


def test_names_only_is_the_default(offline_interpreter):
    """The catalogue lists callables, not signatures or prose."""
    listing = offline_interpreter.toolbox.system_message

    assert "toolbox.web.search" in listing
    assert "toolbox.web.search(" not in listing, "a signature crept back in"
    assert "#" not in listing.split("```python")[1].split("```")[0], "descriptions crept back in"


def test_variadic_arguments_are_visible(offline_interpreter):
    """For several tools `*args` IS the argument, so dropping it hid the point.

    `keyboard.hotkey(interval=0.1)` gave a model no way to pass the keys and
    `mouse.click(button='left', ...)` no way to pass the target; both looked
    complete, so neither prompted a help() call.
    """
    toolbox = offline_interpreter.toolbox
    toolbox.api_listing = "full"
    full = toolbox.system_message

    assert "toolbox.keyboard.hotkey(*args, interval=0.1)" in full
    assert "toolbox.mouse.click(*args, button='left', clicks=1, interval=0.1, **kwargs)" in full
    assert "toolbox.files.search(*args, **kwargs)" in full


def test_browser_entries_are_real_signatures(offline_interpreter):
    """Browser had a hand-rolled branch that leaked `self` and lost defaults.

    It read `co_varnames` off the unbound function, so every browser entry
    advertised `self` as its first argument and `search_google`'s `delays=True`
    disappeared.
    """
    toolbox = offline_interpreter.toolbox
    toolbox.api_listing = "full"
    entries = [e for e in toolbox._get_all_toolbox_tools_signature_and_description() if ".browser." in e]

    assert entries
    for entry in entries:
        assert "(self" not in entry, entry
    assert any(e.startswith("toolbox.browser.search_google(query, delays=True)") for e in entries), entries


def test_internal_and_deprecated_methods_are_not_advertised(offline_interpreter):
    """Naming a method the model must not call buys a wasted turn and tokens.

    `skills.import_skills` says "[INTERNAL METHOD - NOT FOR Assistant USE]" in
    its own docstring and `skills.run` says "DEPRECATED"; the lazy bootstraps
    (`browser.setup`, `browser.driver`, `vision.load`, `ai2.client`) are run by
    the real methods themselves.
    """
    toolbox = offline_interpreter.toolbox
    hidden = [
        "toolbox.skills.import_skills",
        "toolbox.skills.run",
        "toolbox.browser.setup",
        "toolbox.browser.driver",
        "toolbox.vision.load",
        "toolbox.ai2.client",
    ]

    names = toolbox.system_message
    toolbox.api_listing = "full"
    full = toolbox.system_message

    for name in hidden:
        assert name not in names, name
        assert name not in full, name
    assert "toolbox.skills.list" in names, "the real skills methods should survive"


def test_building_the_listing_does_not_start_a_browser(offline_interpreter):
    """`browser.driver` is a property that launches Chrome on first read.

    The catalogue must describe the toolbox, never exercise it, so attributes
    are looked up on the class rather than evaluated on the instance.
    """
    browser = offline_interpreter.toolbox.browser

    offline_interpreter.toolbox.system_message
    offline_interpreter.toolbox.api_listing = "full"
    offline_interpreter.toolbox.system_message

    assert browser._driver is None


def test_every_callable_is_named(offline_interpreter):
    """A model can ask help() for any signature, but only if it knows the name."""
    toolbox = offline_interpreter.toolbox
    listing = toolbox.system_message

    for entry in toolbox._get_all_toolbox_tools_signature_and_description():
        entry = entry.strip()
        if entry.startswith("#") or not entry:
            continue
        # Properties have no parentheses, so trim the description first.
        name = entry.split(" # ")[0].split("(")[0].strip()
        assert name in listing, f"{name} is not discoverable"


def test_listing_stays_within_budget(offline_interpreter):
    """Names for 61 tools should cost hundreds of tokens, not thousands."""
    tokens = len(_enc.encode(offline_interpreter.toolbox.system_message))
    print(f"\ntoolbox listing: {tokens} tokens")
    assert tokens <= TOOLBOX_LISTING_TOKEN_BUDGET


def test_full_listing_is_still_available(offline_interpreter):
    """A profile can ask for the old catalogue back with toolbox.api_listing."""
    offline_interpreter.toolbox.api_listing = "full"
    listing = offline_interpreter.toolbox.system_message

    assert "toolbox.web.search(" in listing, "signatures should return"
    assert len(_enc.encode(listing)) > TOOLBOX_LISTING_TOKEN_BUDGET


def test_entries_carry_no_return_shape(offline_interpreter):
    """Even in full mode, the return shape belongs to help(), not the prompt."""
    offline_interpreter.toolbox.api_listing = "full"
    entries = [
        e
        for e in offline_interpreter.toolbox._get_all_toolbox_tools_signature_and_description()
        if not e.strip().startswith("#")
    ]

    assert entries
    for entry in entries:
        assert "\n" not in entry.strip(), entry
        assert "Returns:" not in entry, entry
