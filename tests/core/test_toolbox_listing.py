"""The toolbox catalogue injected into the system prompt.

It is re-sent with every request, so it carries the one thing a model cannot
recover on its own: the names of the callables. Signatures, parameters and
return shapes come from help() at the moment of use, where the docstring is
live and cannot go stale.
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
