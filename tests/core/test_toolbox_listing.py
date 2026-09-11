"""The toolbox API listing injected into the system prompt.

It is re-sent with every request, so each entry is a signature and one short
sentence; parameters, return shapes and examples come from
help(toolbox.module.method), which the prompt tells the model to use.
"""

import tiktoken

TOOLBOX_LISTING_TOKEN_BUDGET = 1600
_enc = tiktoken.get_encoding("cl100k_base")


def test_listing_stays_within_budget(offline_interpreter):
    listing = offline_interpreter.toolbox.system_message
    tokens = len(_enc.encode(listing))
    print(f"\ntoolbox listing: {tokens} tokens")
    assert tokens <= TOOLBOX_LISTING_TOKEN_BUDGET


def test_entries_are_one_line_and_carry_no_return_shape(offline_interpreter):
    """Each entry is `signature # one sentence`, with no Returns: clause."""
    entries = [
        e
        for e in offline_interpreter.toolbox._get_all_toolbox_tools_signature_and_description()
        if not e.strip().startswith("#")
    ]

    assert entries, "the listing should not be empty"
    for entry in entries:
        assert "\n" not in entry.strip(), entry
        assert "Returns:" not in entry, entry
