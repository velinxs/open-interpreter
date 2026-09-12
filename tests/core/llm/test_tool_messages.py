"""process_messages: shaping the raw LMC-ish message list into paired tool_calls/tool messages.

No test module covered this function before; the two tests below were added
for a specific regression (a minted tool_call_id colliding with a later,
freshly-generated one) rather than as a general survey of process_messages.
"""

from interpreter.core.llm.tool_messages import process_messages


def _function_call(code):
    return {
        "role": "assistant",
        "content": None,
        "function_call": {"name": "execute", "arguments": f'{{"language":"python","code":"{code}"}}'},
    }


def test_a_minted_id_surviving_into_history_does_not_collide_with_a_later_sequence_number():
    """An id minted for a malformed call on an earlier turn is not reissued to a later real call.

    Reproduces .superpowers/sdd/2026-09-12-remaining-defects/probe-id-collision.py:
    dispatch_function_call minted "toolu_2" for a malformed execute call on turn
    1 (because "toolu_1" was already taken that turn) and it stays in history as
    a role:tool message's tool_call_id forever after. process_messages used to
    recompute its toolu_N sequence from scratch on every request, with no idea
    that number was already spoken for, so the second turn's real function_call
    was assigned "toolu_2" again. Two assistant tool_calls sharing an id is
    rejected outright by any pairing-strict provider — asserting on the set of
    ids (not specific numbers) is what keeps this test valid if the id format
    ever changes.
    """
    messages = [
        {"role": "user", "content": "one"},
        _function_call("print(1)"),
        {"role": "function", "name": "execute", "content": "1"},
        {"role": "tool", "tool_call_id": "toolu_2", "content": "execute requires 'language' and 'code'."},
        {"role": "user", "content": "two"},
        _function_call("print(2)"),
        {"role": "function", "name": "execute", "content": "2"},
    ]

    out = process_messages(messages, model="gpt-4o")

    assistant_ids = [call["id"] for message in out for call in (message.get("tool_calls") or [])]
    assert len(assistant_ids) == len(set(assistant_ids)), assistant_ids


def test_consecutive_id_less_tool_messages_get_different_synthetic_ids():
    """Two orphaned tool responses in a row do not share a synthesized assistant id.

    process_messages inserts a synthetic assistant tool_calls message before an
    orphaned tool response (one with no preceding assistant+tool_calls) so
    pairing-strict providers see assistant(tool_calls) -> tool(response). The id
    for that synthetic message used to be computed as
    `generate_tool_id(last_tool_id + 1, model)` without ever advancing
    last_tool_id, so two orphaned tool messages back to back produced two
    different assistant messages sharing the exact same id — the same defect
    family as the minted-id collision above, predating this task.
    """
    messages = [
        {"role": "tool", "content": "first orphaned response, no id"},
        {"role": "tool", "content": "second orphaned response, no id"},
    ]

    out = process_messages(messages, model="gpt-4o")

    synthetic_ids = [message["tool_calls"][0]["id"] for message in out if message.get("role") == "assistant"]
    assert len(synthetic_ids) == 2
    assert synthetic_ids[0] != synthetic_ids[1]
