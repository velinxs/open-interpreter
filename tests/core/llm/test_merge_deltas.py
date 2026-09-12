"""Reassembling a streamed completion from its deltas.

merge_deltas is applied to every chunk of every response. Its failure mode is
silent corruption: a tool call whose arguments are concatenated in the wrong
order or across the wrong index produces JSON that either fails to parse or,
worse, parses into a different call than the model made.
"""

import pytest

from interpreter.core.llm.utils.merge_deltas import merge_deltas, normalize_delta_to_dict


class PydanticV2Like:
    """Stands in for a LiteLLM Delta object (some providers return these, not dicts)."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, exclude_unset=False):
        return dict(self._payload)


class PydanticV1Like:
    def __init__(self, payload):
        self._payload = payload

    def dict(self, exclude_unset=False):
        return dict(self._payload)


def test_string_fields_accumulate_across_chunks():
    """Text content is concatenated, not replaced.

    This is the whole point: each chunk carries a fragment. Replacing would
    leave only the last token of every reply.
    """
    message = {}
    for fragment in ["Hel", "lo ", "world"]:
        merge_deltas(message, {"content": fragment})
    assert message["content"] == "Hello world"


def test_a_null_field_does_not_wipe_what_was_accumulated():
    """Providers send explicit nulls for fields they are not updating.

    Treating null as a value would blank the content on the final chunk, which
    is exactly the chunk most providers send with everything null.
    """
    message = {"content": "so far"}
    merge_deltas(message, {"content": None, "role": None})
    assert message["content"] == "so far"


def test_a_null_accumulator_is_treated_as_empty():
    """Appending to a field whose current value is None yields the new text.

    Some providers open the stream with {"content": null}; without the `or ""`
    the next chunk would raise TypeError mid-response.
    """
    message = {"content": None}
    merge_deltas(message, {"content": "text"})
    assert message["content"] == "text"


def test_nested_dicts_merge_rather_than_replace():
    """A nested object is merged key by key, so partial updates accumulate.

    function_call arrives as {"name": ...} then {"arguments": ...}. Replacing
    would drop the name.
    """
    message = {}
    merge_deltas(message, {"function_call": {"name": "run"}})
    merge_deltas(message, {"function_call": {"arguments": '{"a":'}})
    merge_deltas(message, {"function_call": {"arguments": " 1}"}})
    assert message["function_call"] == {"name": "run", "arguments": '{"a": 1}'}


def test_pydantic_deltas_are_converted_before_merging():
    """LiteLLM returns Pydantic Delta objects for some models; both versions work.

    Without the conversion the merge iterates attributes of an object instead
    of a mapping and produces an empty message — the symptom is a model that
    streams nothing at all.
    """
    message = {}
    merge_deltas(message, PydanticV2Like({"content": "from v2"}))
    assert message["content"] == "from v2"

    message = {}
    merge_deltas(message, PydanticV1Like({"content": "from v1"}))
    assert message["content"] == "from v1"


def test_an_unconvertible_delta_becomes_an_empty_dict():
    """A delta that is neither a mapping nor a Pydantic model is dropped, not fatal.

    One odd chunk from a provider should not end the response.
    """
    assert normalize_delta_to_dict(object()) == {}


def test_nested_pydantic_values_are_merged_as_dicts():
    """A Pydantic value inside a dict delta is converted before being merged.

    Providers mix the two representations within one chunk.
    """
    message = {"function_call": {"name": "run"}}
    merge_deltas(message, {"function_call": PydanticV2Like({"arguments": "{}"})})
    assert message["function_call"] == {"name": "run", "arguments": "{}"}


# --- tool calls -------------------------------------------------------------


def test_tool_call_arguments_accumulate_within_one_index():
    """Argument fragments for the same tool call are concatenated in order.

    Providers stream tool arguments a few characters at a time. Any other
    combination yields JSON the dispatcher cannot parse, and the tool call is
    lost.
    """
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "run", "arguments": ""}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"arguments": '{"code"'}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"arguments": ': "ls"}'}}]})

    call = message["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "run"
    assert call["function"]["arguments"] == '{"code": "ls"}'


def test_parallel_tool_calls_stay_separate():
    """Two indices are two calls; their arguments never bleed into each other.

    Parallel tool calls interleave in the stream. Appending by position
    instead of by index would concatenate one call's arguments onto another's
    and produce a single malformed call.
    """
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "first", "arguments": "{"}}]})
    merge_deltas(message, {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "second", "arguments": "["}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]})
    merge_deltas(message, {"tool_calls": [{"index": 1, "function": {"arguments": "]"}}]})

    by_index = {call["index"]: call for call in message["tool_calls"]}
    assert by_index[0]["function"] == {"name": "first", "arguments": "{}"}
    assert by_index[1]["function"] == {"name": "second", "arguments": "[]"}


def test_a_tool_call_id_is_set_once_and_not_appended():
    """The id arrives on the first fragment only and must not accumulate.

    Concatenating it would produce "call_1call_1", which no provider will
    accept back in the tool result message.
    """
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "call_1", "type": "function"}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "call_1", "type": "function"}]})
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["type"] == "function"


def test_a_tool_call_name_is_kept_from_the_first_fragment():
    """The function name is not re-appended when a later chunk repeats it."""
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"name": "run"}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"name": "run", "arguments": "{}"}}]})
    assert message["tool_calls"][0]["function"]["name"] == "run"


def test_a_missing_index_defaults_to_the_first_call():
    """A tool-call fragment with no index is treated as index 0.

    Some providers omit it for single calls. Creating a separate entry would
    split one call into two half-formed ones.
    """
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]})
    merge_deltas(message, {"tool_calls": [{"function": {"arguments": "}"}}]})
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["arguments"] == "{}"


def test_non_tool_call_lists_are_extended():
    """Ordinary list fields accumulate by extension.

    Only tool_calls needs index-aware merging; anything else (e.g. reasoning
    fragments) is a plain sequence.
    """
    message = {}
    merge_deltas(message, {"parts": ["a"]})
    merge_deltas(message, {"parts": ["b", "c"]})
    assert message["parts"] == ["a", "b", "c"]


def test_the_original_dict_is_mutated_and_returned():
    """merge_deltas both mutates and returns; callers rely on the mutation.

    The streaming loop keeps one message dict and merges into it. Returning a
    copy would leave it empty.
    """
    message = {}
    result = merge_deltas(message, {"content": "x"})
    assert result is message
