"""Pulling token usage out of a stream, and rendering it for %usage.

Usage arrives attached to one arbitrary chunk, in whatever shape the provider
chose — a dict, a Pydantic model, or something only str() understands. If it
is missed, %usage reports nothing; if a reported zero is rendered the same as
an absent field, the user cannot tell "no cache hits" from "this API does not
report cache hits".
"""

import json

import pytest

from interpreter.core.llm.utils.stream_usage import (
    format_last_usage_markdown,
    record_stream_chunk_usage,
)


class Llm:
    def __init__(self):
        self.last_completion_usage = None


class UsageModel:
    """Stands in for a provider's Pydantic usage object."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


class Chunk:
    def __init__(self, usage=None):
        self.usage = usage


def test_usage_on_a_dict_chunk_is_recorded():
    """A mapping chunk with a usage key is stored on the llm.

    The final chunk of an OpenAI stream carries usage this way when
    stream_options.include_usage is set; missing it makes %usage permanently
    empty.
    """
    llm = Llm()
    record_stream_chunk_usage(llm, {"usage": {"total_tokens": 42}})
    assert llm.last_completion_usage == {"total_tokens": 42}


def test_usage_on_an_object_chunk_is_recorded():
    """LiteLLM's chunk objects expose usage as an attribute, not a key."""
    llm = Llm()
    record_stream_chunk_usage(llm, Chunk(UsageModel({"total_tokens": 7})))
    assert llm.last_completion_usage == {"total_tokens": 7}


@pytest.mark.parametrize("chunk", [None, {}, {"usage": None}, Chunk(None), {"usage": {}}])
def test_chunks_without_usage_leave_the_previous_value_alone(chunk):
    """Every chunk goes through here, so the no-usage case must not clear anything.

    Usage is attached to one chunk out of hundreds. Clearing on the others
    would wipe it again immediately after it was recorded.
    """
    llm = Llm()
    llm.last_completion_usage = {"total_tokens": 1}
    record_stream_chunk_usage(llm, chunk)
    assert llm.last_completion_usage == {"total_tokens": 1}


def test_a_usage_object_that_cannot_be_converted_is_stringified_not_dropped():
    """An unrecognised usage shape still reaches the user as text.

    Reporting nothing at all would look like the provider sent no usage, when
    in fact it sent something this code did not understand.
    """
    llm = Llm()
    record_stream_chunk_usage(llm, {"usage": {"weird": object()}})
    assert isinstance(llm.last_completion_usage["weird"], str)


def test_nested_usage_models_are_flattened_to_plain_data():
    """Sub-objects are converted too, so the result is JSON-serialisable.

    %usage dumps leftover fields with json.dumps; a surviving Pydantic object
    would raise there instead of printing.
    """
    llm = Llm()
    record_stream_chunk_usage(llm, {"usage": UsageModel({"details": UsageModel({"cached": 5})})})
    json.dumps(llm.last_completion_usage)
    assert llm.last_completion_usage == {"details": {"cached": 5}}


# --- rendering --------------------------------------------------------------


def test_the_three_headline_numbers_are_shown():
    """Prompt, completion, and total tokens each get their own line."""
    rendered = format_last_usage_markdown(
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    )
    assert "**Prompt tokens:** 100" in rendered
    assert "**Completion tokens:** 20" in rendered
    assert "**Total tokens:** 120" in rendered


def test_a_reported_zero_is_not_shown_as_unreported():
    """0 and null mean different things and must render differently.

    A cache-hit count of 0 means the call missed the cache; a null means the
    API does not report cache hits at all. Collapsing them makes prompt
    caching impossible to debug.
    """
    rendered = format_last_usage_markdown(
        {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": None}}
    )
    assert "**Prompt — Cached Tokens:** 0" in rendered
    assert "**Prompt — Audio Tokens:** *not reported*" in rendered


def test_completion_detail_fields_are_labelled_as_completion():
    """Prompt and completion sub-fields carry different prefixes.

    Both dicts often contain the same key names; without the prefix the
    reader cannot tell which half of the call a number belongs to.
    """
    rendered = format_last_usage_markdown(
        {"completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 3}}
    )
    assert "**Completion — Reasoning Tokens:** 3" in rendered


def test_unknown_fields_are_dumped_rather_than_discarded():
    """Provider-specific usage keys are shown as JSON instead of being swallowed.

    Cost, cache-write tokens, and provider quotas all arrive under names this
    code has never heard of, and they are often the number the user wants.
    """
    rendered = format_last_usage_markdown({"total_tokens": 1, "cost_usd": 0.0004})
    assert "Other usage fields" in rendered
    assert "cost_usd" in rendered


def test_a_usage_object_with_nothing_recognisable_is_dumped_whole():
    """An unfamiliar usage shape still prints something the user can read.

    The fallback is what keeps %usage useful against a provider whose field
    names do not match OpenAI's at all.
    """
    rendered = format_last_usage_markdown({"input": 5, "output": 2})
    assert "```json" in rendered
    assert '"input": 5' in rendered


def test_the_header_explains_that_tool_turns_show_only_the_last_call():
    """The output says it is one HTTP call, not a per-reply total.

    A reply that ran three tools made four model calls. Reading the number as
    a total would understate spend by a factor of four, silently.
    """
    rendered = format_last_usage_markdown({"total_tokens": 1})
    assert "more than once" in rendered
    assert "latest" in rendered
