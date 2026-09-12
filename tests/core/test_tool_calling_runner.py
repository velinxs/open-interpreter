"""Tool-calling mode: OpenAI-style tool_call deltas become code, edit, and view_image chunks."""

import json

from tests.support.fake_llm import install_fake_llm


def _tool_call_stream(name, arguments, call_id="call_1"):
    payload = json.dumps(arguments)
    head, tail = payload[: len(payload) // 2], payload[len(payload) // 2 :]
    yield {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": head},
                        }
                    ]
                }
            }
        ]
    }
    yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": tail}}]}}]}
    yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}


def _text_stream(text):
    yield {"choices": [{"delta": {"content": text}}]}
    yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def _raw_tool_call_stream(name, raw_arguments, call_id="call_1"):
    """Like _tool_call_stream but skips json.dumps, so `raw_arguments` can be
    text that is not valid JSON at all (parse_partial_json cannot repair it).
    """
    head, tail = raw_arguments[: len(raw_arguments) // 2], raw_arguments[len(raw_arguments) // 2 :]
    yield {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": head},
                        }
                    ]
                }
            }
        ]
    }
    yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": tail}}]}}]}
    yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}


def _nameless_tool_call_stream(arguments, call_id="call_1"):
    """A tool_calls delta whose function carries arguments but never a name.

    Simulates a provider (or model) that drops the function name — the last
    arm of dispatch_function_call's if/elif chain is keyed on function_name,
    so this is the one case that used to fall off the end of every branch.
    """
    payload = json.dumps(arguments)
    yield {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {"arguments": payload},
                        }
                    ]
                }
            }
        ]
    }
    yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}


class ScriptedStreams:
    def __init__(self, streams):
        self.streams = list(streams)
        self.calls = []

    def __call__(self, **params):
        self.calls.append(params)
        return self.streams.pop(0)


def test_execute_tool_call_runs_code(offline_interpreter):
    """An execute(language, code) tool call is executed and its output fed back."""
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.llm.completions = ScriptedStreams(
        [
            _tool_call_stream("execute", {"language": "python", "code": "print(6 * 7)"}),
            _text_stream("Done: 42."),
        ]
    )

    messages = offline_interpreter.chat("what is 6*7", display=False, stream=False)

    assert any(m.get("type") == "console" and "42" in str(m.get("content")) for m in messages)
    assert messages[-1]["content"] == "Done: 42."


def test_unknown_tool_call_is_answered_with_a_tool_error(offline_interpreter):
    """A function the runner does not support is reported back as a tool response, not raised."""
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.llm.completions = ScriptedStreams(
        [_tool_call_stream("toolbox.web.search", {"query": "x"}), _text_stream("Understood.")]
    )

    messages = offline_interpreter.chat("search", display=False, stream=False)

    assert messages[-1]["content"] == "Understood."
    assert any(m.get("role") == "tool" for m in offline_interpreter.messages)


# --- malformed calls with no tool_call_id to pair a response to -------------
#
# These three used to end the turn in total silence: parse_partial_json (or the
# empty/non-string code checks) produced an error string, but the branch that
# yields it only fired when a tool_call_id was available, and had no fallback.
# With no id, respond() cannot safely ask the model again — a follow-up request
# with no matching tool response would be rejected by any provider enforcing
# the assistant(tool_calls) -> tool() pairing — so the turn ends after one
# assistant message. The fix is that this message now exists at all; before it,
# the turn ended with nothing yielded and nothing said to anyone.


def test_unparseable_execute_arguments_still_reach_someone(offline_interpreter):
    """Arguments so broken that parse_partial_json gives up no longer vanish silently.

    Before the fix, this combination (bad JSON + no tool_call_id) fell through
    every branch in dispatch_function_call's "execute" arm without yielding
    anything, so the turn ended with no error and no way for the model to
    retry.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.llm.completions = ScriptedStreams(
        [_raw_tool_call_stream("execute", "not json at all {{{", call_id=None)]
    )

    messages = offline_interpreter.chat("run something", display=False, stream=False)

    assert messages[-1]["role"] == "assistant"
    assert "arguments must be a dict" in messages[-1]["content"]


def test_empty_code_with_no_tool_call_id_still_reaches_someone(offline_interpreter):
    """An empty `code` string with no id to pair no longer ends the turn in silence.

    Before the fix, the empty-code branch only yielded a message when a
    tool_call_id was present; with none, nothing was yielded at all.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.llm.completions = ScriptedStreams(
        [_tool_call_stream("execute", {"language": "python", "code": ""}, call_id=None)]
    )

    messages = offline_interpreter.chat("run something", display=False, stream=False)

    assert messages[-1]["role"] == "assistant"
    assert "code is empty" in messages[-1]["content"]
    assert "non-empty" in messages[-1]["content"]


def test_non_string_code_with_no_tool_call_id_still_reaches_someone(offline_interpreter):
    """`code` sent as a non-string with no id to pair no longer ends the turn in silence.

    Same missing-fallback bug as the empty-code case, on the adjacent branch.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.llm.completions = ScriptedStreams(
        [_tool_call_stream("execute", {"language": "python", "code": 5}, call_id=None)]
    )

    messages = offline_interpreter.chat("run something", display=False, stream=False)

    assert messages[-1]["role"] == "assistant"
    assert "code must be a string" in messages[-1]["content"]


def test_a_tool_call_with_no_function_name_is_answered_not_dropped(offline_interpreter):
    """A tool_calls delta whose function never carries a name gets a corrective response.

    dispatch_function_call's chain is a series of `elif function_name ==
    ...:` arms ending in `elif function_name:` — a falsy name fell off the
    end of all of them, so the model's turn ended with nothing said and
    no correction possible.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.llm.completions = ScriptedStreams(
        [
            _nameless_tool_call_stream({"language": "python", "code": "print(1)"}),
            _text_stream("Understood."),
        ]
    )

    messages = offline_interpreter.chat("run something", display=False, stream=False)

    assert messages[-1]["content"] == "Understood."
    tool_errors = [m for m in offline_interpreter.messages if m.get("role") == "tool"]
    assert tool_errors, offline_interpreter.messages
    assert "function name is missing" in tool_errors[0]["content"]
    assert tool_errors[0]["tool_call_id"] == "call_1"
