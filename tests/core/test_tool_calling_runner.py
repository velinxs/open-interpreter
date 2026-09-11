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
