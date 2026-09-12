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


# --- malformed calls where the provider never sent a tool_call_id -----------
#
# These three used to end the turn in total silence: parse_partial_json (or the
# empty/non-string code checks) produced an error string, but the branch that
# yielded it only fired when a tool_call_id was available, with no fallback.
# dispatch_function_call now mints a tool_call_id whenever the provider omits
# one, so the error goes out as a properly paired role:tool message and
# respond() — which only re-prompts when the trailing message has role ==
# "tool" — gives the model a second turn. Scripting and asserting that second
# turn is what proves the model reads the correction, not just the user.


def test_unparseable_execute_arguments_get_a_minted_id_and_a_second_turn(offline_interpreter):
    """Arguments so broken that parse_partial_json gives up still reach the model.

    Before the fix, this combination (bad JSON + no tool_call_id) fell through
    every branch in dispatch_function_call's "execute" arm without yielding
    anything, so the turn ended with no error and no way for the model to
    retry. Now a tool_call_id is minted, and the message distinguishes a
    syntax problem ("not valid JSON") from a shape problem, rather than
    leaking parse_partial_json's `None` failure sentinel as "got NoneType".
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    completions = ScriptedStreams(
        [
            _raw_tool_call_stream("execute", "not json at all {{{", call_id=None),
            _text_stream("Retrying with valid JSON."),
        ]
    )
    offline_interpreter.llm.completions = completions

    messages = offline_interpreter.chat("run something", display=False, stream=False)

    assert len(completions.calls) == 2, "the model was never given a second turn"
    assert messages[-1]["content"] == "Retrying with valid JSON."
    tool_errors = [m for m in offline_interpreter.messages if m.get("role") == "tool"]
    assert tool_errors, offline_interpreter.messages
    assert tool_errors[0]["tool_call_id"], "a tool response with no id would be rejected by a real provider"
    assert "not valid JSON" in tool_errors[0]["content"]


def test_empty_code_gets_a_minted_id_and_a_second_turn(offline_interpreter):
    """An empty `code` string with no provider-supplied id still reaches the model.

    Before the fix, the empty-code branch only yielded a message when a
    tool_call_id was present; with none, nothing was yielded at all.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    completions = ScriptedStreams(
        [
            _tool_call_stream("execute", {"language": "python", "code": ""}, call_id=None),
            _text_stream("Retrying with code."),
        ]
    )
    offline_interpreter.llm.completions = completions

    messages = offline_interpreter.chat("run something", display=False, stream=False)

    assert len(completions.calls) == 2, "the model was never given a second turn"
    assert messages[-1]["content"] == "Retrying with code."
    tool_errors = [m for m in offline_interpreter.messages if m.get("role") == "tool"]
    assert tool_errors, offline_interpreter.messages
    assert tool_errors[0]["tool_call_id"]
    assert "code is empty" in tool_errors[0]["content"]
    assert "non-empty" in tool_errors[0]["content"]


def test_non_string_code_gets_a_minted_id_and_a_second_turn(offline_interpreter):
    """`code` sent as a non-string with no provider-supplied id still reaches the model.

    Same missing-fallback bug as the empty-code case, on the adjacent branch.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    completions = ScriptedStreams(
        [
            _tool_call_stream("execute", {"language": "python", "code": 5}, call_id=None),
            _text_stream("Retrying with a string."),
        ]
    )
    offline_interpreter.llm.completions = completions

    messages = offline_interpreter.chat("run something", display=False, stream=False)

    assert len(completions.calls) == 2, "the model was never given a second turn"
    assert messages[-1]["content"] == "Retrying with a string."
    tool_errors = [m for m in offline_interpreter.messages if m.get("role") == "tool"]
    assert tool_errors, offline_interpreter.messages
    assert tool_errors[0]["tool_call_id"]
    assert "code must be a string" in tool_errors[0]["content"]
    assert "execute requires 'code' as a string" in tool_errors[0]["content"]


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
# --- what the model, and the user, are shown afterwards ---------------------


def test_the_corrective_turn_shows_the_call_the_model_really_made(offline_interpreter):
    """The second request pairs the model's own bad call with the real error.

    dispatch_function_call used to yield only the role:tool error. process_messages
    then had to invent an assistant message to satisfy provider pairing, and what it
    invented was execute(code="pass  # (synthetic; do not run)"), so the model read
    its own history as "I called execute with `pass`" immediately followed by "that
    call was invalid" — about a call it never made. Asserting on the outgoing request
    payload (not on interpreter.messages) is what pins the thing the model actually
    reads. Reproduces
    .superpowers/sdd/2026-09-12-remaining-defects/probe-malformed-call-pairing.py.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    completions = ScriptedStreams(
        [
            _raw_tool_call_stream("execute", "not json at all {{{", call_id=None),
            _text_stream("Retrying with valid JSON."),
        ]
    )
    offline_interpreter.llm.completions = completions

    offline_interpreter.chat("run something", display=False, stream=False)

    assert len(completions.calls) == 2, "the model was never given a second turn"
    outgoing = completions.calls[1]["messages"]
    assert "synthetic; do not run" not in json.dumps(outgoing, default=str)

    assistant_calls = [m for m in outgoing if m.get("tool_calls")]
    tool_responses = [m for m in outgoing if m.get("role") == "tool"]
    assert len(assistant_calls) == 1 and len(tool_responses) == 1, outgoing
    call = assistant_calls[0]["tool_calls"][0]
    assert call["function"]["name"] == "execute"
    assert call["function"]["arguments"] == "not json at all {{{"
    assert call["id"] == tool_responses[0]["tool_call_id"]
    assert "not valid JSON" in tool_responses[0]["content"]


def test_a_nameless_call_is_not_shown_to_the_model_as_an_execute_call(offline_interpreter):
    """A call with no function name is paired with a call carrying no name.

    The worst case of the invented pairing: the model was shown a tool call *named*
    execute and then told, in the very next message, that its function name was
    missing.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    completions = ScriptedStreams(
        [
            _nameless_tool_call_stream({"language": "python", "code": "print(1)"}),
            _text_stream("Understood."),
        ]
    )
    offline_interpreter.llm.completions = completions

    offline_interpreter.chat("run something", display=False, stream=False)

    outgoing = completions.calls[1]["messages"]
    assistant_calls = [m for m in outgoing if m.get("tool_calls")]
    assert len(assistant_calls) == 1, outgoing
    assert assistant_calls[0]["tool_calls"][0]["function"]["name"] == ""
def test_a_malformed_call_reaches_the_terminal_and_still_reaches_the_model(offline_interpreter, capsys):
    """The user sees one line about the bad call; the model still gets the tool response.

    message_stream keeps role:tool messages out of the display on purpose, and the
    assistant-text fallback that used to surface a malformed call was removed when
    every such call was routed through the tool-response path. That left the user
    with nothing at all: a silent pause and another turn, so a model looping on bad
    calls looked like a hang. Both halves are pinned here because fixing either one
    alone is how this broke — the notice must be printed, and the role:tool message
    the model reads must still be stored, untouched.
    """
    from interpreter.terminal_interface.terminal_interface import terminal_interface

    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.plain_text_display = True
    offline_interpreter.llm.completions = ScriptedStreams(
        [
            _raw_tool_call_stream("execute", "not json at all {{{", call_id=None),
            _text_stream("Retrying with valid JSON."),
        ]
    )

    list(terminal_interface(offline_interpreter, "run something"))

    printed = capsys.readouterr().out
    assert "malformed tool call" in printed
    assert "not valid JSON" in printed
    assert "Traceback" not in printed

    tool_messages = [m for m in offline_interpreter.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1, offline_interpreter.messages
    assert "not valid JSON" in tool_messages[0]["content"]
    assert tool_messages[0]["tool_call_id"]
    # The display-only notice must never become a message: the model would then
    # read two accounts of one failure, one of them looking like its own words.
    assert not any(m.get("type") == "notice" for m in offline_interpreter.messages)
