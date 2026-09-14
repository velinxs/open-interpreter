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
    # "{}", not the raw text: litellm's Ollama transform json.loads() this field,
    # so malformed JSON here raises on every later request. Not the old
    # {"_unparsed_arguments": ...} wrapper either — this is the model's own
    # assistant slot and it imitates what it finds there. The text it sent
    # reaches it through the paired error instead.
    arguments = call["function"]["arguments"]
    assert json.loads(arguments) == {}
    assert call["id"] == tool_responses[0]["tool_call_id"]
    assert "not valid JSON" in tool_responses[0]["content"]
    assert "not json at all {{{" in tool_responses[0]["content"]


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


def test_concatenated_argument_objects_are_named_as_the_mistake(offline_interpreter):
    """Two calls crammed into one arguments string get a message naming that.

    A model that wants to run several things at once sometimes concatenates one
    arguments object after another. Being told only "arguments were not valid
    JSON" gave it nothing to act on, and it repeated the same shape for several
    turns before guessing its way out.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    raw = '{"language": "bash", "code": "pwd"}{"language": "python", "code": "print(1)"}'
    completions = ScriptedStreams(
        [_raw_tool_call_stream("execute", raw), _text_stream("Sending one call.")]
    )
    offline_interpreter.llm.completions = completions

    offline_interpreter.chat("run some checks", display=False, stream=False)

    tool_messages = [m for m in offline_interpreter.messages if m.get("role") == "tool"]
    assert tool_messages, "the model was never told anything"
    content = tool_messages[-1]["content"]
    assert "concatenated" in content, content
    assert "one call per turn" in content, content
    assert '"language": "bash"' in content, "the first call must be handed back to resend"


def test_the_unparsed_arguments_wrapper_is_not_read_back_as_fields(offline_interpreter):
    """A model imitating the wrapper from an old conversation is told what is really wrong.

    Malformed arguments used to be recorded as {"_unparsed_arguments": "..."} so
    the outgoing request still parsed. Models copy what they see, so that came
    back as a real call and the reply was "missing required fields, got
    ['_unparsed_arguments']" — a key the model cannot do anything about. Nothing
    writes the wrapper any more, but conversations saved while it did still carry
    it, so it must still be recognised and answered with something usable. The
    key itself stays out of the reply: naming it would be one more place the
    model could read it as a shape worth sending.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    wrapped = json.dumps({"_unparsed_arguments": "not json at all {{{"})
    completions = ScriptedStreams(
        [_raw_tool_call_stream("execute", wrapped), _text_stream("Understood.")]
    )
    offline_interpreter.llm.completions = completions

    offline_interpreter.chat("go", display=False, stream=False)

    tool_messages = [m for m in offline_interpreter.messages if m.get("role") == "tool"]
    assert tool_messages, "the model was never told anything"
    content = tool_messages[-1]["content"]
    assert "_unparsed_arguments" not in content, f"the wrapper leaked to the model: {content}"
    assert "nested inside another object" in content, content
    assert "not json at all {{{" in content, content


def test_a_wrapper_around_valid_arguments_is_corrected_not_executed(offline_interpreter):
    """A wrapped-but-valid payload is refused, so the wrapper never looks like it works.

    This was the reinforcement loop. The wrapper was unwrapped before the
    arguments were validated, so a model that copied it round a payload that
    happened to be fine had its code run with no complaint — the shape was
    rewarded. It only failed later, when it wrapped something broken, and by then
    the habit was established. Refusing it costs one turn and corrects the shape.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    wrapped = json.dumps({"_unparsed_arguments": '{"language": "python", "code": "print(7*7)"}'})
    completions = ScriptedStreams(
        [_raw_tool_call_stream("execute", wrapped), _text_stream("Resending directly.")]
    )
    offline_interpreter.llm.completions = completions

    offline_interpreter.chat("go", display=False, stream=False)

    assert not any(
        m.get("type") == "console" and "49" in str(m.get("content")) for m in offline_interpreter.messages
    ), "the wrapped call ran"
    tool_messages = [m for m in offline_interpreter.messages if m.get("role") == "tool"]
    assert tool_messages, "the model was never told anything"
    assert "nested inside another object" in tool_messages[-1]["content"]


def test_a_malformed_call_never_puts_a_shape_to_copy_in_the_assistant_slot(offline_interpreter):
    """Repeated bad calls leave nothing imitable in the model's own history.

    The loop the fork's owner hit: every failure was written back into the
    assistant slot as {"_unparsed_arguments": "..."}, so by the eighth request
    the model's recent history was seven wrapper-shaped calls of its own and it
    kept sending more. What the model reads as its own prior output must stay a
    shape we are happy for it to repeat.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    completions = ScriptedStreams(
        [_raw_tool_call_stream("execute", "not json at all {{{", call_id=f"c{i}") for i in range(3)]
        + [_text_stream("Giving up on that shape.")]
    )
    offline_interpreter.llm.completions = completions

    offline_interpreter.chat("go", display=False, stream=False)

    assert len(completions.calls) > 1, "the model was never given a second turn"
    for request in completions.calls[1:]:
        for message in request["messages"]:
            for call in message.get("tool_calls") or []:
                assert "_unparsed_arguments" not in call["function"]["arguments"], request["messages"]


# --- more than one tool call in a turn --------------------------------------


def _two_tool_call_stream(first, second, stamp_index=True):
    """Two finished tool calls, in two chunks.

    With stamp_index False the calls carry no index at all, which is what
    litellm's native Ollama path passes through; Delta.__init__ then stamps 0 on
    both, one per chunk.
    """
    from litellm.types.utils import Delta

    for i, (call_id, arguments) in enumerate((("call_a", first), ("call_b", second))):
        call = {
            "id": call_id,
            "type": "function",
            "function": {"name": "execute", "arguments": json.dumps(arguments)},
        }
        if stamp_index:
            call["index"] = i
            yield {"choices": [{"delta": {"tool_calls": [call]}}]}
        else:
            yield {"choices": [{"delta": Delta(tool_calls=[call])}]}
    yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}


def test_index_less_parallel_calls_are_not_blamed_on_the_model(offline_interpreter):
    """Two calls a provider sent without an index still run the first one normally.

    litellm stamps index 0 on every index-less tool call, once per chunk, so a
    provider that streams two finished calls in two chunks hands both of them
    index 0. Merging on index alone concatenated the second call's arguments onto
    the first, and the model was then told it had crammed "2 argument objects"
    into one string — for a malformed call it had not made and could not correct.
    Reproduces the loop the fork's owner hit across five turns.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    completions = ScriptedStreams(
        [
            _two_tool_call_stream(
                {"language": "python", "code": "print(6 * 7)"},
                {"language": "python", "code": "print('second')"},
                stamp_index=False,
            ),
            _text_stream("Done."),
        ]
    )
    offline_interpreter.llm.completions = completions

    offline_interpreter.chat("go", display=False, stream=False)

    tool_messages = [m for m in offline_interpreter.messages if m.get("role") == "tool"]
    assert not any("concatenated" in m["content"] for m in tool_messages), tool_messages
    assert any(
        m.get("type") == "console" and "42" in str(m.get("content")) for m in offline_interpreter.messages
    ), "the first call did not run"


def test_a_second_tool_call_is_answered_rather_than_dropped(offline_interpreter):
    """The call that cannot run this turn is returned to the model, not discarded.

    A turn runs one block: respond() executes interpreter.messages[-1] and
    message_stream folds consecutive code chunks into one message. The extra call
    used to vanish where the stream was converted — the model asked for two
    things, got one, and read a history in which the second never existed. It
    then either assumed it had run or sent it again, which is how the loop
    restarted.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    completions = ScriptedStreams(
        [
            _two_tool_call_stream(
                {"language": "python", "code": "print('FIRST')"},
                {"language": "python", "code": "print('SECOND')"},
            ),
            _text_stream("Sending the second one now."),
        ]
    )
    offline_interpreter.llm.completions = completions

    offline_interpreter.chat("go", display=False, stream=False)

    assert any(
        m.get("type") == "console" and "FIRST" in str(m.get("content")) for m in offline_interpreter.messages
    ), "the first call did not run"
    assert not any(
        m.get("type") == "console" and "SECOND" in str(m.get("content")) for m in offline_interpreter.messages
    ), "two blocks ran in one turn"

    unrun = [m for m in offline_interpreter.messages if m.get("role") == "tool" and "Not run" in m["content"]]
    assert len(unrun) == 1, offline_interpreter.messages
    assert unrun[0]["tool_call_id"] == "call_b"
    # The unrun call is recorded too, or the tool response has no assistant call
    # to pair with and process_messages invents one.
    assert any(
        m.get("type") == "tool_call" and m.get("tool_call_id") == "call_b" for m in offline_interpreter.messages
    ), offline_interpreter.messages


def test_the_unrun_call_is_paired_and_the_first_call_still_runs_last(offline_interpreter):
    """The outgoing history pairs the unrun call, and the code block is stored last.

    Order is load-bearing: respond() only runs code when it is the trailing
    message, so the unrun-call chunks have to be yielded before the code chunk.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    completions = ScriptedStreams(
        [
            _two_tool_call_stream(
                {"language": "python", "code": "print('FIRST')"},
                {"language": "python", "code": "print('SECOND')"},
            ),
            _text_stream("Understood."),
        ]
    )
    offline_interpreter.llm.completions = completions

    offline_interpreter.chat("go", display=False, stream=False)

    outgoing = completions.calls[1]["messages"]
    answered = {m["tool_call_id"] for m in outgoing if m.get("role") == "tool"}
    called = {c["id"] for m in outgoing for c in (m.get("tool_calls") or [])}
    assert answered <= called, (answered, called)
    assert "call_b" in answered


def test_one_call_per_turn_is_declared_to_the_provider(offline_interpreter):
    """The request says parallel_tool_calls: false, so the rule is not a secret.

    The constraint was enforced (extra calls dropped) but never stated: nothing
    in the request or the tool schema said a turn takes one call. A model told
    off for batching had no way to learn the rule, so it batched again.
    """
    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    completions = ScriptedStreams([_text_stream("Nothing to run.")])
    offline_interpreter.llm.completions = completions

    offline_interpreter.chat("hello", display=False, stream=False)

    assert completions.calls[0]["parallel_tool_calls"] is False


def test_the_user_is_told_when_a_call_did_not_run(offline_interpreter, capsys):
    """One terminal line explains why only one of two requested actions happened.

    role:tool messages are never displayed, so without the notice the user
    watches the model ask for two things and silently get one.
    """
    from interpreter.terminal_interface.terminal_interface import terminal_interface

    install_fake_llm(offline_interpreter, [])
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.plain_text_display = True
    offline_interpreter.llm.completions = ScriptedStreams(
        [
            _two_tool_call_stream(
                {"language": "python", "code": "print('FIRST')"},
                {"language": "python", "code": "print('SECOND')"},
            ),
            _text_stream("Understood."),
        ]
    )

    list(terminal_interface(offline_interpreter, "go"))

    printed = capsys.readouterr().out
    assert "only the first ran" in printed, printed
