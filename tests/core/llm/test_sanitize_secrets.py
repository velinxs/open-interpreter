"""Which messages get scanned for secrets before they are sent to a remote model.

Redaction is deliberately narrow: only the output of code the interpreter ran,
because that is where a printed environment variable or a cat'd credentials
file turns up. Scanning everything would rewrite the user's own words whenever
a detector matched a benign phrase. The selection rule below is therefore the
whole feature — a message it misses is a secret that leaves the machine.
"""

from interpreter.core.llm.utils.sanitize_secrets import sanitize_messages


def _scanner(text):
    """Stand-in for the detect-secrets pass, so these tests pin selection, not detection."""
    return text.replace("hunter2", "[REDACTED]")


def test_a_tool_result_is_scanned():
    """Code output arriving as a `tool` message is sanitized.

    Tool-calling models receive execution output as role "tool"; only the
    legacy `function`/`execute` shape used to be covered, so on that path
    every secret in command output went out verbatim.
    """
    messages = [{"role": "tool", "tool_call_id": "call_1", "content": "API_KEY=hunter2"}]
    sanitize_messages(messages, scanner=_scanner)
    assert messages[0]["content"] == "API_KEY=[REDACTED]"


def test_a_legacy_execute_result_is_still_scanned():
    """The old `function`/`execute` message shape keeps working.

    Saved conversations and the non-tool-calling path still produce it, so
    dropping it while adding the tool role would trade one blind spot for
    another.
    """
    messages = [{"role": "function", "name": "execute", "content": "API_KEY=hunter2"}]
    sanitize_messages(messages, scanner=_scanner)
    assert messages[0]["content"] == "API_KEY=[REDACTED]"


def test_the_conversation_around_the_output_is_left_alone():
    """System, user and assistant text are never rewritten by default.

    The detectors match on shape, not meaning. Editing the user's own
    question or the system prompt because a word looked like a credential
    would change what the model is being asked.
    """
    messages = [
        {"role": "system", "content": "hunter2"},
        {"role": "user", "content": "hunter2"},
        {"role": "assistant", "content": "hunter2"},
        {"role": "function", "name": "other_tool", "content": "hunter2"},
    ]
    sanitize_messages(messages, scanner=_scanner)
    assert [m["content"] for m in messages] == ["hunter2"] * 4


def test_structured_tool_content_is_scanned_part_by_part():
    """Text parts inside a list-shaped content are redacted; other parts are untouched.

    Tool results can carry text alongside an image. Handing the list to the
    scanner as a whole would either skip it or corrupt the image payload.
    """
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,hunter2"}}
    messages = [{"role": "tool", "content": [{"type": "text", "text": "API_KEY=hunter2"}, image]}]
    sanitize_messages(messages, scanner=_scanner)
    assert messages[0]["content"][0]["text"] == "API_KEY=[REDACTED]"
    assert messages[0]["content"][1] is image


def test_everything_is_scanned_when_the_narrow_filter_is_switched_off():
    """only_code_output=False sanitizes every message, whatever its role."""
    messages = [{"role": "user", "content": "hunter2"}]
    sanitize_messages(messages, scanner=_scanner, only_code_output=False)
    assert messages[0]["content"] == "[REDACTED]"
