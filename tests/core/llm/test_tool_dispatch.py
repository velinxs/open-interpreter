"""Turning a finished tool call into chunks — including every way the model gets it wrong.

The point of this module is that a malformed tool call comes back as a tool
*response* the model can read and correct, never as an exception that kills
the turn. The tool_call_id matters as much as the message: the chat APIs
require assistant(tool_calls) -> tool(response) ordering, and a response
without an id cannot be matched to its call, so the next request is rejected
outright.
"""

from types import SimpleNamespace

import pytest

from interpreter.core.llm.tool_dispatch import dispatch_function_call
from interpreter.core.tools.file_edit import EDIT_LANGUAGES


@pytest.fixture
def llm():
    return SimpleNamespace(interpreter=SimpleNamespace(_view_image_approval="n"))


def _dispatch(llm, name, arguments, tool_call_id="call_1", request_params=None, language=None, verbose=False):
    return list(
        dispatch_function_call(
            llm,
            {"function_call": {"name": name, "arguments": arguments}},
            request_params or {"messages": []},
            tool_call_id,
            verbose,
            language,
        )
    )


def _tool_response(chunks):
    """The one role=tool chunk in a malformed call's output.

    A malformed call yields three chunks now — the assistant tool_call record
    the model really sent, the tool response it reads, and a display-only
    notice for the user — so tests that care about the message the model reads
    ask for it by role instead of by position.
    """
    responses = [chunk for chunk in chunks if chunk.get("role") == "tool"]
    assert len(responses) == 1, chunks
    return responses[0]


def test_no_pending_call_yields_nothing(llm):
    """A reply with no function call produces no chunks.

    This runs after every stream, tool call or not.
    """
    assert list(dispatch_function_call(llm, {}, {"messages": []}, None, False, None)) == []


# --- execute ----------------------------------------------------------------


def test_a_valid_execute_call_becomes_a_code_chunk(llm):
    """execute with a language and code yields one assistant code chunk.

    That chunk is what respond() runs. Anything else in its place means the
    model asked to run code and nothing ran.
    """
    chunks = _dispatch(llm, "execute", '{"language": "python", "code": "print(1)"}')
    assert chunks == [{"role": "assistant", "type": "code", "format": "python", "content": "print(1)"}]


def test_the_streamed_language_wins_over_the_arguments(llm):
    """A language already known from the stream is not overwritten by the payload.

    The caller passes `language` when it was set during streaming; re-reading
    it from partially-parsed JSON risks a truncated value.
    """
    chunks = _dispatch(llm, "execute", '{"language": "bash", "code": "ls"}', language="python")
    assert chunks[0]["format"] == "python"


def test_a_dict_of_arguments_is_accepted_as_well_as_a_json_string(llm):
    """Arguments already parsed into a dict skip the JSON step.

    Some providers hand back structured arguments; requiring a string would
    reject every call from them.
    """
    chunks = _dispatch(llm, "execute", {"language": "python", "code": "print(1)"})
    assert chunks[0]["content"] == "print(1)"


@pytest.mark.parametrize(
    "arguments,expected_fragments",
    [
        ('{"language": "python"}', ["missing required fields", "execute requires 'language' and 'code'"]),
        ('{"language": "python", "code": ""}', ["code is empty", "execute requires a non-empty 'code' string"]),
        ('{"language": "python", "code": 5}', ["code must be a string", "execute requires 'code' as a string"]),
        ('"not an object"', ["arguments must be a dict"]),
        ("not json at all {{{", ["not valid JSON", "execute requires a JSON object with 'language' and 'code'"]),
    ],
)
def test_malformed_execute_calls_come_back_as_tool_responses(llm, arguments, expected_fragments):
    """Every bad execute payload becomes a role=tool message the model can read.

    Raising instead would end the turn with a traceback and no way for the
    model to correct itself; staying silent would leave the API waiting for a
    tool response that never arrives, which fails the *next* request. Pinning
    every fragment (not just one) is what would catch a regression that undoes
    the "say what is required, not just what failed" rewrite of these messages.
    """
    chunks = _dispatch(llm, "execute", arguments)
    response = _tool_response(chunks)
    assert response["tool_call_id"] == "call_1"
    for fragment in expected_fragments:
        assert fragment in response["content"]


def test_a_missing_tool_call_id_gets_one_minted(llm):
    """With no id to answer, dispatch_function_call mints one rather than degrading to an assistant message.

    A tool response with no tool_call_id is rejected by pairing-strict
    providers, so the old "fall back to role: assistant" behavior told the
    user but never the model. Minting an id instead means every error still
    goes out as a properly paired role:tool message, which is what lets
    respond() give the model a turn to correct itself.
    """
    chunks = _dispatch(llm, "execute", '{"language": "python"}', tool_call_id=None)
    response = _tool_response(chunks)
    assert response["tool_call_id"]
    assert "missing required fields" in response["content"]


def test_a_missing_tool_call_id_is_recovered_from_the_request(llm):
    """The id is dug out of the last assistant message when the caller did not pass one.

    Without it the error response is unmatched and the provider rejects the
    whole next request, turning a recoverable mistake into a dead session.
    """
    request_params = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "recovered_id"}]},
        ]
    }
    chunks = _dispatch(llm, "execute", '{"language": "python"}', tool_call_id=None, request_params=request_params)
    assert _tool_response(chunks)["tool_call_id"] == "recovered_id"


def test_a_minted_id_does_not_collide_with_one_already_in_the_request(llm):
    """Minting an id checks the request's existing ids first, so two tool_calls never share one.

    A collision would be a new pairing bug layered on top of the one this
    fixes: the provider would see two different messages claiming the same
    id. This message has no preceding assistant+tool_calls entry, so the
    tool_call_id-recovery step above (which looks for exactly that shape)
    finds nothing and minting is what actually runs; "toolu_1" — the id
    minting would pick first — is already taken by an orphaned tool message,
    so the mint must skip it.
    """
    request_params = {
        "messages": [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "an earlier, unrelated tool response"},
        ]
    }
    chunks = _dispatch(llm, "execute", '{"language": "python"}', tool_call_id=None, request_params=request_params)
    assert _tool_response(chunks)["tool_call_id"] not in ("toolu_1", None, "")


def test_an_empty_string_id_is_treated_as_no_id_and_gets_one_minted(llm):
    """"" is not a usable tool_call_id; it is treated as absent and a real id is minted instead."""
    chunks = _dispatch(llm, "execute", '{"language": "python"}', tool_call_id="")
    assert _tool_response(chunks)["tool_call_id"]


def test_a_whitespace_only_id_is_treated_as_no_id_and_gets_one_minted(llm):
    """"   " is not a usable tool_call_id either.

    A bare truthiness check would accept it, and sending it as a
    tool_call_id is exactly the blank-id pairing failure the id
    normalisation exists to prevent.
    """
    chunks = _dispatch(llm, "execute", '{"language": "python"}', tool_call_id="   ")
    assert _tool_response(chunks)["tool_call_id"].strip()


# --- edit -------------------------------------------------------------------


def test_a_valid_edit_call_becomes_an_edit_chunk(llm, tmp_path):
    """A well-formed edit yields an assistant edit chunk carrying its target."""
    target = str(tmp_path / "file.py")
    chunks = _dispatch(llm, "edit", {"language": "write", "code": "body", "target": target})
    assert chunks == [{"role": "assistant", "type": "edit", "format": "write", "content": "body", "target": target}]


def test_write_may_produce_an_empty_file(llm, tmp_path):
    """An empty body is valid for `write` and only for `write`.

    Truncating a file is a legitimate edit. For sed/patch-style languages an
    empty program is always a mistake.
    """
    target = str(tmp_path / "file.txt")
    chunks = _dispatch(llm, "edit", {"language": "write", "code": "", "target": target})
    assert chunks[0]["type"] == "edit"


@pytest.mark.parametrize(
    "arguments,expected_fragment",
    [
        ({"language": "nonsense", "code": "x", "target": "/tmp/a"}, "invalid language"),
        ({"language": "write", "target": "/tmp/a"}, "'code' is required"),
        ({"language": "write", "code": 5, "target": "/tmp/a"}, "must be a string"),
        ({"language": "sed", "code": "  ", "target": "/tmp/a"}, "cannot be empty"),
        ({"language": "write", "code": "x"}, "'target' is required"),
        ({"language": "write", "code": "x", "target": "relative.py"}, "must be an absolute path"),
        ("not a dict", "must be a JSON object"),
    ],
)
def test_malformed_edit_calls_come_back_as_tool_responses(llm, arguments, expected_fragment):
    """Every bad edit payload is reported to the model rather than applied or raised.

    The relative-path case matters most: an edit resolved against whatever cwd
    the kernel happens to be in would write to a file nobody named.
    """
    chunks = _dispatch(llm, "edit", arguments)
    assert expected_fragment in _tool_response(chunks)["content"]


def test_the_edit_language_list_is_named_in_the_error(llm):
    """A wrong language error lists the valid ones, so the model can retry correctly."""
    chunks = _dispatch(llm, "edit", {"language": "nope", "code": "x", "target": "/tmp/a"})
    for language in EDIT_LANGUAGES:
        assert language in _tool_response(chunks)["content"]


# --- view_image -------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected_fragment",
    [
        (None, "path is required"),
        (5, "path is required"),
        ("relative.png", "must be absolute"),
        ("/definitely/missing/file.png", "file not found"),
    ],
)
def test_bad_view_image_paths_are_rejected_before_any_prompt(llm, path, expected_fragment):
    """A path that cannot be an image never reaches the approval prompt.

    Asking the user to approve showing a file that does not exist wastes the
    one thing the prompt is for.
    """
    chunks = _dispatch(llm, "view_image", {"path": path} if path is not None else {})
    assert expected_fragment in _tool_response(chunks)["content"]
    assert not any(chunk.get("type") == "view_image_approval" for chunk in chunks)


def test_an_unsupported_image_format_is_rejected_with_the_supported_list(llm, tmp_path):
    """A PDF is refused, and the message says what is accepted.

    The model will otherwise retry the same unsupported file; naming the
    formats is what lets it convert or pick another.
    """
    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF")
    chunks = _dispatch(llm, "view_image", {"path": str(document)})
    response = _tool_response(chunks)
    assert "unsupported file format" in response["content"]
    assert "png" in response["content"]


def test_a_valid_image_asks_for_approval_and_records_the_call(llm, tmp_path):
    """view_image emits the tool_call record *before* the approval prompt.

    Without that record, history holds a role=tool response with no preceding
    assistant tool call, and process_messages synthesises a fake execute call
    that the model then echoes on its next turn. The record carries the call
    as the model sent it — name and raw arguments — because it is the same
    record every malformed branch writes; view_image no longer has a chunk
    type of its own.
    """
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG")
    chunks = _dispatch(llm, "view_image", {"path": str(image)})

    assert chunks[0]["type"] == "tool_call"
    assert chunks[0]["name"] == "view_image"
    assert chunks[0]["tool_call_id"] == "call_1"
    assert chunks[0]["arguments"] == {"path": str(image)}
    assert chunks[1]["type"] == "view_image_approval"
    assert chunks[2]["role"] == "tool"
    assert chunks[2]["tool_call_id"] == "call_1"
    assert "declined" in chunks[2]["content"]


@pytest.mark.parametrize("answer,shrink", [("f", False), ("y", False), ("r", True)])
def test_approving_an_image_queues_it_with_the_right_resize_flag(llm, tmp_path, answer, shrink):
    """f and the legacy y send full resolution; r sets the shrink flag.

    The flag is read later when the image is encoded. Losing it means a
    multi-megabyte data URL is resent on every subsequent turn.
    """
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    llm.interpreter._view_image_approval = answer

    chunks = _dispatch(llm, "view_image", {"path": str(image)})

    assert llm.interpreter._pending_view_image_path == str(image)
    assert llm.interpreter._pending_view_image_shrink is shrink
    assert "Image added" in chunks[-1]["content"]


def test_declining_an_image_leaves_nothing_pending(llm, tmp_path):
    """A declined image is not queued, and the model is told so."""
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG")
    llm.interpreter._view_image_approval = "n"
    chunks = _dispatch(llm, "view_image", {"path": str(image)})
    assert not hasattr(llm.interpreter, "_pending_view_image_path")
    assert "declined" in chunks[-1]["content"]


# --- anything else ----------------------------------------------------------


def test_an_unknown_function_is_explained_rather_than_ignored(llm):
    """A call to a function that is not a direct tool gets a corrective response.

    Models routinely try to call toolbox helpers directly. The error names the
    three real tools and shows how to reach the rest from Python, which is
    what stops the model repeating the same call.
    """
    chunks = _dispatch(llm, "toolbox.web.search", '{"query": "x"}')
    response = _tool_response(chunks)
    assert "Unsupported function call" in response["content"]
    assert "execute" in response["content"]
    assert "toolbox.web.search" in response["content"]
# --- the record of the call the model actually made -------------------------


def test_a_malformed_call_yields_the_real_call_then_the_error_then_a_notice(llm):
    """A bad call produces three chunks: the model's own call, the error, one user line.

    dispatch_function_call used to yield only the tool response. process_messages
    then had to invent an assistant message to pair with it, and what it invented
    was execute(code="pass  # (synthetic; do not run)") — so the model read its own
    history as "I called execute with `pass`" immediately followed by "that call was
    invalid", two statements that contradict each other about a call it never made.
    The notice is the other half: the role:tool message is never displayed, so
    without it the user sees nothing at all when the model sends a bad call.
    """
    chunks = _dispatch(llm, "execute", "not json at all {{{")

    assert [chunk["type"] for chunk in chunks] == ["tool_call", "message", "notice"]
    record, response, notice = chunks
    assert record["role"] == "assistant"
    assert record["name"] == "execute"
    assert record["tool_call_id"] == response["tool_call_id"] == "call_1"
    assert "not valid JSON" in notice["content"]


def test_the_recorded_arguments_are_the_unrepaired_ones_the_model_sent(llm):
    """Unparseable arguments are recorded verbatim, not repaired into valid JSON.

    The recorded call is paired with an error saying the JSON could not be parsed.
    Substituting anything parseable would put a call the model never made in front
    of that error and bring back the contradiction this record exists to remove.
    """
    chunks = _dispatch(llm, "execute", '{"language": "python", "code": ')
    assert chunks[0]["arguments"] == '{"language": "python", "code": '


def test_a_call_with_no_function_name_is_recorded_without_inventing_one(llm):
    """A nameless call is recorded with an empty name, never with a tool's name.

    This was the worst case of the fabricated pairing: the model was shown a tool
    call *named* execute and then told, in the very next message, that its function
    name was missing.
    """
    chunks = _dispatch(llm, "", '{"language": "python", "code": "print(1)"}')

    assert chunks[0]["type"] == "tool_call"
    assert chunks[0]["name"] == ""
    assert "function name is missing" in _tool_response(chunks)["content"]


def test_a_valid_call_records_no_tool_call_chunk(llm):
    """A call that works yields its code chunk and nothing else.

    The code chunk is already rebuilt as an assistant tool call by
    convert_to_openai_messages, so an extra record here would put the same call
    into history twice, and a notice would tell the user a working call failed.
    """
    chunks = _dispatch(llm, "execute", '{"language": "python", "code": "print(1)"}')
    assert [chunk["type"] for chunk in chunks] == ["code"]
