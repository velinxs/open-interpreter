"""The conversation-contribution prompt decides whether private chats leave the machine.

Every branch here is a privacy boundary: a wrong default, a cache that never
records "already asked", or a payload built from the wrong list would upload
the user's whole history without them agreeing to it. These tests pin the
answer to "when, exactly, does anything get sent".
"""

import json

import pytest

import interpreter.terminal_interface.contributing_conversations as cc


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the contribution cache at the test's own directory.

    The real path is ~/.cache/open-interpreter/contribute.json. Without this a
    test would overwrite the developer's record of what they have already been
    asked, and re-prompt them on their next real run.
    """
    cache = tmp_path / "cache" / "contribute.json"
    cache.parent.mkdir(parents=True)
    monkeypatch.setattr(cc, "contribute_cache_path", str(cache))
    return cache


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Record uploads instead of performing them."""
    posts = []
    monkeypatch.setattr(cc.requests, "post", lambda url, json=None, **kw: posts.append((url, json)))
    return posts


class FakeInterpreter:
    def __init__(self, will_contribute=False, history_path=None):
        self.will_contribute = will_contribute
        self.conversation_history_path = history_path


def test_first_launch_writes_the_cache_with_everything_unasked(_isolate_cache):
    """A missing cache file is created with all three flags False.

    These flags are the only memory of what the user has been asked. If a
    fresh file came back with them set, the contribution prompt would never
    appear; if the file were not written, the user would be asked on every
    single launch.
    """
    cache = cc.get_contribute_cache_contents()
    assert cache == {
        "asked_to_contribute_past": False,
        "displayed_contribution_message": False,
        "asked_to_contribute_future": False,
    }
    assert json.loads(_isolate_cache.read_text()) == cache


def test_cache_read_requires_the_parent_directory_to_already_exist(tmp_path, monkeypatch):
    """Characterisation: the cache is created with open(), which does not mkdir.

    Today ~/.cache/open-interpreter always exists by the time this runs only
    because importing interpreter.core.utils.telemetry creates it at import
    time for the telemetry user id. This test pins that hidden dependency:
    if telemetry's directory creation is ever made lazy or removed, startup
    raises FileNotFoundError here, and this test says why.
    """
    monkeypatch.setattr(cc, "contribute_cache_path", str(tmp_path / "missing" / "contribute.json"))
    with pytest.raises(FileNotFoundError):
        cc.get_contribute_cache_contents()


def test_existing_cache_is_read_back_not_overwritten(_isolate_cache):
    """An existing cache wins, so a user who declined is not asked again."""
    _isolate_cache.write_text(
        json.dumps(
            {
                "asked_to_contribute_past": True,
                "displayed_contribution_message": True,
                "asked_to_contribute_future": True,
            }
        )
    )
    assert cc.get_contribute_cache_contents()["asked_to_contribute_past"] is True


def test_launch_logic_shows_the_advert_once_then_never_again(monkeypatch, capsys, _isolate_cache):
    """The "we're training a model" pitch is displayed exactly once per machine.

    It is the only thing a non-contributing user sees from this module. If the
    flag were not persisted, every launch would reprint it; if it were set too
    early, nobody would ever see it.
    """
    monkeypatch.setattr(cc.time, "sleep", lambda *_: None)
    interpreter = FakeInterpreter(will_contribute=False)

    cc.contribute_conversation_launch_logic(interpreter)
    first = capsys.readouterr().out
    assert "training an open-source language model" in first.replace("\n", " ")
    assert json.loads(_isolate_cache.read_text())["displayed_contribution_message"] is True

    cc.contribute_conversation_launch_logic(interpreter)
    assert "training" not in capsys.readouterr().out


def test_contributing_user_is_asked_about_past_and_future_once_each(monkeypatch, _isolate_cache):
    """A user who opted in is asked the two follow-up questions exactly once.

    Both questions are recorded in the cache whatever the answer. Answering
    "no" must still mark them asked, or the user is nagged on every launch;
    the asked flags are pinned here for the declining case specifically.
    """
    asked = []

    def _no(*args, **kwargs):
        asked.append(True)
        return False

    monkeypatch.setattr(cc, "user_wants_to_contribute_past", _no)
    monkeypatch.setattr(cc, "user_wants_to_contribute_future", _no)
    monkeypatch.setattr(cc, "display_contributing_current_message", lambda: None)

    interpreter = FakeInterpreter(will_contribute=True)
    cc.contribute_conversation_launch_logic(interpreter)

    assert len(asked) == 2
    cache = json.loads(_isolate_cache.read_text())
    assert cache["asked_to_contribute_past"] is True
    assert cache["asked_to_contribute_future"] is True

    cc.contribute_conversation_launch_logic(interpreter)
    assert len(asked) == 2


def test_saying_yes_to_past_uploads_history_only_after_a_second_confirmation(monkeypatch, tmp_path, _no_network):
    """Uploading past conversations needs *two* yeses, not one.

    user_wants_to_contribute_past() only opens the door; send_past_conversations
    then shows what is about to leave the machine and asks again. Collapsing
    that into one prompt would ship history the user never reviewed.
    """
    history = tmp_path / "conversations"
    history.mkdir()
    (history / "chat.json").write_text(json.dumps([{"role": "user", "content": "secret"}]))

    monkeypatch.setattr(cc.time, "sleep", lambda *_: None)
    monkeypatch.setattr(cc, "user_wants_to_contribute_past", lambda: True)
    monkeypatch.setattr(cc, "user_wants_to_contribute_future", lambda: False)
    monkeypatch.setattr(cc, "display_contributing_current_message", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *_: "n")

    interpreter = FakeInterpreter(will_contribute=True, history_path=str(history))
    cc.contribute_conversation_launch_logic(interpreter)
    assert _no_network == []

    monkeypatch.setattr("builtins.input", lambda *_: "y")
    cc.send_past_conversations(interpreter)
    assert len(_no_network) == 1
    url, payload = _no_network[0]
    assert payload["conversations"] == [[{"role": "user", "content": "secret"}]]


def test_get_all_conversations_reads_only_json_and_tolerates_a_missing_directory(tmp_path):
    """Non-JSON files in the history directory are skipped, and no directory means no conversations.

    The history directory also collects stray files. Handing a .txt to
    json.load would abort the upload mid-way; raising on a missing directory
    would break a brand-new install.
    """
    history = tmp_path / "conversations"
    assert cc.get_all_conversations(FakeInterpreter(history_path=str(history))) == []

    history.mkdir()
    (history / "a.json").write_text(json.dumps([{"role": "user"}]))
    (history / "notes.txt").write_text("not json")
    assert cc.get_all_conversations(FakeInterpreter(history_path=str(history))) == [[{"role": "user"}]]


def test_empty_conversations_are_never_posted(_no_network):
    """Nothing is sent for an empty list or an empty first conversation.

    The endpoint is public; posting empty payloads every time the user quits a
    fresh session would be pure noise, and the early return is what prevents
    it.
    """
    assert cc.contribute_conversations([]) is None
    assert cc.contribute_conversations([[]]) is None
    assert _no_network == []


def test_payload_carries_version_feedback_and_conversation_id(_no_network):
    """The upload includes the metadata the training pipeline keys on.

    Dropping conversation_id would make duplicate uploads indistinguishable,
    and dropping feedback would lose the thumbs-up/down the user gave on exit.
    """
    cc.contribute_conversations([[{"role": "user", "content": "hi"}]], feedback=True, conversation_id="abc")
    url, payload = _no_network[0]
    assert url == "https://api.openinterpreter.com/v0/contribute/"
    assert payload["feedback"] is True
    assert payload["conversation_id"] == "abc"
    assert payload["oi_version"]


def test_a_flat_message_list_is_rejected_before_it_is_sent(_no_network):
    """conversations must be a list *of conversations*, and the assert catches it.

    Passing interpreter.messages instead of [interpreter.messages] is the easy
    mistake at every call site. Sending it would corrupt the training set with
    a conversation per message, so failing loudly is the intended behaviour.
    """
    with pytest.raises(AssertionError):
        cc.contribute_conversations([[{"role": "user"}], {"role": "user"}])
    assert _no_network == []


def test_upload_failures_are_swallowed(monkeypatch, _no_network):
    """A dead network must not take the session down on exit.

    This runs from the Ctrl-C handler in main(); an exception escaping here
    would replace a clean exit with a traceback.
    """

    def _boom(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(cc.requests, "post", _boom)
    cc.contribute_conversations([[{"role": "user", "content": "hi"}]])


def test_opting_in_for_the_future_writes_the_profile_not_just_the_cache(monkeypatch):
    """"Contribute future conversations" is persisted to the profile, not the cache.

    The cache only remembers that the question was asked. If the answer were
    stored there too, the setting would be lost the moment the cache file was
    cleared, silently turning contribution back off.
    """
    written = []
    monkeypatch.setattr(cc, "write_key_to_profile", lambda k, v: written.append((k, v)))
    cc.set_send_future_conversations(FakeInterpreter(), True)
    assert written == [("contribute_conversation", True)]
