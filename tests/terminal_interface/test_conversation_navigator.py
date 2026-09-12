"""`interpreter --conversations` is the only way back into a saved session.

The navigator maps files on disk to menu entries, replays the chosen one, and
then hands a *resumed* history to the model. Getting the last part wrong is
the dangerous one: the Python kernel and the working directory have both been
reset, and the model only learns that from the alert this module injects.
"""

import json
import os

import pytest

import interpreter.terminal_interface.conversation_navigator as cn


class FakeInterpreter:
    """Just enough interpreter to record what the navigator did to it."""

    def __init__(self):
        self.messages = []
        self.displayed = []
        self.chats = 0
        self.conversation_filename = None
        self._conversation_title_upgraded = False

    def display_message(self, message):
        self.displayed.append(message)

    def chat(self):
        self.chats += 1


@pytest.fixture
def conversations_dir(tmp_path, monkeypatch):
    """Redirect the conversation store into the test's directory.

    get_storage_path() normally resolves to the user's real
    ~/.config/open-interpreter/conversations, which holds their actual chats.
    """
    directory = tmp_path / "conversations"
    directory.mkdir()
    monkeypatch.setattr(cn, "get_storage_path", lambda sub=None: str(directory))
    return directory


@pytest.fixture
def answers(monkeypatch):
    """Feed scripted answers to inquirer.prompt, in order."""
    scripted = []

    def _prompt(questions):
        if not scripted:
            raise AssertionError("inquirer.prompt called more times than the test scripted")
        return scripted.pop(0)

    monkeypatch.setattr(cn.inquirer, "prompt", _prompt)
    monkeypatch.setattr(cn, "render_past_conversation", lambda messages: None)
    return scripted


def _write(directory, name, messages):
    path = directory / name
    path.write_text(json.dumps(messages))
    return path


def test_missing_directory_reports_instead_of_raising(tmp_path, monkeypatch, capsys):
    """A user who has never saved a conversation gets a message, not a traceback.

    os.listdir on a missing directory raises, so the existence check is the
    only thing standing between a fresh install and a crash on
    `interpreter --conversations`.
    """
    monkeypatch.setattr(cn, "get_storage_path", lambda sub=None: str(tmp_path / "nope"))
    interpreter = FakeInterpreter()
    assert cn.conversation_navigator(interpreter) is None
    assert "No conversations found" in capsys.readouterr().out


def test_cancelling_the_search_prompt_exits_without_touching_the_interpreter(conversations_dir, answers):
    """Ctrl-C at the search box returns cleanly and starts nothing.

    inquirer returns None when the user aborts. Treating that as an empty
    search would drop them into the file list they just tried to leave.
    """
    _write(conversations_dir, "a.json", [])
    answers.append(None)
    interpreter = FakeInterpreter()
    assert cn.conversation_navigator(interpreter) is None
    assert interpreter.chats == 0


def test_new_conversation_entry_starts_a_chat_without_loading_history(conversations_dir, answers):
    """"New Conversation" is a menu entry, not a filename, and must not be looked up.

    It shares the list with real files; if it fell through to the filename
    lookup it would raise KeyError instead of starting a session.
    """
    _write(conversations_dir, "old.json", [{"role": "user", "content": "x"}])
    answers.append({"search": ""})
    answers.append({"name": "New Conversation →"})

    interpreter = FakeInterpreter()
    cn.conversation_navigator(interpreter)
    assert interpreter.chats == 1
    assert interpreter.messages == []


def test_open_folder_entry_opens_the_directory_and_starts_nothing(conversations_dir, answers, monkeypatch):
    """"Open Folder" shells out and returns; it must not also start a chat."""
    opened = []
    monkeypatch.setattr(cn, "open_folder", lambda path: opened.append(path))
    answers.append({"search": ""})
    answers.append({"name": "Open Folder →"})

    interpreter = FakeInterpreter()
    cn.conversation_navigator(interpreter)
    assert opened == [str(conversations_dir)]
    assert interpreter.chats == 0


def test_filenames_become_readable_menu_entries(conversations_dir, answers):
    """Stored filenames are rewritten into "First few words... (date)" for the menu.

    The mapping is round-tripped back to the filename on selection, so if the
    two halves ever disagree the user picks one conversation and opens
    another.
    """
    _write(conversations_dir, "Fix_the_build__September_23rd.json", [{"role": "user", "content": "hi"}])
    answers.append({"search": ""})
    answers.append({"name": "Fix the build... (September 23rd)"})

    interpreter = FakeInterpreter()
    cn.conversation_navigator(interpreter)
    assert interpreter.conversation_filename == "Fix_the_build__September_23rd.json"


def test_search_filters_the_list_case_insensitively(conversations_dir, monkeypatch):
    """Typing a search term narrows the menu; matching ignores case.

    With hundreds of saved conversations the list is unusable without this,
    and a case-sensitive match would miss the capitalised first word of most
    auto-generated titles.
    """
    _write(conversations_dir, "Fix_the_build__Sep.json", [])
    _write(conversations_dir, "Write_a_poem__Sep.json", [])

    seen = {}

    def _prompt(questions):
        if questions[0].name == "search":
            return {"search": "POEM"}
        seen["choices"] = list(questions[0].choices)
        return None

    monkeypatch.setattr(cn.inquirer, "prompt", _prompt)
    monkeypatch.setattr(cn, "render_past_conversation", lambda messages: None)
    cn.conversation_navigator(FakeInterpreter())
    assert seen["choices"] == ["Write a poem... (Sep)"]


def test_a_search_with_no_matches_asks_again(conversations_dir, answers, capsys):
    """An unmatched search re-prompts; it never shows a zero-choice list.

    inquirer.List with no choices is unnavigable — the user would be stuck in
    a menu with nothing to select and no way back.
    """
    _write(conversations_dir, "Fix_the_build__Sep.json", [])
    answers.append({"search": "zzz"})
    answers.append({"search": ""})
    answers.append(None)

    cn.conversation_navigator(FakeInterpreter())
    assert 'No conversations match "zzz"' in capsys.readouterr().out
    assert answers == []


def test_resuming_injects_exactly_one_reset_alert_even_across_repeated_resumes(conversations_dir, answers):
    """The resumed-session alert is injected once and old copies are dropped first.

    The alert tells the model its Python state is gone and the cwd has moved.
    Because the alert is saved into history, resuming a resumed conversation
    would stack copies; the filter on alert_kind is what prevents an ever
    growing pile of contradictory "this was just resumed" notices.
    """
    history = [
        {"role": "user", "type": "message", "content": "hello"},
        {
            "role": "user",
            "type": "message",
            "content": "stale alert",
            "source": "terminal",
            "format": "system_alert",
            "alert_kind": "conversation_resumed",
        },
    ]
    _write(conversations_dir, "Chat__Sep.json", history)
    answers.append({"search": ""})
    answers.append({"name": "Chat... (Sep)"})

    interpreter = FakeInterpreter()
    cn.conversation_navigator(interpreter)

    alerts = [m for m in interpreter.messages if m.get("alert_kind") == "conversation_resumed"]
    assert len(alerts) == 1
    assert "stale alert" not in alerts[0]["content"]
    assert alerts[0]["source"] == "terminal"
    assert alerts[0]["format"] == "system_alert"
    assert "CWD Reset" in alerts[0]["content"]
    assert "Python REPL Reset" in alerts[0]["content"]


def test_resuming_marks_the_title_as_already_chosen(conversations_dir, answers):
    """Resuming suppresses the one-shot LLM rename.

    The file on disk already has a name. Leaving the flag unset would spend an
    extra model call renaming it, and then write the conversation out under a
    second filename.
    """
    _write(conversations_dir, "Chat__Sep.json", [{"role": "user", "content": "hi"}])
    answers.append({"search": ""})
    answers.append({"name": "Chat... (Sep)"})

    interpreter = FakeInterpreter()
    cn.conversation_navigator(interpreter)
    assert interpreter._conversation_title_upgraded is True
    assert interpreter.chats == 1


def test_cancelling_the_file_menu_leaves_history_untouched(conversations_dir, answers):
    """Aborting at the file list must not load or start anything."""
    _write(conversations_dir, "Chat__Sep.json", [{"role": "user", "content": "hi"}])
    answers.append({"search": ""})
    answers.append(None)

    interpreter = FakeInterpreter()
    assert cn.conversation_navigator(interpreter) is None
    assert interpreter.messages == []
    assert interpreter.chats == 0


def test_conversations_are_listed_newest_first(conversations_dir, monkeypatch):
    """The most recently modified conversation is the first real entry.

    Resuming is almost always "the one I was just in". Sorting by name instead
    of mtime would bury it among alphabetically earlier titles.
    """
    old = _write(conversations_dir, "AAA_old__Sep.json", [])
    new = _write(conversations_dir, "ZZZ_new__Sep.json", [])
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    seen = {}

    def _prompt(questions):
        if questions[0].name == "search":
            return {"search": ""}
        seen["choices"] = list(questions[0].choices)
        return None

    monkeypatch.setattr(cn.inquirer, "prompt", _prompt)
    cn.conversation_navigator(FakeInterpreter())
    assert seen["choices"] == [
        "New Conversation →",
        "Open Folder →",
        "ZZZ new... (Sep)",
        "AAA old... (Sep)",
    ]


@pytest.mark.parametrize(
    "system,expected",
    [("Darwin", ["open", "/some/dir"]), ("Linux", ["xdg-open", "/some/dir"])],
)
def test_open_folder_uses_the_platform_file_manager(monkeypatch, system, expected):
    """Each platform gets its own opener; Linux is the fallback, not a no-op.

    A missing branch here silently does nothing, and "Open Folder" that opens
    no folder is indistinguishable from a hang.
    """
    calls = []
    monkeypatch.setattr(cn.platform, "system", lambda: system)
    monkeypatch.setattr(cn.subprocess, "run", lambda args: calls.append(args))
    cn.open_folder("/some/dir")
    assert calls == [expected]
