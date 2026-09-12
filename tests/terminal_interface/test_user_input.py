"""What happens to a typed line before it becomes a message to the model.

_prepare_message sits between the prompt and chat(). It swallows magic
commands, catches the two commands people habitually type at the prompt
instead of the shell, and turns a dragged-in image path into an image message
— but only after asking. That last part is an upload gate: getting it wrong
sends a file off the machine that the user did not agree to send.
"""

from types import SimpleNamespace

import pytest

import interpreter.terminal_interface.user_input as user_input
from interpreter.terminal_interface.user_input import _prepare_message, _print_mode_banner


class FakeInterpreter:
    def __init__(self, supports_vision=False, **kwargs):
        self.llm = SimpleNamespace(supports_vision=supports_vision, vision_renderer=None)
        self.messages = []
        self.displayed = []
        self.auto_run_mode = kwargs.get("auto_run_mode", "prompt")
        self.offline = kwargs.get("offline", False)
        self.safe_mode = kwargs.get("safe_mode", "off")
        self.plain_text_display = kwargs.get("plain_text_display", False)
        self.magic_commands = []

    def display_message(self, message):
        self.displayed.append(message)


@pytest.fixture(autouse=True)
def _no_magic(monkeypatch):
    """Record magic commands instead of executing them."""
    recorded = []
    monkeypatch.setattr(user_input, "handle_magic_command", lambda interp, text: recorded.append(text))
    return recorded


def test_an_empty_line_sends_nothing(_no_magic):
    """Pressing enter on an empty prompt does not start a model call.

    Every turn costs a request. An empty message would spend one to ask the
    model to respond to nothing.
    """
    assert _prepare_message(FakeInterpreter(), "", interactive=True) is None


def test_magic_commands_are_handled_locally_and_not_sent(_no_magic):
    """A line starting with % is executed here, and the model never sees it.

    %undo and friends manipulate the conversation. Forwarding them would both
    waste a call and put the literal text into the history they are supposed
    to be editing.
    """
    assert _prepare_message(FakeInterpreter(), "%undo", interactive=True) is None
    assert _no_magic == ["%undo"]


def test_magic_commands_are_ordinary_text_when_not_interactive(_no_magic):
    """In non-interactive mode a leading % is just text.

    A piped-in document or a scripted message can legitimately start with %,
    and silently running it as a command would rewrite the caller's session.
    """
    assert _prepare_message(FakeInterpreter(), "%undo", interactive=False) == "%undo"
    assert _no_magic == []


@pytest.mark.parametrize(
    "typed",
    ["interpreter --local", "pip install --upgrade open-interpreter"],
)
def test_shell_commands_typed_at_the_prompt_are_explained_not_forwarded(typed, capsys, _no_magic):
    """The two commands users habitually type at the prompt get an explanation.

    Both are shell commands from the documentation. Sending either to the
    model produces a confident, useless answer about how to run it, when the
    real problem is that the user is in the wrong place.
    """
    assert _prepare_message(FakeInterpreter(), f"  {typed}  ", interactive=True) is None
    assert "exit this conversation" in capsys.readouterr().out


def test_a_plain_message_passes_through_unchanged(_no_magic):
    """Anything else is returned as typed."""
    assert _prepare_message(FakeInterpreter(), "what is 21*2", interactive=True) == "what is 21*2"


# --- image upload gate ------------------------------------------------------


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "photo.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    return str(path)


def test_image_paths_are_ignored_when_the_model_cannot_see(_no_magic, image):
    """A text-only model gets the path as text, with no prompt.

    Asking to upload an image to a model that would reject it is a question
    with no right answer.
    """
    interpreter = FakeInterpreter(supports_vision=False)
    assert _prepare_message(interpreter, f"look at {image}", interactive=True) == f"look at {image}"
    assert interpreter.messages == []


def test_declining_the_upload_sends_the_text_only(monkeypatch, _no_magic, image):
    """Answering "n" leaves the message as plain text and uploads nothing.

    This is the whole point of the prompt. A path that leaks through as an
    image message would send the file contents to the provider anyway.
    """
    monkeypatch.setattr(user_input, "_prompt_or_skip", lambda prompt, choices: "n")
    interpreter = FakeInterpreter(supports_vision=True)
    result = _prepare_message(interpreter, f"look at {image}", interactive=True)
    assert result == f"look at {image}"
    assert interpreter.messages == []


def test_accepting_the_upload_splits_the_text_from_the_image(monkeypatch, _no_magic, image):
    """"y" queues the typed text as a message and hands the image to chat() separately.

    chat() only processes the object it is given, so the text has to be pushed
    into history first or the question that came with the picture is lost.
    """
    monkeypatch.setattr(user_input, "_prompt_or_skip", lambda prompt, choices: "y")
    interpreter = FakeInterpreter(supports_vision=True)
    result = _prepare_message(interpreter, f"look at {image}", interactive=True)

    assert interpreter.messages == [
        {
            "role": "user",
            "type": "message",
            "content": f"look at {image}",
            "sent_at": interpreter.messages[0]["sent_at"],
        }
    ]
    assert result["type"] == "image"
    assert result["format"] == "path"
    assert result["content"] == image
    assert result["shrink"] is False


def test_several_images_are_all_queued(monkeypatch, _no_magic, tmp_path):
    """Every path in the line becomes an image message, not just the first.

    The first is returned for chat() to process and the rest are appended;
    dropping the tail would silently ignore images the user clearly attached.
    """
    paths = []
    for index in range(3):
        path = tmp_path / f"shot{index}.png"
        path.write_bytes(b"\x89PNG" + bytes([index]) * 50)
        paths.append(str(path))

    monkeypatch.setattr(user_input, "_prompt_or_skip", lambda prompt, choices: "y")
    interpreter = FakeInterpreter(supports_vision=True)
    result = _prepare_message(interpreter, " ".join(paths), interactive=True)

    queued_images = [m for m in interpreter.messages if m.get("type") == "image"]
    assert len(queued_images) == 2
    assert {result["content"], *(m["content"] for m in queued_images)} == set(paths)


def test_a_large_image_is_offered_full_or_resized(monkeypatch, _no_magic, tmp_path):
    """Over the shrink threshold the choices become f/r/n, and "r" sets shrink.

    A full-resolution screenshot can be several megabytes of base64 in every
    subsequent request. The resize option is the only way to send it without
    paying for it on every turn, so the flag has to reach the message.
    """
    big = tmp_path / "huge.png"
    big.write_bytes(b"\x89PNG" + b"0" * 4_000_000)

    asked = {}

    def _prompt(prompt, choices):
        asked["choices"] = choices
        return "r"

    monkeypatch.setattr(user_input, "_prompt_or_skip", _prompt)
    interpreter = FakeInterpreter(supports_vision=True)
    result = _prepare_message(interpreter, str(big), interactive=True)

    assert asked["choices"] == ("f", "r", "n")
    assert result["shrink"] is True


def test_full_resolution_is_uploaded_without_the_shrink_flag(monkeypatch, _no_magic, tmp_path):
    """"f" uploads the large image untouched.

    Resizing an image the user explicitly asked to send at full resolution
    would destroy the detail they wanted the model to look at.
    """
    big = tmp_path / "huge.png"
    big.write_bytes(b"\x89PNG" + b"0" * 4_000_000)
    monkeypatch.setattr(user_input, "_prompt_or_skip", lambda prompt, choices: "f")
    result = _prepare_message(FakeInterpreter(supports_vision=True), str(big), interactive=True)
    assert result["shrink"] is False


def test_a_vision_renderer_counts_as_vision_support(monkeypatch, _no_magic, image):
    """A local vision renderer enables the image path even without a vision model.

    That is the whole point of `--local --vision`: the renderer describes the
    image so a text-only model can use it.
    """
    monkeypatch.setattr(user_input, "_prompt_or_skip", lambda prompt, choices: "y")
    interpreter = FakeInterpreter(supports_vision=False)
    interpreter.llm.vision_renderer = object()
    result = _prepare_message(interpreter, image, interactive=True)
    assert result["type"] == "image"


# --- the approval banner ----------------------------------------------------


def test_the_banner_is_skipped_when_everything_runs_automatically():
    """Full auto-run says nothing; there is no approval to explain."""
    interpreter = FakeInterpreter(auto_run_mode="all")
    _print_mode_banner(interpreter)
    assert interpreter.displayed == []


def test_the_banner_is_skipped_for_a_one_shot_command():
    """The "i {command}" entry starts with one message already queued and stays quiet.

    That mode prints one answer. A paragraph about approval modes ahead of it
    would be most of the output.
    """
    interpreter = FakeInterpreter()
    interpreter.messages = [{"role": "user", "type": "message", "content": "hi"}]
    _print_mode_banner(interpreter)
    assert interpreter.displayed == []


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("prompt", "interpreter -y"),
        ("allowlist", "Allowlist mode"),
        ("denylist", "Denylist mode"),
    ],
)
def test_each_approval_mode_explains_itself(mode, expected):
    """The banner names the mode in force and how it behaves.

    Allowlist and denylist have opposite defaults. A user who thinks they are
    in the other one either waits for prompts that never come or is surprised
    by code that ran.
    """
    interpreter = FakeInterpreter(auto_run_mode=mode)
    _print_mode_banner(interpreter)
    assert expected in interpreter.displayed[0]


def test_safe_mode_without_semgrep_says_so(monkeypatch):
    """Safe mode advertises its missing dependency instead of silently not scanning.

    Without semgrep installed, --safe scans nothing. Saying so is the only
    thing standing between the user and a false sense of security.
    """
    monkeypatch.setattr(user_input, "check_for_package", lambda package: False)
    interpreter = FakeInterpreter(safe_mode="ask")
    _print_mode_banner(interpreter)
    assert "pip install semgrep" in interpreter.displayed[0]


def test_the_ctrl_c_hint_is_dropped_in_plain_text_mode():
    """Piped output does not get a "Press CTRL-C to exit" line.

    plain_text_display is the proxy for stdin mode, where nothing is watching
    the terminal and the hint is noise in someone's data.
    """
    interpreter = FakeInterpreter(plain_text_display=True)
    _print_mode_banner(interpreter)
    assert "CTRL-C" not in interpreter.displayed[0]
