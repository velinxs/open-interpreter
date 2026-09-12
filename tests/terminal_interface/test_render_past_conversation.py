"""Replaying a saved conversation must show the same story the live session showed.

render_past_conversation() is what `--conversations` and `%load_message` use to
redraw history. If it drops a message type, duplicates the terminal's own
injected notices, or detaches a code block from its output, a resumed session
looks nothing like the one the user left.
"""

import pytest

from interpreter.terminal_interface.render_past_conversation import (
    render_past_conversation,
)


@pytest.fixture
def render(capsys):
    def _render(messages):
        render_past_conversation(messages)
        return capsys.readouterr().out

    return _render


def test_user_and_assistant_messages_are_both_rendered(render):
    """Both halves of the dialogue appear, with the user's prefixed by '>'.

    A replay that only showed one side would make a resumed conversation
    unreadable, and the '>' is the only visual cue separating who said what.
    """
    out = render(
        [
            {"role": "user", "type": "message", "content": "hello there"},
            {"role": "assistant", "type": "message", "content": "hi back"},
        ]
    )
    assert "> hello there" in out
    assert "hi back" in out


def test_code_and_its_console_output_render_together(render):
    """Console output is attached to the code block that produced it.

    The renderer buffers a code chunk and every console chunk after it, then
    emits them as one group. If the buffering broke, output would appear
    before its code or be dropped entirely when the conversation ends on a
    code block.
    """
    out = render(
        [
            {"role": "assistant", "type": "code", "format": "python", "content": "print(2+2)"},
            {"role": "computer", "type": "console", "format": "output", "content": "4"},
        ]
    )
    assert "print(2+2)" in out
    assert "4" in out
    assert out.index("print(2+2)") < out.index("4")


def test_trailing_code_block_is_flushed_at_the_end(render):
    """A conversation that ends on a code block still renders that block.

    Pending code is only emitted by a flush; without the final flush after the
    loop, the last thing the model did would silently vanish from the replay.
    """
    out = render([{"role": "assistant", "type": "code", "format": "python", "content": "print('last')"}])
    assert "print('last')" in out


def test_active_line_chunks_are_not_rendered_as_output(render):
    """active_line markers are UI state, not program output.

    They carry line numbers used to highlight the running line live. Replaying
    them would splice stray integers into the saved program output.
    """
    out = render(
        [
            {"role": "assistant", "type": "code", "format": "python", "content": "x=1\ny=2"},
            {"role": "computer", "type": "console", "format": "active_line", "content": 2},
            {"role": "computer", "type": "console", "format": "output", "content": "done"},
        ]
    )
    assert "done" in out
    assert "\n2\n" not in out


def test_terminal_injected_user_messages_are_suppressed(render):
    """Messages the terminal injected as the user are not replayed as user speech.

    The terminal writes context (file dumps, tool notices) into history as
    role=user with source=terminal. Replaying them as '> ...' would make it
    look like the user typed text they never typed.
    """
    out = render(
        [
            {
                "role": "user",
                "type": "message",
                "content": "INJECTED CONTEXT",
                "source": "terminal",
            },
            {"role": "user", "type": "message", "content": "real question"},
        ]
    )
    assert "INJECTED CONTEXT" not in out
    assert "> real question" in out


def test_resumed_session_alert_is_the_one_terminal_message_that_replays(render):
    """The "conversation was resumed" alert is user-facing and must survive replay.

    conversation_navigator appends it with source=terminal and
    format=system_alert. It explains why the Python environment and cwd reset,
    so hiding it on a second resume would leave the reader confused about a
    visible discontinuity in the log.
    """
    out = render(
        [
            {
                "role": "user",
                "type": "message",
                "content": "SYSTEM ALERT: resumed",
                "source": "terminal",
                "format": "system_alert",
            }
        ]
    )
    assert "SYSTEM ALERT" in out


def test_blank_system_alert_renders_nothing(render):
    """An empty alert produces no output rather than an empty horizontal rule."""
    out = render(
        [
            {
                "role": "user",
                "type": "message",
                "content": "   ",
                "source": "terminal",
                "format": "system_alert",
            }
        ]
    )
    assert out.strip() == ""


def test_view_image_call_chunks_are_skipped(render):
    """Image tool-call plumbing is not part of the transcript.

    A view_image_call carries the request, not the picture. Rendering it would
    show raw call machinery in the middle of the conversation.
    """
    out = render(
        [
            {"role": "assistant", "type": "view_image_call", "content": "/tmp/x.png"},
            {"role": "assistant", "type": "message", "content": "after"},
        ]
    )
    assert "/tmp/x.png" not in out
    assert "after" in out


def test_user_image_is_replayed_as_a_labelled_panel(render):
    """An image the user supplied is replayed as a placeholder, not silently.

    The terminal cannot redraw the picture, but dropping it would hide the
    fact that the model was given one — which is often the reason a later
    answer makes sense.
    """
    out = render([{"role": "user", "type": "image", "content": "/tmp/photo.png"}])
    assert "Image viewed" in out


def test_reasoning_text_is_boxed_separately_from_the_answer(render):
    """Reasoning is visually separated from the model's actual reply.

    Reasoning content is not an answer, and a replay that mixed the two would
    make the model look like it said things it only thought.
    """
    out = render(
        [
            {
                "role": "assistant",
                "type": "message",
                "format": "reasoning",
                "content": "thinking about it",
            },
            {"role": "assistant", "type": "message", "content": "the answer"},
        ]
    )
    assert "Thinking" in out
    assert "the answer" in out


def test_blank_assistant_messages_do_not_produce_separators(render):
    """Empty assistant chunks are dropped instead of emitting blank paragraphs.

    Streamed histories often contain empty message chunks; rendering one per
    chunk would pad the replay with dozens of blank lines.
    """
    out = render(
        [
            {"role": "assistant", "type": "message", "content": ""},
            {"role": "assistant", "type": "message", "content": "   \n "},
        ]
    )
    assert out.strip() == ""


def test_messages_are_separated_by_a_blank_line_but_not_led_by_one(render):
    """Separators go between messages only, so a replay never starts with a blank line.

    render_separator() suppresses the first newline with a has_rendered flag;
    losing that flag would add a leading gap to every resumed conversation.
    """
    out = render(
        [
            {"role": "user", "type": "message", "content": "one"},
            {"role": "user", "type": "message", "content": "two"},
        ]
    )
    assert not out.startswith("\n")
    assert "\n\n" in out


def test_unknown_roles_and_types_are_ignored_rather_than_crashing(render):
    """Unrecognised chunks are skipped so an old or future log still replays.

    Saved conversations outlive the code that wrote them. A KeyError here
    would make an entire history file unopenable.
    """
    out = render(
        [
            {"role": "system", "type": "message", "content": "system thing"},
            {"role": "computer", "type": "image", "content": "base64..."},
            {},
            {"role": "user", "type": "message", "content": "still here"},
        ]
    )
    assert "> still here" in out
