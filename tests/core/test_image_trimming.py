"""Trimming has to keep working once a conversation contains an image.

A vision turn reaches the provider with `content` as a list of parts rather
than a string. tokentrim assumed a string and raised TypeError on that shape,
and the bare `except` around the trimming block swallowed it and handed the
messages on untrimmed -- so on any vision turn the context window stopped being
enforced at all. The conversation then grew without limit until the provider
rejected it on context length.

These drive the interpreter's own trimming path rather than the trimming
library, so they fail if that path stops coping with multimodal content.
"""

import json

from tests.support.fake_llm import install_fake_llm

SCREENSHOT = "A" * 200_000
OLDEST_TURN = "the oldest message 8675309"


def _vision_session(interpreter, replies, *, context_window, max_tokens):
    fake = install_fake_llm(interpreter, replies)
    # install_fake_llm defaults vision off, and images are dropped during
    # conversion for a model that cannot see -- which would hide the list-shaped
    # content this is about.
    interpreter.llm.supports_vision = True
    interpreter.llm.context_window = context_window
    interpreter.llm.max_tokens = max_tokens
    return fake


def _long_vision_history():
    """A history far larger than the window below, ending in a screenshot."""
    messages = [{"role": "user", "type": "message", "content": OLDEST_TURN}]
    for index in range(40):
        messages.append(
            {"role": "assistant", "type": "message", "content": f"filler reply {index} " * 60}
        )
    messages.append(
        {"role": "user", "type": "image", "format": "base64.png", "content": SCREENSHOT}
    )
    return messages


def test_a_conversation_with_an_image_is_still_trimmed(offline_interpreter):
    """An over-long vision history gets cut down instead of passing through whole.

    This is the regression that matters: tokentrim raised on list-shaped content
    and the swallowing `except` meant nothing was trimmed, so the request went
    out at full size. With trimming working, the oldest turn is dropped.
    """
    fake = _vision_session(
        offline_interpreter, ["Understood."], context_window=2000, max_tokens=500
    )
    offline_interpreter.messages = _long_vision_history()

    offline_interpreter.chat("what is on my screen?", display=False, stream=False)

    sent = json.dumps(fake.calls[0]["messages"])
    assert OLDEST_TURN not in sent, "nothing was trimmed: the whole history went out"


def test_trimming_an_image_turn_keeps_the_system_message(offline_interpreter):
    """Cutting a vision conversation down must not cut the instructions out.

    Trimming to a small window is exactly when the system message is at risk,
    and a request that loses it loses the model's instructions with it.
    """
    fake = _vision_session(
        offline_interpreter, ["Understood."], context_window=2000, max_tokens=500
    )
    offline_interpreter.messages = _long_vision_history()

    offline_interpreter.chat("what is on my screen?", display=False, stream=False)

    first = fake.calls[0]["messages"][0]
    assert first["role"] == "system"
    assert first["content"].strip()


def test_a_screenshot_does_not_evict_a_recent_turn(offline_interpreter):
    """With room to spare, an image does not push out the turn before it.

    The image's real cost is around a hundred tokens, so a short history plus a
    screenshot fits a modest window comfortably.
    """
    fake = _vision_session(
        offline_interpreter, ["Understood."], context_window=8000, max_tokens=1000
    )
    offline_interpreter.messages = [
        {"role": "user", "type": "message", "content": OLDEST_TURN},
        {"role": "assistant", "type": "message", "content": "Noted."},
        {"role": "user", "type": "image", "format": "base64.png", "content": SCREENSHOT},
    ]

    offline_interpreter.chat("what is on my screen?", display=False, stream=False)

    assert OLDEST_TURN in json.dumps(fake.calls[0]["messages"])
