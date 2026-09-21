"""Locating something on screen by asking the model that can already see.

The original path predates vision models: it downloads a sentence-transformer,
OCRs with pytesseract, embeds candidate labels and matches them -- several
gigabytes of dependencies -- and when those are missing it POSTs a screenshot
of the user's desktop to a hosted endpoint instead. Its local screenshot helper
shells out to `screencapture`, which only exists on macOS.

A session whose model has vision does not need any of that. It already sends
images to a model that reads them, so the same model can be asked where a thing
is. That is local, private, works on every platform, needs nothing installed,
and on a 27B local model lands within a few pixels of the target.

Coordinates come back normalized to 0-1, which is the contract the rest of the
toolbox expects: mouse.move multiplies by display.width/height.
"""

import base64
import json
import re
from io import BytesIO

_PROMPT = (
    "This is a {width}x{height} screenshot. Give the pixel coordinates of the "
    "center of: {description}\n\n"
    'Reply with ONLY JSON: {{"x": <int>, "y": <int>}} — or {{"found": false}} '
    "if it is not visible. No other text."
)


class GroundingUnavailable(Exception):
    """Raised when the session has no model that can look at a screenshot."""


def _encode(screenshot):
    buffer = BytesIO()
    screenshot.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _coordinates_from(reply, width, height):
    """The last JSON object in the reply, normalized, or None.

    Thinking models narrate before answering and may show a worked coordinate
    mid-thought, so the *last* object is the answer. Anything outside the image
    is a miss rather than a click in the wrong place.
    """
    for candidate in reversed(re.findall(r"\{[^{}]*\}", reply or "")):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if parsed.get("found") is False:
            return None
        if "x" in parsed and "y" in parsed:
            try:
                x, y = int(parsed["x"]), int(parsed["y"])
            except (TypeError, ValueError):
                continue
            if 0 <= x <= width and 0 <= y <= height:
                return (x / width, y / height)
    return None


def locate(llm, screenshot, description):
    """Where `description` is on `screenshot`, as (x, y) in 0-1, or None.

    `llm` is the session's Llm; its model, endpoint and key are reused so the
    lookup runs wherever the session already runs, with no second provider and
    no second bill.
    """
    if not getattr(llm, "supports_vision", False):
        raise GroundingUnavailable(
            "This model cannot see screenshots, so it cannot be asked where "
            "something is. Set llm.supports_vision if it can (ollama reports "
            "this under /api/show capabilities), or pass explicit coordinates."
        )

    from litellm import completion

    width, height = screenshot.size
    request = {
        "model": llm.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _PROMPT.format(
                            width=width, height=height, description=description
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + _encode(screenshot)},
                    },
                ],
            }
        ],
        # Room for a thinking model to reason before the JSON, not for an essay.
        "max_tokens": 600,
    }
    for attribute, key in (("api_base", "api_base"), ("api_key", "api_key")):
        value = getattr(llm, attribute, None)
        if value:
            request[key] = value

    response = completion(**request)
    return _coordinates_from(response.choices[0].message.content, width, height)
