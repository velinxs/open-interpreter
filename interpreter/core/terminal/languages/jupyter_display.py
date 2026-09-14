"""Turning a Jupyter display payload into one Open Interpreter chunk."""

import json


def display_chunk(data, vision):
    """Pick the one representation of a display to forward, richest first.

    A display carries the same object in several mimetypes. Images go to a
    model that can see them and are dropped in favour of the text repr for one
    that can't — sending base64 to a text-only model wastes the context and
    arrives as nothing, since the message conversion drops it anyway. Anything
    with no branch of its own (latex, json, a custom mimetype) used to vanish
    silently; it now falls through to whatever text is on offer.
    """
    if vision:
        for mimetype, image_format in (("image/png", "base64.png"), ("image/jpeg", "base64.jpeg")):
            if mimetype in data:
                return {"type": "image", "format": image_format, "content": data[mimetype]}
    if "text/markdown" in data:
        return {"type": "console", "format": "output", "content": data["text/markdown"]}
    if "text/html" in data:
        return {"type": "code", "format": "html", "content": data["text/html"]}
    if "text/plain" in data:
        return {"type": "console", "format": "output", "content": data["text/plain"]}
    if "application/javascript" in data:
        return {"type": "code", "format": "javascript", "content": data["application/javascript"]}
    for mimetype, value in data.items():
        if mimetype.startswith("image/"):
            continue
        if not isinstance(value, str):
            try:
                value = json.dumps(value, default=str)
            except Exception:
                value = str(value)
        return {"type": "console", "format": "output", "content": value}
    return None
