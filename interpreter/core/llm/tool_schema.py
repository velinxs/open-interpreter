"""The tool definitions sent to function-calling models.

execute runs code, edit applies a file edit, view_image shows the model an
image. build_request_tools() picks which of them a given turn offers.
"""

import copy

from ..terminal.base_language import format_execute_language_description
from .tool_messages import _inline_user_image_in_turn_after_last_assistant_text

tool_schema = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": "Executes code on the user's machine **in the user's local environment** and returns the output",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "The programming language (required parameter to the `execute` function)",
                    "enum": [
                        # This will be filled dynamically with the languages OI has access to.
                    ],
                },
                "code": {
                    "type": "string",
                    "description": "The code to execute (required)",
                },
            },
            "required": ["language", "code"],
        },
    },
}

from ..tools.file_edit import EDIT_LANGUAGES

EDIT_LANGUAGES_ENUM = sorted(EDIT_LANGUAGES)

# Raster image formats supported by view_image (vision APIs and Pillow). PDF and other documents are not supported.
VIEW_IMAGE_ALLOWED_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "gif", "webp", "bmp"})

view_image_tool_schema = {
    "type": "function",
    "function": {
        "name": "view_image",
        "description": "Load an image from disk so you can see it. Only for files that actually exist on the user's machine at an absolute path they gave you or that code wrote. Do not invent paths (e.g. /mnt/data/, /tmp/placeholder). If the user already attached an image in this chat (inline / base64), describe that attachment directly—do not call view_image. Supported formats: PNG, JPEG, GIF, WebP, BMP. PDF is not supported. Path must be absolute (Windows: r'C:\\Users\\...', Mac/Linux: '/home/...').",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to an image file. Supported: PNG, JPEG, GIF, WebP, BMP. Must be an absolute path.",
                },
            },
            "required": ["path"],
        },
    },
}


edit_tool_schema = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Edit or create a file at an absolute path.\n"
            "Languages:\n"
            "  write — create a NEW file; code is the full file body (UTF-8). Errors if target exists.\n"
            "  sed   — sed script (in-place); line-oriented text edits, multi-line OK. Use for regex search/replace.\n"
            "  gawk  — GNU awk program (in-place). General text processing, columnar data, multi-line patterns.\n"
            "  jq    — jq filter for JSON files (in-place). Wrap if-expressions in parentheses in object literals.\n"
            "  yq    — yq (mikefarah) expression (multi-line OK). Supports YAML, JSON, TOML, XML, CSV, "
            "properties files.\n"
            "  poke  — GNU poke statements for binary files; do not use .file (auto-opened).\n"
            "  comby — structural match/replace: match template, then --- line, then rewrite "
            "(or match on line 1, rewrite on following lines).\n"
            "  patch — unified diff body to apply to the existing target file.\n"
            "Rules: target must be absolute. Never wrap these in bash."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": EDIT_LANGUAGES_ENUM,
                    "description": "write | sed | gawk | jq | yq | poke | comby | patch",
                },
                "code": {
                    "type": "string",
                    "description": "File body (write) or language-specific edit code",
                },
                "target": {
                    "type": "string",
                    "description": "Absolute path to the file",
                },
            },
            "required": ["language", "code", "target"],
        },
    },
}


def build_request_tools(interpreter, messages=None):
    """Build the ``tools`` array sent on the chat completions API request (deep copy)."""
    import copy

    languages = interpreter.terminal.languages
    execute_tool = copy.deepcopy(tool_schema)
    execute_tool["function"]["parameters"]["properties"]["language"]["enum"] = [lang.name.lower() for lang in languages]
    execute_tool["function"]["parameters"]["properties"]["language"]["description"] = (
        format_execute_language_description(languages)
    )
    tools = [execute_tool, copy.deepcopy(edit_tool_schema)]
    if getattr(interpreter.llm, "supports_vision", None) is True:
        if messages is None or not _inline_user_image_in_turn_after_last_assistant_text(messages):
            tools.append(copy.deepcopy(view_image_tool_schema))
    return tools
