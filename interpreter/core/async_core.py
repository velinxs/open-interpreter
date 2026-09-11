"""Compatibility path. The server lives in interpreter.server since the rework."""

from ..server import AsyncInterpreter, Server, create_router
from ..server.auth import authenticate_function
from ..server.openai_compat import (
    OPENAI_CODE_APPROVAL_DECLINED,
    OPENAI_CODE_APPROVAL_INVALID_TEMPLATE,
    OPENAI_CODE_APPROVAL_PROMPT,
    OPENAI_SHELL_OUTPUT_NOTE,
)

__all__ = [
    "AsyncInterpreter",
    "Server",
    "create_router",
    "authenticate_function",
    "OPENAI_CODE_APPROVAL_DECLINED",
    "OPENAI_CODE_APPROVAL_INVALID_TEMPLATE",
    "OPENAI_CODE_APPROVAL_PROMPT",
    "OPENAI_SHELL_OUTPUT_NOTE",
]
