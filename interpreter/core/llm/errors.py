"""Errors the LLM layer raises for respond.py to handle."""


class FunctionCallingNotSupportedError(Exception):
    """Raised when a model doesn't support function calling"""

    pass


class ModelNotFoundError(Exception):
    """Raised when a model doesn't exist or isn't accessible"""

    pass


class AccessDeniedError(Exception):
    """Raised when access to a model is denied"""

    pass
