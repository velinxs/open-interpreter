"""Errors the LLM layer raises for respond.py to handle.

These exist so the turn loop can tell a model problem from a provider problem
without matching on message text. respond.py catches them where it can retry,
fall back, or tell the user something specific; everything else it treats as a
provider error and renders with the usual panel.
"""


class FunctionCallingNotSupportedError(Exception):
    """The model was asked for tool calls and cannot do them.

    Raised mid-request, because some providers accept the tools parameter and
    only fail once the model replies. Llm.run() catches it, turns off
    supports_functions, and re-runs the turn in markdown mode rather than
    losing it.
    """


class ModelNotFoundError(Exception):
    """The model name does not exist at this provider, so retrying cannot help."""


class AccessDeniedError(Exception):
    """The key is valid but not entitled to this model (billing, tier, region)."""
