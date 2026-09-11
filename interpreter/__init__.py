"""Open Interpreter.

Importing this package has no side effects. The module-level singleton
(`from interpreter import interpreter`) and the public classes are created
on first access via PEP 562 so `import interpreter` stays cheap and never
touches the network or starts a mode.
"""

import importlib
import os
import warnings

# OpenRouter (via LiteLLM): optional HTTP-Referer and app title for openrouter.ai rankings.
os.environ.setdefault("OR_SITE_URL", "https://github.com/velinxs/open-interpreter")
os.environ.setdefault("OR_APP_NAME", "Open Interpreter")

# Suppress pydantic warning from litellm about fields being removed in V2
warnings.filterwarnings(
    "ignore",
    message="Valid config keys have changed in V2:*",
    module="pydantic.*",
)

_CLASSES = {
    "OpenInterpreter": ("interpreter.core.core", "OpenInterpreter"),
    "AsyncInterpreter": ("interpreter.server", "AsyncInterpreter"),
    "BaseLanguage": ("interpreter.core.terminal.base_language", "BaseLanguage"),
}

_singleton = None
_constructing = False


def _get_singleton():
    global _singleton, _constructing
    if _singleton is None:
        if _constructing:
            # OpenInterpreter.__init__ imports modules that may reach back for
            # `interpreter`; without this the recursion would build a second
            # instance and hand out whichever finished last.
            raise RuntimeError("interpreter.interpreter was accessed while it was being constructed")
        _constructing = True
        try:
            module = importlib.import_module("interpreter.core.core")
            _singleton = module.OpenInterpreter()
        finally:
            _constructing = False
    return _singleton


def __getattr__(name):
    if name == "interpreter":
        return _get_singleton()
    if name == "toolbox":
        return _get_singleton().toolbox
    if name == "ai2":
        return _get_singleton().toolbox.ai2
    if name in _CLASSES:
        module_name, attr = _CLASSES[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module 'interpreter' has no attribute {name!r}")


__all__ = ["interpreter", "toolbox", "ai2", "OpenInterpreter", "AsyncInterpreter", "BaseLanguage"]
