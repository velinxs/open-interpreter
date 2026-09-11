"""Profile that replaces the model with a two-reply script. Used by the CLI smoke test."""

from interpreter import interpreter

_replies = ["Sure.\n```python\nprint(21 * 2)\n```", "The answer is 42."]


def _fake_completions(**params):
    text = _replies.pop(0)
    yield {"choices": [{"delta": {"content": text}}]}
    yield {"choices": [{"delta": {}}]}


interpreter.llm.completions = _fake_completions
interpreter.llm.model = "openai/fake"
interpreter.llm.api_key = "fake"
interpreter.llm.supports_functions = False
interpreter.llm.supports_vision = False
interpreter.llm.context_window = 32000
interpreter.llm.max_tokens = 4096
interpreter.llm._is_loaded = True
interpreter.offline = True
interpreter.disable_telemetry = True
interpreter.auto_run = True
