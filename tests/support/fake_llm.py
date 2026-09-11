"""A scripted stand-in for Llm.completions.

Llm.run() only ever touches the provider through `llm.completions(**params)`,
which normally is litellm.completion. Replacing that one attribute lets the
whole loop (system message assembly, trimming, code execution, output
feedback) run for real while the model's replies come from a list.
"""


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # params dict per request, in order

    def __call__(self, **params):
        self.calls.append(params)
        if not self.replies:
            raise AssertionError("FakeCompletions ran out of scripted replies")
        text = self.replies.pop(0)
        step = max(1, len(text) // 3)  # a few chunks, like a real stream
        for i in range(0, len(text), step):
            yield {"choices": [{"delta": {"content": text[i : i + step]}}]}
        yield {"choices": [{"delta": {}}]}


def install_fake_llm(interpreter, replies):
    fake = FakeCompletions(replies)
    llm = interpreter.llm
    llm.completions = fake
    llm.model = "openai/fake"
    llm.api_key = "fake"
    llm.api_base = "http://127.0.0.1:9"
    llm.supports_functions = False
    llm.supports_vision = False
    llm.context_window = 32000
    llm.max_tokens = 4096
    llm._is_loaded = True
    interpreter.offline = True
    interpreter.disable_telemetry = True
    interpreter.auto_run = True
    return fake
