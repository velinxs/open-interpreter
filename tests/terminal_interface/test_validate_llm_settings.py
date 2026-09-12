"""The last gate before a session starts: is there a usable model?

validate_llm_settings runs on every interactive launch. If it prompts when it
should not, the CLI hangs in a pipeline; if it stays silent when a key is
missing, the user's first message dies with a provider error instead of a
question they can answer.
"""

import pytest

import interpreter.terminal_interface.validate_llm_settings as validate_module
from interpreter.terminal_interface.validate_llm_settings import (
    display_welcome_message_once,
    validate_llm_settings,
)


class FakeLlm:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.api_key = None
        self.api_base = None
        self.loads = 0

    def load(self):
        self.loads += 1


class FakeInterpreter:
    def __init__(self, model="gpt-4o", offline=False, auto_run=False, messages=None):
        self.llm = FakeLlm(model)
        self.offline = offline
        self.auto_run = auto_run
        self.messages = messages if messages is not None else []
        self.displayed = []

    def display_message(self, message):
        self.displayed.append(message)


@pytest.fixture(autouse=True)
def _no_waiting_and_no_key(monkeypatch):
    """Strip the sleeps, the ambient API key, and the once-only welcome flag.

    display_welcome_message_once memoises on a function attribute, which
    outlives the test that set it; leaving it in place would make a later test
    silently take a different branch.
    """
    monkeypatch.setattr(validate_module.time, "sleep", lambda *_: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delattr(display_welcome_message_once, "_displayed", raising=False)


@pytest.fixture
def never_prompts(monkeypatch):
    """Fail loudly if the code asks the user anything."""

    def _prompt(*args, **kwargs):
        raise AssertionError("validate_llm_settings prompted when it should not have")

    monkeypatch.setattr(validate_module, "prompt", _prompt)


def test_offline_mode_never_asks_for_a_key(never_prompts):
    """--offline short-circuits the whole check.

    An offline run points at a local server that needs no key. Prompting would
    block a setup that is already complete, and the prompt has no way out but
    Ctrl-C.
    """
    interpreter = FakeInterpreter(model="gpt-4o", offline=True)
    validate_llm_settings(interpreter)
    assert interpreter.llm.api_key is None


def test_an_unknown_model_is_not_interrogated(never_prompts):
    """Only the handful of listed OpenAI models trigger the key prompt.

    Every other model — a local one, an Anthropic one, an OpenRouter one —
    carries its own credentials elsewhere, so asking for an OpenAI key would
    be both useless and confusing.
    """
    interpreter = FakeInterpreter(model="anthropic/claude-sonnet-4-6")
    validate_llm_settings(interpreter)
    assert interpreter.llm.api_key is None


@pytest.mark.parametrize("model", ["gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
def test_listed_openai_models_ask_for_a_key_and_keep_it(monkeypatch, model):
    """A listed OpenAI model with no credentials prompts, and the answer is used.

    This is the first-run experience. If the typed key were not assigned to
    llm.api_key, the user would answer the prompt and still get an
    authentication error on their first message.
    """
    monkeypatch.setattr(validate_module, "prompt", lambda *a, **kw: "sk-typed-in")
    interpreter = FakeInterpreter(model=model)
    validate_llm_settings(interpreter)
    assert interpreter.llm.api_key == "sk-typed-in"


@pytest.mark.parametrize("preset", ["env", "api_key", "api_base"])
def test_existing_credentials_suppress_the_prompt(monkeypatch, never_prompts, preset):
    """Any of an env var, a configured key, or a custom api_base counts as "set up".

    api_base is included because a local OpenAI-compatible server needs no
    key at all; treating it as unconfigured would prompt on every launch of a
    working local setup.
    """
    interpreter = FakeInterpreter(model="gpt-4o")
    if preset == "env":
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    elif preset == "api_key":
        interpreter.llm.api_key = "sk-configured"
    else:
        interpreter.llm.api_base = "http://localhost:1234/v1"

    validate_llm_settings(interpreter)


def test_typing_the_local_command_at_the_prompt_exits(monkeypatch, capsys):
    """Answering "interpreter --local" escapes the key prompt instead of storing it as a key.

    Someone who does not have an OpenAI key needs a way out of a prompt that
    otherwise only accepts one. Storing that string as the API key would send
    it to OpenAI as a credential.
    """
    monkeypatch.setattr(validate_module, "prompt", lambda *a, **kw: "interpreter --local")
    interpreter = FakeInterpreter(model="gpt-4o")
    with pytest.raises(SystemExit):
        validate_llm_settings(interpreter)
    assert "interpreter --local" in capsys.readouterr().out


def test_the_chosen_model_is_announced_for_an_interactive_session(never_prompts):
    """An ordinary interactive launch says which model it is about to use.

    It is the only confirmation that a profile or flag took effect, and
    getting it wrong (or silently loading a different model) is the single
    most common source of "why is it behaving like that".
    """
    interpreter = FakeInterpreter(model="anthropic/claude-sonnet-4-6")
    validate_llm_settings(interpreter)
    assert any("claude-sonnet-4-6" in message for message in interpreter.displayed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"auto_run": True},
        {"offline": True},
        {"messages": [{"role": "user", "content": "hi"}]},
    ],
)
def test_the_model_banner_is_suppressed_for_non_interactive_runs(never_prompts, kwargs):
    """auto-run, offline, and the one-shot "i {command}" entry all stay quiet.

    Each is a mode where the output is being read by a script or is meant to
    be a single answer. A banner line would be noise in the first two and
    would prefix the answer in the third.
    """
    interpreter = FakeInterpreter(model="anthropic/claude-sonnet-4-6", **kwargs)
    validate_llm_settings(interpreter)
    assert not any("Model set to" in message for message in interpreter.displayed)


def test_the_hosted_i_model_warns_that_conversations_are_used_for_training(never_prompts):
    """`--model i` says out loud that the chat will be used as training data.

    It is the one model where using it is itself consent. Losing this notice
    would mean data leaves the machine with nothing on screen saying so.
    """
    interpreter = FakeInterpreter(model="i")
    validate_llm_settings(interpreter)
    assert any("train our open-source model" in message for message in interpreter.displayed)


def test_ollama_models_are_loaded_eagerly(never_prompts):
    """An ollama model is loaded here rather than on the first message.

    Loading a local model takes tens of seconds. Deferring it would make the
    user's first message look like a hang with no explanation.
    """
    interpreter = FakeInterpreter(model="ollama/llama3")
    validate_llm_settings(interpreter)
    assert interpreter.llm.loads == 1


def test_non_ollama_models_are_not_loaded_here(never_prompts):
    """Hosted models are not pre-loaded; there is nothing to warm up."""
    interpreter = FakeInterpreter(model="gpt-4o")
    interpreter.llm.api_key = "sk-configured"
    validate_llm_settings(interpreter)
    assert interpreter.llm.loads == 0


def test_the_welcome_banner_is_shown_only_once_per_process(monkeypatch):
    """The "Welcome to Open Interpreter" splash does not repeat.

    It memoises on a function attribute. That state is process-global, so a
    second interpreter constructed in the same process (a test, a server, a
    notebook) must not replay the splash.
    """
    first = FakeInterpreter()
    second = FakeInterpreter()
    display_welcome_message_once(first)
    display_welcome_message_once(second)
    assert len(first.displayed) == 1
    assert second.displayed == []
