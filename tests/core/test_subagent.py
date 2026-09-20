"""A sub-agent must run on the session's model, not on the default.

The kernel is a separate process, so `OpenInterpreter()` written there builds a
*fresh* instance carrying the package defaults -- `gpt-4o-mini` and no
api_base. A sub-agent spawned from a local or self-hosted session would
therefore call OpenAI, on whatever key happens to be in the environment, and
nothing says so: the run succeeds and the only evidence is the bill.

The session publishes its settings into the kernel's environment and the
helper reads them back, so these tests work on that environment rather than on
a live kernel.
"""

import os
from types import SimpleNamespace

import pytest

from interpreter.core.subagent import _DEPTH_VARIABLE, SubagentError, build, kernel_env
from interpreter.core.terminal.languages.jupyter_language import JupyterLanguage


def _session_llm(**overrides):
    settings = {
        "model": "ollama/qwen3",
        "api_base": "http://127.0.0.1:11434",
        "api_key": "session-key",
        "api_version": None,
        "context_window": 32000,
        "max_tokens": 4096,
    }
    settings.update(overrides)
    return SimpleNamespace(**settings)


def test_the_session_publishes_its_model_to_the_kernel():
    """The kernel's environment carries the settings a sub-agent needs.

    Read off the live Llm rather than the profile file on purpose: a `.py`
    profile has no llm section to copy, and --model overrides never reach a
    file at all.
    """
    env = kernel_env(SimpleNamespace(llm=_session_llm()))

    assert env["OI_LLM_MODEL"] == "ollama/qwen3"
    assert env["OI_LLM_API_BASE"] == "http://127.0.0.1:11434"
    assert env["OI_LLM_CONTEXT_WINDOW"] == "32000"
    # The rest of the environment is still there; the kernel needs PATH et al.
    assert "PATH" in env


def test_an_unset_setting_is_left_out_rather_than_sent_as_none():
    """A missing api_base must be absent, not the string "None".

    Written as "None" it would be inherited as a real api_base and every
    sub-agent request would go to a host by that name.
    """
    env = kernel_env(SimpleNamespace(llm=_session_llm(api_base=None, api_version=None)))

    assert "OI_LLM_API_BASE" not in env
    assert "OI_LLM_API_VERSION" not in env


@pytest.mark.timeout(120)
def test_a_real_kernel_receives_the_settings():
    """The settings survive the trip into an actual kernel process.

    kernel_env only builds a dict; whether start_kernel honours `env` is
    jupyter_client's business, and if it ever stops doing so every sub-agent
    silently falls back to the default model again.
    """
    language = JupyterLanguage(
        SimpleNamespace(llm=_session_llm(), verbose=False, debug=False)
    )
    try:
        output = "".join(
            str(chunk.get("content", ""))
            for chunk in language.run(
                "import os, json\n"
                "print(json.dumps({k: v for k, v in os.environ.items() "
                "if k.startswith('OI_LLM_')}))"
            )
        )
    finally:
        language.terminate()

    assert '"OI_LLM_MODEL": "ollama/qwen3"' in output, output
    assert '"OI_LLM_API_BASE": "http://127.0.0.1:11434"' in output, output


def test_a_subagent_inherits_the_session_model(monkeypatch):
    """build() produces an interpreter pointed at the session's model.

    Without this it would carry the package default, gpt-4o-mini, and bill
    OpenAI from a session that never intended to use it.
    """
    monkeypatch.setenv("OI_LLM_MODEL", "ollama/qwen3")
    monkeypatch.setenv("OI_LLM_API_BASE", "http://127.0.0.1:11434")
    monkeypatch.setenv("OI_LLM_API_KEY", "session-key")
    monkeypatch.setenv("OI_LLM_CONTEXT_WINDOW", "32000")
    monkeypatch.delenv(_DEPTH_VARIABLE, raising=False)

    child = build()

    assert child.llm.model == "ollama/qwen3"
    assert child.llm.api_base == "http://127.0.0.1:11434"
    assert child.llm.api_key == "session-key"
    assert child.llm.context_window == 32000
    assert child.auto_run is True


def test_an_explicit_model_overrides_the_inherited_one(monkeypatch):
    """A task that needs a different model can still ask for one."""
    monkeypatch.setenv("OI_LLM_MODEL", "ollama/qwen3")
    monkeypatch.delenv(_DEPTH_VARIABLE, raising=False)

    child = build(model="openai/gpt-4.1")

    assert child.llm.model == "openai/gpt-4.1"


def test_a_subagent_cannot_spawn_another(monkeypatch):
    """Depth is capped, because a chain of sub-agents stops only at a budget."""
    monkeypatch.setenv(_DEPTH_VARIABLE, "1")

    with pytest.raises(SubagentError, match="deep"):
        build()


def test_the_default_model_is_the_thing_being_avoided(monkeypatch):
    """Pins why this exists: a bare interpreter really does default to gpt-4o-mini.

    This is the failure being prevented -- a sub-agent written the obvious way
    in the kernel gets this instance. If the default ever becomes something
    harmless, this test should be what notices, rather than the inheritance
    machinery quietly guarding nothing.
    """
    from interpreter import OpenInterpreter

    for variable in list(os.environ):
        if variable.startswith("OI_LLM_"):
            monkeypatch.delenv(variable, raising=False)

    assert OpenInterpreter().llm.model == "gpt-4o-mini"
