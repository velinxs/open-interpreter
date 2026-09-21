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
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import interpreter.core.subagent as subagent_module
from interpreter.core.llm.session_env import DEPTH_VARIABLE as _DEPTH_VARIABLE
from interpreter.core.llm.session_env import apply_to, kernel_env
from interpreter.core.subagent import SubagentError, build
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


def test_sub_agents_started_side_by_side_are_siblings_not_nested(monkeypatch):
    """Several sub-agents from one kernel are all depth 1, not 1, 2, 3...

    The depth counter used to live in os.environ and be incremented around each
    run, which is process-global: three of these started together raced, and
    whichever ran second saw the first one's increment and was refused as "too
    deep". The docstring recommends exactly this ThreadPoolExecutor shape, so
    the guard was rejecting the documented usage.
    """
    monkeypatch.delenv(_DEPTH_VARIABLE, raising=False)
    monkeypatch.setenv("OI_LLM_MODEL", "ollama/qwen3")

    with ThreadPoolExecutor(max_workers=3) as pool:
        children = list(pool.map(lambda _: build(), range(3)))

    assert len(children) == 3
    # Each is one level below the kernel that built them -- all the same level.
    assert [child._subagent_depth for child in children] == [1, 1, 1]


def test_a_childs_kernel_is_told_it_is_one_level_down(monkeypatch):
    """The depth reaches the child through its own kernel, not the caller's env.

    That is what makes the cap work without a shared counter: the child's
    kernel carries depth 1, so a sub-agent started inside it is refused.
    """
    monkeypatch.delenv(_DEPTH_VARIABLE, raising=False)

    child = build()

    assert kernel_env(child)[_DEPTH_VARIABLE] == "1"
    # ...and the caller's own environment was never touched.
    assert _DEPTH_VARIABLE not in os.environ


def test_running_a_subagent_never_mutates_the_shared_environment(monkeypatch):
    """os.environ must be untouched *while* the sub-agent runs, not just after.

    The depth counter used to be written into os.environ for the duration of
    the run and restored afterwards. Restoring hid it from any check that ran
    after the call -- but os.environ is process-global, so a sibling starting
    during that window read the raised count and was refused as "too deep".
    That is the failure that made a ThreadPoolExecutor over sub-agents blow up
    on the second task, so the observation has to happen mid-flight.
    """
    monkeypatch.delenv(_DEPTH_VARIABLE, raising=False)
    seen = {}

    def _chat(*args, **kwargs):
        # What a concurrent sibling would see at its own build() moment.
        seen["during"] = os.environ.get(_DEPTH_VARIABLE)
        return [{"role": "assistant", "type": "message", "content": "done"}]

    monkeypatch.setattr(
        subagent_module,
        "build",
        lambda model=None, **settings: SimpleNamespace(
            chat=_chat, computer=SimpleNamespace(terminate=lambda: None)
        ),
    )

    assert subagent_module.run("anything") == "done"
    assert seen["during"] is None, (
        f"run() raised the shared depth to {seen['during']!r} mid-flight; a "
        f"sibling starting now would be refused"
    )
    assert _DEPTH_VARIABLE not in os.environ


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


def test_the_kernels_own_interpreter_is_pointed_at_the_session(monkeypatch):
    """apply_session_llm fixes the throwaway instance toolbox.ai delegates to.

    `from interpreter import interpreter` inside the kernel builds a *new*
    singleton in that process, at package defaults. toolbox.ai routes through
    it, so before this an ai.chat() from a local session called OpenAI at
    gpt-4o-mini -- real spend, with nothing in the session to suggest it.
    """
    monkeypatch.setenv("OI_LLM_MODEL", "ollama/qwen3")
    monkeypatch.setenv("OI_LLM_API_BASE", "http://127.0.0.1:11434")

    host = SimpleNamespace(model="gpt-4o-mini", api_base=None, api_key=None,
                           api_version=None, context_window=None, max_tokens=None)
    apply_to(host)

    assert host.model == "ollama/qwen3"
    assert host.api_base == "http://127.0.0.1:11434"


def test_ai2_defaults_to_the_session_model(monkeypatch):
    """ai2 answers helper calls on the launched model, not its own default.

    It used to default to gpt-4.1-nano regardless, so every boolean_query from
    an ollama session was an OpenAI call.
    """
    from interpreter.core.toolbox.ai2 import Ai2

    monkeypatch.setenv("OI_LLM_MODEL", "ollama/qwen3")
    monkeypatch.setenv("OI_LLM_API_BASE", "http://127.0.0.1:11434")
    monkeypatch.delenv("AI2_MODEL", raising=False)

    assert Ai2().default_model == "ollama/qwen3"
    # An explicit choice still wins, so a job can pick the model that suits it.
    assert Ai2(default_model="openai/gpt-4.1").default_model == "openai/gpt-4.1"
    monkeypatch.setenv("AI2_MODEL", "openai/gpt-4.1-nano")
    assert Ai2().default_model == "openai/gpt-4.1-nano"


def test_ai2_falls_back_when_there_is_no_session(monkeypatch):
    """Imported outside a session there is nothing to inherit, so keep a default."""
    from interpreter.core.toolbox.ai2 import Ai2

    for variable in ("OI_LLM_MODEL", "OI_LLM_API_BASE", "AI2_MODEL"):
        monkeypatch.delenv(variable, raising=False)

    assert Ai2().default_model == "gpt-4.1-nano"


def test_a_plain_openinterpreter_inherits_the_session(monkeypatch):
    """The naive SDK snippet is enough on its own -- no helper required.

    This is the point of applying the session in Llm.__init__ rather than in a
    wrapper: someone who writes the obvious three lines in a kernel gets the
    session's model, instead of silently getting gpt-4o-mini and a bill.
    """
    monkeypatch.setenv("OI_LLM_MODEL", "ollama/qwen3")
    monkeypatch.setenv("OI_LLM_API_BASE", "http://127.0.0.1:11434")
    from interpreter import OpenInterpreter

    plain = OpenInterpreter()

    assert plain.llm.model == "ollama/qwen3"
    assert plain.llm.api_base == "http://127.0.0.1:11434"
