"""Run a task in a second interpreter that matches this session's settings.

The model is told it can spawn sub-agents in Python, and writing that by hand
is three lines -- until you notice that a fresh `OpenInterpreter()` is *fresh*:
the kernel is a separate process, so its instance carries the defaults
(`gpt-4o-mini`, no api_base) rather than the session's model. A sub-agent
spawned from a local or self-hosted session therefore calls OpenAI instead, on
whatever key is in the environment, and the only symptom is the bill.

This reads the settings the session published into the kernel's environment
(see JupyterLanguage._kernel_env) so a sub-agent runs on the same model as its
parent. It also stops the kernel it started, which hand-written code reliably
forgets -- each sub-agent owns one, and a leaked kernel holds its sockets until
the process dies.
"""

import os

# Settings the session publishes into the kernel's environment, and that a
# sub-agent reads back. Both halves live here so the names cannot drift apart.
_INHERITED = {
    "OI_LLM_MODEL": "model",
    "OI_LLM_API_BASE": "api_base",
    "OI_LLM_API_KEY": "api_key",
    "OI_LLM_API_VERSION": "api_version",
}
_INHERITED_NUMERIC = {
    "OI_LLM_CONTEXT_WINDOW": "context_window",
    "OI_LLM_MAX_TOKENS": "max_tokens",
}


def kernel_env(interpreter):
    """The environment for a kernel: ours, plus this session's model settings.

    Read off the live Llm rather than the profile file: a `.py` profile carries
    its configuration as executed code with no `llm` section to copy, and
    command-line overrides never reach a file at all. They travel in the
    environment rather than in the kernel's setup code because that code is
    printed when the toolbox runs verbose, and the api key would go with it.
    """
    env = dict(os.environ)
    llm = getattr(interpreter, "llm", None)
    for variable, attribute in {**_INHERITED, **_INHERITED_NUMERIC}.items():
        value = getattr(llm, attribute, None)
        # Left unset rather than written as "None": the reader checks for
        # presence, and an empty api_base is not the same as no api_base.
        if value not in (None, ""):
            env[variable] = str(value)
    return env

# A sub-agent that spawns sub-agents is nearly always a mistake, and an
# expensive one: nothing in a chain of them stops until a budget does.
_DEPTH_VARIABLE = "OI_SUBAGENT_DEPTH"
_MAX_DEPTH = 1


class SubagentError(Exception):
    """Raised when a sub-agent cannot be run."""


def _inherit(llm):
    """Copy the parent session's model settings onto a new Llm."""
    for variable, attribute in _INHERITED.items():
        value = os.environ.get(variable)
        if value:
            setattr(llm, attribute, value)
    for variable, attribute in _INHERITED_NUMERIC.items():
        value = os.environ.get(variable)
        if value:
            try:
                setattr(llm, attribute, int(value))
            except ValueError:
                pass


def build(model=None, **settings):
    """An interpreter configured like this session, without running anything.

    Use it when the task needs more than one message, or settings this
    function does not take. Remember to call `.computer.terminate()` when you
    are done with it, which `run()` does for you.
    """
    from interpreter import OpenInterpreter

    depth = int(os.environ.get(_DEPTH_VARIABLE, "0"))
    if depth >= _MAX_DEPTH:
        raise SubagentError(
            f"Already {depth} sub-agent(s) deep. A sub-agent spawning sub-agents "
            f"runs away quietly and the bill is the first sign, so this is capped "
            f"at {_MAX_DEPTH}. Do this work directly instead."
        )

    interpreter = OpenInterpreter()
    _inherit(interpreter.llm)
    if model:
        interpreter.llm.model = model
    interpreter.auto_run = True
    interpreter.offline = True  # no telemetry or version checks for a child
    interpreter.conversation_history = False
    for key, value in settings.items():
        setattr(interpreter, key, value)
    return interpreter


def run(task, model=None, **settings):
    """Run one task to completion and return the sub-agent's final reply.

    Each sub-agent owns its own kernel, so several of these run in parallel
    happily -- a ThreadPoolExecutor over a list of tasks is the usual shape.
    """
    interpreter = build(model=model, **settings)
    # Children see the incremented depth and refuse to spawn further.
    previous = os.environ.get(_DEPTH_VARIABLE)
    os.environ[_DEPTH_VARIABLE] = str(int(previous or "0") + 1)
    try:
        messages = interpreter.chat(task, display=False, stream=False)
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("type") == "message":
                return message.get("content", "")
        return ""
    finally:
        if previous is None:
            os.environ.pop(_DEPTH_VARIABLE, None)
        else:
            os.environ[_DEPTH_VARIABLE] = previous
        try:
            interpreter.computer.terminate()
        except Exception:
            pass
