"""Carrying a session's model settings across a process boundary.

Code the model writes runs in a Jupyter kernel, which is a separate process.
Anything built there -- a sub-agent, the kernel's own `interpreter` singleton
that `toolbox.ai` delegates to -- starts from the package defaults, because
profiles are applied by the terminal interface and a library that read a user's
config file on import would be badly behaved. Defaults mean `gpt-4o-mini` and
no api_base, so a local or self-hosted session would quietly call OpenAI on
whatever key is in the environment, and the only evidence would be the bill.

So the session publishes its settings into the kernel's environment on the way
out (`kernel_env`) and every Llm reads them back on the way in (`apply_to`,
called from Llm.__init__). An ordinary `pip install` user is unaffected: these
variables only exist inside a kernel an OI session started.

Read off the live Llm rather than the active profile, because the profile is
not a complete record -- a `.py` profile carries its configuration as executed
code with no `llm` section at all, and --model overrides never reach a file.
Carried in the environment rather than in the kernel's setup code because that
code is printed when the toolbox runs verbose, and the api key would go with it.
"""

import os

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

# How deep in sub-agents a kernel is. Published per kernel rather than tracked
# in os.environ, so sub-agents started side by side are siblings at one depth
# instead of racing a counter that is shared by the whole process.
DEPTH_VARIABLE = "OI_SUBAGENT_DEPTH"


def kernel_env(interpreter):
    """The environment for a kernel: ours, plus this session's model settings."""
    env = dict(os.environ)
    llm = getattr(interpreter, "llm", None)
    for variable, attribute in {**_INHERITED, **_INHERITED_NUMERIC}.items():
        value = getattr(llm, attribute, None)
        # Left unset rather than written as "None": apply_to checks for
        # presence, and an empty api_base is not the same as no api_base.
        if value not in (None, ""):
            env[variable] = str(value)
    env[DEPTH_VARIABLE] = str(getattr(interpreter, "_subagent_depth", 0))
    return env


def apply_to(llm):
    """Point an Llm at the session's model, if one published its settings.

    Called from Llm.__init__, so `OpenInterpreter()` written in a kernel
    inherits the session without the author having to know any of this. Outside
    a session nothing is set and the package defaults stand.
    """
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
