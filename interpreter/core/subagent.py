"""Run a task in a second interpreter.

`OpenInterpreter()` already inherits the session's model when it is built
inside a kernel the session started (see llm/session_env.py), so a sub-agent
written by hand is three honest lines. This exists for the two things those
three lines leave out:

    - stopping the kernel afterwards, which each sub-agent owns and which
      hand-written code reliably forgets; a leaked kernel holds its sockets
      until the process dies.
    - refusing to nest, since a chain of sub-agents spawning sub-agents stops
      only when a budget does.

Nothing here is required. `from interpreter import OpenInterpreter` still
works, and now runs on the right model.
"""

import os

from .llm.session_env import DEPTH_VARIABLE

_MAX_DEPTH = 1


class SubagentError(Exception):
    """Raised when a sub-agent cannot be run."""


def build(model=None, **settings):
    """An interpreter configured like this session, without running anything.

    Use it when the task needs more than one message, or settings this function
    does not take. Call `.computer.terminate()` when done, which `run()` does
    for you.
    """
    from interpreter import OpenInterpreter

    depth = int(os.environ.get(DEPTH_VARIABLE, "0"))
    if depth >= _MAX_DEPTH:
        raise SubagentError(
            f"Already {depth} sub-agent(s) deep. A sub-agent spawning sub-agents "
            f"runs away quietly and the bill is the first sign, so this is capped "
            f"at {_MAX_DEPTH}. Do this work directly instead."
        )

    interpreter = OpenInterpreter()  # inherits the session's model on its own
    # A level below the kernel that built it. Recorded on the object so it
    # reaches only this sub-agent's own kernel; nothing in the caller's
    # environment changes, so sub-agents built side by side stay siblings
    # rather than tripping over a shared counter.
    interpreter._subagent_depth = depth + 1
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
    try:
        messages = interpreter.chat(task, display=False, stream=False)
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("type") == "message":
                return message.get("content", "")
        return ""
    finally:
        try:
            interpreter.computer.terminate()
        except Exception:
            pass
