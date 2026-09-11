"""The system prompt: size, and that it is identical on every request.

The prompt is the largest fixed cost of a session (it is re-sent with every
request), and an identical prefix is what lets a provider serve it from its
prompt cache. A rule change is fine; growth and per-turn variation are not.
"""

import os
import tempfile

import tiktoken

from tests.support.fake_llm import install_fake_llm
from tests.support.scripted_session import SCRIPT

# Includes the per-language notes appended after the base prompt. The budget is
# a guard against drift, not a target: it buys the worked command shapes in
# Execution Style and the line naming the running model, both of which earn
# their tokens back (one truncated output costs ~700 tokens and rides along in
# every later request; without the model line the model invents an answer when
# asked what it runs on). Raise it only for a stated addition, never to make a
# red test green.
SYSTEM_PROMPT_TOKEN_BUDGET = 1650
_enc = tiktoken.get_encoding("cl100k_base")


def _run_session(interpreter):
    install_fake_llm(interpreter, [r for _, replies in SCRIPT for r in replies])
    workdir = tempfile.mkdtemp()
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        for message, _ in SCRIPT:
            interpreter.chat(message, display=False, stream=False)
    finally:
        os.chdir(cwd)
    return interpreter.llm.completions.calls


def test_system_prompt_is_byte_identical_on_every_request(offline_interpreter):
    """One distinct system message per session, so the cached prefix keeps matching."""
    calls = _run_session(offline_interpreter)

    distinct = {call["messages"][0]["content"] for call in calls}
    assert len(calls) == 12
    assert len(distinct) == 1, f"{len(distinct)} different system prompts in one session"


def test_system_prompt_stays_within_budget(offline_interpreter):
    """The prompt sent with every request stays under the budget."""
    calls = _run_session(offline_interpreter)

    prompt = calls[0]["messages"][0]["content"]
    tokens = len(_enc.encode(prompt))
    print(f"\nsystem prompt: {tokens} tokens")
    assert tokens <= SYSTEM_PROMPT_TOKEN_BUDGET, f"{tokens} tokens exceeds {SYSTEM_PROMPT_TOKEN_BUDGET}"
