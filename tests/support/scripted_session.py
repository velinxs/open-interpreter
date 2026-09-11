"""A fixed six-turn session for measuring prompt tokens per request.

Turns: run python, produce long console output, hit an error and recover,
write a file, read it back, and a plain chat reply. The fake replies are
constant, so any change in token counts comes from the code, not the model.
"""

import os

import tiktoken

from tests.support.fake_llm import install_fake_llm

SCRIPT = [
    ("print hello", ["```python\nprint('hello')\n```", "Done."]),
    (
        "print 400 numbered lines",
        ["```python\nfor i in range(400):\n    print('line', i)\n```", "That was long."],
    ),
    (
        "import a module that does not exist, then recover",
        [
            "```python\nimport nonexistent_module_xyz\n```",
            "```python\nprint('recovered')\n```",
            "Recovered.",
        ],
    ),
    (
        "write the word ok to notes.txt",
        ["```python\nopen('notes.txt', 'w').write('ok')\n```", "Written."],
    ),
    ("read notes.txt back", ["```python\nprint(open('notes.txt').read())\n```", "It says ok."]),
    ("thanks", ["You're welcome."]),
]

_enc = tiktoken.get_encoding("cl100k_base")


def count_prompt_tokens(messages):
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(_enc.encode(content))
        elif isinstance(content, list):  # vision-style parts
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(_enc.encode(part["text"]))
    return total


def run_scripted_session(interpreter, workdir):
    """Drive SCRIPT through the interpreter; return prompt tokens per request."""
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        fake = install_fake_llm(interpreter, [r for _, replies in SCRIPT for r in replies])
        for user_message, _ in SCRIPT:
            interpreter.chat(user_message, display=False, stream=False)
    finally:
        os.chdir(cwd)
    return [count_prompt_tokens(call["messages"]) for call in fake.calls]
