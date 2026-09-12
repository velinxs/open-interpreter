"""The System Information block: what it says, and where it sits.

It answers two questions the model could not answer before — what am I, and
where will my code run — and it sits at the very end of the prompt because the
working directory is the one line that can change mid-session.
"""

import os
import tempfile

from interpreter.core.utils.assemble_system_message import assemble_system_message


def test_the_prompt_names_the_model_and_the_working_directory(offline_interpreter):
    """Asked what it runs on, the model should not have to invent an answer."""
    offline_interpreter.llm.model = "ollama_chat/qwen-thinking:latest"

    prompt = assemble_system_message(offline_interpreter)

    assert "- You are: ollama_chat/qwen-thinking:latest" in prompt
    assert f"- $PWD: {os.getcwd()}" in prompt


def test_system_information_is_last(offline_interpreter):
    """Everything stable sits in front of the one volatile line."""
    offline_interpreter.toolbox.import_toolbox_api = True

    prompt = assemble_system_message(offline_interpreter)
    after = prompt.rsplit("## System Information", 1)[1]

    assert "toolbox" not in after, "the toolbox listing must come before the volatile block"
    assert after.strip().endswith(os.getcwd())


def test_changing_directory_only_changes_the_tail(offline_interpreter):
    """A cd must not invalidate the cached prefix of the prompt.

    Providers cache a prefix, so a working directory named early would throw
    away the instructions, the language notes and the toolbox listing on every
    cd. Named last, it costs only its own line.
    """
    offline_interpreter.toolbox.import_toolbox_api = True
    before = assemble_system_message(offline_interpreter)

    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    try:
        after = assemble_system_message(offline_interpreter)
    finally:
        os.chdir(cwd)

    shared = len(os.path.commonprefix([before, after]))
    assert shared > len(before) - 200, f"a cd invalidated {len(before) - shared} characters of prompt"


def test_a_deleted_working_directory_does_not_crash_the_prompt(offline_interpreter):
    """os.getcwd() raises if the directory was removed under us; say so instead."""
    from interpreter.core.default_system_message import working_directory

    cwd = os.getcwd()
    doomed = tempfile.mkdtemp()
    os.chdir(doomed)
    try:
        os.rmdir(doomed)
        assert "unknown" in working_directory()
    except OSError:
        pass  # some platforms refuse to remove the directory you are standing in
    finally:
        os.chdir(cwd)
