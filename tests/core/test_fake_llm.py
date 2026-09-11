"""The fake LLM must drive a full turn, including code execution, with no network."""

from tests.support.fake_llm import install_fake_llm


def test_fake_llm_runs_code_and_records_requests(offline_interpreter):
    """A scripted reply with a python block is executed and its output fed back.

    Locks the injection point every other characterization test relies on:
    `llm.completions` is the only seam between the loop and the provider.
    """
    fake = install_fake_llm(
        offline_interpreter,
        ["Sure.\n```python\nprint(21 * 2)\n```", "The answer is 42."],
    )

    messages = offline_interpreter.chat("what is 21*2", display=False, stream=False)

    outputs = [m["content"] for m in messages if m.get("type") == "console"]
    assert any("42" in str(o) for o in outputs)
    assert messages[-1]["content"] == "The answer is 42."
    assert len(fake.calls) == 2
    assert fake.calls[0]["messages"][0]["role"] == "system"
