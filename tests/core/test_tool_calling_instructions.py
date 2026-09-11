import unittest

from interpreter.core.llm.llm import _TOOL_CALLING_INSTRUCTIONS


class TestToolCallingInstructions(unittest.TestCase):
    def test_instructions_do_not_duplicate_tool_schemas(self):
        self.assertNotIn("Two tools are available", _TOOL_CALLING_INSTRUCTIONS)
        self.assertNotIn("persistent REPL", _TOOL_CALLING_INSTRUCTIONS)
        self.assertNotIn("write`, `sed`", _TOOL_CALLING_INSTRUCTIONS)
        self.assertIn("JSON schema", _TOOL_CALLING_INSTRUCTIONS)
        self.assertIn("internal storage", _TOOL_CALLING_INSTRUCTIONS)


def test_only_one_mode_of_execution_instructions_reaches_the_request(offline_interpreter):
    """A request carries the tool schemas or the markdown instructions, never both.

    Both describe how to run code; sending them together wastes the smaller one's
    tokens on every request and gives the model two contradictory protocols.
    """
    from tests.support.fake_llm import install_fake_llm

    fake = install_fake_llm(offline_interpreter, ["Hello.", "Hello again."])

    offline_interpreter.llm.supports_functions = False
    offline_interpreter.chat("hi", display=False, stream=False)
    text_mode = fake.calls[-1]

    assert "tools" not in text_mode
    assert offline_interpreter.llm.execution_instructions in text_mode["messages"][0]["content"]
    assert _TOOL_CALLING_INSTRUCTIONS not in text_mode["messages"][0]["content"]

    offline_interpreter.reset()
    offline_interpreter.llm.supports_functions = True
    offline_interpreter.chat("hi", display=False, stream=False)
    tool_mode = fake.calls[-1]

    assert tool_mode["tools"], "tool mode should send the schemas"
    assert _TOOL_CALLING_INSTRUCTIONS in tool_mode["messages"][0]["content"]
    assert offline_interpreter.llm.execution_instructions not in tool_mode["messages"][0]["content"]


if __name__ == "__main__":
    unittest.main()
