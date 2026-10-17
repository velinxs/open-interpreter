import unittest

from interpreter.core.default_system_message import default_system_message
from interpreter.core.llm.run_function_calling_llm import function_schema
from interpreter.core.llm.tool_schema import tool_schema
from interpreter.core.terminal.base_language import (
    format_execute_language_description,
)
from interpreter.core.terminal.languages.bash import Bash
from interpreter.core.terminal.languages.html import HTML
from interpreter.core.terminal.languages.java import Java
from interpreter.core.terminal.languages.python import Python


class TestExecutionStatePrompt(unittest.TestCase):
    def test_tool_schema_requires_incremental_persistent_calls(self):
        """Verify the current execute schema tells models to continue live sessions rather than restart them."""
        code_description = tool_schema["function"]["parameters"]["properties"]["code"]["description"]
        self.assertIn("next missing operation", code_description)
        self.assertIn("Jupyter/IPython kernel", code_description)
        self.assertIn("stateless/display-only", code_description)

    def test_legacy_function_schema_matches_tool_schema_contract(self):
        """Verify older function-calling models receive the same persistent-session contract."""
        code_description = function_schema["parameters"]["properties"]["code"]["description"]
        self.assertIn("next missing operation", code_description)
        self.assertIn("Jupyter/IPython kernel", code_description)
        self.assertIn("stateless/display-only", code_description)

    def test_language_description_covers_all_execution_modes(self):
        """Verify the dynamic language list distinguishes persistent, stateless, and display-only execution."""
        description = format_execute_language_description([Python, Bash, Java, HTML])
        self.assertIn("persistent REPL", description)
        self.assertIn("next missing operation", description)
        self.assertIn("Jupyter/IPython kernel", description)
        self.assertIn("complete, self-contained block", description)
        self.assertIn("complete display source", description)

    def test_system_message_tells_the_model_to_build_on_the_live_session(self):
        """The prompt must tell the model to carry REPL state forward, not redo setup.

        This asserted a specific set of phrases, including a worked
        define-once/call-later example built around `def mean(values)`. The
        Execution Style section replaced that example with the general rule it
        was illustrating, so the phrases are gone while the instruction is not.
        Pinned here on the rule rather than on the example, because the example
        is presentation and the rule is the contract: a prompt that stops
        telling the model to reuse live state makes it re-import and re-load on
        every block, which is the behaviour this guards against.
        """
        self.assertIn("persistent REPL", default_system_message)
        self.assertIn("let the REPL carry state forward", default_system_message)
        self.assertIn("Reuse what you are already holding", default_system_message)
        self.assertIn("do not re-extract or hardcode what you have", default_system_message)


if __name__ == "__main__":
    unittest.main()
