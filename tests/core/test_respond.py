"""What respond() does when the loop cannot continue.

The budget handler is the case that bites hardest: it only runs once real money
has already been spent, so a defect in it is invisible until the worst moment.
"""

from types import SimpleNamespace
from unittest import mock

import litellm
import pytest

import interpreter.core.respond as respond_module
from interpreter.core.respond import respond


def _budget_exhausted_interpreter(max_budget):
    """An interpreter whose first LLM call reports the budget is already spent.

    max_budget is set on `llm`, which is where the real object carries it: the
    --max_budget flag (arguments.py) and the profile migrator both write
    interpreter.llm.max_budget, and OpenInterpreter.__init__ never defines a
    top-level one. A fake that carried it at the top level would mirror the bug
    instead of the product.
    """

    def _run(messages):
        raise litellm.exceptions.BudgetExceededError(current_cost=9.99, max_budget=max_budget)
        yield  # unreachable; makes _run a generator like the real llm.run

    return SimpleNamespace(
        messages=[{"role": "user", "type": "message", "content": "hi"}],
        llm=SimpleNamespace(run=_run, max_budget=max_budget),
        display_message=mock.Mock(),
        _last_rendered_system_message=None,
        loop=False,
        loop_message=None,
        loop_breakers=[],
        verbose=False,
        debug=False,
    )


@pytest.fixture(autouse=True)
def _stub_system_message(monkeypatch):
    """The system message is assembled from the whole product; not under test here."""
    monkeypatch.setattr(respond_module, "assemble_system_message", lambda interpreter: "SYSTEM")


def test_the_budget_message_reports_the_limit_that_was_set():
    """Hitting the budget shows the configured limit instead of crashing.

    max_budget lives on interpreter.llm, but the handler read
    interpreter.max_budget. Building the message therefore raised AttributeError
    from inside the except block -- so the moment the budget was actually
    reached, the user got a traceback instead of the explanation, and because
    litellm._current_cost is process-global every later chat() in the process
    failed the same way until the budget was raised.
    """
    interpreter = _budget_exhausted_interpreter(max_budget=1.25)

    list(respond(interpreter))

    interpreter.display_message.assert_called()
    shown = "\n".join(str(call.args[0]) for call in interpreter.display_message.call_args_list)
    assert "Max budget exceeded" in shown
    assert "1.25" in shown, shown
