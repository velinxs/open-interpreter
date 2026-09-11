"""Upper bound on prompt tokens sent during the scripted session.

Phase 0 sets TOKEN_BUDGET to the measured total plus ten percent so any
accidental growth fails; Phase 3 lowers it as the prompts shrink.
"""

from tests.support.scripted_session import SCRIPT, run_scripted_session

TOKEN_BUDGET = 46500  # measured 42,272 on 2026-09-11 plus ten percent; Phase 3 lowers this


def test_scripted_session_stays_under_token_budget(offline_interpreter, tmp_path):
    per_request = run_scripted_session(offline_interpreter, tmp_path)

    expected_requests = sum(len(replies) for _, replies in SCRIPT)
    assert len(per_request) == expected_requests

    total = sum(per_request)
    print(f"\nprompt tokens per request: {per_request}\ntotal: {total}")
    assert total <= TOKEN_BUDGET, f"{total} prompt tokens exceeds budget {TOKEN_BUDGET}"
