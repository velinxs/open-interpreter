import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires an LLM API key (not run in default CI)")
    config.addinivalue_line(
        "markers",
        "network: makes real HTTP calls to a provider (run with RUN_NETWORK_TESTS=1)",
    )


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(reason="OPENAI_API_KEY not set; skipping integration test")
    skip_network = pytest.mark.skip(reason="RUN_NETWORK_TESTS not set; test makes real provider calls")
    for item in items:
        if "integration" in item.keywords and not os.environ.get("OPENAI_API_KEY"):
            item.add_marker(skip_integration)
        if "network" in item.keywords and not os.environ.get("RUN_NETWORK_TESTS"):
            item.add_marker(skip_network)


@pytest.fixture
def offline_interpreter():
    """An OpenInterpreter that never talks to a provider or telemetry."""
    from interpreter.core.core import OpenInterpreter
    from tests.support.fake_llm import install_fake_llm

    interp = OpenInterpreter()
    interp.offline = True
    interp.disable_telemetry = True
    interp.auto_run = True
    interp.script = lambda replies: install_fake_llm(interp, replies)
    yield interp
    try:
        interp.terminal.terminate()  # stop() only halts running code; terminate() kills the kernels
    except Exception:
        pass
