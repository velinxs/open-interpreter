"""`interpreter --local`: picking a local provider and pointing the LLM at it.

local_setup is one 400-line function of interactive branches, and every branch
ends by writing api_base / api_key / model. Get one wrong and the session
talks to the wrong port — or, with a bare model name and no prefix, to the
public OpenAI endpoint. None of it was covered.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest

import interpreter.terminal_interface.local_setup as local_setup_module
from interpreter.terminal_interface.local_setup import local_setup


class FakeLlm:
    model = None
    api_base = None
    api_key = None
    temperature = None
    supports_functions = None
    max_tokens = None
    context_window = None


class FakeInterpreter:
    def __init__(self, oi_dir, auto_run=True):
        self.llm = FakeLlm()
        self.auto_run = auto_run
        self.displayed = []
        self.pings = []
        self._oi_dir = oi_dir
        self.toolbox = SimpleNamespace(ai=SimpleNamespace(chat=self.pings.append))

    def display_message(self, message):
        self.displayed.append(message)

    def get_oi_dir(self):
        return self._oi_dir


@pytest.fixture
def interpreter(tmp_path):
    """An interpreter whose models directory is inside the test's tmp_path.

    The real get_oi_dir() is ~/.config/open-interpreter, and the Llamafile
    branch creates directories and downloads gigabyte files into it.
    """
    return FakeInterpreter(str(tmp_path))


@pytest.fixture
def answers(monkeypatch):
    """Scripted inquirer answers, consumed in order."""
    scripted = []

    def _prompt(questions):
        assert scripted, "inquirer.prompt called more times than the test scripted"
        return scripted.pop(0)

    monkeypatch.setattr(local_setup_module.inquirer, "prompt", _prompt)
    return scripted


@pytest.fixture(autouse=True)
def _fast_and_offline(monkeypatch):
    """No sleeps, and a machine with plenty of RAM and disk unless a test says otherwise."""
    monkeypatch.setattr(local_setup_module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        local_setup_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=32 * 1024**3),
    )
    monkeypatch.setattr(
        local_setup_module.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=500 * 1024**3),
    )


def test_cancelling_the_provider_menu_exits(interpreter, answers):
    """Ctrl-C at the provider list stops the run instead of falling through.

    inquirer returns None on abort. Continuing would index None and raise a
    TypeError on top of the abort the user just asked for.
    """
    answers.append(None)
    with pytest.raises(SystemExit):
        local_setup(interpreter)


def test_lm_studio_points_at_its_default_port_with_a_dummy_key(interpreter, answers):
    """LM Studio is configured entirely from its documented defaults.

    The key must be non-empty (LiteLLM refuses to send without one) and
    functions must be off, because LM Studio's OpenAI shim does not implement
    them and a function call would fail every request.
    """
    answers.append({"model": "LM Studio"})
    local_setup(interpreter)
    assert interpreter.llm.api_base == "http://localhost:1234/v1"
    assert interpreter.llm.api_key == "dummy"
    assert interpreter.llm.supports_functions is False


def test_jan_lists_the_models_the_running_server_reports(interpreter, answers, monkeypatch):
    """Jan's model list comes from its own /models endpoint, not a hard-coded list.

    Jan model ids are whatever the user downloaded. Guessing would give a
    model id the server rejects.
    """
    monkeypatch.setattr(
        local_setup_module.requests,
        "get",
        lambda url: SimpleNamespace(json=lambda: {"data": [{"id": "mistral-ins-7b-q4"}]}),
    )
    answers.append({"model": "Jan"})
    answers.append({"jan_model_name": "mistral-ins-7b-q4"})
    local_setup(interpreter)
    assert interpreter.llm.api_base == "http://localhost:1337/v1"
    assert interpreter.llm.model == "mistral-ins-7b-q4"
    assert interpreter.llm.api_key == "dummy"


def test_jan_offers_a_custom_model_id(interpreter, answers, monkeypatch):
    """The custom-id escape hatch is offered first and read from stdin.

    A model Jan has not finished registering will not be in /models, and
    without this the user could not select it at all.
    """
    offered = {}

    def _prompt(questions):
        if questions[0].name == "jan_model_name":
            offered["choices"] = list(questions[0].choices)
            return {"jan_model_name": ">> Type Custom Model ID"}
        return {"model": "Jan"}

    monkeypatch.setattr(
        local_setup_module.requests,
        "get",
        lambda url: SimpleNamespace(json=lambda: {"data": [{"id": "known-model"}]}),
    )
    monkeypatch.setattr(local_setup_module.inquirer, "prompt", _prompt)
    monkeypatch.setattr("builtins.input", lambda *a: "typed-model-id")

    local_setup(interpreter)
    assert offered["choices"][0] == ">> Type Custom Model ID"
    assert interpreter.llm.model == "typed-model-id"


def test_ollama_lists_installed_models_and_offers_downloads(interpreter, answers, monkeypatch):
    """`ollama list` output is parsed into a menu: header dropped, :latest trimmed, favourites first.

    The parsing is positional and unguarded, so a stray header row or a
    ":latest" suffix would either appear as a menu entry or produce a model id
    ollama does not recognise. The priority reorder exists because the list is
    otherwise in arbitrary order.
    """
    listing = "NAME\tID\tSIZE\nmistral:latest\tabc\t4GB\nllama3.1:latest\tdef\t5GB\n"
    monkeypatch.setattr(
        local_setup_module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=listing, returncode=0),
    )

    seen = {}

    def _prompt(questions):
        if questions[0].name == "model":
            return {"model": "Ollama"}
        seen["choices"] = list(questions[0].choices)
        return {"name": "mistral"}

    monkeypatch.setattr(local_setup_module.inquirer, "prompt", _prompt)
    local_setup(interpreter)

    assert "NAME" not in seen["choices"]
    assert seen["choices"][0] == "llama3.1"  # priority model moved to the front
    assert "mistral" in seen["choices"]
    assert "↓ Download phi3" in seen["choices"]
    assert seen["choices"][-1] == "Browse Models ↗"
    assert interpreter.llm.model == "ollama/mistral"


def test_ollama_pings_the_model_and_restores_the_token_limits(interpreter, monkeypatch):
    """Loading the model uses a one-token ping, then puts the real limits back.

    The ping exists so ollama pulls the weights into memory now rather than
    during the user's first message. Leaving max_tokens at 1 afterwards would
    truncate every reply to a single token.
    """
    monkeypatch.setattr(
        local_setup_module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout="NAME\nmistral:latest\n", returncode=0),
    )
    monkeypatch.setattr(
        local_setup_module.inquirer,
        "prompt",
        lambda questions: {"model": "Ollama"} if questions[0].name == "model" else {"name": "mistral"},
    )
    interpreter.llm.max_tokens = 4096
    interpreter.llm.context_window = 128000

    local_setup(interpreter)

    assert interpreter.pings == ["ping"]
    # The final RAM-based block overwrites both, which is what a local model gets.
    assert interpreter.llm.max_tokens == 1200
    assert interpreter.llm.context_window == 8000


def test_selecting_a_download_entry_pulls_the_model(interpreter, monkeypatch):
    """"↓ Download phi3" runs `ollama pull phi3` rather than setting a bogus model name.

    The download entries share the menu with installed models. Treating one as
    a model id would set the model to "↓ Download phi3".
    """
    calls = []

    def _run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="NAME\n", returncode=0)

    monkeypatch.setattr(local_setup_module.subprocess, "run", _run)
    monkeypatch.setattr(
        local_setup_module.inquirer,
        "prompt",
        lambda questions: {"model": "Ollama"} if questions[0].name == "model" else {"name": "↓ Download phi3"},
    )

    local_setup(interpreter)
    assert ["ollama", "pull", "phi3"] in calls
    assert interpreter.llm.model == "ollama/phi3"


def test_a_missing_ollama_binary_explains_itself_and_exits(interpreter, answers, monkeypatch, capsys):
    """No ollama installed produces an install link and exit 1, not a traceback.

    FileNotFoundError from subprocess is the normal case for a user who picked
    Ollama without having it. The message is the only thing telling them what
    to do next.
    """

    def _run(*args, **kwargs):
        raise FileNotFoundError("ollama")

    monkeypatch.setattr(local_setup_module.subprocess, "run", _run)
    answers.append({"model": "Ollama"})

    with pytest.raises(SystemExit) as excinfo:
        local_setup(interpreter)
    assert excinfo.value.code == 1
    assert "not installed" in capsys.readouterr().out
    assert any("ollama.com" in message for message in interpreter.displayed)


def test_a_provider_error_from_the_ping_is_shown_in_a_panel_and_exits(interpreter, answers, monkeypatch):
    """A LiteLLM error while loading the model exits rather than starting a broken session.

    Reaching the chat loop with a model that failed to load means every
    message fails identically, with the real cause scrolled away.
    """
    monkeypatch.setattr(
        local_setup_module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout="NAME\nmistral:latest\n", returncode=0),
    )
    monkeypatch.setattr(
        local_setup_module.inquirer,
        "prompt",
        lambda questions: {"model": "Ollama"} if questions[0].name == "model" else {"name": "mistral"},
    )

    def _boom(_):
        raise local_setup_module.litellm.exceptions.NotFoundError(
            message="model not found", model="mistral", llm_provider="ollama"
        )

    interpreter.toolbox.ai.chat = _boom
    with pytest.raises(SystemExit) as excinfo:
        local_setup(interpreter)
    assert excinfo.value.code == 1


def test_an_unrelated_exception_from_the_ping_is_not_swallowed(interpreter, answers, monkeypatch):
    """Bugs in our own code still surface as tracebacks.

    The broad except exists to pretty-print provider errors. If it also ate
    TypeErrors and AttributeErrors, every bug in this function would look like
    a model problem.
    """
    monkeypatch.setattr(
        local_setup_module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout="NAME\nmistral:latest\n", returncode=0),
    )
    monkeypatch.setattr(
        local_setup_module.inquirer,
        "prompt",
        lambda questions: {"model": "Ollama"} if questions[0].name == "model" else {"name": "mistral"},
    )

    def _boom(_):
        raise ValueError("a real bug")

    interpreter.toolbox.ai.chat = _boom
    with pytest.raises(ValueError):
        local_setup(interpreter)


@pytest.mark.parametrize(
    "ram_gb,max_tokens,context_window",
    [(32, 1200, 8000), (8, 1000, 3000)],
)
def test_token_limits_are_chosen_from_available_ram(interpreter, answers, monkeypatch, ram_gb, max_tokens, context_window):
    """A local model's context window is sized from the machine, not the model.

    Local backends will happily accept a context larger than they can hold and
    then either swap or return garbage, so the cap is deliberately
    conservative on small machines.
    """
    monkeypatch.setattr(
        local_setup_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=ram_gb * 1024**3),
    )
    answers.append({"model": "LM Studio"})
    local_setup(interpreter)
    assert interpreter.llm.max_tokens == max_tokens
    assert interpreter.llm.context_window == context_window


def test_manual_approval_is_advertised_when_auto_run_is_off(tmp_path, answers):
    """A local setup that will ask before running code says so up front.

    Otherwise the first code block looks like a hang: the model has answered
    and the terminal is waiting for a keypress the user does not know about.
    """
    interpreter = FakeInterpreter(str(tmp_path), auto_run=False)
    answers.append({"model": "LM Studio"})
    local_setup(interpreter)
    assert any("require approval" in message for message in interpreter.displayed)


# --- llamafile ---------------------------------------------------------------


@pytest.fixture
def llamafile(monkeypatch):
    """Neutralise the Llamafile branch's side effects: Xcode check, download, exec."""
    monkeypatch.setattr(local_setup_module.platform, "system", lambda: "Linux")
    downloads = []
    monkeypatch.setattr(local_setup_module.wget, "download", lambda url, path: downloads.append((url, path)))
    monkeypatch.setattr(local_setup_module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    launched = []

    def _popen(command, **kwargs):
        launched.append(command)
        return SimpleNamespace(stdout=iter(["llama server listening at http://localhost:8080\n"]), kill=lambda: None)

    monkeypatch.setattr(local_setup_module.subprocess, "Popen", _popen)
    return SimpleNamespace(downloads=downloads, launched=launched)


def test_an_already_downloaded_llamafile_is_launched(interpreter, answers, llamafile, tmp_path):
    """Choosing an existing llamafile starts it and points the LLM at port 8080.

    The model name must be "openai/local" — a bare name with an api_base set
    would be rewritten and routed by LiteLLM as a hosted OpenAI model.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "tiny.llamafile").write_text("#!/bin/sh\n")

    answers.append({"model": "Llamafile"})
    answers.append({"model": "tiny.llamafile"})
    local_setup(interpreter)

    assert llamafile.launched, "the selected llamafile was never started"
    assert interpreter.llm.model == "openai/local"
    assert interpreter.llm.api_base == "http://localhost:8080/v1"
    assert interpreter.llm.supports_functions is False
    assert interpreter.llm.temperature == 0


def test_a_first_run_download_never_starts_the_server(interpreter, answers, llamafile, tmp_path):
    """Characterisation bug: on a first run the model downloads but is never launched.

    The `if model_path:` block that starts the llamafile process lives inside
    the `else:` branch taken when models already exist on disk. A user with no
    models therefore downloads one, has api_base set to localhost:8080, and
    then gets a connection error on every message because nothing is
    listening. Compare test_an_already_downloaded_llamafile_is_launched: the
    only difference is whether a file was already there.
    """
    answers.append({"model": "Llamafile"})
    answers.append({"model": "Phi-3-mini (2.42GB)"})
    local_setup(interpreter)

    assert len(llamafile.downloads) == 1
    url, path = llamafile.downloads[0]
    assert "Phi-3-mini" in url
    assert path.startswith(str(tmp_path / "models"))
    assert interpreter.llm.api_base == "http://localhost:8080/v1"
    assert llamafile.launched == [], "a fix would start the downloaded llamafile here"


@pytest.mark.parametrize("choice", ["Mistral-7B-Instruct (4.40GB)", "Gemma-2-27b (16.70GB)", "TinyLlama-1.1B (0.70GB)"])
def test_llamafile_sizes_that_need_a_trailing_zero_cannot_be_selected(interpreter, answers, llamafile, choice):
    """Characterisation bug: menu entries are formatted with %.2f, the lookup is not.

    The menu is built with f"{size:.2f}GB" but the selected model is found with
    f"{size}GB", so every size whose repr differs from its two-decimal form
    (4.40, 16.7, 0.70) raises StopIteration inside download_model. That is
    caught by a bare `except Exception`, printed as an empty line, and turned
    into a None model path, which then crashes on .split(). Three of the
    twelve offered models are unselectable.
    """
    answers.append({"model": "Llamafile"})
    answers.append({"model": choice})
    with pytest.raises(AttributeError):
        local_setup(interpreter)
    assert llamafile.downloads == []


def test_models_too_large_for_the_disk_are_not_offered(interpreter, answers, llamafile, monkeypatch, capsys):
    """A machine with no free space is told so rather than offered a download it cannot finish.

    wget would fill the disk and fail partway through, leaving a truncated
    file that later looks like an installed model.
    """
    monkeypatch.setattr(
        local_setup_module.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=1 * 1024**2),
    )
    answers.append({"model": "Llamafile"})
    with pytest.raises(AttributeError):
        local_setup(interpreter)
    assert "not have enough storage" in capsys.readouterr().out
    assert llamafile.downloads == []
