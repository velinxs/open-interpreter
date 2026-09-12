"""The CLI entry point: what a flag actually does to the interpreter.

start_terminal_interface() is the only place where argv, the profile file, and
the hard-coded model defaults are reconciled. It is 200+ lines of ordering-
sensitive assignment with no return value, so a mistake here is invisible
until a session behaves nothing like the flags asked for. These tests drive
the real function against a real OpenInterpreter with only the I/O boundaries
(profile loading, the update check, the provider, and chat) replaced.
"""

import sys
from types import SimpleNamespace

import pytest

import interpreter.terminal_interface.contributing_conversations as contributing
import interpreter.terminal_interface.conversation_navigator as navigator
import interpreter.terminal_interface.start_terminal_interface as sti
import interpreter.terminal_interface.validate_llm_settings as validate_module
from interpreter.core.core import OpenInterpreter


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Run start_terminal_interface() against a real interpreter, offline.

    Every boundary that would touch the user's machine is replaced: the
    profile loader (which reads ~/.config/open-interpreter and can rewrite
    files there), the PyPI update check, the API-key prompt, the contribution
    cache, and chat() itself. What is left running is the real flag table and
    the real assignment logic, which is the part under test.
    """
    monkeypatch.chdir(tmp_path)

    class Run:
        def __init__(self):
            self.interpreter = OpenInterpreter()
            self.profile_loaded = None
            self.profile_hook = None
            self.chats = []
            self.validated = 0
            self.update_checked = 0
            self.navigated = 0
            self.storage_dirs = []
            self.reset_profiles = []

        def _profile(self, interp, filename):
            self.profile_loaded = filename
            if self.profile_hook:
                self.profile_hook(interp)
            return interp

        def __call__(self, *argv):
            monkeypatch.setattr(sys, "argv", ["interpreter", *argv])
            sti.start_terminal_interface(self.interpreter)
            return self.interpreter

    run = Run()
    interpreter = run.interpreter
    interpreter.disable_telemetry = True
    monkeypatch.setattr(interpreter, "chat", lambda *a, **kw: run.chats.append(a))
    monkeypatch.setattr(interpreter, "display_message", lambda *a, **kw: None)

    def _count(field):
        def _counter(*args, **kwargs):
            setattr(run, field, getattr(run, field) + 1)
            return False

        return _counter

    monkeypatch.setattr(sti, "profile", run._profile)
    monkeypatch.setattr(sti, "check_for_update", _count("update_checked"))
    monkeypatch.setattr(sti, "open_storage_dir", lambda d: run.storage_dirs.append(d))
    monkeypatch.setattr(sti, "reset_profile", lambda p=None: run.reset_profiles.append(p))
    monkeypatch.setattr(sti.time, "sleep", lambda *_: None)
    monkeypatch.setattr(validate_module, "validate_llm_settings", _count("validated"))
    monkeypatch.setattr(contributing, "contribute_conversation_launch_logic", lambda interp: None)
    monkeypatch.setattr(navigator, "conversation_navigator", _count("navigated"))
    monkeypatch.delenv("DISABLE_TELEMETRY", raising=False)
    # --stdin calls input(); pytest's capture makes a real read an OSError.
    monkeypatch.setattr("builtins.input", lambda *a: "")

    yield run

    try:
        interpreter.terminal.terminate()
    except Exception:
        pass


# --- quick-exit flags: these must return before anything heavy loads ---------


def test_version_prints_and_returns_without_an_interpreter(monkeypatch, capsys):
    """--version works against a bare stub, not a constructed interpreter.

    main() deliberately hands start_terminal_interface an empty stub for
    --version and --help so the multi-second interpreter import is skipped.
    If anything before the version branch touched a real attribute, that
    optimisation would break with an AttributeError instead of printing.
    """

    class Stub:
        pass

    stub = Stub()
    stub.llm = Stub()
    monkeypatch.setattr(sys, "argv", ["interpreter", "--version"])
    sti.start_terminal_interface(stub)
    assert "Open Interpreter" in capsys.readouterr().out


def test_unknown_flags_exit_nonzero_instead_of_being_ignored(cli, capsys):
    """A typo'd flag stops the run; it is never silently dropped.

    Without the parse_known_args check, `interpreter --no-auto-run` would
    start a normal session and the user would think they had disabled
    something they had not.
    """
    with pytest.raises(SystemExit) as excinfo:
        cli("--definitely-not-a-flag")
    assert excinfo.value.code == 1
    assert "Unrecognized argument" in capsys.readouterr().out


def test_profiles_and_local_models_open_a_directory_and_stop(cli):
    """--profiles and --local_models are directory shortcuts, not session starts."""
    cli("--profiles")
    assert cli.storage_dirs == ["profiles"]
    assert cli.chats == []

    cli("--local_models")
    assert cli.storage_dirs == ["profiles", "models"]
    assert cli.chats == []


def test_reset_profile_with_a_name_resets_that_profile_and_stops(cli):
    """`--reset_profile default.yaml` rewrites one profile and does not chat."""
    cli("--reset_profile", "default.yaml")
    assert cli.reset_profiles == ["default.yaml"]
    assert cli.chats == []


def test_bare_reset_profile_resets_every_default_and_stops(cli):
    """`--reset_profile` with no argument does what its help text promises.

    nargs="?" with no const makes the value None when the flag is passed
    bare, and the guard used to exclude None as well as the absent-flag
    sentinel, so the documented form fell through and started a chat session
    instead. reset_profile(None) means "every default profile".
    """
    cli("--reset_profile")
    assert cli.reset_profiles == [None]
    assert cli.chats == []


# --- argv rewriting ---------------------------------------------------------


def test_deprecated_debug_mode_is_rewritten_to_verbose(cli, capsys):
    """--debug_mode still works and says what replaced it.

    Silently ignoring a renamed flag would leave a user staring at a session
    with no logs and no explanation.
    """
    cli("--debug_mode")
    assert cli.interpreter.verbose is True
    assert "--verbose" in capsys.readouterr().out


def test_stdin_implies_plain_text_output(cli):
    """--stdin turns on --plain, because nothing is reading the Rich panels.

    In stdin mode the output is being piped somewhere. Live-updating panels
    and ANSI boxes would corrupt whatever consumes it.
    """
    cli("--stdin")
    assert cli.interpreter.plain_text_display is True


def test_a_bare_message_becomes_a_one_shot_request(cli, tmp_path):
    """`interpreter fix the build` queues the message and switches to fast mode.

    This is the "i {command}" shortcut. It must consume the words before
    argparse sees them (they are not flags), prefix the message with "I", and
    attach the current directory listing so the model can resolve "the build"
    against real filenames.
    """
    (tmp_path / "Makefile").write_text("all:\n")
    cli("fix", "the", "build")

    queued = cli.interpreter.messages[0]
    assert queued["content"] == "I fix the build"
    assert queued["role"] == "user"
    assert "Makefile" in cli.interpreter.custom_instructions
    assert "ULTRA FAST" in cli.interpreter.custom_instructions


# --- flags reaching the interpreter -----------------------------------------


def test_cli_flags_override_the_profile(cli):
    """Flags are applied again *after* the profile, so the CLI always wins.

    set_attributes runs twice for exactly this reason. If the second call were
    dropped, `interpreter --model X` would be silently overridden by whatever
    the profile happens to say, which is the opposite of what a flag means.
    """
    cli.profile_hook = lambda interp: setattr(interp.llm, "model", "from-the-profile")
    cli("--model", "from-the-cli", "--temperature", "0.25")
    assert cli.interpreter.llm.model == "from-the-cli"
    assert cli.interpreter.llm.temperature == 0.25


def test_omitted_flags_do_not_clobber_profile_settings(cli):
    """A flag the user did not pass leaves the profile's value alone.

    set_attributes skips None, which is why most flags default to None. A
    default of False instead would quietly switch off every boolean the
    profile turned on — disable_telemetry is called out in the flag table for
    this reason.
    """
    cli.profile_hook = lambda interp: setattr(interp, "disable_telemetry", True)
    cli()
    assert cli.interpreter.disable_telemetry is True


def test_disable_telemetry_env_var_is_honoured(cli, monkeypatch):
    """DISABLE_TELEMETRY=true switches telemetry off without a flag or profile edit.

    It is the only opt-out available to someone running OI from a script they
    do not control.
    """
    cli.profile_hook = lambda interp: setattr(interp, "disable_telemetry", False)
    monkeypatch.setenv("DISABLE_TELEMETRY", "TRUE")
    cli()
    assert cli.interpreter.disable_telemetry is True


def test_no_highlight_active_line_flag(cli):
    """--no_highlight_active_line reaches the interpreter even though it has no attribute row."""
    cli("--no_highlight_active_line")
    assert cli.interpreter.highlight_active_line is False


def test_auto_run_and_safe_mode_cannot_both_be_on(cli):
    """`-y --safe ask` downgrades to prompting rather than scanning nothing.

    Full auto-run would run code before the safety scan could ask about it,
    which makes --safe a lie. Prompting is the conservative resolution.
    """
    cli("-y", "--safe_mode", "ask")
    assert cli.interpreter.auto_run_mode == "prompt"


@pytest.mark.parametrize("mode", ["allowlist", "denylist"])
def test_partial_auto_run_modes_survive_safe_mode(cli, mode):
    """allowlist/denylist still gate some code on approval, so they are kept.

    Downgrading these to "prompt" as well would make --safe silently disable
    the allowlist the user configured.
    """
    cli("--auto_run_mode", mode, "--safe_mode", "auto")
    assert cli.interpreter.auto_run_mode == mode


# --- model-specific defaults ------------------------------------------------


@pytest.mark.parametrize(
    "model,context_window,supports_functions",
    [
        ("gpt-4", 6500, True),
        ("openai/gpt-4", 6500, True),
        ("gpt-4o", 123000, True),
        ("gpt-4-vision-preview", 123000, False),
        ("gpt-3.5-turbo", 16000, True),
    ],
)
def test_known_openai_models_get_context_window_defaults(cli, model, context_window, supports_functions):
    """OpenAI models get a context window and function support filled in.

    LiteLLM does not always know these, and a missing context window makes
    trimming fall back to a guess — the model then either wastes most of its
    window or overflows it mid-conversation. "vision" in the name means no
    function calling.
    """
    cli("--model", model)
    assert cli.interpreter.llm.context_window == context_window
    assert cli.interpreter.llm.max_tokens == 4096
    assert cli.interpreter.llm.supports_functions is supports_functions


def test_explicit_settings_beat_the_model_defaults(cli):
    """A context window the user asked for is not overwritten by the defaults.

    Every default is guarded by `is None`. Dropping a guard would make
    `--context_window` unusable for exactly the models that most need it.
    """
    cli("--model", "gpt-4", "--context_window", "100000", "--max_tokens", "8000")
    assert cli.interpreter.llm.context_window == 100000
    assert cli.interpreter.llm.max_tokens == 8000


@pytest.mark.parametrize("alias", ["claude-3.5", "claude-3-5", "claude-3.5-sonnet", "claude-3-5-sonnet"])
def test_retired_claude_aliases_are_remapped(cli, alias):
    """Short claude-3.5 names resolve to a model that still exists.

    These aliases were never valid LiteLLM model ids and the versions they
    referred to are retired; without the remap the run dies on the first
    request with a provider error.
    """
    cli("--model", alias)
    assert cli.interpreter.llm.model == "claude-sonnet-4-6"


# --- api_base and provider prefixes -----------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("my-local-model", "openai/my-local-model"),
        ("openai/gpt-4o", "openai/gpt-4o"),
        ("azure/deployment", "azure/deployment"),
        ("deepseek/chat", "deepseek/chat"),
        ("dashscope-us/qwen", "dashscope-us/qwen"),
        ("ollama/llama3", "ollama/llama3"),
        ("local/whatever", "local/whatever"),
        ("jan/mistral", "mistral"),
    ],
)
def test_custom_api_base_routes_unqualified_models_through_openai(cli, model, expected):
    """With --api_base, a bare model name is assumed to be an OpenAI-compatible server.

    Almost every local server (LM Studio, llama.cpp, vLLM) speaks the OpenAI
    protocol, and LiteLLM needs the prefix to pick a client. The exceptions
    are providers that already route themselves; "jan/" is stripped because
    Jan's ids are bare. Getting this wrong sends the request to the public
    OpenAI endpoint instead of localhost.
    """
    cli("--api_base", "http://localhost:1234/v1", "--model", model)
    assert cli.interpreter.llm.model == expected


def test_without_api_base_model_names_are_left_alone(cli):
    """No --api_base means no rewriting: LiteLLM resolves the name itself."""
    cli("--model", "some-model")
    assert cli.interpreter.llm.model == "some-model"


# --- profile shortcuts ------------------------------------------------------


@pytest.mark.parametrize(
    "argv,expected",
    [
        ([], "develop.yaml"),
        (["--profile", "custom.yaml"], "custom.yaml"),
        (["--fast"], "fast.yaml"),
        (["--vision"], "vision.yaml"),
        (["--local"], "local.py"),
        (["--local", "--vision"], "local.py"),
        (["--codestral"], "codestral.py"),
        (["--codestral", "--vision"], "codestral-vision.py"),
        (["--assistant"], "assistant.py"),
        (["--llama3"], "llama3.py"),
        (["--llama3", "--vision"], "llama3-vision.py"),
        (["--groq"], "groq.py"),
    ],
)
def test_profile_shortcut_flags_select_a_profile_file(cli, argv, expected, monkeypatch):
    """Each shortcut flag resolves to the profile file it advertises.

    These are the documented one-word entry points. A wrong filename here
    loads somebody else's settings — or, for a name that does not exist,
    fails at startup with a file error the user cannot connect to the flag
    they typed.
    """
    monkeypatch.setattr(cli.interpreter.toolbox.vision, "load", lambda *a, **kw: None)
    cli(*argv)
    assert cli.profile_loaded == expected


def test_the_default_profile_is_the_one_the_flag_table_declares(cli):
    """No --profile falls back to _DEFAULT_PROFILE, not to a hard-coded string.

    This branch carries a DEVELOP-BRANCH-ONLY warning: the default differs
    from upstream's default.yaml on purpose. The test makes the coupling
    visible so the two cannot drift apart unnoticed.
    """
    cli()
    assert cli.profile_loaded == sti._DEFAULT_PROFILE


# --- update check -----------------------------------------------------------


def test_update_check_is_skipped_when_offline_or_piped(cli):
    """--offline and --stdin both suppress the PyPI version check.

    --offline is a promise that nothing reaches the network. --stdin means
    the output is being consumed by another program, which should not receive
    an upgrade advert in the middle of its data.
    """
    cli("--offline")
    assert cli.update_checked == 0

    cli("--stdin", "-y")
    assert cli.update_checked == 0


def test_update_check_runs_for_a_normal_session(cli):
    """An ordinary interactive run does check for updates."""
    cli()
    assert cli.update_checked == 1


# --- mode dispatch ----------------------------------------------------------


def test_conversations_flag_opens_the_navigator_and_stops(cli):
    """--conversations hands off to the navigator; it does not also start a chat.

    The navigator starts the chat itself once a conversation is chosen.
    Falling through would start a second, empty session on top of it.
    """
    cli("--conversations")
    assert cli.navigated == 1
    assert cli.chats == []


def test_a_normal_run_validates_the_model_then_chats(cli):
    """The API-key check runs before the session, once.

    Skipping it drops the user into a prompt that fails on their first
    message with a provider error instead of asking for a key up front.
    """
    cli()
    assert cli.validated == 1
    assert cli.chats == [()]
    assert cli.interpreter.in_terminal_interface is True


def test_stdin_mode_reads_one_line_and_sends_it(cli, monkeypatch):
    """--stdin takes a single line as the whole request.

    This is the scripting entry point. Reading zero lines would hang, and
    passing the line to chat() is what distinguishes it from an interactive
    session that happens to have no tty.
    """
    monkeypatch.setattr("builtins.input", lambda *a: "what is 21*2")
    cli("--stdin", "-y")
    assert cli.chats == [("what is 21*2",)]


def test_server_mode_skips_the_api_key_prompt(cli, monkeypatch):
    """--server starts the HTTP server and never blocks on an interactive prompt.

    validate_llm_settings can call prompt() for a missing key. In a server
    that is a hang with no output, which is why the branch is explicitly
    skipped — the comment in the source says so.
    """
    started = []

    class FakeServer:
        def run(self):
            started.append(True)

    class FakeAsyncInterpreter:
        def __init__(self):
            self.llm = SimpleNamespace(
                model="gpt-4o",
                api_base=None,
                context_window=None,
                max_tokens=None,
                supports_functions=None,
            )
            self.server = FakeServer()
            self.messages = []
            self.auto_run_mode = "prompt"
            self.safe_mode = "off"
            self.offline = False
            self.disable_telemetry = True
            self.highlight_active_line = True

        def display_message(self, *a, **kw):
            pass

    import interpreter.server as server_module

    monkeypatch.setattr(server_module, "AsyncInterpreter", FakeAsyncInterpreter)
    monkeypatch.setattr(sys, "argv", ["interpreter", "--server"])
    sti.start_terminal_interface(cli.interpreter)

    assert started == [True]
    assert cli.validated == 0
