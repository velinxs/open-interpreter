import argparse
import os
import sys

# Disable Intel Fortran runtime's Ctrl+C handler on Windows so Ctrl+C exits cleanly
# without "forrtl: error (200): program aborting due to control-C event" (Intel docs).
if sys.platform == "win32":
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import time
import traceback
from importlib.metadata import PackageNotFoundError, version

# validate_llm_settings, contributing_conversations, and conversation_navigator
# import litellm (directly or transitively) and are deferred to after
# argparse parses args, so --help / --version exit before they load.
from .arguments import (
    _DEFAULT_PROFILE,
    DEPRECATED_FLAGS,
    build_arguments,
    get_argument_dictionary,
    set_attributes,
)
from .profiles.profiles import open_storage_dir, profile, reset_profile
from .utils.check_for_update import check_for_update


def start_terminal_interface(interpreter):
    """
    Meant to be used from the command line. Parses arguments, starts OI's terminal interface.
    """

    # Instead use an async interpreter, which has a server. Set settings on that
    if "--server" in sys.argv:
        from interpreter import AsyncInterpreter

        interpreter = AsyncInterpreter()

    arguments = build_arguments(interpreter)

    if "--stdin" in sys.argv and "--plain" not in sys.argv:
        sys.argv += ["--plain"]

    # i shortcut
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        message = " ".join(sys.argv[1:])
        interpreter.messages.append(
            {
                "role": "user",
                "type": "message",
                "content": "I " + message,
                "sent_at": time.time(),
            }
        )
        sys.argv = sys.argv[:1]

        interpreter.custom_instructions = "UPDATED INSTRUCTIONS: You are in ULTRA FAST, ULTRA CERTAIN mode. Do not ask the user any questions or run code to gathet information. Go as quickly as you can. Run code quickly. Do not plan out loud, simply start doing the best thing. The user expects speed. Trust that the user knows best. Just interpret their ambiguous command as quickly and certainly as possible and try to fulfill it IN ONE COMMAND, assuming they have the right information. If they tell you do to something, just do it quickly in one command, DO NOT try to get more information (for example by running `cat` to get a file's infomration— this is probably unecessary!). DIRECTLY DO THINGS AS FAST AS POSSIBLE."

        files_in_directory = os.listdir()[:100]
        interpreter.custom_instructions += "\nThe files in CWD, which THE USER MAY BE REFERRING TO, are: " + ", ".join(
            files_in_directory
        )

        # interpreter.debug = True

    for old_flag, new_flag in DEPRECATED_FLAGS.items():
        if old_flag in sys.argv:
            print(f"\n`{old_flag}` has been renamed to `{new_flag}`.\n")
            time.sleep(1.5)
            sys.argv.remove(old_flag)
            sys.argv.append(new_flag)

    class CustomHelpParser(argparse.ArgumentParser):
        def print_help(self, *args, **kwargs):
            super().print_help(*args, **kwargs)
            special_help_message = '''
Open Interpreter, 2024

Use """ to write multi-line messages.
            '''
            print(special_help_message)

    parser = CustomHelpParser(description="Open Interpreter", usage="%(prog)s [options]")

    # Add arguments
    for arg in arguments:
        default = arg.get("default")
        action = arg.get("action", "store_true")
        nickname = arg.get("nickname")

        name_or_flags = [f"--{arg['name']}"]
        if nickname:
            name_or_flags.append(f"-{nickname}")

        # Construct argument name flags
        flags = [f"-{nickname}", f"--{arg['name']}"] if nickname else [f"--{arg['name']}"]

        if arg["type"] == bool:
            parser.add_argument(
                *flags,
                dest=arg["name"],
                help=arg["help_text"],
                action=action,
                default=default,
            )
        else:
            choices = arg.get("choices")
            parser.add_argument(
                *flags,
                dest=arg["name"],
                help=arg["help_text"],
                type=arg["type"],
                choices=choices,
                default=default,
                nargs=arg.get("nargs"),
            )

    args, unknown_args = parser.parse_known_args()

    # handle unknown arguments
    if unknown_args:
        print(f"\nUnrecognized argument(s): {unknown_args}")
        parser.print_usage()
        print(
            "For detailed documentation of supported arguments, please visit: https://docs.openinterpreter.com/settings/all-settings"
        )
        sys.exit(1)

    if args.profiles:
        open_storage_dir("profiles")
        return

    if args.local_models:
        open_storage_dir("models")
        return

    # nargs="?" gives None when the flag is passed with no value, and the
    # sentinel default when it is absent. Testing for None as well meant the
    # documented bare `--reset_profile` fell through and started a session.
    if args.reset_profile != "NOT_PROVIDED":
        reset_profile(args.reset_profile)  # None means "every default profile"
        return

    if args.version:
        oi_version = version("open-interpreter")
        update_name = "Developer Preview"  # Change this with each major update
        print(f"Open Interpreter {oi_version} {update_name}")
        return

    # Deferred: these import litellm (directly or transitively). Placed after all
    # quick-exit flags so --help / --version / --profiles etc. never load them.
    from interpreter.core.utils.prompt_choice import prompt_choice
    from interpreter.terminal_interface.contributing_conversations import (
        contribute_conversation_launch_logic,
        contribute_conversations,
    )

    from .conversation_navigator import conversation_navigator
    from .validate_llm_settings import validate_llm_settings

    if args.no_highlight_active_line:
        interpreter.highlight_active_line = False

    ### Set attributes on interpreter, so that a profile script can read the arguments passed in via the CLI

    set_attributes(args, arguments)

    ### Apply profile

    # Profile shortcuts, which should probably not exist:

    if args.fast:
        args.profile = "fast.yaml"

    if args.vision:
        args.profile = "vision.yaml"

    if args.os:
        args.profile = "os.py"

    if args.local:
        args.profile = "local.py"
        if args.vision:
            # This is local vision, set up moondream!
            interpreter.toolbox.vision.load()
        if args.os:
            args.profile = "local-os.py"

    if args.codestral:
        args.profile = "codestral.py"
        if args.vision:
            args.profile = "codestral-vision.py"
        if args.os:
            args.profile = "codestral-os.py"

    if args.assistant:
        args.profile = "assistant.py"

    if args.llama3:
        args.profile = "llama3.py"
        if args.vision:
            args.profile = "llama3-vision.py"
        if args.os:
            args.profile = "llama3-os.py"

    if args.groq:
        args.profile = "groq.py"

    interpreter = profile(
        interpreter,
        args.profile or get_argument_dictionary(arguments, "profile")["default"],
    )

    ### Set attributes on interpreter, because the arguments passed in via the CLI should override profile

    set_attributes(args, arguments)
    interpreter.disable_telemetry = (
        interpreter.disable_telemetry
        or os.getenv("DISABLE_TELEMETRY", "false").lower() == "true"
        or bool(args.disable_telemetry)
    )

    # Full auto-run and safe_mode scanning are incompatible; allowlist and
    # denylist modes are fine, since both still gate some code on approval.
    if interpreter.auto_run_mode == "all" and interpreter.safe_mode in ("ask", "auto"):
        interpreter.auto_run_mode = "prompt"

    ### Set some helpful settings we know are likely to be true

    if interpreter.llm.model == "gpt-4" or interpreter.llm.model == "openai/gpt-4":
        if interpreter.llm.context_window is None:
            interpreter.llm.context_window = 6500
        if interpreter.llm.max_tokens is None:
            interpreter.llm.max_tokens = 4096
        if interpreter.llm.supports_functions is None:
            interpreter.llm.supports_functions = False if "vision" in interpreter.llm.model else True

    elif interpreter.llm.model.startswith("gpt-4") or interpreter.llm.model.startswith("openai/gpt-4"):
        if interpreter.llm.context_window is None:
            interpreter.llm.context_window = 123000
        if interpreter.llm.max_tokens is None:
            interpreter.llm.max_tokens = 4096
        if interpreter.llm.supports_functions is None:
            interpreter.llm.supports_functions = False if "vision" in interpreter.llm.model else True

    if interpreter.llm.model.startswith("gpt-3.5-turbo") or interpreter.llm.model.startswith("openai/gpt-3.5-turbo"):
        if interpreter.llm.context_window is None:
            interpreter.llm.context_window = 16000
        if interpreter.llm.max_tokens is None:
            interpreter.llm.max_tokens = 4096
        if interpreter.llm.supports_functions is None:
            interpreter.llm.supports_functions = True

    ### Check for update

    try:
        if not interpreter.offline and not args.stdin:
            # This message should actually be pushed into the utility
            if check_for_update():
                interpreter.display_message(
                    "> **A new version of Open Interpreter is available.**\n>Please run: `pip install --upgrade open-interpreter`\n\n---"
                )
    except:
        # Doesn't matter
        pass

    if interpreter.llm.api_base:
        if (
            "/" not in interpreter.llm.model
            and not interpreter.llm.model.lower().startswith("openai/")
            and not interpreter.llm.model.lower().startswith("azure/")
            and not interpreter.llm.model.lower().startswith("deepseek/")
            and not interpreter.llm.model.lower().startswith("dashscope-us/")
            and not interpreter.llm.model.lower().startswith("dashscope-intl/")
            and not interpreter.llm.model.lower().startswith("ollama")
            and not interpreter.llm.model.lower().startswith("jan")
            and not interpreter.llm.model.lower().startswith("local")
        ):
            interpreter.llm.model = "openai/" + interpreter.llm.model
        elif interpreter.llm.model.lower().startswith("jan/"):
            # Strip jan/ from the model name
            interpreter.llm.model = interpreter.llm.model[4:]

    # If --conversations is used, run conversation_navigator
    if args.conversations:
        conversation_navigator(interpreter)
        return

    if interpreter.llm.model in [
        "claude-3.5",
        "claude-3-5",
        "claude-3.5-sonnet",
        "claude-3-5-sonnet",
    ]:
        interpreter.llm.model = "claude-sonnet-4-6"

    if not args.server:
        # This SHOULD RUN WHEN THE SERVER STARTS. But it can't rn because
        # if you don't have an API key, a prompt shows up, breaking the whole thing.
        validate_llm_settings(
            interpreter
        )  # This should actually just run interpreter.llm.load() once that's == to validate_llm_settings

    if args.server:
        interpreter.server.run()
        return

    interpreter.in_terminal_interface = True

    contribute_conversation_launch_logic(interpreter)

    # Standard in mode
    if args.stdin:
        stdin_input = input()
        interpreter.plain_text_display = True
        interpreter.chat(stdin_input)
    else:
        interpreter.chat()


def _run_computer_use_mode():
    """`interpreter --os`: the computer-use loop, previously started from interpreter/__init__.py at import time."""
    from rich import print as rich_print
    from rich.markdown import Markdown
    from rich.rule import Rule

    def print_markdown(message):
        for line in message.split("\n"):
            line = line.strip()
            if line == "":
                print("")
            elif line == "---":
                rich_print(Rule(style="white"))
            else:
                try:
                    rich_print(Markdown(line))
                except UnicodeEncodeError:
                    print("Error displaying line:", line)
        if "\n" not in message and message.startswith(">"):
            print("")

    try:
        if check_for_update():
            print_markdown(
                "> **A new version of Open Interpreter is available.**\n>Please run: `pip install --upgrade open-interpreter`\n\n---"
            )
    except Exception:
        pass  # offline or PyPI unreachable: the check is advisory

    if "--voice" in sys.argv:
        print("Coming soon...")

    from interpreter.computer_use.loop import run_async_main

    run_async_main()


def main():
    if "--os" in sys.argv:
        _run_computer_use_mode()
        return

    # --help / -h / --version exit immediately inside start_terminal_interface
    # (argparse calls sys.exit), so skip the heavy interpreter import entirely.
    _FAST_EXIT_FLAGS = {"--help", "-h", "--version"}
    if _FAST_EXIT_FLAGS.intersection(sys.argv):

        class _Stub:
            pass

        _stub = _Stub()
        _stub.llm = _Stub()
        start_terminal_interface(_stub)
        return  # unreachable; already exited

    from interpreter import interpreter

    try:
        start_terminal_interface(interpreter)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    except KeyboardInterrupt:
        try:
            from interpreter.core.utils.prompt_choice import NoInteractiveInput, prompt_choice
            from interpreter.terminal_interface.contributing_conversations import contribute_conversations

            interpreter.terminal.terminate()

            if not interpreter.offline and not interpreter.disable_telemetry:
                feedback = None
                if len(interpreter.messages) > 3:
                    try:
                        raw = prompt_choice(
                            "\n\nWas Open Interpreter helpful? (y/n): ",
                            ("y", "n"),
                        )
                        feedback = raw == "y"
                        if feedback is not None and not interpreter.contribute_conversation:
                            if interpreter.llm.model == "i":
                                contribute = "y"
                            else:
                                print(
                                    "\nThanks for your feedback! Would you like to send us this chat so we can improve?\n"
                                )
                                contribute = prompt_choice("(y/n): ", ("y", "n"))

                            if contribute == "y":
                                interpreter.contribute_conversation = True
                                interpreter.display_message("\n*Thank you for contributing!*\n")
                    except NoInteractiveInput:
                        # Optional end-of-session survey. Nobody is there to
                        # answer it, so leave feedback unset and carry on.
                        feedback = None

                if (interpreter.contribute_conversation or interpreter.llm.model == "i") and interpreter.messages != []:
                    conversation_id = interpreter.conversation_id if hasattr(interpreter, "conversation_id") else None
                    contribute_conversations([interpreter.messages], feedback, conversation_id)

        except KeyboardInterrupt:
            pass
    finally:
        interpreter.terminal.terminate()
