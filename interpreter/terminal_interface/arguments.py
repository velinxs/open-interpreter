"""The CLI flag table, and applying parsed flags onto the interpreter.

Each entry names a flag, how argparse should read it, and (when the flag is
just a setting) the object and attribute it writes to, so adding a flag is a
row here rather than a branch in the entry point.
"""

import argparse

# =============================================================================
# DEVELOP BRANCH ONLY — DO NOT MERGE TO main / classic/main
#
# classic/develop loads `develop.yaml` by default so day-to-day fork settings
# stay separate from upstream `default.yaml`. Revert `_DEFAULT_PROFILE` to
# `"default.yaml"` before merging this file into main.
# =============================================================================
_DEFAULT_PROFILE = "develop.yaml"

# Flags renamed since; the entry point rewrites them with a notice.
DEPRECATED_FLAGS = {"--debug_mode": "--verbose"}


def build_arguments(interpreter):
    """The flag table. Takes the interpreter because rows point at its attributes."""
    return [
        {
            "name": "profile",
            "nickname": "p",
            "help_text": "name of profile. run `--profiles` to open profile directory",
            "type": str,
            "default": _DEFAULT_PROFILE,
        },
        {
            "name": "custom_instructions",
            "nickname": "ci",
            "help_text": "custom instructions for the language model. will be appended to the system_message",
            "type": str,
            "attribute": {"object": interpreter, "attr_name": "custom_instructions"},
        },
        {
            "name": "system_message",
            "nickname": "sm",
            "help_text": "(we don't recommend changing this) base prompt for the language model",
            "type": str,
            "attribute": {"object": interpreter, "attr_name": "system_message"},
        },
        {
            "name": "auto_run",
            "nickname": "y",
            "help_text": "automatically run generated code (same as --auto_run_mode all)",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "auto_run"},
        },
        {
            "name": "auto_run_mode",
            "help_text": "when to run code without asking: prompt, all, allowlist, or denylist",
            "type": str,
            "choices": ["prompt", "all", "allowlist", "denylist"],
            "default": None,
            "attribute": {"object": interpreter, "attr_name": "auto_run_mode"},
        },
        {
            "name": "no_highlight_active_line",
            "nickname": "nhl",
            "help_text": "turn off active line highlighting in code blocks",
            "type": bool,
            "action": "store_true",
            "default": False,  # Default to False, meaning highlighting is on by default
        },
        {
            "name": "verbose",
            "nickname": "v",
            "help_text": "print detailed logs",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "verbose"},
        },
        {
            "name": "model",
            "nickname": "m",
            "help_text": "language model to use (e.g. gpt-4o, openrouter/..., deepseek/..., dashscope-us/...)",
            "type": str,
            "attribute": {"object": interpreter.llm, "attr_name": "model"},
        },
        {
            "name": "temperature",
            "nickname": "t",
            "help_text": "optional temperature setting for the language model",
            "type": float,
            "attribute": {"object": interpreter.llm, "attr_name": "temperature"},
        },
        {
            "name": "llm_supports_vision",
            "nickname": "lsv",
            "help_text": "inform OI that your model supports vision, and can receive vision inputs",
            "type": bool,
            "action": argparse.BooleanOptionalAction,
            "attribute": {"object": interpreter.llm, "attr_name": "supports_vision"},
        },
        {
            "name": "llm_supports_functions",
            "nickname": "lsf",
            "help_text": "inform OI that your model supports OpenAI-style functions, and can make function calls",
            "type": bool,
            "action": argparse.BooleanOptionalAction,
            "attribute": {"object": interpreter.llm, "attr_name": "supports_functions"},
        },
        {
            "name": "shrink_images",
            "nickname": "si",
            "help_text": "resize large images before sending (may reduce detail). Default: off. When you upload images or approve view_image, you are asked y/n; this flag sets the default for images without that prompt (e.g. replay).",
            "type": bool,
            "action": argparse.BooleanOptionalAction,
            "attribute": {"object": interpreter, "attr_name": "shrink_images"},
        },
        {
            "name": "context_window",
            "nickname": "cw",
            "help_text": "optional context window size for the language model",
            "type": int,
            "attribute": {"object": interpreter.llm, "attr_name": "context_window"},
        },
        {
            "name": "max_tokens",
            "nickname": "x",
            "help_text": "optional maximum number of tokens for the language model",
            "type": int,
            "attribute": {"object": interpreter.llm, "attr_name": "max_tokens"},
        },
        {
            "name": "retention_ratio",
            "nickname": "rr",
            "help_text": "enable cache-aware truncation: when the prompt outgrows the context window, drop a variable number of oldest whole turns down to this fraction of the window (e.g. 0.8 keeps 80%% of the window and drops the oldest 20%% at once), keeping the prefix stable so the provider's KV cache stays warm",
            "type": float,
            "attribute": {"object": interpreter.llm, "attr_name": "retention_ratio"},
        },
        {
            "name": "max_budget",
            "nickname": "b",
            "help_text": "optionally set the max budget (in USD) for your llm calls",
            "type": float,
            "attribute": {"object": interpreter.llm, "attr_name": "max_budget"},
        },
        {
            "name": "api_base",
            "nickname": "ab",
            "help_text": "optionally set the API base URL for your llm calls (this will override environment variables)",
            "type": str,
            "attribute": {"object": interpreter.llm, "attr_name": "api_base"},
        },
        {
            "name": "api_key",
            "nickname": "ak",
            "help_text": "optionally set the API key for your llm calls (this will override environment variables)",
            "type": str,
            "attribute": {"object": interpreter.llm, "attr_name": "api_key"},
        },
        {
            "name": "api_version",
            "nickname": "av",
            "help_text": "optionally set the API version for your llm calls (this will override environment variables)",
            "type": str,
            "attribute": {"object": interpreter.llm, "attr_name": "api_version"},
        },
        {
            "name": "sanitize_secrets",
            "help_text": "redact API keys/passwords from messages sent to the LLM (auto=only for API models, on=always, off=never)",
            "type": str,
            "choices": ["auto", "on", "off"],
            "default": "auto",
            "attribute": {"object": interpreter.llm, "attr_name": "sanitize_secrets"},
        },
        {
            "name": "max_output",
            "nickname": "xo",
            "help_text": "optional maximum number of characters for code outputs",
            "type": int,
            "attribute": {"object": interpreter, "attr_name": "max_output"},
        },
        {
            "name": "loop",
            "help_text": "runs OI in a loop, requiring it to admit to completing/failing task",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "loop"},
        },
        {
            "name": "disable_telemetry",
            "nickname": "dt",
            "help_text": "disables sending of basic anonymous usage stats",
            "type": bool,
            # default must be None (not False) so set_attributes does not overwrite
            # disable_telemetry: true from the user's profile when the flag is omitted.
            "attribute": {"object": interpreter, "attr_name": "disable_telemetry"},
        },
        {
            "name": "offline",
            "nickname": "o",
            "help_text": "turns off all online features (except the language model, if it's hosted)",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "offline"},
        },
        {
            "name": "speak_messages",
            "nickname": "sp",
            "help_text": "(Mac only, experimental) use the applescript `say` command to read messages aloud",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "speak_messages"},
        },
        {
            "name": "safe_mode",
            "nickname": "safe",
            "help_text": "optionally enable safety mechanisms like code scanning; valid options are off, ask, and auto",
            "type": str,
            "choices": ["off", "ask", "auto"],
            "default": "off",
            "attribute": {"object": interpreter, "attr_name": "safe_mode"},
        },
        {
            "name": "debug",
            "nickname": "debug",
            "help_text": "debug mode for open interpreter developers",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "debug"},
        },
        {
            "name": "fast",
            "nickname": "f",
            "help_text": "runs `interpreter --model gpt-4o-mini` and asks OI to be extremely concise (shortcut for `interpreter --profile fast`)",
            "type": bool,
        },
        {
            "name": "multi_line",
            "nickname": "ml",
            "help_text": "enable multi-line inputs starting and ending with ```",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "multi_line"},
        },
        {
            "name": "local",
            "nickname": "l",
            "help_text": "setup a local model (shortcut for `interpreter --profile local`)",
            "type": bool,
        },
        {
            "name": "codestral",
            "help_text": "shortcut for `interpreter --profile codestral`",
            "type": bool,
        },
        {
            "name": "assistant",
            "help_text": "shortcut for `interpreter --profile assistant.py`",
            "type": bool,
        },
        {
            "name": "llama3",
            "help_text": "shortcut for `interpreter --profile llama3`",
            "type": bool,
        },
        {
            "name": "groq",
            "help_text": "shortcut for `interpreter --profile groq`",
            "type": bool,
        },
        {
            "name": "vision",
            "nickname": "vi",
            "help_text": "experimentally use vision for supported languages (shortcut for `interpreter --profile vision`)",
            "type": bool,
        },
        {
            "name": "os",
            "nickname": "os",
            "help_text": "experimentally let Open Interpreter control your mouse and keyboard (shortcut for `interpreter --profile os`)",
            "type": bool,
        },
        # Special commands
        {
            "name": "reset_profile",
            "help_text": "reset a profile file. run `--reset_profile` without an argument to reset all default profiles",
            "type": str,
            "default": "NOT_PROVIDED",
            "nargs": "?",  # This means you can pass in nothing if you want
        },
        {"name": "profiles", "help_text": "opens profiles directory", "type": bool},
        {
            "name": "local_models",
            "help_text": "opens local models directory",
            "type": bool,
        },
        {
            "name": "conversations",
            "help_text": "list conversations to resume",
            "type": bool,
        },
        {
            "name": "server",
            "help_text": "start open interpreter as a server",
            "type": bool,
        },
        {
            "name": "version",
            "help_text": "get Open Interpreter's version number",
            "type": bool,
        },
        {
            "name": "contribute_conversation",
            "help_text": "let Open Interpreter use the current conversation to train an Open-Source LLM",
            "type": bool,
            "attribute": {
                "object": interpreter,
                "attr_name": "contribute_conversation",
            },
        },
        {
            "name": "plain",
            "nickname": "pl",
            "help_text": "set output to plain text",
            "type": bool,
            "attribute": {
                "object": interpreter,
                "attr_name": "plain_text_display",
            },
        },
        {
            "name": "stdin",
            "nickname": "s",
            "help_text": "Run OI in stdin mode",
            "type": bool,
        },
    ]


def set_attributes(args, arguments):
    for argument_name, argument_value in vars(args).items():
        if argument_value is not None:
            if argument_dictionary := get_argument_dictionary(arguments, argument_name):
                if "attribute" in argument_dictionary:
                    attr_dict = argument_dictionary["attribute"]
                    setattr(attr_dict["object"], attr_dict["attr_name"], argument_value)

                    if args.verbose:
                        print(
                            f"Setting attribute {attr_dict['attr_name']} on {attr_dict['object'].__class__.__name__.lower()} to '{argument_value}'..."
                        )


def get_argument_dictionary(arguments: list[dict], key: str) -> dict:
    if len(argument_dictionary_list := list(filter(lambda x: x["name"] == key, arguments))) > 0:
        return argument_dictionary_list[0]
    return {}
