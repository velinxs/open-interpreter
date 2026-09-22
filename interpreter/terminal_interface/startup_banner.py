"""What the session tells you about itself before the first prompt.

Starting the interpreter used to print nothing at all in the common case. The
one-time welcome is only reached from a branch in validate_llm_settings, and
the approval notice is suppressed whenever `offline` is set -- so a local-model
session, the configuration most likely to be unusual, was the one that showed
least. You could not tell which model answered, whether code would run without
asking, or how to leave.

So: the three facts that change what happens next -- which model, whether it
asks before running code, and where it will run it -- and the three keys worth
knowing on the first run.
"""

from rich.box import ROUNDED
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .utils.display_constants import PADDING_PANEL

# What each approval mode means in one word, from the user's side of the
# screen: the question is "will this thing run something without asking me".
_APPROVAL = {
    "all": ("runs code without asking", "yellow"),
    "prompt": ("asks before running code", "green"),
    "allowlist": ("asks unless allowlisted", "green"),
    "denylist": ("runs unless denylisted", "yellow"),
}


def _version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("open-interpreter")
    except PackageNotFoundError:
        return ""


def _working_directory():
    """The cwd, with $HOME collapsed -- an absolute path is mostly noise."""
    import os

    cwd = os.getcwd()
    home = os.path.expanduser("~")
    return "~" + cwd[len(home) :] if cwd.startswith(home) else cwd


def startup_banner(interpreter):
    """The panel shown once, before the first prompt."""
    model = getattr(getattr(interpreter, "llm", None), "model", None) or "no model set"
    mode = getattr(interpreter, "auto_run_mode", "prompt")
    approval, approval_colour = _APPROVAL.get(mode, (mode, "white"))

    facts = Table.grid(padding=(0, 1))
    facts.add_column(style="bright_black", justify="right")
    facts.add_column()
    facts.add_row("model", Text(model, style="cyan"))
    facts.add_row("approval", Text(approval, style=approval_colour))
    facts.add_row("cwd", Text(_working_directory(), style="white"))

    keys = Text.assemble(
        ("%help", "bold"),
        (" commands", "bright_black"),
        ("   ·   ", "bright_black"),
        ("/exit", "bold"),
        (" leave", "bright_black"),
        ("   ·   ", "bright_black"),
        ("esc", "bold"),
        (" stop a running command", "bright_black"),
    )

    body = Table.grid()
    body.add_column()
    body.add_row(facts)
    body.add_row("")
    body.add_row(keys)

    version = _version()
    title = Text.assemble(
        ("● ", "bold"),
        ("Open Interpreter", "bold"),
        (f" {version}" if version else "", "bright_black"),
    )
    Console().print(
        Padding(Panel(body, title=title, title_align="left", box=ROUNDED, padding=(0, 2)), PADDING_PANEL)
    )


def print_startup_banner(interpreter):
    """Show the banner, unless nothing is watching the terminal.

    plain_text_display is the proxy for --stdin, where output is somebody's
    data rather than a screen, so a panel of box-drawing characters would be
    corruption rather than decoration.
    """
    if getattr(interpreter, "plain_text_display", False):
        return
    # `i {command}` starts with the message already queued and prints one
    # answer; a banner would be most of that output.
    if len(getattr(interpreter, "messages", [])) == 1:
        return
    startup_banner(interpreter)
