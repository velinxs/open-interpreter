"""Asking the user to approve running code or applying an edit.

respond() pauses on a confirmation chunk whenever the auto-run policy says a
human must decide; this is the terminal's answer to it: y runs, n declines
with a notice the model can read, e opens an editor, a adds the command to
the allowlist.
"""

import os
import platform
import shutil
import subprocess
import tempfile
import time

from rich.box import ROUNDED
from rich.console import Console as RichConsole
from rich.padding import Padding as RichPadding
from rich.panel import Panel
from rich.text import Text as RichText

from ..core.utils.execution_allowlist import (
    persist_allowlist_rule,
    should_require_execution_confirmation,
)
from ..core.utils.prompt_choice import NoInteractiveInput, prompt_choice
from ..core.utils.scan_code import scan_code
from .components.code_block import CodeBlock
from .utils.display_markdown_message import display_markdown_message
from .utils.target_file_lexer import syntax_lang_for_dry_run


def _prompt_or_skip(prompt, choices, skipped="n"):
    """
    prompt_choice, but returns `skipped` when there is no TTY to ask.

    Only for prompts where "no" is a harmless outcome the model can work
    around: not showing an image, not scanning code. Prompts where skipping
    would lose work or hide the reason handle NoInteractiveInput themselves.
    """
    try:
        return prompt_choice(prompt, choices)
    except NoInteractiveInput:
        return skipped


# Shown to the model instead of running/editing when nothing can approve the step.
# It has to say "don't retry", or the model loops on the same rejected action.
NO_APPROVER_NOTICE = (
    "[This step needed approval and there is no interactive terminal to give"
    " it, so it was not performed. Do not retry the same action. Either"
    " continue with something that does not require approval, or stop and say"
    " what needs approving. To approve automatically, start Open Interpreter"
    " with -y (or set auto_run).]"
)


def _display_edit_dry_run(output, *, interpreter, target, edit_language, ok=True):
    if not output:
        return
    # File-format highlighting only for successful previews; errors are plain text.
    fence_lang = syntax_lang_for_dry_run(target, edit_language) if ok else "text"
    fence = "````" if "```" in output else "```"
    block = f"{fence}{fence_lang}\n{output}\n{fence}"
    if interpreter.plain_text_display:
        print("  Dry run:", flush=True)
        print(block, flush=True)
        print("", flush=True)
    else:
        display_markdown_message(f"**Dry run:**\n\n{block}\n")


def handle_confirmation(interpreter, chunk, active_block):
    """Answer one confirmation chunk.

    Returns the (possibly new) active block and "break" to end the turn or
    "continue" to move on to the next chunk.
    """
    removed_notice = None
    confirm_content = chunk.get("content")
    if isinstance(confirm_content, dict):
        removed_notice = confirm_content.get("removed")

    if should_require_execution_confirmation(interpreter, chunk):
        # OI is about to execute code or apply an edit. The user wants to approve this

        # End the active block so you can run input() below it.
        # finalize() must be called first (if the block has it) to
        # properly clear and stop the Rich Live display and flush any
        # buffered content to the permanent console.  Without it,
        # live.stop() fires while the Live viewport still has rendered
        # content, leaving the terminal cursor inside/below the panel
        # rather than at a clean line boundary — causing the subsequent
        # input("> ") prompt to appear on the wrong line.
        if active_block and not interpreter.plain_text_display:
            if hasattr(active_block, "finalize"):
                active_block.finalize()
            active_block.end()
            active_block = None

        if chunk.get("format") == "edit":
            # Edit tool confirmation — ask y/n (no "edit" option since the
            # content is structured JSON). Target path is already shown
            # above the code block; dry-run output (if any) is below.
            edit_info = chunk["content"]
            language = edit_info["format"]
            code = edit_info["content"]
            target = edit_info["target"]

            _display_edit_dry_run(
                edit_info.get("dry_run_output"),
                interpreter=interpreter,
                target=target,
                edit_language=language,
                ok=edit_info.get("dry_run_ok", True),
            )

            # Target path is already shown above the code block.
            print("", flush=True)
            edit_prompt = (
                "Would you like to apply this edit? (y/n)\n\n"
                if interpreter.plain_text_display
                else "  Would you like to apply this edit? (y/n)\n\n  "
            )
            try:
                response = prompt_choice(edit_prompt, ("y", "n"))
                declined_notice = "[User declined to apply this edit.]"
            except NoInteractiveInput:
                response = "n"
                declined_notice = NO_APPROVER_NOTICE

            if response == "y":
                active_block = CodeBlock(interpreter)
                active_block.margin_top = False
                active_block.language = language
                active_block.target_path = target
                # Don't repeat the code — it was already shown in the
                # streaming preview block above the confirmation prompt.
            else:
                interpreter.messages.append(
                    {
                        "role": "user",
                        "type": "message",
                        "content": declined_notice,
                        "sent_at": time.time(),
                        "source": "terminal",
                    }
                )
                return active_block, "break"
            return active_block, "continue"

        code_to_run = chunk["content"]
        language = code_to_run["format"]
        code = code_to_run["content"]

        should_scan_code = False

        if not interpreter.safe_mode == "off":
            if interpreter.safe_mode == "auto":
                should_scan_code = True
            elif interpreter.safe_mode == "ask":
                print("", flush=True)
                response = _prompt_or_skip(
                    "  Would you like to scan this code? (y/n)\n\n  ",
                    ("y", "n"),
                )
                if response == "y":
                    should_scan_code = True

        if should_scan_code:
            scan_code(code, language, interpreter)

        # Blank line before approval: leading \n inside input() is often lost after
        # Rich/console output on Windows (cursor sits on last panel row).
        print("", flush=True)
        if removed_notice:
            print(f"  [{removed_notice}]", flush=True)
            print("", flush=True)
        if interpreter.auto_run_mode == "allowlist":
            run_prompt = (
                "Would you like to run this code? (y/n/e = edit / a = add to allowlist)\n\n"
                if interpreter.plain_text_display
                else "  Would you like to run this code? (y/n/e = edit / a = add to allowlist)\n\n  "
            )
            run_choices = ("y", "n", "e", "a")
        else:
            run_prompt = (
                "Would you like to run this code? (y/n/e = edit)\n\n"
                if interpreter.plain_text_display
                else "  Would you like to run this code? (y/n/e = edit)\n\n  "
            )
            run_choices = ("y", "n", "e")
        try:
            response = prompt_choice(run_prompt, run_choices)
            declined_notice = "[User declined to run this code.]"
        except NoInteractiveInput:
            # Never fall back to a choice here. run_choices ends in
            # "e" (edit) or "a" (add to allowlist), so "the last
            # option" would open an editor or permanently approve
            # the command. Reject the step instead.
            response = "n"
            declined_notice = NO_APPROVER_NOTICE

        if response == "a":
            rule, added = persist_allowlist_rule(interpreter, language, code)
            if added:
                print(
                    f'  Added to allowlist: {rule["language"]} exact "{rule["pattern"]}"',
                    flush=True,
                )
                print("", flush=True)
            active_block = CodeBlock(interpreter)
            active_block.margin_top = False
            active_block.language = language
            should_highlight = (
                interpreter.highlight_active_line
                if hasattr(interpreter, "highlight_active_line") and interpreter.highlight_active_line is not None
                else True
            )
            if should_highlight:
                active_block.code = code
        elif response == "y":
            # Create a new, identical block where the code will actually be run
            # Conveniently, the chunk includes everything we need to do this:
            active_block = CodeBlock(interpreter)
            active_block.margin_top = False  # <- Aesthetic choice
            active_block.language = language

            should_highlight = (
                interpreter.highlight_active_line
                if hasattr(interpreter, "highlight_active_line") and interpreter.highlight_active_line is not None
                else True
            )
            if should_highlight:
                active_block.code = code
            # If should_highlight is False and the code hasn't been edited,
            # we leave active_block.code empty to avoid printing a duplicate
            # static code block below the y/n prompt.
        elif response == "e":
            # Edit - use mkstemp with the correct extension so the editor
            # can apply syntax highlighting. mkstemp is used instead of
            # NamedTemporaryFile because on Windows NamedTemporaryFile holds
            # an exclusive lock that prevents other processes from opening it.
            extension_map = {
                "python": ".py",
                "javascript": ".js",
                "typescript": ".ts",
                "cmd": ".bat",
                "bash": ".sh",
                "r": ".r",
                "ruby": ".rb",
                "java": ".java",
                "html": ".html",
                "css": ".css",
                "sql": ".sql",
                "powershell": ".ps1",
                "applescript": ".applescript",
            }
            suffix = extension_map.get(language.lower(), f".{language.lower()}")
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            try:
                # Close the fd immediately so the editor can open the file
                # on Windows (where open fds prevent other processes from
                # accessing the file).
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(code)

                editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
                if editor:
                    subprocess.call([editor, tmp_path])
                elif platform.system() == "Windows":
                    # "open" runs scripts or opens .html in a browser; use "edit".
                    # Other types (e.g. .py) often have no "edit" verb registered.
                    if suffix.lower() in (
                        ".bat",
                        ".cmd",
                        ".ps1",
                        ".vbs",
                        ".html",
                        ".htm",
                    ):
                        try:
                            os.startfile(tmp_path, "edit")
                        except OSError:
                            notepad = os.path.join(
                                os.environ.get("SystemRoot", r"C:\Windows"),
                                "System32",
                                "notepad.exe",
                            )
                            subprocess.call([notepad, tmp_path])
                    else:
                        os.startfile(tmp_path)
                else:
                    # Try common Linux editors in order
                    # Skip GUI editors if no display is available
                    has_display = os.environ.get("DISPLAY") is not None
                    editors = ["gedit", "kate"] if has_display else []
                    editors.extend(["nano", "vi"])

                    for try_editor in editors:
                        if shutil.which(try_editor):
                            subprocess.call([try_editor, tmp_path])
                            break
                    else:
                        # No editor found - show error and pause
                        print("\n" + "=" * 60)
                        print("ERROR: Could not find a suitable text editor.")
                        print("Please set one of these environment variables:")
                        print("  - VISUAL (preferred, e.g., gedit, kate)")
                        print("  - EDITOR (e.g., nano, vi)")
                        print("Or install one of: gedit, kate, nano, vi")
                        print("=" * 60 + "\n")
                        input("Press Enter to continue without editing...")
                        return active_block, "continue"

                # On Windows, GUI editors like Notepad++ may be set as a
                # Notepad replacement and return immediately by opening the
                # file in an existing instance. We wait for explicit
                # confirmation so the user has time to finish editing before
                # we read the file, regardless of editor type.
                input("  Press Enter when done editing...")
                print("")

                with open(tmp_path, encoding="utf-8") as f:
                    code = f.read()
            finally:
                os.unlink(tmp_path)

            original_code = interpreter.messages[-1]["content"]
            interpreter.messages[-1]["content"] = code  # Give it code

            # Let the LLM know the code was user-edited before running,
            # including the original so it can see what changed.
            # source="terminal" marks this as UI-injected so %undo skips it.
            interpreter.messages.append(
                {
                    "role": "user",
                    "type": "message",
                    "content": (
                        f"[The user edited your code before running it. "
                        f"Your original code was:\n```{language}\n{original_code}\n```\n"
                        f"The version above is what actually ran.]"
                    ),
                    "sent_at": time.time(),
                    "source": "terminal",
                }
            )

            active_block = CodeBlock(interpreter)
            active_block.margin_top = False  # <- Aesthetic choice
            active_block.language = language
            active_block.code = code
        else:
            # User declined to run code. source="terminal"
            # marks this as UI-injected so %undo skips it
            # when finding the last real user message.
            interpreter.messages.append(
                {
                    "role": "user",
                    "type": "message",
                    "content": declined_notice,
                    "sent_at": time.time(),
                    "source": "terminal",
                }
            )
            return active_block, "break"

    else:
        # Auto-run mode — `auto_run_mode` is "all" or the code is
        # allowlisted, so no confirmation prompt appears. This is
        # the only chance to tell the user that redundant
        # boilerplate was stripped; in the confirmation path above
        # the same notice is printed right before the run prompt.
        if removed_notice:
            print(f"  [{removed_notice}]", flush=True)
            print("", flush=True)

    # The confirmation chunk has been fully handled above (y/n/e all
    # either break, continue, or fall into the active_block setup).
    # Skip the rest of the loop body so the chunk isn't re-processed
    # by the plain-text or rich-display paths below.
    return active_block, "continue"
