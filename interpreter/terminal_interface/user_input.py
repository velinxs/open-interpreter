"""What happens to a line the user typed before it becomes a message.

Magic commands, the two habitual mistakes people type at the prompt, and
dragged-in image paths are handled here, along with the one-time banner
saying which code this session will ask about.
"""

import time

from rich.box import ROUNDED
from rich.console import Console as RichConsole
from rich.padding import Padding as RichPadding
from rich.panel import Panel
from rich.text import Text as RichText

from ..core.llm.utils.convert_to_openai_messages import image_path_exceeds_shrink_threshold
from .approval import _prompt_or_skip
from .magic_commands import handle_magic_command
from .utils.check_for_package import check_for_package
from .utils.display_constants import PADDING_PANEL
from .utils.find_image_path import find_image_path


def _print_mode_banner(interpreter):
    """The one-time notice about which code needs approval in this session."""
    if interpreter.auto_run_mode != "all" and not interpreter.offline and not (len(interpreter.messages) == 1):
        interpreter_intro_message = ["**Open Interpreter** will require approval before running code."]

        if interpreter.auto_run_mode == "allowlist":
            interpreter_intro_message.append(
                "**Allowlist mode**: only exact allowlisted commands run without approval."
            )

        if interpreter.auto_run_mode == "denylist":
            interpreter_intro_message.append(
                "**Denylist mode**: everything runs without approval *except* commands matching a denylist rule."
            )

        if interpreter.safe_mode == "ask" or interpreter.safe_mode == "auto":
            if not check_for_package("semgrep"):
                interpreter_intro_message.append(
                    f"**Safe Mode**: {interpreter.safe_mode}\n\n>Note: **Safe Mode** requires `semgrep` (`pip install semgrep`)"
                )
        elif interpreter.auto_run_mode == "prompt":
            interpreter_intro_message.append("Use `interpreter -y` to bypass this.")

        if (
            not interpreter.plain_text_display
        ):  # A proxy/heuristic for standard in mode, which isn't tracked (but prob should be)
            interpreter_intro_message.append("Press `CTRL-C` to exit.")

        interpreter.display_message("\n\n".join(interpreter_intro_message) + "\n")
        print()


def _prepare_message(interpreter, message, interactive):
    """Handle a typed line before it becomes a user message.

    Magic commands, the two habitual mistakes users type at the prompt, and
    dragged-in image paths are all dealt with here. Returns the message to
    send, or None when there is nothing to send this time round.
    """
    if message == "":
        # Ignore empty messages when user presses enter without typing anything
        return None

    if message.startswith("%") and interactive:
        handle_magic_command(interpreter, message)
        return None

    # Many users do this
    if message.strip() == "interpreter --local":
        print("Please exit this conversation, then run `interpreter --local`.")
        return None
    if message.strip() == "pip install --upgrade open-interpreter":
        print("Please exit this conversation, then run `pip install --upgrade open-interpreter`.")
        return None

    if interpreter.llm.supports_vision or interpreter.llm.vision_renderer != None:
        # Is the input a path to an image? Like they just dragged it into the terminal?
        image_paths = find_image_path(message)

        ## If we found images, ask for approval before uploading
        if image_paths:
            _console = RichConsole(emoji=False)
            _content = RichText()
            for p in image_paths:
                _content.append(p + "\n")
            _any_large = any(image_path_exceeds_shrink_threshold(p) for p in image_paths)
            if _any_large:
                _content.append("\nf = upload full resolution\nr = upload with resize if large\nn = don't upload")
                _panel = Panel(_content, title="Image Detected", box=ROUNDED, padding=(0, 1))
                _console.print(RichPadding(_panel, PADDING_PANEL))
                response = _prompt_or_skip("  ", ("f", "r", "n"))
            else:
                _content.append("\ny = upload image\nn = don't upload")
                _panel = Panel(_content, title="Image Detected", box=ROUNDED, padding=(0, 1))
                _console.print(RichPadding(_panel, PADDING_PANEL))
                response = _prompt_or_skip("  ", ("y", "n"))

            if (_any_large and response in ("f", "r")) or (not _any_large and response == "y"):
                _shrink = _any_large and response == "r"
                # Add the text message to interpreter's message history
                interpreter.messages.append(
                    {
                        "role": "user",
                        "type": "message",
                        "content": message,
                        "sent_at": time.time(),
                    }
                )

                # Add remaining image messages (first one will be added by chat() when we pass it)
                for image_path in image_paths[1:]:
                    interpreter.messages.append(
                        {
                            "role": "user",
                            "type": "image",
                            "format": "path",
                            "content": image_path,
                            "sent_at": time.time(),
                            "shrink": _shrink,
                        }
                    )

                # Pass the first image dict to chat to trigger processing
                message = {
                    "role": "user",
                    "type": "image",
                    "format": "path",
                    "content": image_paths[0],
                    "sent_at": time.time(),
                    "shrink": _shrink,
                }
            else:
                # User declined, just process the text message normally
                pass
    return message
