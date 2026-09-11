"""Running the code block the model just wrote.

Repairs a few habitual malformations first (a functions.execute wrapper, a
JSON language block, a stray token), asks for confirmation unless the policy
says otherwise, runs it, and feeds the output back as console messages.

Returns "break" to end the turn, or None to let the model take another one.
"""

import json
import os
import re
import time
import traceback

from .toolbox.web.results import ApiKeyError, WebToolboxError


def run_pending_code(interpreter, state):
    """Execute interpreter.messages[-1] (a code message) and yield its chunks."""
    if interpreter.verbose:
        print("Running code:", interpreter.messages[-1])

    try:
        # What language/code do you want to run?
        language = interpreter.messages[-1]["format"].lower().strip()
        code = interpreter.messages[-1]["content"]

        # Short messages shown with the confirmation prompt whenever the
        # code the LLM wrote was rewritten before running — from the
        # hardcoded fixes below (functions.execute wrapper, stray
        # tokens, JSON language blocks) or from the redundant-boilerplate
        # strip. Accumulated so a single prompt can list several.
        notices = []

        if code.startswith("`\n"):
            code = code[2:].strip()
            if interpreter.verbose:
                print("Removing `\n")
            interpreter.messages[-1]["content"] = code  # So the LLM can see it.

        # A common hallucination
        if code.startswith("functions.execute("):
            edited_code = code.replace("functions.execute(", "").rstrip(")")
            try:
                code_dict = json.loads(edited_code)
                language = code_dict.get("language", language)
                code = code_dict.get("code", code)
                interpreter.messages[-1]["content"] = code  # So the LLM can see it.
                interpreter.messages[-1]["format"] = language  # So the LLM can see it.
                if "code" in code_dict:
                    notices.append("Extracted code from a `functions.execute()` wrapper.")
            except:
                pass

        # print(code)
        # print("---")
        # time.sleep(2)

        if code.strip().endswith("executeexecute"):
            code = code.replace("executeexecute", "")
            try:
                interpreter.messages[-1]["content"] = code  # So the LLM can see it.
            except:
                pass
            notices.append("Removed a stray trailing `executeexecute`.")

        if code.replace("\n", "").replace(" ", "").startswith('{"language":'):
            try:
                code_dict = json.loads(code)
                if set(code_dict.keys()) == {"language", "code"}:
                    language = code_dict["language"]
                    code = code_dict["code"]
                    interpreter.messages[-1]["content"] = code  # So the LLM can see it.
                    interpreter.messages[-1]["format"] = language  # So the LLM can see it.
                    notices.append("Extracted code from a JSON `{language: ...}` block.")
            except:
                pass

        if code.replace("\n", "").replace(" ", "").startswith("{language:"):
            try:
                code = code.replace("language: ", '"language": ').replace("code: ", '"code": ')
                code_dict = json.loads(code)
                if set(code_dict.keys()) == {"language", "code"}:
                    language = code_dict["language"]
                    code = code_dict["code"]
                    interpreter.messages[-1]["content"] = code  # So the LLM can see it.
                    interpreter.messages[-1]["format"] = language  # So the LLM can see it.
                    notices.append("Extracted code from a JSON `{language: ...}` block.")
            except:
                pass

        if language == "text" or language == "markdown" or language == "plaintext":
            # It does this sometimes just to take notes. Let it, it's useful.
            # In the future we should probably not detect this behavior as code at all.
            real_content = interpreter.messages[-1]["content"]
            interpreter.messages[-1] = {
                "role": "assistant",
                "type": "message",
                "content": f"```\n{real_content}\n```",
            }
            return "continue"

        # Is this language enabled/supported?
        if interpreter.terminal.get_language(language) is None:
            output = f"`{language}` disabled or not supported."

            yield {
                "role": "computer",
                "type": "console",
                "format": "output",
                "content": output,
            }

            # Let the response continue so it can deal with the unsupported code in another way. Also prevent looping on the same piece of code.
            if code != state.last_unsupported_code:
                state.last_unsupported_code = code
                return "continue"
            else:
                return "break"

        # Is there any code at all?
        if code.strip() == "":
            yield {
                "role": "computer",
                "type": "console",
                "format": "output",
                "content": "Code block was empty. Please try again, be sure to write code before executing.",
            }
            return "continue"

        # Strip redundant boilerplate (already-imported modules, cd to
        # the current dir) so the user previews — and the LLM records —
        # exactly what will run. The short notice travels with the
        # confirmation chunk and is shown beside the run prompt, not in
        # the command's terminal output.
        lang = interpreter.terminal.get_language_instance(language)
        if lang is not None:
            try:
                stripped, strip_notice = lang.strip_boilerplate(code)
                # Always adopt the stripped version when something was
                # removed — even if it reduces the block to nothing (a
                # lone redundant cd, or only already-imported modules).
                # The preview, the recorded message and the execution
                # must all agree: if there's nothing left, nothing runs.
                if strip_notice:
                    code = stripped
                    interpreter.messages[-1]["content"] = code
                    notices.append(strip_notice)
            except Exception:
                pass

        removed_notice = "; ".join(notices) if notices else None

        # Yield a message, such that the user can stop code execution if they want to
        try:
            yield {
                "role": "computer",
                "type": "confirmation",
                "format": "execution",
                "content": {
                    "type": "code",
                    "format": language,
                    "content": code,
                    "removed": removed_notice,
                },
            }
        except GeneratorExit:
            # The user might exit here.
            # We need to tell python what we (the generator) should do if they exit
            return "break"

        # They may have edited the code! Grab it again
        code = [m for m in interpreter.messages if m["type"] == "code"][-1]["content"]

        # don't let it import toolbox — we handle that!
        if interpreter.toolbox.import_toolbox_api and language == "python":
            # Check for nested imports like "from toolbox.ai2 import client"
            nested_import_match = re.search(r"from toolbox\.(\w+) import (.+)", code)
            if nested_import_match:
                module = nested_import_match.group(1)
                items = [item.strip() for item in nested_import_match.group(2).split(",")]
                first_item = items[0]
                raise ValueError(
                    f"Cannot import from `toolbox`. The `toolbox` object is already available as a variable in your namespace.\n"
                    f"Instead of: `from toolbox.{module} import {', '.join(items)}`\n"
                    f"Use directly: `toolbox.{module}.{first_item}` (and similarly for other items)\n"
                    f"For example, instead of `from toolbox.ai2 import client`, use `toolbox.ai2.client` directly.\n"
                    f"Do NOT import `toolbox` or try to import any of its sub-modules. The `toolbox` object is already available."
                )

            # Check for direct imports from toolbox
            direct_import_match = re.search(r"from toolbox import (.+)", code)
            if direct_import_match:
                items = [item.strip() for item in direct_import_match.group(1).split(",")]
                first_item = items[0]
                raise ValueError(
                    f"Cannot import from `toolbox`. The `toolbox` object is already available as a variable in your namespace.\n"
                    f"Instead of: `from toolbox import {', '.join(items)}`\n"
                    f"Use directly: `toolbox.{first_item}` (and similarly for other items)\n"
                    f"Do NOT import `toolbox` or try to import any of its sub-modules. The `toolbox` object is already available."
                )

            # Check for simple import statements
            if re.search(r"^import toolbox\b", code, re.MULTILINE):
                raise ValueError(
                    "Cannot import `toolbox`. The `toolbox` object is already available as a variable in your namespace.\n"
                    "Do NOT import `toolbox`. It is already available as a variable named `toolbox`.\n"
                    "Use `toolbox` directly without any import statement."
                )

            # Check for import toolbox.something
            if re.search(r"^import toolbox\.\w+", code, re.MULTILINE):
                raise ValueError(
                    "Cannot import from `toolbox`. The `toolbox` object is already available as a variable in your namespace.\n"
                    "Do NOT import `toolbox` or try to import any of its sub-modules. The `toolbox` object is already available.\n"
                    "Use `toolbox` directly without any import statement."
                )
            # If it does this it sees the screenshot twice (which is expected jupyter behavior)
            if any(
                code.strip().split("\n")[-1].startswith(text)
                for text in [
                    "toolbox.display.view",
                    "toolbox.display.screenshot",
                    "toolbox.view",
                    "toolbox.screenshot",
                ]
            ):
                code = code + "\npass"

        # sync up some things (is this how we want to do this?)
        interpreter.toolbox.verbose = interpreter.verbose
        interpreter.toolbox.debug = interpreter.debug
        interpreter.toolbox.emit_images = interpreter.llm.supports_vision
        interpreter.toolbox.max_output = interpreter.max_output

        # sync up the interpreter's toolbox with your toolbox
        try:
            if interpreter.sync_computer and language == "python":
                toolbox_dict = interpreter.toolbox.to_dict()
                if "_hashes" in toolbox_dict:
                    toolbox_dict.pop("_hashes")
                if "system_message" in toolbox_dict:
                    toolbox_dict.pop("system_message")
                toolbox_json = json.dumps(toolbox_dict)
                sync_code = f"""import json\ntoolbox.load_dict(json.loads('''{toolbox_json}'''))"""
                interpreter.terminal.run("python", sync_code)
        except Exception as e:
            if interpreter.debug:
                raise
            print(str(e))
            print("Failed to sync iToolbox with your Toolbox. Continuing...")

        ## ↓ CODE IS RUN HERE

        for line in interpreter.terminal.run(language, code, stream=True):
            yield {"role": "computer", **line}

        ## ↑ CODE IS RUN HERE

        # sync up your toolbox with the interpreter's toolbox
        try:
            if interpreter.sync_computer and language == "python":
                # sync up the interpreter's toolbox with your toolbox
                result = interpreter.terminal.run(
                    "python",
                    """
                    import json
                    toolbox_dict = toolbox.to_dict()
                    if '_hashes' in toolbox_dict:
                        toolbox_dict.pop('_hashes')
                    if "system_message" in toolbox_dict:
                        toolbox_dict.pop("system_message")
                    print(json.dumps(toolbox_dict))
                    """,
                )
                result = result[-1]["content"]
                interpreter.toolbox.load_dict(json.loads(result.strip('"').strip("'")))
        except Exception as e:
            if interpreter.debug:
                raise
            print(str(e))
            print("Failed to sync your Computer with iComputer. Continuing.")

        # yield final "active_line" message, as if to say, no more code is running. unhighlight active lines
        # (is this a good idea? is this our responsibility? i think so — we're saying what line of code is running! ...?)
        # Always yield end-of-execution signal so core can add empty output when needed.
        yield {
            "role": "computer",
            "type": "console",
            "format": "active_line",
            "content": None,
        }

    except KeyboardInterrupt:
        return "break"  # It's fine.
    except Exception as e:
        # For expected toolbox web/API errors, surface only the concise error
        # message to avoid cluttering the LLM context with full tracebacks.
        if isinstance(e, (WebToolboxError, ApiKeyError)):
            content = str(e)
        else:
            content = traceback.format_exc()
        yield {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": content,
        }
