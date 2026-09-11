"""
This is NOT jupyter language, this is just python.
Gotta split this out, generalize it, and move all the python additions to python.py, which imports this
"""

import ast
import logging
import os
import queue
import re
import sys
import threading
import time
import traceback

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
# litellm reads this at its first import. The package imports litellm lazily,
# at the first model call, so setting it here (when the language loads) is early
# enough for the local cost map to take effect.
from jupyter_client import KernelManager

from ..base_language import BaseLanguage
from .python_preprocess import (
    add_active_line_prints,
    preprocess_python,
    string_to_python,
    strip_redundant_imports,
    wrap_in_try_except,
)

DEBUG_MODE = False

# When running from an executable, ipykernel calls itself infinitely
# This is a workaround to detect it and launch it manually
if "ipykernel_launcher" in sys.argv:
    if sys.path[0] == "":
        del sys.path[0]

    from ipykernel import kernelapp as app

    app.launch_new_instance()
    sys.exit(0)


class JupyterLanguage(BaseLanguage):
    file_extension = "py"
    name = "python"

    def __init__(self, interpreter):
        self.interpreter = interpreter

        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        while not self.kc.is_alive():
            time.sleep(0.1)
        time.sleep(0.5)

        self.listener_thread = None
        self.finish_flag = False

        # Modules known to be bound in the kernel's user namespace. Populated
        # from each block's imports and refreshed from the REPL-state line the
        # kernel reports after every run, so redundant top-level `import X`
        # lines can be stripped before execution.
        self.imported_modules = set()

        # Use Inline by default for broad compatibility. Users can opt into a GUI backend
        # (e.g. TkAgg/QtAgg) by setting INTERPRETER_MPL_BACKEND or MPLBACKEND.
        # INTERPRETER_MPL_BACKEND takes precedence so Open Interpreter can control behavior.
        configured_backend = os.environ.get("INTERPRETER_MPL_BACKEND", "").strip()
        backend_source = "INTERPRETER_MPL_BACKEND"
        if not configured_backend:
            configured_backend = os.environ.get("MPLBACKEND", "").strip()
            backend_source = "MPLBACKEND"

        if configured_backend:
            code = f"""
import matplotlib
matplotlib.use({configured_backend!r})
import matplotlib.pyplot as plt
_oi_mpl_backend_hint_shown = False
_oi_mpl_original_show = plt.show
def _oi_mpl_show_with_hint(*args, **kwargs):
    global _oi_mpl_backend_hint_shown
    if not _oi_mpl_backend_hint_shown:
        print("Matplotlib backend set by {backend_source}={configured_backend!r}.")
        _oi_mpl_backend_hint_shown = True
    # GUI backends (e.g. TkAgg) can block the OI loop on show(); default to
    # non-blocking so users can interact with the figure while code execution continues.
    kwargs.setdefault("block", False)
    return _oi_mpl_original_show(*args, **kwargs)
plt.show = _oi_mpl_show_with_hint
""".strip()
        else:
            code = """
%matplotlib inline
import matplotlib.pyplot as plt
_oi_mpl_backend_hint_shown = False
_oi_mpl_original_show = plt.show
def _oi_mpl_show_with_hint(*args, **kwargs):
    global _oi_mpl_backend_hint_shown
    if not _oi_mpl_backend_hint_shown:
        print("Matplotlib backend not set; defaulting to inline. Set INTERPRETER_MPL_BACKEND=TkAgg (or MPLBACKEND=TkAgg) for interactive GUI plots (pan/zoom).")
        _oi_mpl_backend_hint_shown = True
    return _oi_mpl_original_show(*args, **kwargs)
plt.show = _oi_mpl_show_with_hint
""".strip()

        for _ in self.run(code):
            pass

        # Configure IPython display formatter to prefer markdown or plain text
        # so that pandas doesn't output dataframes as HTML tables (which are
        # then themselves executed).
        ipython_config = """
from IPython import get_ipython
ip = get_ipython()
ip.display_formatter.active_types = ['text/markdown', 'text/plain']
""".strip()

        for _ in self.run(ipython_config):
            pass

        # DISABLED because it doesn't work??
        # Disable color outputs in the terminal, which don't look good in OI and aren't useful
        # code = """
        # from IPython.core.getipython import get_ipython
        # get_ipython().colors = 'NoColor'
        # """
        # self.run(code)

    def terminate(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel()

    def run(self, code):
        while not self.kc.is_alive():
            time.sleep(0.1)

        ################################################################
        ### OFFICIAL OPEN INTERPRETER GOVERNMENT ISSUE SKILL LIBRARY ###
        ################################################################

        # try:
        #     functions = string_to_python(code)
        # except:
        #     # Non blocking
        #     functions = {}

        # if self.toolbox.save_skills and functions:
        #     skill_library_path = self.toolbox.skills.path

        #     if not os.path.exists(skill_library_path):
        #         os.makedirs(skill_library_path)

        #     for filename, function_code in functions.items():
        #         with open(f"{skill_library_path}/{filename}.py", "w") as file:
        #             file.write(function_code)

        self.finish_flag = False
        try:
            try:
                preprocessed_code = self.preprocess_code(code)
            except:
                # Any errors produced here are our fault.
                # Also, for python, you don't need them! It's just for active_line and stuff. Just looks pretty.
                preprocessed_code = code
            message_queue = queue.Queue()
            self._execute_code(preprocessed_code, message_queue)
            for output in self._capture_output(message_queue):
                self._maybe_update_imported_modules(output)
                yield output

            if getattr(self, "kc", None) and self.kc.is_alive():
                for output in self._get_active_state():
                    self._maybe_update_imported_modules(output)
                    yield output
        except GeneratorExit:
            raise  # gotta pass this up!
        except KeyboardInterrupt:
            # Properly handle KeyboardInterrupt: interrupt kernel, clear queue, set finish flag
            self.finish_flag = True
            try:
                self.km.interrupt_kernel()
            except:
                pass
            # Clear any remaining messages from the queue to prevent output sync issues
            while not message_queue.empty():
                try:
                    message_queue.get_nowait()
                except queue.Empty:
                    break
            yield {"type": "console", "format": "output", "content": "KeyboardInterrupt\n"}
        except:
            content = traceback.format_exc()
            yield {"type": "console", "format": "output", "content": content}

    def _execute_code(self, code, message_queue):
        def iopub_message_listener():
            max_retries = 100
            while True:
                # If self.finish_flag = True, and we didn't set it (we do below), we need to stop. That's our "stop"
                if self.finish_flag == True:
                    if DEBUG_MODE:
                        print("interrupting kernel!!!!!")
                    self.km.interrupt_kernel()
                    return
                # For async usage
                if hasattr(self.interpreter, "stop_event") and self.interpreter.stop_event.is_set():
                    self.km.interrupt_kernel()
                    self.finish_flag = True
                    return
                try:
                    msg = self.kc.iopub_channel.get_msg(timeout=0.05)
                except queue.Empty:
                    continue
                except Exception as e:
                    max_retries -= 1
                    if max_retries < 0:
                        raise
                    print("Jupyter error, retrying:", str(e))
                    continue

                if DEBUG_MODE:
                    print("-----------" * 10)
                    print("Message received:", msg["content"])
                    print("-----------" * 10)

                if msg["header"]["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                    # Set finish_flag and return when the kernel becomes idle
                    if DEBUG_MODE:
                        print("from thread: kernel is idle")
                    self.finish_flag = True
                    return

                content = msg["content"]

                if msg["msg_type"] == "stream":
                    line, active_line = self.detect_active_line(content["text"])
                    active_line_enabled = os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower() == "true"
                    if active_line and active_line_enabled:
                        message_queue.put(
                            {
                                "type": "console",
                                "format": "active_line",
                                "content": active_line,
                            }
                        )
                    message_queue.put({"type": "console", "format": "output", "content": line})
                elif msg["msg_type"] == "error":
                    content = "\n".join(content["traceback"])
                    # Remove color codes
                    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
                    content = ansi_escape.sub("", content)
                    message_queue.put(
                        {
                            "type": "console",
                            "format": "output",
                            "content": content,
                        }
                    )
                elif msg["msg_type"] in ["display_data", "execute_result"]:
                    data = content["data"]
                    if "image/png" in data:
                        message_queue.put(
                            {
                                "type": "image",
                                "format": "base64.png",
                                "content": data["image/png"],
                            }
                        )
                    elif "image/jpeg" in data:
                        message_queue.put(
                            {
                                "type": "image",
                                "format": "base64.jpeg",
                                "content": data["image/jpeg"],
                            }
                        )
                    elif "text/html" in data:
                        message_queue.put(
                            {
                                "type": "code",
                                "format": "html",
                                "content": data["text/html"],
                            }
                        )
                    elif "text/plain" in data:
                        message_queue.put(
                            {
                                "type": "console",
                                "format": "output",
                                "content": data["text/plain"],
                            }
                        )
                    elif "application/javascript" in data:
                        message_queue.put(
                            {
                                "type": "code",
                                "format": "javascript",
                                "content": data["application/javascript"],
                            }
                        )

        self.listener_thread = threading.Thread(target=iopub_message_listener)
        # self.listener_thread.daemon = True
        self.listener_thread.start()

        if DEBUG_MODE:
            print("thread is on:", self.listener_thread.is_alive(), self.listener_thread)

        self.kc.execute(code)

    def detect_active_line(self, line):
        if "##active_line" in line:
            # Split the line by "##active_line" and grab the last element
            last_active_line = line.split("##active_line")[-1]
            # Split the last active line by "##" and grab the first element
            try:
                active_line = int(last_active_line.split("##")[0])
            except:
                active_line = 0
            # Remove all ##active_line{number}##\n
            line = re.sub(r"##active_line\d+##\n", "", line)
            return line, active_line
        return line, None

    def _capture_output(self, message_queue):
        while True:
            time.sleep(0.1)

            # For async usage
            if hasattr(self.interpreter, "stop_event") and self.interpreter.stop_event.is_set():
                self.finish_flag = True
                break

            if self.listener_thread:
                try:
                    output = message_queue.get(timeout=0.1)
                    if DEBUG_MODE:
                        print(output)
                    yield output

                except queue.Empty:
                    if self.finish_flag:
                        time.sleep(0.1)

                        try:
                            output = message_queue.get(timeout=0.1)
                            if DEBUG_MODE:
                                print(output)
                            yield output
                        except queue.Empty:
                            if DEBUG_MODE:
                                print("we're done")
                            break
                except KeyboardInterrupt:
                    # Handle KeyboardInterrupt during output capture
                    self.finish_flag = True
                    # Clear any remaining messages to prevent sync issues
                    while not message_queue.empty():
                        try:
                            message_queue.get_nowait()
                        except queue.Empty:
                            break
                    break

    def stop(self):
        self.finish_flag = True

    def _get_active_state(self):
        state_code = """
import types as __oi_types
import os as __oi_os
__oi_globals = globals()
__oi_cwd = __oi_os.getcwd()
__oi_exclude = ['In', 'Out', 'get_ipython', 'exit', 'quit', 'open', 'original_ps1', 'is_wsl', 'REPLHooks', 'get_last_command', 'PS1', 'ip', 'plt']
__oi_mods = []
__oi_funcs = []
__oi_vars = []

for __oi_k, __oi_v in __oi_globals.items():
    if __oi_k.startswith('_') or __oi_k in __oi_exclude:
        continue
    if isinstance(__oi_v, __oi_types.ModuleType):
        __oi_mods.append(__oi_k)
    elif callable(__oi_v):
        __oi_funcs.append(__oi_k)
    else:
        __oi_vars.append(__oi_k)

__oi_parts = [f"CWD: {__oi_cwd}"]
if __oi_mods:
    __oi_parts.append(f"Already imported: {', '.join(__oi_mods)}")
if __oi_vars:
    __oi_parts.append(f"Variables: {', '.join(__oi_vars)}")
if __oi_funcs:
    __oi_parts.append(f"Functions/Classes: {', '.join(__oi_funcs)}")

__oi_res = f"\\n[Python REPL State: {' | '.join(__oi_parts)}]"
print(__oi_res)
"""
        message_queue = queue.Queue()
        self.finish_flag = False
        self._execute_code(state_code.strip(), message_queue)

        for output in self._capture_output(message_queue):
            if output.get("type") == "console" and output.get("format") == "output":
                yield output

    _STATE_MODULES_RE = re.compile(r"Already imported:\s*([^|\]]*)")

    def _maybe_update_imported_modules(self, output):
        """Refresh the tracked module set from the kernel's REPL-state line.

        The kernel reports exactly which modules are bound in its user
        namespace after every run, so when that line appears we adopt it
        wholesale — this corrects optimistic entries recorded from blocks that
        failed to execute (e.g. an `import sklearn` that raised).
        """
        if not isinstance(output, dict):
            return
        content = output.get("content")
        if not isinstance(content, str):
            return
        m = self._STATE_MODULES_RE.search(content)
        if not m:
            return
        modules = [name.strip() for name in m.group(1).split(",") if name.strip()]
        if modules:
            self.imported_modules = set(modules)

    def strip_boilerplate(self, code):
        """Return (stripped_code, notice) after removing redundant top-level imports.

        Removes plain ``import X`` lines for allowlisted boilerplate modules
        that the kernel has reported as already bound (via ``_get_active_state``
        after each run). This method deliberately does NOT learn new imports
        from the code it is handed: recording ``import time`` here would make a
        second ``strip_boilerplate`` call (e.g. from ``preprocess_code`` during
        the same execution) strip that import before it ever ran, breaking code
        that genuinely needs it. Only the kernel's authoritative REPL-state
        line drives ``imported_modules``.
        """
        stripped, removed = strip_redundant_imports(code, self.imported_modules)
        if removed:
            distinct = sorted(set(removed))[:4]
            label = "imports" if len(set(removed)) > 1 else "import"
            return stripped, f"Removed redundant {label} {', '.join(distinct)} (already imported)."
        return code, None

    def preprocess_code(self, code):
        code, _ = self.strip_boilerplate(code)
        return preprocess_python(code)
