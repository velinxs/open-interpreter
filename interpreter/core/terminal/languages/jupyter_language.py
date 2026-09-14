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
from .jupyter_display import display_chunk
from .python_preprocess import (
    add_active_line_prints,
    preprocess_python,
    string_to_python,
    strip_redundant_definitions_and_assignments,
    strip_redundant_imports,
    wrap_in_try_except,
)
from .python_state import PythonStateMixin
from .subprocess_language import DEFAULT_IDLE_TIMEOUT, _env_seconds

DEBUG_MODE = False

# When running from an executable, ipykernel calls itself infinitely
# This is a workaround to detect it and launch it manually
if "ipykernel_launcher" in sys.argv:
    if sys.path[0] == "":
        del sys.path[0]

    from ipykernel import kernelapp as app

    app.launch_new_instance()
    sys.exit(0)


class JupyterLanguage(PythonStateMixin, BaseLanguage):
    file_extension = "py"
    name = "python"

    # A kernel that exited takes the whole session's state with it. Say so once,
    # plainly, so the model recreates what it needs instead of assuming its
    # variables are still there.
    RESTART_NOTICE = (
        "\n[The Python kernel exited (exit(), sys.exit(), quit() or a crash) and has been "
        "restarted. Every variable, import and definition from earlier blocks is gone — "
        "recreate whatever you still need. Calling exit() ends the Python session, not the "
        "task; don't call it again.]\n"
    )
    DIED_NOTICE = "\n[The Python kernel exited before this block finished; its remaining output is lost.]\n"

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
        # The id of the execute_request currently being listened for.
        self._execution_id = None
        # Guard against a restart triggering another restart from the setup
        # code it runs on the new kernel.
        self._restarting = False

        # Modules known to be bound in the kernel's user namespace. Populated
        # from each block's imports and refreshed from the REPL-state line the
        # kernel reports after every run, so redundant top-level `import X`
        # lines can be stripped before execution.
        self.imported_modules = set()

        # Fingerprints of the top-level definitions the kernel currently binds,
        # refreshed wholesale from the hidden `##oi_fp##` marker after every
        # run: functions by their normalized-source fingerprint, scalars by
        # their repr fingerprint. An identical `def f` or `x = 5` in a later
        # block is a no-op, so it can be stripped before it runs.
        self.function_fingerprints = {}
        self.variable_fingerprints = {}

        self._configure_kernel()

    def _configure_kernel(self):
        """Run the per-kernel setup. Re-run verbatim after a restart."""
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
        # then themselves executed). The image formatters stay on: turning them
        # off (they are off by default once active_types is set) made every plot
        # and screenshot reach the model as "<Figure size 640x480 ...>". Whether
        # the picture is forwarded or dropped in favour of that text repr is
        # decided per output, by the model's vision support, in _display_chunk.
        ipython_config = """
from IPython import get_ipython
ip = get_ipython()
ip.display_formatter.active_types = ['text/markdown', 'text/plain', 'image/png', 'image/jpeg']
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

    def _kernel_is_ready(self, timeout=10):
        """True once the kernel process is up and its channels are running."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if not self.km.is_alive():
                    return False
            except Exception:
                return False
            if self.kc.is_alive():
                return True
            time.sleep(0.1)
        return False

    def _restart_kernel(self):
        """Replace a dead kernel with a fresh one. True if the new one is usable."""
        if self._restarting:
            return False
        self._restarting = True
        try:
            try:
                self.kc.stop_channels()
            except Exception:
                pass
            try:
                self.km.restart_kernel(now=True)
            except Exception:
                self.km = KernelManager(kernel_name="python3")
                self.km.start_kernel()
            self.kc = self.km.client()
            self.kc.start_channels()
            try:
                # Without this the first execute_request after a restart is sent
                # before the new kernel binds its shell socket, and is dropped.
                self.kc.wait_for_ready(timeout=60)
            except Exception:
                return False
            # The new kernel's namespace is empty, so nothing is imported or
            # defined any more and the toolbox/skills injections have to run
            # again. Keeping a fingerprint here would strip the re-definition
            # the model has to send to get its helper back.
            self.imported_modules = set()
            self.function_fingerprints = {}
            self.variable_fingerprints = {}
            toolbox = getattr(self.interpreter, "toolbox", None)
            if toolbox is not None:
                toolbox._has_imported_toolbox_api = False
                toolbox._has_imported_skills = False
            self.finish_flag = False
            self._execution_id = None
            self._configure_kernel()
            return True
        finally:
            self._restarting = False

    def run(self, code):
        # A block that calls exit()/sys.exit()/quit(), or one the OS kills, takes
        # the kernel with it. This used to spin here forever — the channels of a
        # dead kernel never come back alive — so the session was over with no
        # message and no way out but Ctrl-C.
        if not self._kernel_is_ready():
            if not self._restart_kernel():
                yield {
                    "type": "console",
                    "format": "output",
                    "content": "\n[The Python kernel exited and could not be restarted. Python is unavailable.]\n",
                }
                return
            yield {"type": "console", "format": "output", "content": self.RESTART_NOTICE}

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
        # Bash has had an idle timeout since a `find /` looked hung; Python never
        # did, so a call that blocks forever — a web request with no timeout, a
        # notification on a headless box, an input() nobody can answer — wedged
        # the session until the user found Ctrl-C. Same env var and default as
        # the shell, because "no output for this long" means the same thing in
        # both and one knob is easier to reason about than two.
        idle_timeout = _env_seconds("INTERPRETER_COMMAND_IDLE_TIMEOUT", DEFAULT_IDLE_TIMEOUT)

        # How long silence lasts before we say something. A block that runs for two
        # minutes without printing is indistinguishable from a hung one, and the
        # honest answer — "still going, here is how long, here is when I will stop
        # it" — is what turns a suspected hang back into a wait.
        heartbeat = min(15.0, idle_timeout / 4) if idle_timeout else 15.0

        def iopub_message_listener():
            max_retries = 100
            last_activity = time.monotonic()
            last_heartbeat = last_activity
            dead_since = None
            while True:
                silent_for = time.monotonic() - last_activity
                if heartbeat and silent_for >= heartbeat and time.monotonic() - last_heartbeat >= heartbeat:
                    last_heartbeat = time.monotonic()
                    limit = f", will interrupt at {idle_timeout:.0f}s" if idle_timeout else ""
                    message_queue.put(
                        {
                            # A notice is shown and then dropped. Putting these in the
                            # model's context would spend tokens telling it nothing it
                            # can act on, several times per slow command.
                            "type": "notice",
                            "format": "output",
                            "content": f"Still running, no output for {silent_for:.0f}s{limit}.",
                        }
                    )
                if idle_timeout and silent_for > idle_timeout:
                    self.km.interrupt_kernel()
                    message_queue.put(
                        {
                            "type": "console",
                            "format": "output",
                            "content": (
                                f"\n[Interrupted: no output for {idle_timeout:.0f}s. The code was still "
                                "running — it is most likely blocked on something that will not finish "
                                "(a request with no timeout, or a prompt nobody can answer). Set "
                                "INTERPRETER_COMMAND_IDLE_TIMEOUT to raise this limit, or pass an explicit "
                                "timeout to the call.]\n"
                            ),
                        }
                    )
                    self.finish_flag = True
                    return
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
                    # A kernel killed outright (os._exit, a segfault, the OOM
                    # killer) never sends its "idle", so waiting for one only
                    # burns the whole idle timeout before giving up. Nothing is
                    # queued right now either, so confirm over a second to let
                    # anything still in flight arrive, then stop.
                    if self.km.is_alive():
                        dead_since = None
                        continue
                    if dead_since is None:
                        dead_since = time.monotonic()
                        continue
                    if time.monotonic() - dead_since < 1:
                        continue
                    message_queue.put({"type": "console", "format": "output", "content": self.DIED_NOTICE})
                    self.finish_flag = True
                    return
                except Exception as e:
                    max_retries -= 1
                    if max_retries < 0:
                        raise
                    print("Jupyter error, retrying:", str(e))
                    continue

                # Only this execution's replies count. An interrupted block keeps
                # emitting its tail — traceback, REPL state, final idle status —
                # after its listener has gone, and those messages arrive while the
                # NEXT block is running. Reading them made the next command print
                # the previous one's output and, worse, stop at the previous one's
                # "idle": the code had been sent to the kernel and was running, but
                # nothing was left listening for its result, so it looked like the
                # command simply never ran. Jupyter stamps every reply with the id
                # of the request that caused it, which settles it exactly.
                parent_id = (msg.get("parent_header") or {}).get("msg_id")
                if parent_id and parent_id != self._execution_id:
                    continue

                # Any message from the kernel is proof it is still doing something.
                last_activity = time.monotonic()

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
                    chunk = self._display_chunk(content["data"])
                    if chunk:
                        message_queue.put(chunk)

        # Send first, listen second. The channel buffers, so nothing is missed, and
        # this is what gives us the request id before any message can be read.
        execution_id = self.kc.execute(code)
        self._execution_id = execution_id

        self.listener_thread = threading.Thread(target=iopub_message_listener)
        # self.listener_thread.daemon = True
        self.listener_thread.start()

        if DEBUG_MODE:
            print("thread is on:", self.listener_thread.is_alive(), self.listener_thread)

    def _display_chunk(self, data):
        """Forward a display, with images only if the model can actually see them."""
        vision = getattr(getattr(self.interpreter, "llm", None), "supports_vision", False) is True
        return display_chunk(data, vision)

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

    def strip_boilerplate(self, code):
        """Return (stripped_code, notice) after removing redundant top-level boilerplate.

        Two passes, both driven only by what the kernel reported after the last
        run:

        - ``strip_redundant_imports`` drops plain ``import X`` lines for
          allowlisted boilerplate modules the kernel has reported as already
          bound. It deliberately does NOT learn new imports from the code it is
          handed: recording ``import time`` here would make a second
          ``strip_boilerplate`` call (e.g. from ``preprocess_code`` during the
          same execution) strip that import before it ever ran, breaking code
          that genuinely needs it.
        - ``strip_redundant_definitions_and_assignments`` drops a top-level
          ``def``/``async def`` or scalar assignment that re-creates, byte for
          byte, something already bound in the kernel (matched by fingerprint).
          A re-definition with different code is always kept.
        """
        stripped, removed = strip_redundant_imports(code, self.imported_modules)
        notices = []
        if removed:
            distinct = sorted(set(removed))[:4]
            label = "imports" if len(set(removed)) > 1 else "import"
            notices.append(f"Removed redundant {label} {', '.join(distinct)} (already imported).")
        stripped, defs_removed, vars_removed = strip_redundant_definitions_and_assignments(
            stripped, self.function_fingerprints, self.variable_fingerprints
        )
        if defs_removed:
            distinct = sorted(set(defs_removed))[:4]
            label = "definitions of" if len(defs_removed) > 1 else "definition of"
            notices.append(f"Removed redundant {label} {', '.join(distinct)} (already defined identically).")
        if vars_removed:
            distinct = sorted(set(vars_removed))[:4]
            label = "assignments to" if len(vars_removed) > 1 else "assignment to"
            notices.append(f"Removed redundant {label} {', '.join(distinct)} (already set to that value).")
        if notices:
            return stripped, " ".join(notices)
        return code, None

    def preprocess_code(self, code):
        code, _ = self.strip_boilerplate(code)
        return preprocess_python(code)
