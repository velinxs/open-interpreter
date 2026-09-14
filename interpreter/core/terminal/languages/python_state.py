"""What the kernel knows about itself, reported back after every block.

After each block the kernel runs a small snippet that prints one REPL-state
line — cwd, bound modules, variables, functions — and the client adopts it
wholesale. That line is what lets a redundant ``import os`` in the next block
be stripped before it runs.
"""

import queue
import re

STATE_CODE = """
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


class PythonStateMixin:
    """Runs the state snippet and keeps the client's copy of the kernel's namespace."""

    # Default for instances built without __init__ (tests that skip kernel
    # startup). __init__ replaces it with an instance attribute, and the update
    # below rebinds rather than mutates, so this is never written to.
    imported_modules = frozenset()

    _STATE_MODULES_RE = re.compile(r"Already imported:\s*([^|\]]*)")

    def _get_active_state(self):
        message_queue = queue.Queue()
        self.finish_flag = False
        self._execute_code(STATE_CODE.strip(), message_queue)

        for output in self._capture_output(message_queue):
            if output.get("type") == "console" and output.get("format") == "output":
                yield output

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
