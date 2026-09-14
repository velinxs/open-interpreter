"""What the kernel knows about itself, reported back after every block.

After each block the kernel runs a small snippet that prints one REPL-state
line — cwd, bound modules, variables, functions — and, hidden behind a
``##oi_fp##`` marker, a sha1 fingerprint of each top-level definition it holds.
The client adopts both wholesale. The state line is what lets a redundant
``import os`` be stripped before it runs; the fingerprints are what let a
re-definition that is byte-for-byte identical to the one already bound be
stripped too, so the same helper is not carried in the transcript twice.

The marker is swallowed here and never reaches the model or the terminal.
"""

import queue
import re

FINGERPRINT_MARKER = "##oi_fp##"

# The kernel fingerprints at most 50 functions and 50 scalars per run. The work
# is a getsource + ast.parse per function, measured at ~0.1 ms for fifty of
# them; the cap mostly bounds the marker's size on the wire (~6.6 KB at 50+50).
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

# Fingerprint the top-level user definitions so the client can drop a
# re-definition identical to the one already bound here. Functions and classes
# are fingerprinted from their source, scalar variables from their repr.
# Anything that cannot be fingerprinted is simply left out, and the client then
# never strips that name — the safe direction to fail in.
import ast as __oi_ast
import inspect as __oi_inspect
import hashlib as __oi_hashlib
import re as __oi_re
__oi_fp_parts = []
__oi_fn_n = 0
for __oi_k in __oi_funcs:
    if __oi_fn_n >= 50:
        break
    __oi_o = __oi_globals[__oi_k]
    try:
        __oi_src = __oi_inspect.getsource(__oi_o)
        if len(__oi_src) > 20000:
            continue
        # What ran carries the client's `print('##active_lineN##')` markers.
        # They are not part of the definition the client fingerprints, so they
        # must come back out before the source is parsed.
        __oi_src = __oi_re.sub(
            r"^\\s*print\\('##active_line\\d+##'\\)\\s*$", "", __oi_src, flags=__oi_re.M
        )
        __oi_fp_parts.append(
            __oi_k + '=fn:' + __oi_hashlib.sha1(
                __oi_ast.dump(__oi_ast.parse(__oi_src).body[0]).encode()
            ).hexdigest()
        )
        __oi_fn_n += 1
    except Exception:
        pass
__oi_var_n = 0
for __oi_k in __oi_vars:
    if __oi_var_n >= 50:
        break
    __oi_v = __oi_globals[__oi_k]
    if not (__oi_v is None or isinstance(__oi_v, (bool, int, float, complex, str, bytes))):
        continue
    __oi_r = repr(__oi_v)
    if len(__oi_r) > 200:
        continue
    __oi_fp_parts.append(
        __oi_k + '=var:' + __oi_hashlib.sha1(__oi_r.encode()).hexdigest()
    )
    __oi_var_n += 1
if __oi_fp_parts:
    print('__OI_FINGERPRINT_MARKER__' + ','.join(__oi_fp_parts))

__oi_res = f"\\n[Python REPL State: {' | '.join(__oi_parts)}]"
print(__oi_res)
""".replace("__OI_FINGERPRINT_MARKER__", FINGERPRINT_MARKER)


class PythonStateMixin:
    """Runs the state snippet and keeps the client's copy of the kernel's namespace."""

    # Defaults for instances built without __init__ (tests that skip kernel
    # startup). __init__ replaces each with an instance attribute, and every
    # update below rebinds rather than mutates, so these are never written to.
    imported_modules = frozenset()
    function_fingerprints = {}
    variable_fingerprints = {}

    _STATE_MODULES_RE = re.compile(r"Already imported:\s*([^|\]]*)")

    def _get_active_state(self):
        message_queue = queue.Queue()
        self.finish_flag = False
        self._execute_code(STATE_CODE.strip(), message_queue)

        # The marker and the REPL-state line are printed by the same cell and
        # arrive in one stream chunk (measured at 6.6 KB for the full fifty
        # functions and fifty scalars), but nothing guarantees that, so collect
        # the snippet's whole output before cutting the marker out of it. That
        # also keeps the newlines around the state line exactly as they were.
        chunks = [
            output
            for output in self._capture_output(message_queue)
            if output.get("type") == "console" and output.get("format") == "output"
        ]
        if not chunks:
            return
        text = "".join(str(chunk.get("content") or "") for chunk in chunks)
        text = self._absorb_fingerprints(text)
        if text.strip():
            yield {**chunks[0], "content": text}

    def _absorb_fingerprints(self, text):
        """Record every ``##oi_fp##`` line in ``text`` and return the rest of it.

        The marker is internal bookkeeping: it must not reach the terminal or
        the model, so it is taken out of the state output here.
        """
        if FINGERPRINT_MARKER not in text:
            return text
        kept = []
        for line in text.split("\n"):
            if line.startswith(FINGERPRINT_MARKER):
                self._update_fingerprints(line)
            else:
                kept.append(line)
        return "\n".join(kept)

    def _update_fingerprints(self, marker):
        """Parse one ``##oi_fp##`` marker into the tracked fingerprint dicts.

        Entries are comma-separated ``name=fn:<sha1>`` (function source) and
        ``name=var:<sha1>`` (scalar repr). They are adopted wholesale each run
        so the client's view always matches the kernel's live namespace — a
        later cell that rebinds ``f = other`` changes f's fingerprint, and the
        stale ``def f`` stops being stripped.
        """
        function_fingerprints, variable_fingerprints = {}, {}
        for token in marker[len(FINGERPRINT_MARKER) :].split(","):
            name, _, value = token.partition("=")
            if not name or not value:
                continue
            if value.startswith("fn:"):
                function_fingerprints[name] = value[len("fn:") :]
            elif value.startswith("var:"):
                variable_fingerprints[name] = value[len("var:") :]
        self.function_fingerprints = function_fingerprints
        self.variable_fingerprints = variable_fingerprints

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
