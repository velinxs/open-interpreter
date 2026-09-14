"""Action modules: capabilities that cost nothing until they are used.

A toolbox module is listed in the system message, so every method it exposes
is paid for in tokens on every request whether or not the session needs it.
That is the right trade for a handful of general capabilities; it is the wrong
one for a long tail of specific ones — a deploy script, a report generator, a
client's API wrapper.

An action is an ordinary Python file in the actions directory. Only its name
and one-line summary reach the model; the code is read when the model asks for
it. Fifty actions cost about what five toolbox methods cost.

Writing one is writing a module:

    \"\"\"Roll the staging deployment back to the previous release.\"\"\"

    def run(release=None):
        ...

The model then writes, in an ordinary code block:

    rollback = toolbox.actions.load("rollback_staging")
    rollback.run()

which matters for approval: importing defines functions and does nothing else,
so the work happens on the `run()` line, in the block the user approved. An
action that acts at import time breaks that, which is why load() says so.
"""

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

from ....terminal_interface.utils.oi_dir import oi_dir


class ActionError(Exception):
    """Raised when an action cannot be found or loaded."""


def _summary(path):
    """The action's one-line description, taken from its module docstring.

    Parsed rather than imported: the listing must not execute anything, or
    merely asking what is available would run every action on disk.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return "(unreadable)"
    doc = ast.get_docstring(tree)
    if not doc:
        return "(no description)"
    return doc.strip().splitlines()[0].strip()


def _acts_at_import(tree):
    """True when a module body does more than define things.

    Imports, definitions, assignments and docstrings are inert. A call or any
    other statement at the top level runs the moment the module is loaded —
    before the user has seen a line of it in an approved block.
    """
    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,  # a bare docstring, checked below
            ),
        ):
            if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
                return True
            continue
        return True
    return False


class Actions:
    """Python files the model can look up and load on demand.

    list()        names and one-line summaries
    show(name)    the source, so the model (or user) can read before running
    load(name)    import it and return the module
    """

    def __init__(self, toolbox):
        self.toolbox = toolbox
        self.path = Path(oi_dir) / "actions"

    def _file(self, name):
        """The action's path, rejecting anything that escapes the directory."""
        candidate = (self.path / f"{name}.py").resolve()
        root = self.path.resolve()
        if root not in candidate.parents:
            raise ActionError(f"'{name}' is not an action name.")
        if not candidate.is_file():
            available = ", ".join(sorted(a["name"] for a in self.list())) or "none yet"
            raise ActionError(f"No action named '{name}'. Available: {available}")
        return candidate

    def list(self):
        """Every available action, as {"name", "summary"} — without running any."""
        if not self.path.is_dir():
            return []
        return [
            {"name": p.stem, "summary": _summary(p)}
            for p in sorted(self.path.glob("*.py"))
            if not p.stem.startswith("_")
        ]

    def show(self, name):
        """The action's source code, for reading before running it."""
        return self._file(name).read_text(encoding="utf-8")

    def create(self, name, source, overwrite=False):
        """Write a new action, refusing one that could not later be loaded.

        The same rules load() enforces are checked here, so a file that would
        be rejected on use cannot be written in the first place: it must parse,
        it must carry a docstring for the listing, and it must do nothing at
        import time.
        """
        if not name.isidentifier() or name.startswith("_"):
            raise ActionError(
                f"'{name}' is not a usable action name. Use a Python identifier, "
                f"such as 'deploy_staging'."
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            raise ActionError(f"That action has a syntax error: {error}") from error

        if not ast.get_docstring(tree):
            raise ActionError(
                "An action needs a module docstring — its first line is the "
                "description shown in the listing, and is all anyone sees "
                "before loading it."
            )

        if _acts_at_import(tree):
            raise ActionError(
                "An action must only define things at the top level. Move the "
                "work into a function so it runs when it is called, in a code "
                "block the user has approved."
            )

        self.path.mkdir(parents=True, exist_ok=True)
        destination = self.path / f"{name}.py"
        if destination.exists() and not overwrite:
            raise ActionError(
                f"'{name}' already exists. Read it with toolbox.actions.show('{name}'), "
                f"or pass overwrite=True to replace it."
            )
        destination.write_text(source, encoding="utf-8")
        return str(destination)

    def load(self, name):
        """Import an action and return its module.

        Loading only defines things. Whatever the action *does* happens when
        one of its functions is called, on a line the user can see in the
        approved code block — so an action that acts at import time is
        reported rather than silently run.
        """
        path = self._file(name)
        source = path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            raise ActionError(f"Action '{name}' has a syntax error: {error}") from error

        if _acts_at_import(tree):
            raise ActionError(
                f"Action '{name}' runs code at import time, so loading it would act "
                f"before you had seen what it does. Move that work into a function "
                f"and call it explicitly. Read it first with "
                f"toolbox.actions.show('{name}')."
            )

        # Namespaced so an action cannot shadow a real module, and keyed by
        # content so editing a file takes effect without restarting the kernel.
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        module_name = f"oi_action_{name}_{digest}"
        if module_name in sys.modules:
            return sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[module_name]
            raise
        return module
