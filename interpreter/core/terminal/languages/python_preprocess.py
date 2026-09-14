"""Rewriting a Python block before the kernel runs it.

Strips imports the kernel already holds and definitions it already holds
verbatim, injects the active-line markers the terminal highlights with, and
wraps the block so a traceback comes back as output instead of killing the cell.
"""

import ast
import hashlib
import os
import re
import sys
import traceback

# Modules we're willing to drop a redundant top-level `import X` for. Only
# side-effect-free imports that LLMs habitually repeat at the top of every cell
# are listed. Deliberately NOT included: modules whose import has side effects
# (e.g. matplotlib picks a backend), and the near-universal aliased imports
# (`import numpy as np`, `import pandas as pd`, `import matplotlib.pyplot as plt`)
# — those have aliases/dots and are never stripped anyway. Stripping also
# requires the module to already be bound in the kernel, so a first import
# (with any real side effect) is never affected.
REMOVABLE_BOILERPLATE_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "re",
        "json",
        "time",
        "datetime",
        "random",
        "math",
        "subprocess",
        "glob",
        "shutil",
        "pathlib",
        "string",
        "typing",
        "tempfile",
        "uuid",
        "hashlib",
        "base64",
        "io",
        "csv",
        "statistics",
        "secrets",
        "pprint",
        "copy",
        "warnings",
        "platform",
        "textwrap",
        "collections",
        "functools",
        "itertools",
        "logging",
        "argparse",
        "heapq",
        "bisect",
        "decimal",
        "fractions",
        "operator",
        "queue",
        "threading",
        "struct",
        "zlib",
        "gzip",
        "pickle",
        "sqlite3",
        "socket",
        "requests",
    }
)


def strip_redundant_imports(code, imported_modules):
    """Drop top-level ``import X`` lines for boilerplate modules already bound.

    Only plain, single-line, column-0 imports without aliases or dotted names
    are considered (``import os``, ``import os, sys``), and only while they sit
    in the *leading* block — before the first executable statement (comments
    and blank lines may precede them). An ``import`` after any other statement
    is never stripped: it may re-bind a name that was assigned earlier (e.g.
    ``os = "string"`` followed by ``import os``) or be an intentional mid-cell
    import, so removing it could break the code. The module must also be in
    REMOVABLE_BOILERPLATE_IMPORTS: a deliberately tiny allowlist of
    side-effect-free stdlib modules that LLMs habitually re-import and that the
    kernel already binds at startup. Anything else — matplotlib and other
    imports with side effects, aliased/dotted/``from X import y`` lines, and
    indented function-scope imports — is always left alone.

    Returns ``(stripped_code, removed_module_names)`` so the caller can notify
    the user about what was dropped.
    """
    removed = []
    if not imported_modules:
        return code, removed
    kept_lines = []
    in_leading = True
    for line in code.splitlines():
        if line.strip() == "" or line.lstrip().startswith("#"):
            kept_lines.append(line)  # blank/comment — stay in the leading block
            continue
        if in_leading:
            m = re.match(r"^import\s+(.+)$", line)
            if m:
                names = [name.split("#")[0].strip() for name in m.group(1).split(",")]
                if not names or any(" as " in name or "." in name for name in names):
                    kept_lines.append(line)
                    continue
                if all(name in imported_modules and name in REMOVABLE_BOILERPLATE_IMPORTS for name in names):
                    removed.extend(names)  # redundant boilerplate — drop the line
                    continue
                kept_lines.append(line)
                continue
            if re.match(r"^from\s+\S+\s+import\s", line):
                kept_lines.append(line)  # never stripped, but still part of the leading block
                continue
            in_leading = False  # first executable statement ends the leading block
        kept_lines.append(line)
    return "\n".join(kept_lines), removed


def strip_redundant_definitions_and_assignments(code, function_fps, variable_fps):
    """Drop top-level definitions and scalar assignments already bound identically.

    Removes a top-level ``def``/``async def`` whose normalized source
    fingerprint matches the one the kernel already binds to that name, and a
    single-name scalar assignment (``x = 5``, ``x = "abc"``, ``x = None``)
    whose repr fingerprint matches. A fingerprint match means re-running the
    statement is a no-op, so removing it cannot change behaviour. A
    re-definition with different code always survives.

    Safety rules:
    - Only *top-level* statements are considered; anything inside a function,
      class or block is untouched.
    - A statement is stripped only if its name is not bound earlier in the same
      block (``def f`` then a *different* ``def f`` keeps both; ``x = 6`` then
      ``x = 5`` keeps both — stripping the second would change the result).
      Kept statements mark the names they bind (assigns, defs, imports,
      for/with targets, walrus, ``del``), so a later same-name candidate is
      conservatively kept.
    - Mutable objects, non-literal right-hand sides (``2 + 3``, ``int("5")``,
      ``f(x)``) and anything the kernel could not fingerprint are never
      stripped. The kernel only fingerprints immutable scalars, so ``x = [1,
      2]`` has no matching entry and is left alone — rebinding a fresh list is
      not a no-op.
    - The block is re-parsed after each removal; if that produces invalid
      Python the statement is put back.

    Returns ``(stripped_code, removed_function_names, removed_var_names)``.
    """
    removed_funcs, removed_vars = [], []
    if not function_fps and not variable_fps:
        return code, removed_funcs, removed_vars

    work = code
    bound = set()
    while True:
        try:
            tree = ast.parse(work)
        except SyntaxError:
            break  # magics, `!cmd`, incomplete code — never touch it
        if not tree.body:
            break
        for stmt in tree.body:
            span, kind = _redundant_span(stmt, work, function_fps, variable_fps, bound)
            if span is None:
                bound |= _bound_names(stmt)
                continue
            # Try the removal; re-parse to check it did not break the block.
            candidate = work[: span[0]] + work[span[1] :]
            try:
                ast.parse(candidate)
            except SyntaxError:
                bound |= _bound_names(stmt)  # unsafe — keep it, treat as bound
                continue
            work = candidate
            if kind == "fn":
                removed_funcs.append(stmt.name)
            else:
                removed_vars.append(stmt.targets[0].id)
            break  # offsets have shifted — restart the walk on the new text
        else:
            break  # nothing left to remove
    return work, removed_funcs, removed_vars


def _bound_names(stmt):
    """Names a top-level statement binds, for in-block redundancy tracking.

    Conservative: anything assigned anywhere in the statement's subtree counts,
    even inside nested function bodies — over-marking only suppresses stripping,
    it can never cause a wrong strip.
    """
    names = set()
    for node in ast.walk(stmt):
        # `del x` uses Del ctx, not Store; either way the name's earlier
        # binding no longer holds after the statement, so a later identical
        # re-assignment has to be kept.
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(stmt.name)
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for alias in stmt.names:
            names.add(alias.asname or alias.name.split(".")[0])
    return names


def _redundant_span(stmt, source, function_fps, variable_fps, bound):
    """Return (span, kind) if ``stmt`` is a redundant definition, else (None, None).

    ``span`` is a (start_offset, end_offset) pair into ``source``. ``kind`` is
    ``"fn"`` for functions or ``"var"`` for scalar assignments.
    """
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if stmt.name not in function_fps or stmt.name in bound:
            return None, None
        try:
            fingerprint = _function_fingerprint(stmt)
        except Exception:
            return None, None
        if fingerprint != function_fps[stmt.name]:
            return None, None
        start_line = stmt.lineno
        if stmt.decorator_list:
            start_line = min(d.lineno for d in stmt.decorator_list)
        return _span_offsets(source, start_line, 0, stmt.end_lineno, stmt.end_col_offset), "fn"

    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        name = stmt.targets[0].id
        if name not in variable_fps or name in bound:
            return None, None
        try:
            value = ast.literal_eval(stmt.value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None, None  # not a literal — cannot be a no-op re-assignment
        if _value_fingerprint(value) != variable_fps[name]:
            return None, None
        start, end = _span_offsets(source, stmt.lineno, stmt.col_offset, stmt.end_lineno, stmt.end_col_offset)
        # Only strip an assignment that has its line to itself. A shared line
        # (`x = 5; y = 6`) leaves a dangling `;` or an orphaned sibling when one
        # target goes, and re-parsing cannot catch the semantics, so require
        # nothing but whitespace before and after the span on its line.
        line_start = _span_offsets(source, stmt.lineno, 0, stmt.lineno, 0)[0]
        line_tail = source[end:].split("\n", 1)[0]
        if source[line_start:start].strip() or line_tail.strip():
            return None, None
        return (start, end), "var"

    return None, None


def _span_offsets(source, start_line, start_col, end_line, end_col):
    """Convert line/column positions to absolute string offsets."""
    offsets = [0]
    for line in source.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)  # +1 for the newline
    start = offsets[start_line - 1] + start_col
    end = offsets[end_line - 1] + end_col
    return start, end


def _function_fingerprint(stmt):
    """Normalized-source fingerprint of a FunctionDef/AsyncFunctionDef node.

    ``ast.dump`` of the node — structure only, no line numbers or formatting —
    matching the kernel's ``ast.dump(ast.parse(src).body[0])``.
    """
    return hashlib.sha1(ast.dump(stmt).encode()).hexdigest()


def _value_fingerprint(value):
    """Repr fingerprint of an immutable scalar, matching the kernel's ``var:`` entries."""
    return hashlib.sha1(repr(value).encode()).hexdigest()


def preprocess_python(code):
    """
    Add active line markers
    Wrap in a try except
    """

    # Stripping the preamble drops whole lines, and the markers name lines in
    # the code the model wrote — the terminal highlights that copy, not this
    # one. Count what goes so the numbers still point at the right line.
    dropped_lines = code[: len(code) - len(code.lstrip())].count("\n")
    code = code.strip()

    # Add print commands that tell us what the active line is
    # but don't do this if any line starts with ! or %
    if (
        not any(line.strip().startswith(("!", "%")) for line in code.split("\n"))
        and os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower() == "true"
    ):
        code = add_active_line_prints(code, line_offset=dropped_lines)

    # Wrap in a try except (DISABLED)
    # code = wrap_in_try_except(code)

    # Remove any whitespace lines, as this will break indented blocks
    # (are we sure about this? test this)
    code_lines = code.split("\n")
    code_lines = [c for c in code_lines if c.strip() != ""]
    code = "\n".join(code_lines)

    return code


def add_active_line_prints(code, line_offset=0):
    """
    Add print statements indicating line numbers to a python string.

    The markers carry the line numbers of the *original* code: they come from
    the ``lineno`` ast records for each statement, so comments and blank lines
    are counted even though they parse to nothing. ``line_offset`` accounts for
    lines the caller removed ahead of this code. Nothing is rewritten before
    parsing — text that merely looks like a comment (a ``#`` line inside a YAML
    or Markdown string literal) belongs to the string and must be left alone.
    """
    tree = ast.parse(code)
    transformer = AddLinePrints(line_offset)
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)


# Nodes whose first statement may be a docstring. A marker inserted ahead of it
# would demote the docstring to a plain expression and leave ``__doc__`` None.
DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_docstring(node):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _is_future_import(node):
    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


class AddLinePrints(ast.NodeTransformer):
    """
    Transformer to insert print statements indicating the line number
    before every executable line in the AST.
    """

    def __init__(self, line_offset=0):
        super().__init__()
        self.line_offset = line_offset

    def insert_print_statement(self, line_number):
        """Inserts a print statement for a given line number."""
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[ast.Constant(value=f"##active_line{line_number + self.line_offset}##")],
                keywords=[],
            )
        )

    def leading_statements(self, node, body):
        """How many statements at the start of ``body`` must keep their position.

        A docstring stops being one the moment anything precedes it, and
        ``from __future__ import ...`` is a syntax error anywhere but at the
        top of a module. Both are left un-marked rather than corrupted.
        """
        count = 0
        if isinstance(node, DOCSTRING_OWNERS) and body and _is_docstring(body[0]):
            count = 1
        if isinstance(node, ast.Module):
            while count < len(body) and _is_future_import(body[count]):
                count += 1
        return count

    def process_body(self, body, skip=0):
        """Processes a block of statements, adding print calls."""
        new_body = list(body[:skip])

        for sub_node in body[skip:]:
            if hasattr(sub_node, "lineno"):
                new_body.append(self.insert_print_statement(sub_node.lineno))
            new_body.append(sub_node)

        return new_body

    def visit(self, node):
        """Overridden visit to transform nodes."""
        new_node = super().visit(node)

        # If node has a block of statements, process it. `body`/`orelse` are
        # single expressions on a lambda or a ternary, where there is no line to
        # mark and inserting a statement would be a type error.
        if isinstance(getattr(new_node, "body", None), list):
            new_node.body = self.process_body(new_node.body, self.leading_statements(new_node, new_node.body))

        # If node has an orelse block (like in for, while, if), process it
        if isinstance(getattr(new_node, "orelse", None), list) and new_node.orelse:
            new_node.orelse = self.process_body(new_node.orelse)

        # `finalbody` is the one block no node type reaches on its own: `body`
        # and `orelse` are handled above, and each except clause is an
        # ExceptHandler node whose own visit marks it.
        if isinstance(new_node, ast.Try):
            if new_node.finalbody:
                new_node.finalbody = self.process_body(new_node.finalbody)

        return new_node


def wrap_in_try_except(code):
    # Add import traceback
    code = "import traceback\n" + code

    # Parse the input code into an AST
    parsed_code = ast.parse(code)

    # Wrap the entire code's AST in a single try-except block
    try_except = ast.Try(
        body=parsed_code.body,
        handlers=[
            ast.ExceptHandler(
                type=ast.Name(id="Exception", ctx=ast.Load()),
                name=None,
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="traceback", ctx=ast.Load()),
                                attr="print_exc",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        )
                    ),
                ],
            )
        ],
        orelse=[],
        finalbody=[],
    )

    # Assign the try-except block as the new body
    parsed_code.body = [try_except]

    # Convert the modified AST back to source code
    return ast.unparse(parsed_code)


def string_to_python(code_as_string):
    parsed_code = ast.parse(code_as_string)

    # Initialize containers for different categories
    import_statements = []
    functions = []
    functions_dict = {}

    # Traverse the AST
    for node in ast.walk(parsed_code):
        # Check for import statements
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for alias in node.names:
                # Handling the alias in import statements
                if alias.asname:
                    import_statements.append(f"import {alias.name} as {alias.asname}")
                else:
                    import_statements.append(f"import {alias.name}")
        # Check for function definitions
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                # ignore private functions
                continue
            docstring = ast.get_docstring(node)
            body = node.body
            if docstring:
                body = body[1:]

            code_body = ast.unparse(body[0]).replace("\n", "\n    ")

            func_info = {
                "name": node.name,
                "docstring": docstring,
                "body": code_body,
            }
            functions.append(func_info)

    for func in functions:
        # Consolidating import statements and function definition
        function_content = "\n".join(import_statements) + "\n\n"
        function_content += f'def {func["name"]}():\n    """{func["docstring"]}"""\n    {func["body"]}\n'

        # Adding to dictionary
        functions_dict[func["name"]] = function_content

    return functions_dict
