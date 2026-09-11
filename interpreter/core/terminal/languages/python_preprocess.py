"""Rewriting a Python block before the kernel runs it.

Strips imports the kernel already holds, injects the active-line markers the
terminal highlights with, and wraps the block so a traceback comes back as
output instead of killing the cell.
"""

import ast
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


def preprocess_python(code):
    """
    Add active line markers
    Wrap in a try except
    """

    code = code.strip()

    # Add print commands that tell us what the active line is
    # but don't do this if any line starts with ! or %
    if (
        not any(line.strip().startswith(("!", "%")) for line in code.split("\n"))
        and os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower() == "true"
    ):
        code = add_active_line_prints(code)

    # Wrap in a try except (DISABLED)
    # code = wrap_in_try_except(code)

    # Remove any whitespace lines, as this will break indented blocks
    # (are we sure about this? test this)
    code_lines = code.split("\n")
    code_lines = [c for c in code_lines if c.strip() != ""]
    code = "\n".join(code_lines)

    return code


def add_active_line_prints(code):
    """
    Add print statements indicating line numbers to a python string.
    """
    # Replace newlines and comments with pass statements, so the line numbers are accurate (ast will remove them otherwise)
    code_lines = code.split("\n")
    in_multiline_string = False
    for i in range(len(code_lines)):
        line = code_lines[i]
        if '"""' in line or "'''" in line:
            in_multiline_string = not in_multiline_string
        if not in_multiline_string and (line.strip().startswith("#") or line == ""):
            whitespace = len(line) - len(line.lstrip(" "))
            code_lines[i] = " " * whitespace + "pass"
    processed_code = "\n".join(code_lines)
    try:
        tree = ast.parse(processed_code)
    except:
        # If you can't parse the processed version, try the unprocessed version before giving up
        tree = ast.parse(code)
    transformer = AddLinePrints()
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)


class AddLinePrints(ast.NodeTransformer):
    """
    Transformer to insert print statements indicating the line number
    before every executable line in the AST.
    """

    def insert_print_statement(self, line_number):
        """Inserts a print statement for a given line number."""
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[ast.Constant(value=f"##active_line{line_number}##")],
                keywords=[],
            )
        )

    def process_body(self, body):
        """Processes a block of statements, adding print calls."""
        new_body = []

        # In case it's not iterable:
        if not isinstance(body, list):
            body = [body]

        for sub_node in body:
            if hasattr(sub_node, "lineno"):
                new_body.append(self.insert_print_statement(sub_node.lineno))
            new_body.append(sub_node)

        return new_body

    def visit(self, node):
        """Overridden visit to transform nodes."""
        new_node = super().visit(node)

        # If node has a body, process it
        if hasattr(new_node, "body"):
            new_node.body = self.process_body(new_node.body)

        # If node has an orelse block (like in for, while, if), process it
        if hasattr(new_node, "orelse") and new_node.orelse:
            new_node.orelse = self.process_body(new_node.orelse)

        # Special case for Try nodes as they have multiple blocks
        if isinstance(new_node, ast.Try):
            for handler in new_node.handlers:
                handler.body = self.process_body(handler.body)
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
