"""No module over 600 lines, so every file can be held in one head at a time.

Two files are exempt: the vision-model pointer (point.py) and the file-edit
tool, both single-purpose and left as they are by the rework spec.
"""

from pathlib import Path

LIMIT = 600
EXEMPT = {
    "interpreter/core/toolbox/display/point/point.py",
    "interpreter/core/tools/file_edit.py",
}
ROOT = Path(__file__).resolve().parents[1]


def test_no_module_over_the_line_budget():
    over = {}
    for path in (ROOT / "interpreter").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        lines = sum(1 for _ in path.open(encoding="utf-8"))
        if lines > LIMIT and rel not in EXEMPT:
            over[rel] = lines
    assert not over, f"modules over {LIMIT} lines: {over}"
