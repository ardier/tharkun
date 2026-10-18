"""Lay a program out like an SV-COMP Python task (shared by the swallowing catalog and the engine audit)."""

from __future__ import annotations

import ast
import textwrap


def benchmark_shape(src: str) -> str:
    """``src`` laid out like an SV-COMP Python task: imports, classes and functions at the top, every
    other top-level statement (the draws and the behaviour under test) inside the
    ``if __name__ == "__main__":`` block that the concolic harness lifts draws from."""
    tree = ast.parse(src)
    lines = src.splitlines()
    top, main = [], []
    for node in tree.body:
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        chunk = "\n".join(lines[first - 1:node.end_lineno])
        declaration = isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))
        (top if declaration else main).append(chunk)
    body = textwrap.indent("\n".join(main) or "pass", "    ")
    return "\n".join(top) + '\nif __name__ == "__main__":\n' + body + "\n"
