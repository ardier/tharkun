"""[internal ref]: every stage that reads or parses the SV-COMP task's source file must
go through ``tharkun.sv.engine.read_source`` (or ``read_source_bytes`` for a
raw-byte hash), which detects the file's own PEP 263 encoding cookie the way
CPython does. A stage that instead does its own ``open(program, encoding=
"utf-8")`` decodes a non-UTF-8 source differently from how it actually runs --
exactly the gap that let absint prove a Latin-1/UTF-8 key mismatch's KeyError
"impossible" (see ``test_a_latin1_key_mismatch_is_not_proved_safe`` in
``tests/test_witness_validation.py``).

This is a structural regression guard, not a behavioral test: it scans every
module under ``tharkun`` for an ``open(...)`` call whose argument is the
conventional "the task's source file" parameter (named ``program``, or an
attribute/subscript ending in ``.program``) and fails if one exists outside
the small, explicitly reviewed allowlist below -- so a FUTURE stage that adds
its own ad hoc ``open(program, ...)`` breaks this test instead of silently
reintroducing the bug.

The allowlist is PER FUNCTION, not per file ([internal ref] follow-up, 2026-09-28: a
whole-file allowlist for ``sv/engine.py`` hid a second ``open(program,
encoding="utf-8")`` a still-unmerged branch ([internal ref]/[internal ref]) added to a NEW
function, ``demands()``'s caller, in that same file -- a per-file allowlist
cannot tell the canonical primitive apart from a future stage that lands
beside it). Naming ``engine.py`` here again would repeat that mistake.
"""
from __future__ import annotations

import ast
import os

import tharkun

SRC_ROOT = os.path.dirname(os.path.abspath(tharkun.__file__))

# (relative file path, enclosing function name) allowed to open the task
# source file directly, and why:
ALLOWED = {
    # Defines read_source()/read_source_bytes() themselves (the canonical
    # PEP-263-aware primitives every other stage must call instead).
    (os.path.join("sv", "engine.py"), "read_source"),
    (os.path.join("sv", "engine.py"), "read_source_bytes"),
    # Runs as a script in a FRESH interpreter and "imports nothing from
    # tharkun" by design (its own module docstring) -- it duplicates a
    # private, stdlib-only equivalent rather than importing
    # ``tharkun.sv.engine``.
    (os.path.join("sv", "_probe.py"), "_read_source"),
    # ``python -m tharkun.solve.absint`` dev CLI: ``tharkun.solve`` must not
    # import ``tharkun.sv`` (tests/test_layering.py), so this duplicates the
    # same primitive again, inline, under its own name.
    (os.path.join("solve", "absint", "__main__.py"), "_read_source"),
    # Same "imports nothing from tharkun" design (its own module docstring).
    # Reads raw bytes ("rb", no decode) and feeds them straight to
    # ``compile()``, which does its own PEP 263 encoding-cookie detection
    # internally -- the same bytes ``read_source_bytes`` would return, and
    # never separately decoded with a hardcoded "utf-8", so there is no
    # encoding-mismatch gap here to route through ``read_source``.
    (os.path.join("sv", "_child.py"), "run"),
}


def _enclosing_function_opens(tree: ast.AST):
    """Every ``open(X, ...)`` call in *tree* whose first argument X is named
    (or attribute-accessed as) ``program`` -- the parameter name every fixed
    stage uses for "the task's source file" -- paired with the name of the
    function it is lexically inside (``"<module>"`` if none)."""

    class _Visit(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []
            self.found: list[tuple[str, int]] = []

        def _enter(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _enter
        visit_AsyncFunctionDef = _enter

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
                target = node.args[0]
                is_program = (isinstance(target, ast.Name) and target.id == "program") or (
                    isinstance(target, ast.Attribute) and target.attr == "program")
                if is_program:
                    self.found.append((self.stack[-1] if self.stack else "<module>", node.lineno))
            self.generic_visit(node)

    visitor = _Visit()
    visitor.visit(tree)
    return visitor.found


def test_no_stage_opens_the_program_source_outside_read_source():
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(SRC_ROOT):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            rel = os.path.relpath(path, SRC_ROOT)
            with open(path, encoding="utf-8") as fh:
                source = fh.read()
            tree = ast.parse(source, path)
            for func_name, lineno in _enclosing_function_opens(tree):
                if (rel, func_name) in ALLOWED:
                    continue
                offenders.append(f"{rel}:{lineno} (in {func_name})")
    assert not offenders, (
        "these open(program, ...) calls bypass tharkun.sv.engine.read_source "
        "(PEP 263 encoding-cookie detection) -- route them through read_source "
        "(or read_source_bytes for a raw-byte hash) instead, or add the "
        "(file, function) pair to ALLOWED above with a reason: " + ", ".join(offenders)
    )
