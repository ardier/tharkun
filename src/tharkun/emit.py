"""Render ``BranchSolution`` objects as runnable pytest test files.

Given a ``CoverageResult`` from ``branch_solver``, emits a Python source file
containing one test function per solved branch.  Each test:

- Calls the target function with the satisfying inputs.
- Wraps with ``@patch`` decorators when ``mock_patches`` are present.
- For ``__sequence__`` solutions (state guards): constructs the object,
  calls setup methods, then invokes the target method.

Usage::

    from tharkun.emit import emit_branch_tests

    coverage = solve_uncovered_branches("module.py")
    source = emit_branch_tests(coverage)
    Path("test_generated.py").write_text(source)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from tharkun.branch_solver import BranchSolution, CoverageResult
from tharkun.mock_gen import PatchSpec, render_mock_params, render_patch_decorators


# ---------------------------------------------------------------------------
# Value rendering
# ---------------------------------------------------------------------------


def _render_value(val: Any) -> str:
    """Render a Python value as source code for embedding in a test."""
    if val is None:
        return "None"
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, (int, float)):
        return repr(val)
    if isinstance(val, str):
        return repr(val)
    if isinstance(val, bytes):
        return repr(val)
    if isinstance(val, list):
        items = ", ".join(_render_value(v) for v in val)
        return f"[{items}]"
    if isinstance(val, dict):
        items = ", ".join(f"{_render_value(k)}: {_render_value(v)}" for k, v in val.items())
        return "{" + items + "}"
    if isinstance(val, tuple):
        items = ", ".join(_render_value(v) for v in val)
        if len(val) == 1:
            return f"({items},)"
        return f"({items})"
    if isinstance(val, set):
        if not val:
            return "set()"
        items = ", ".join(_render_value(v) for v in val)
        return "{" + items + "}"
    # SimpleNamespace or other objects — try to reconstruct
    tp = type(val).__name__
    if tp == "SimpleNamespace":
        attrs = {k: v for k, v in val.__dict__.items() if not k.startswith("_")}
        items = ", ".join(f"{k}={_render_value(v)}" for k, v in attrs.items())
        return f"SimpleNamespace({items})"
    # AST nodes
    if hasattr(val, "_fields") and hasattr(val, "lineno"):
        tp_name = type(val).__name__
        return f"ast.{tp_name}()"
    return repr(val)


# ---------------------------------------------------------------------------
# Import inference
# ---------------------------------------------------------------------------


def _infer_imports(solutions: list[BranchSolution], module_name: str) -> list[str]:
    """Determine what imports the generated test file needs."""
    imports: set[str] = set()
    imports.add(f"from {module_name} import *")

    needs_mock = False
    needs_mock_open = False
    needs_namespace = False
    needs_ast = False

    for sol in solutions:
        if not sol.solved:
            continue
        if sol.mock_patches:
            needs_mock = True
            for p in sol.mock_patches:
                if p.target == "builtins.open":
                    needs_mock_open = True
        if sol.best_inputs:
            for v in sol.best_inputs.values():
                if isinstance(v, dict) and "__sequence__" in sol.best_inputs:
                    pass
                tp = type(v).__name__
                if tp == "SimpleNamespace":
                    needs_namespace = True
                if hasattr(v, "_fields") and hasattr(v, "lineno"):
                    needs_ast = True

    if needs_mock and needs_mock_open:
        imports.add("from unittest.mock import Mock, patch, mock_open")
    elif needs_mock:
        imports.add("from unittest.mock import Mock, patch")
    if needs_namespace:
        imports.add("from types import SimpleNamespace")
    if needs_ast:
        imports.add("import ast")

    return sorted(imports)


def _module_name_from_path(source_path: str) -> str:
    """Best-effort module name from file path."""
    p = Path(source_path)
    name = p.stem
    # Walk up looking for __init__.py to build dotted name
    parts = [name]
    parent = p.parent
    for _ in range(10):
        init = parent / "__init__.py"
        if init.exists():
            parts.append(parent.name)
            parent = parent.parent
        else:
            break
    parts.reverse()
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Test function rendering
# ---------------------------------------------------------------------------


def _sanitize_name(s: str) -> str:
    """Turn a string into a valid Python identifier fragment."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _emit_regular_test(
    sol: BranchSolution,
    index: int,
) -> list[str]:
    """Emit a test function for a regular (non-sequence) solved branch."""
    func_name = sol.callable_id or "unknown"
    guard_short = (sol.guard_source or "")[:40].replace("\n", " ")
    test_name = f"test_{_sanitize_name(func_name)}_line{sol.lineno}_{index}"

    lines: list[str] = []
    lines.append("")

    # Patch decorators
    mock_params: list[str] = []
    if sol.mock_patches:
        for dec in render_patch_decorators(sol.mock_patches):
            lines.append(dec)
        mock_params = render_mock_params(sol.mock_patches)

    # Function signature
    params = ["self"] + mock_params
    lines.append(f"    def {test_name}({', '.join(params)}):")
    lines.append(f'        """Branch at line {sol.lineno}: {guard_short}"""')

    # Build call arguments
    inputs = sol.best_inputs or {}
    if "self" in inputs:
        # Skip 'self' — we pass the other params
        inputs = {k: v for k, v in inputs.items() if k != "self"}

    arg_parts = []
    for k, v in inputs.items():
        arg_parts.append(f"{k}={_render_value(v)}")
    call_args = ", ".join(arg_parts)
    lines.append(f"        result = {func_name}({call_args})")

    # Mock setup lines
    for patch in sol.mock_patches:
        for setup in patch.setup_lines:
            lines.append(f"        {setup}")

    lines.append("")
    return lines


def _emit_sequence_test(
    sol: BranchSolution,
    index: int,
) -> list[str]:
    """Emit a test for a state-sequence solution (self.xxx guards)."""
    seq = sol.best_inputs.get("__sequence__", {})
    class_name = seq.get("class", "TargetClass")
    setup_methods = seq.get("setup_methods", [])
    func_name = sol.callable_id or "target_method"
    guard_short = (sol.guard_source or "")[:40].replace("\n", " ")
    test_name = f"test_{_sanitize_name(func_name)}_line{sol.lineno}_seq_{index}"

    lines: list[str] = []
    lines.append("")

    # Patch decorators
    mock_params: list[str] = []
    if sol.mock_patches:
        for dec in render_patch_decorators(sol.mock_patches):
            lines.append(dec)
        mock_params = render_mock_params(sol.mock_patches)

    params = ["self"] + mock_params
    lines.append(f"    def {test_name}({', '.join(params)}):")
    lines.append(f'        """State sequence for line {sol.lineno}: {guard_short}"""')

    # Construct
    lines.append(f"        obj = {class_name}()")

    # Setup methods
    for method in setup_methods:
        lines.append(f"        obj.{method}()")

    # Target call
    lines.append(f"        obj.{func_name}()")

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emit_branch_tests(
    coverage: CoverageResult,
    *,
    class_name: str | None = None,
    module_name: str | None = None,
) -> str:
    """Render a ``CoverageResult`` as a runnable pytest test file.

    Args:
        coverage:     Result from ``solve_uncovered_branches()``.
        class_name:   Name for the test class. Auto-generated if ``None``.
        module_name:  Module to import from. Inferred from ``coverage.source_path``
                      if ``None``.

    Returns:
        Python source code for a test file.
    """
    source_path = coverage.source_path
    if module_name is None:
        module_name = _module_name_from_path(source_path)

    solved = [s for s in coverage.solutions if s.solved and s.best_inputs]
    if not solved:
        return f'"""No solved branches to test for {source_path}."""\n'

    if class_name is None:
        stem = Path(source_path).stem
        class_name = f"TestGenerated{_sanitize_name(stem).title()}"

    # Build imports
    import_lines = _infer_imports(solved, module_name)

    # Build test methods
    test_methods: list[str] = []
    for i, sol in enumerate(solved):
        if sol.best_inputs and "__sequence__" in sol.best_inputs:
            test_methods.extend(_emit_sequence_test(sol, i))
        else:
            test_methods.extend(_emit_regular_test(sol, i))

    # Assemble
    out: list[str] = []
    out.append(f'"""Auto-generated branch coverage tests for {Path(source_path).name}."""')
    out.append("")
    out.append("import unittest")
    for imp in import_lines:
        out.append(imp)
    out.append("")
    out.append("")
    out.append(f"class {class_name}(unittest.TestCase):")
    out.extend(test_methods)
    out.append("")
    out.append("")
    out.append('if __name__ == "__main__":')
    out.append("    unittest.main()")
    out.append("")

    return "\n".join(out)


def emit_branch_tests_to_file(
    coverage: CoverageResult,
    output_path: str,
    *,
    class_name: str | None = None,
    module_name: str | None = None,
) -> str:
    """Render and write branch tests to a file.

    Returns:
        The output file path.
    """
    source = emit_branch_tests(
        coverage,
        class_name=class_name,
        module_name=module_name,
    )
    Path(output_path).write_text(source, encoding="utf-8")
    return output_path
