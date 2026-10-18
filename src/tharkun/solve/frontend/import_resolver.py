"""Shared import resolution utilities for cross-module analysis.

Resolves Python import statements to file paths and loads AST definitions
from imported modules.  Used by both ``type_solver`` (cross-module class
lookup) and ``guard_decomposer`` (cross-module function tracing).
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any


def _package_roots(from_dir: str) -> list[str]:
    """Directories an absolute intra-package import could resolve against.

    Walks up from *from_dir* for as long as each level is a package (has
    ``__init__.py``), returning each parent — so ``.../pkg/sub`` yields ``.../pkg`` and
    ``...``, which is where ``import pkg.sub.mod`` is rooted.
    """
    roots: list[str] = []
    cur = os.path.realpath(from_dir) if from_dir else ""
    for _ in range(12):                              # bounded: no unbounded fs walk
        if not cur or not os.path.isfile(os.path.join(cur, "__init__.py")):
            break
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        roots.append(parent)
        cur = parent
    return roots


def resolve_module_path(module_name: str, from_dir: str) -> str | None:
    """Resolve a dotted module name to its ``.py`` file path.

    Searches *from_dir* first, then ``sys.path``.  Returns ``None`` if the
    module cannot be found.

    Handles both ``foo.bar`` (→ ``foo/bar.py`` or ``foo/bar/__init__.py``)
    and plain ``foo`` (→ ``foo.py`` or ``foo/__init__.py``).

    A module inside a package normally imports its siblings by the package's *absolute*
    name — ``ntlm_auth/compute_keys.py`` says ``from ntlm_auth.constants import ...`` —
    which resolves against the package's parent, not against the module's own directory.
    So the enclosing package roots are searched too, found by walking up while
    ``__init__.py`` marks each level as part of the package.
    """
    parts = module_name.split(".")
    rel = os.path.join(*parts) if parts else module_name

    search_dirs = [from_dir, *_package_roots(from_dir), *sys.path]
    for base in search_dirs:
        if not base:
            continue
        # foo/bar.py
        candidate = os.path.join(base, rel + ".py")
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
        # foo/bar/__init__.py
        candidate = os.path.join(base, rel, "__init__.py")
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    return None


def parse_imports(source_path: str) -> dict[str, str]:
    """Parse import statements in *source_path* and resolve to file paths.

    Returns ``{local_name: resolved_module_path}`` for all imports whose
    source module can be found on disk.  Handles both::

        from foo import Bar          → {"Bar": "/path/to/foo.py"}
        from foo.sub import helper   → {"helper": "/path/to/foo/sub.py"}
        from foo import sub          → {"sub": "/path/to/foo/sub.py"}   (a SUBMODULE,
                                       not the package __init__ that merely exposes it)
        import foo                   → {"foo": "/path/to/foo.py"}

    Relative imports are resolved relative to the directory of *source_path*.

    Imports nested inside ``with`` / ``try`` blocks count: every Atheris harness writes
    ``with atheris.instrument_imports(): from pkg import mod``, so scanning only
    top-level statements missed the imports that matter most.
    """
    rp = os.path.realpath(source_path)
    from_dir = os.path.dirname(rp)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}

    def _import_stmts(root: ast.AST):
        """Import statements at module level, including inside with/try wrappers."""
        for node in ast.iter_child_nodes(root):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                yield node
            elif isinstance(node, (ast.With, ast.AsyncWith, ast.Try)):
                yield from _import_stmts(node)

    result: dict[str, str] = {}
    for node in _import_stmts(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            # Handle relative imports.
            level = node.level or 0
            if level > 0:
                parent = from_dir
                for _ in range(level - 1):
                    parent = os.path.dirname(parent)
                if module_name:
                    mod_path = resolve_module_path(module_name, parent)
                else:
                    # `from . import X` — each name is a submodule in parent dir
                    for alias in node.names:
                        local_name = alias.asname or alias.name
                        sub_path = resolve_module_path(local_name, parent)
                        if sub_path:
                            result[local_name] = sub_path
                    continue
            else:
                mod_path = resolve_module_path(module_name, from_dir)
            if mod_path is None:
                continue
            for alias in node.names:
                local_name = alias.asname or alias.name
                # `from pkg import sub` where pkg/sub.py exists imports the SUBMODULE.
                # Mapping it to pkg/__init__.py instead points every lookup at the
                # wrong file — and a harness names its callee `sub.func`.
                sub = resolve_module_path(f"{module_name}.{alias.name}", from_dir) \
                    if module_name and level == 0 else None
                result[local_name] = sub or mod_path

        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name
                mod_path = resolve_module_path(alias.name, from_dir)
                if mod_path:
                    result[local_name] = mod_path

    return result


def load_function_def(
    module_path: str,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Load a function definition by *name* from *module_path*.

    Searches top-level functions and class methods (one level deep).
    Returns ``None`` if not found or if the file can't be parsed.
    """
    rp = os.path.realpath(module_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == name or f"{node.name}.{child.name}" == name:
                        return child
    return None


def load_class_def(
    module_path: str,
    name: str,
) -> ast.ClassDef | None:
    """Load a class definition by *name* from *module_path*."""
    rp = os.path.realpath(module_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None
