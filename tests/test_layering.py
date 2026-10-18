"""Layering contracts for tharkun (#0603 / #0606, plan #0600).

* ``tharkun.solve`` is the fenced constraint-solving core: every module under
  ``solve/`` imports only ``solve/`` (one documented lazy back-edge remains
  until #0607), and ``solve/condition.py`` eagerly imports only its leaves;
* the tracer is a leaf on the other side -- ``trace_api`` imports no tharkun
  module at all;
* the fuzzing layer has no inbound edges from outside itself --
  ``redqueen`` is imported only by ``driver``, ``hybrid`` only from ``hybrid/``;
* the package root is lazy, and the old module paths are shims that only
  external consumers use -- tharkun's own code imports the real homes.

Pure AST scans: nothing is imported or executed.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "tharkun"
SOLVE_ROOT = PACKAGE_ROOT / "solve"
TESTS_ROOT = Path(__file__).resolve().parent

# Old module paths kept as aliasing shims (#0606).  External consumers pin
# them (tests/test_public_surface.py); nothing inside tharkun may import them.
SHIM_PATHS = frozenset(
    {
        "tharkun.condition_solver", "tharkun.guard_decomposer", "tharkun.attr_tracer",
        "tharkun.constants", "tharkun.types", "tharkun.config",
        "tharkun.canonicalize", "tharkun.rewrite_catalog",
        "tharkun.import_resolver", "tharkun.field_resolver", "tharkun.constructor",
        "tharkun.type_hints", "tharkun.type_propagation", "tharkun.structure_infer",
        "tharkun.proxy", "tharkun.exception_feedback", "tharkun.call_walker",
        "tharkun.call_mapper",
        "tharkun.minleaf.guard_extract", "tharkun.minleaf.guard_normalize",
        "tharkun.minleaf.guard_trace",
    }
)


def _py_files(root: Path = PACKAGE_ROOT):
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def _all_imports(path: Path) -> set[str]:
    """Every top-level module name ``path`` imports, any prefix (not just tharkun)."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def _tharkun_imports(path: Path, *, eager_only: bool) -> set[str]:
    """tharkun modules imported by ``path`` (module body only when ``eager_only``)."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = list(tree.body) if eager_only else list(ast.walk(tree))
    found: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("tharkun"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tharkun"):
            found.add(node.module)
    return found


def _importers_of(prefix: str) -> set[str]:
    """Package-relative paths of every module importing ``prefix`` (any depth)."""

    return {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in _py_files()
        if any(m == prefix or m.startswith(prefix + ".") for m in _tharkun_imports(path, eager_only=False))
    }


def test_condition_eager_imports_are_leaves() -> None:
    eager = _tharkun_imports(SOLVE_ROOT / "condition.py", eager_only=True)
    assert eager <= {
        "tharkun.solve.constants",
        "tharkun.solve.types",
        "tharkun.solve.guard",
        "tharkun.solve.loop_exit",
    }, eager


def test_solve_imports_only_solve() -> None:
    """No edge out of the core, eager or lazy (#0607 removed the last one:
    condition now loads targets through solve/loader.py, not branch_solver)."""

    bad = set()
    for path in _py_files(SOLVE_ROOT):
        rel = path.relative_to(SOLVE_ROOT).as_posix()
        for mod in _tharkun_imports(path, eager_only=False):
            if not mod.startswith("tharkun.solve"):
                bad.add((rel, mod))
    assert bad == set(), sorted(bad)


# --- #0607: the core never runs the target except through solve/loader.py -----

_EXEC_MODULES = {"importlib", "subprocess", "runpy"}
_EXEC_CALLS = {"exec", "__import__"}
_EXEC_ATTRS = {("sys", "settrace"), ("sys", "setprofile"), ("sys", "monitoring")}


def _execution_primitives(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.split(".")[0] in _EXEC_MODULES)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _EXEC_MODULES:
                found.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _EXEC_CALLS:
            found.add(node.func.id + "()")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if (node.value.id, node.attr) in _EXEC_ATTRS:
                found.add(f"{node.value.id}.{node.attr}")
    return found


def test_solve_has_no_execution_primitives_outside_the_loader() -> None:
    bad = {}
    for path in _py_files(SOLVE_ROOT):
        if path.name == "loader.py":
            continue  # the one sanctioned place
        if path.name == "__init__.py":
            continue  # the lazy facade imports tharkun's own modules
        found = _execution_primitives(path)
        if found:
            bad[path.relative_to(SOLVE_ROOT).as_posix()] = sorted(found)
    assert bad == {}, bad


_FOREIGN_IDENTIFIERS = {"IRExpr", "IRFlatStep", "Program", "GeneratorSpec", "EnumCall"}


def _foreign(name: str) -> bool:
    return name in _FOREIGN_IDENTIFIERS or name.startswith("eregion_") or name == "amonhen"


def test_solve_has_no_eregion_or_ir_identifiers() -> None:
    """The core speaks predicates and value sets; eregion's IR vocabulary never appears."""

    bad = set()
    for path in _py_files(SOLVE_ROOT):
        rel = path.relative_to(SOLVE_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and _foreign(node.id):
                bad.add((rel, node.id))
            elif isinstance(node, ast.Attribute) and _foreign(node.attr):
                bad.add((rel, node.attr))
            elif isinstance(node, ast.Import):
                bad.update((rel, a.name) for a in node.names if _foreign(a.name.split(".")[0]))
            elif isinstance(node, ast.ImportFrom) and node.module and _foreign(node.module.split(".")[0]):
                bad.add((rel, node.module))
    assert bad == set(), sorted(bad)


def test_solve_imports_without_the_observation_layers() -> None:
    """Every module under solve/ imports in a fresh process without loading the
    tracer, the driver, the search adapters or the fuzzing layer."""

    import subprocess
    import sys

    code = (
        "import importlib, pkgutil, sys\n"
        "import tharkun.solve as s\n"
        "for m in pkgutil.walk_packages(s.__path__, 'tharkun.solve.'):\n"
        "    importlib.import_module(m.name)\n"
        "loaded = {m for m in sys.modules if m.startswith('tharkun.')}\n"
        "bad = loaded & {'tharkun.trace_api', 'tharkun.branch_solver', 'tharkun.guard_analyzer',\n"
        "                'tharkun.driver', 'tharkun.hybrid', 'tharkun.redqueen', 'tharkun.differential',\n"
        "                'tharkun.hypothesis_adapter'}\n"
        "assert not bad, sorted(bad)\n"
        "assert not any(m.startswith('tharkun.') and not m.startswith('tharkun.solve') for m in loaded), sorted(loaded)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_internal_importers_use_the_solve_paths() -> None:
    """The old paths are shims for external consumers; tharkun imports the real homes."""

    bad = set()
    for root in (PACKAGE_ROOT, TESTS_ROOT):
        for path in _py_files(root):
            if path.name == "test_public_surface.py":
                continue  # pins the shim paths on purpose
            mods = _tharkun_imports(path, eager_only=False)
            if mods == {"tharkun._shims"}:
                continue  # the shim file itself
            bad.update((path.relative_to(root).as_posix(), m) for m in mods & SHIM_PATHS)
    assert bad == set(), sorted(bad)


def test_trace_api_imports_no_tharkun() -> None:
    assert _tharkun_imports(PACKAGE_ROOT / "trace_api.py", eager_only=False) == set()


def test_package_root_has_no_eager_imports() -> None:
    """#0605: ``tharkun/__init__.py`` is a PEP 562 lazy root."""

    eager = _tharkun_imports(PACKAGE_ROOT / "__init__.py", eager_only=True)
    assert eager == set(), eager


def test_fuzzing_layer_has_no_inbound_edges_from_outside() -> None:
    assert _importers_of("tharkun.redqueen") <= {"driver.py"}
    outside = {f for f in _importers_of("tharkun.hybrid") if not f.startswith("hybrid/")}
    assert outside <= {"sv/hybrid.py"}, outside


# --- avengers is DRIVE, and nothing above it depends on it -------------------

AVENGERS_ROOT = PACKAGE_ROOT / "avengers"

#: Layers at or below DRIVE, which ``avengers`` may import (docs/LAYERS.md).
_AT_OR_BELOW_DRIVE = (
    "tharkun.solve",
    "tharkun.trace_api",
    "tharkun.hypothesis_adapter",
    "tharkun.minleaf",
    "tharkun.strategies",
    "tharkun.facts",
    "tharkun.lattice",
    "tharkun.knowledge",
    "tharkun.warmup",
    "tharkun.egraph",
    "tharkun.branch_solver",
    "tharkun.guard_analyzer",
    "tharkun.driver",
    "tharkun.differential",
    "tharkun.mock_gen",
    "tharkun.mutation_input_gen",
    "tharkun.avengers",
)

#: Layers above DRIVE.  An import of one of these from ``avengers`` would make
#: the object-construction surface depend on the fuzzer or the tool layer.
_ABOVE_DRIVE = ("tharkun.redqueen", "tharkun.hybrid", "tharkun.cli", "tharkun.mcp_server")


def test_avengers_imports_nothing_above_the_drive_layer() -> None:
    """``avengers/*`` is DRIVE (docs/LAYERS.md), so FUZZ and TOOLS are closed to it."""

    bad = set()
    for path in _py_files(AVENGERS_ROOT):
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        for module in _tharkun_imports(path, eager_only=False):
            if module.startswith(_ABOVE_DRIVE):
                bad.add((rel, module))
            elif module.startswith("tharkun.") and not module.startswith(_AT_OR_BELOW_DRIVE):
                bad.add((rel, module))
    assert bad == set(), sorted(bad)


def test_avengers_uses_the_real_module_homes_not_the_shims() -> None:
    bad = set()
    for path in _py_files(AVENGERS_ROOT):
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        for module in _tharkun_imports(path, eager_only=False):
            if module in SHIM_PATHS:
                bad.add((rel, module))
    assert bad == set(), sorted(bad)


def test_only_named_drive_callers_import_avengers() -> None:
    """Only the callers the owner named may use the object-construction
    surface directly: SV's own whole-file seed stage, and driver.py's
    on-demand object flip for a guard minleaf cannot mutate.

    [internal ref] slice 3 ([internal ref], owner ruling, corrected 2026-09-29):
    Avengers is ALREADY Tharkun's generic, implemented object-construction
    line -- every caller (SV, driver.py, and Eregion's [internal ref] code in a
    different repo) reaches it through its OWN existing public API directly;
    no separate service module sits between them and it. This test still
    pins WHICH files may reach for it, so a stray new import elsewhere in
    DRIVE is a deliberate decision, not an accident."""

    importers = {
        path
        for path in _importers_of("tharkun.avengers")
        if not path.startswith("avengers/")
    }
    assert all(path.startswith("sv/") or path == "driver.py"
               for path in importers), sorted(importers)


def test_nothing_outside_avengers_imports_py_obj_flatting() -> None:
    """The flattener bridge is Avengers-internal ([internal ref], docs/LAYERS.md): SV, the
    solver and the fuzzing layer reach a flattened object only through Avengers
    APIs (``synthesize_object`` / ``ObjectGoal``) and the recipes Avengers
    materializes, never by importing ``py_obj_flatting`` themselves."""

    bad = set()
    for path in _py_files():
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        if rel.startswith("avengers/"):
            continue
        if "py_obj_flatting" in _all_imports(path):
            bad.add(rel)
    assert bad == set(), sorted(bad)


def test_the_worker_is_reachable_as_a_module_for_the_child_interpreter() -> None:
    """``python -m tharkun.avengers._worker`` is the containment boundary."""

    worker = AVENGERS_ROOT / "_worker.py"
    assert worker.is_file()
    tree = ast.parse(worker.read_text(encoding="utf-8"), filename=str(worker))
    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    ), "the worker must be runnable as a script"


# --- sv is DRIVE ([internal ref], [internal ref] slice 2) ----------------------------------------

SV_ROOT = PACKAGE_ROOT / "sv"


def test_sv_imports_only_layers_at_or_below_drive() -> None:
    """``tharkun.sv`` (docs/SV.md) is a DRIVE-layer front end (docs/LAYERS.md): like
    ``avengers`` it may import solve, trace, search, kb and drive -- Tharkun's own
    solvers are what it points at SV tasks -- but nothing above DRIVE; and nothing
    outside it imports it except the tools layer."""

    for path in _py_files(SV_ROOT):
        for module in _tharkun_imports(path, eager_only=False):
            if path.name == "hybrid.py" and module.startswith("tharkun.hybrid"):
                continue  # the SV stage adapter may call the FUZZ backend
            assert not module.startswith(_ABOVE_DRIVE), (path, module)
            assert module.startswith(_AT_OR_BELOW_DRIVE) or module.startswith("tharkun.sv"), (
                path, module)
    importers = {p for p in _importers_of("tharkun.sv") if not p.startswith("sv/")}
    assert importers <= {"cli.py", "mcp_server.py"}, sorted(importers)


def test_the_sv_child_imports_no_tharkun() -> None:
    """``sv/_child.py`` runs as a script in the program's own interpreter: no tharkun import."""

    child = SV_ROOT / "_child.py"
    assert _tharkun_imports(child, eager_only=False) == set()
