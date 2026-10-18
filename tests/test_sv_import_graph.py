"""The competition front end and its subprocesses cannot reach removed solvers."""

import ast
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "tharkun"
FORBIDDEN = {"touchstone", "crosshair", "tharkun.touchstone_adapter", "tharkun.crosshair_adapter"}


def _imports(path, module):
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name("." * node.level + (node.module or ""), package)
            yield base
            yield from (base + "." + a.name for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Includes lazy export tables, literal dynamic imports and -m workers.
            name = node.value
            if name.startswith("tharkun.") or name.split(".")[0] in FORBIDDEN:
                yield resolve_name(name, package)


def test_sv_import_graph_excludes_removed_solvers():
    metadata = (ROOT.parents[1] / "pyproject.toml").read_text().lower()
    assert not any(name in metadata for name in ("touchstone", "crosshair"))
    modules = {}
    for path in ROOT.rglob("*.py"):
        parts = path.relative_to(ROOT.parent).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        modules[module] = path
    # Workers run by filename are not ordinary import edges: seed the whole SV package.
    pending = [(m, (m,)) for m in modules if m == "tharkun.sv" or m.startswith("tharkun.sv.")]
    seen = set()
    while pending:
        module, trail = pending.pop()
        assert not any(module == f or module.startswith(f + ".") for f in FORBIDDEN), " -> ".join(trail)
        if module in seen or module not in modules:
            continue
        seen.add(module)
        parent = module.rpartition(".")[0]
        if parent:
            pending.append((parent, (*trail, parent)))
        pending.extend((m, (*trail, m)) for m in _imports(modules[module], module))
    assert {"tharkun.driver", "tharkun.branch_solver", "tharkun.sv._symexec"} <= seen
    assert not any("touchstone" in p.name or "crosshair" in p.name for p in ROOT.rglob("*.py"))


def test_import_edges_include_lazy_relative_and_dynamic_imports(tmp_path):
    path = tmp_path / "worker.py"
    path.write_text(
        "def lazy():\n"
        "    from .. import touchstone_adapter\n"
        "    import crosshair.core\n"
        "    importlib.import_module('touchstone')\n"
    )
    assert {"tharkun.touchstone_adapter", "crosshair.core", "touchstone"} <= set(
        _imports(path, "tharkun.sv.worker")
    )
