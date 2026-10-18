"""``tharkun.solve.loader`` (#0607, plan #0600): the one place the solving core
imports or executes a target module, and the ``no_targets()`` switch that turns
every such load into its absent value so the core answers purely statically.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tharkun.solve import loader
from tharkun.solve.loader import no_targets, targets_disabled


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_no_targets_is_scoped_and_reentrant() -> None:
    assert not targets_disabled()
    with no_targets():
        assert targets_disabled()
        with no_targets():
            assert targets_disabled()
        assert targets_disabled()
    assert not targets_disabled()


def test_import_dotted() -> None:
    import json

    assert loader.import_dotted("json") is json
    assert loader.import_dotted(None) is None
    assert loader.import_dotted("") is None
    assert loader.import_dotted("no.such.module.anywhere") is None
    with no_targets():
        assert loader.import_dotted("json") is None


def test_load_from_file_and_exec_source(tmp_path: Path) -> None:
    target = _write(tmp_path, "target_0607.py", "X = 41\n\ndef f():\n    return X + 1\n")
    mod = loader.load_from_file(str(target))
    assert mod is not None and mod.f() == 42
    module_globals = loader.exec_source(str(target))
    assert module_globals is not None and module_globals["f"]() == 42 and "__builtins__" in module_globals
    assert loader.load_from_file(str(tmp_path / "missing.py")) is None
    assert loader.exec_source(str(tmp_path / "missing.py")) is None
    with no_targets():
        assert loader.load_from_file(str(target)) is None
        assert loader.exec_source(str(target)) is None


def test_load_module_for_execution_uses_the_package_system_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "pkg_0607"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    _write(pkg, "sibling.py", "V = 7\n")
    mod_path = _write(pkg, "mod.py", "from . import sibling\nVALUE = sibling.V\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert loader.file_path_to_dotted_module(str(mod_path)) == "pkg_0607.mod"
    mod, module_globals = loader.load_module_for_execution(str(mod_path))
    assert mod is not None and mod.VALUE == 7  # the relative import only works via strategy 1
    assert module_globals["VALUE"] == 7 and "__builtins__" in module_globals
    with no_targets():
        assert loader.load_module_for_execution(str(mod_path)) == (None, {})
    assert loader.load_module_for_execution(str(tmp_path / "missing.py")) == (None, {})


@pytest.mark.parametrize("active_context", [False, True])
def test_source_only_loading_preserves_metadata_and_encoding(tmp_path, monkeypatch, active_context):
    from contextlib import nullcontext
    from importlib.machinery import SourceFileLoader
    from tharkun.solve.environment import confined

    target = tmp_path / "encoded.py"
    target.write_bytes(b"# coding: latin-1\nVALUE = 'caf\xe9'\nclass Box: pass\ndef f(x: VALUE): pass\n")
    flags = set()

    def no_cache(*args, **kwargs):
        pytest.fail("source-only loading consulted the bytecode loader")

    monkeypatch.setattr(SourceFileLoader, "get_code", no_cache)
    with confined([str(target)], flags=flags) if active_context else nullcontext():
        mod = loader.load_from_file(str(target), source_only=not active_context)
        assert mod is not None
        assert mod.VALUE == "café" and mod.f.__annotations__ == {"x": "café"}
        assert mod.__name__ == loader.EXEC_TARGET_NAME
        assert mod.__file__ == str(target) and mod.__spec__.origin == str(target)
        assert mod.__package__ == "" and mod.Box.__module__ == mod.__name__
    assert flags == set()
    with no_targets():
        assert loader.load_from_file(str(target), source_only=True) is None


@pytest.mark.parametrize("effect", ["read", "unlink"])
def test_source_only_loading_keeps_target_effects_confined(tmp_path, effect):
    from tharkun.solve.environment import confined

    victim = tmp_path / "victim"
    victim.write_text("keep")
    directory = tmp_path / "declared"
    directory.mkdir()
    action = f"open({str(victim)!r}).read()" if effect == "read" else f"os.unlink({str(victim)!r})"
    target = _write(directory, "target.py", f"import os\ntry:\n    {action}\nexcept OSError:\n    REFUSED = True\n")
    flags = set()
    with confined([str(target)], flags=flags):
        mod, namespace = loader.load_module_for_execution(str(target))
        assert mod is not None and namespace["REFUSED"] is True
    assert "file_unmodelled" in flags
    assert victim.read_text() == "keep"


def test_branch_solver_delegates_to_the_loader() -> None:
    from tharkun import branch_solver

    assert branch_solver._load_module_for_execution is loader.load_module_for_execution
    assert branch_solver._file_path_to_dotted_module is loader.file_path_to_dotted_module


def test_no_targets_solves_an_isinstance_guard_without_running_the_target(tmp_path: Path) -> None:
    """The static answer survives with target loading off; the module body never runs."""

    marker = tmp_path / "ran.marker"
    src = _write(
        tmp_path,
        "guarded_0607.py",
        f'''
        import pathlib

        pathlib.Path({str(marker)!r}).write_text("ran")


        class Foo:
            def __init__(self, n: int = 0):
                self.n = n


        def f(x):
            if isinstance(x, Foo):
                return 1
            return 0
        ''',
    )
    lineno = next(i for i, line in enumerate(src.read_text().splitlines(), 1) if "isinstance(x, Foo)" in line)
    from tharkun.solve import ConditionSolution, solve_condition

    with no_targets():
        sol = solve_condition(str(src), lineno)
    assert isinstance(sol, ConditionSolution)
    assert not marker.exists(), "the target module body ran under no_targets()"
    assert any(v.name == "x" for v in sol.variables)

    # Outside the guard the solver may load the target to construct a real Foo.
    solve_condition(str(src), lineno)
    assert marker.exists(), "expected the unguarded solve to load the target (the test would otherwise prove nothing)"


def test_a_path_component_that_is_not_an_identifier_yields_no_dotted_name(
    tmp_path: Path, monkeypatch
) -> None:
    """A directory can be on ``sys.path`` and still name no package."""

    root = tmp_path / "root"
    awkward = root / ".venv" / "lib" / "python3.12" / "site-packages"
    awkward.mkdir(parents=True)
    (awkward / "thing.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(root))
    assert loader.file_path_to_dotted_module(str(awkward / "thing.py")) is None


def test_the_longest_matching_sys_path_entry_wins(tmp_path: Path, monkeypatch) -> None:
    """A target under the current directory must still be named by its own root.

    A virtual environment inside a checkout is the case that matters: the
    checkout is on ``sys.path`` too, and naming the module relative to IT would
    import something else, or nothing.
    """

    root = tmp_path / "root"
    site = root / "env" / "site_packages"
    package = site / "pkg_longest"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text("VALUE = 3\n")

    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(root))  # searched first, and it matches too
    assert loader.file_path_to_dotted_module(str(package / "mod.py")) == "pkg_longest.mod"
