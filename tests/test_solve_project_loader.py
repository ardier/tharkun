"""#1132b: the project loader -- the ONE answer to "import this file as the member of its package it is".

``solve/loader.py::project_root_and_module`` is static; ``load_project_module`` is the real import in the
calling process behind a confining facade; ``import_project_isolated`` is the same import in a disposable
child.  Each test builds a throwaway package so the attack shapes are exact: an ``__init__`` that mutates a
sibling (the reason a synthetic parent package is unsound), two copies of one package, a sibling named like the
standard library, a failing import.
"""
from __future__ import annotations

import contextlib
import os
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from tharkun.solve import loader
from tharkun.solve.loader import (
    ConfinedModule,
    ProjectImportError,
    import_project_isolated,
    load_project_module,
    no_targets,
    project_root_and_module,
    relative_import_level,
)


def write(root: Path, rel: str, body: str = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


class Recorder:
    """A confine factory that records every entry and exit."""

    def __init__(self):
        self.events: list[str] = []
        self.depth = 0

    def __call__(self):
        @contextlib.contextmanager
        def cm():
            self.events.append("enter")
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1
                self.events.append("exit")
        return cm()


def nothing():
    return contextlib.nullcontext()


@pytest.fixture(autouse=True)
def _clean_modules():
    before = set(sys.modules)
    path = list(sys.path)
    yield
    for name in set(sys.modules) - before:
        sys.modules.pop(name, None)
    sys.path[:] = path


# -- the static split ---------------------------------------------------------


def test_a_module_in_nested_packages_is_named_by_its_full_dotted_path(tmp_path):
    f = write(tmp_path, "top/__init__.py")
    write(tmp_path, "top/sub/__init__.py")
    mod = write(tmp_path, "top/sub/m.py", "X = 1\n")
    assert project_root_and_module(str(mod)) == (str(tmp_path.resolve()), "top.sub.m")
    assert project_root_and_module(str(f)) == (str(tmp_path.resolve()), "top")


def test_a_standalone_file_is_named_by_its_stem(tmp_path):
    f = write(tmp_path, "loose.py", "X = 1\n")
    assert project_root_and_module(str(f)) == (str(tmp_path.resolve()), "loose")


def test_relative_imports_in_a_directory_without_init_make_it_a_namespace_package(tmp_path):
    f = write(tmp_path, "ns/m.py", "from . import sibling\n")
    write(tmp_path, "ns/sibling.py")
    assert relative_import_level(str(f)) == 1
    assert project_root_and_module(str(f)) == (str(tmp_path.resolve()), "ns.m")


def test_a_deeper_relative_import_reaches_further_up(tmp_path):
    f = write(tmp_path, "a/b/m.py", "from .. import x\n")
    assert project_root_and_module(str(f)) == (str(tmp_path.resolve()), "a.b.m")


def test_no_import_name_for_a_non_python_file_a_missing_file_or_a_non_identifier_directory(tmp_path):
    txt = write(tmp_path, "notes.txt", "x")
    bad = write(tmp_path, "my-dir/__init__.py")
    inner = write(tmp_path, "my-dir/m.py")
    assert project_root_and_module(str(txt)) is None
    assert project_root_and_module(str(tmp_path / "missing.py")) is None
    assert project_root_and_module(str(inner)) is None
    assert bad.exists()


def test_avengers_uses_the_same_walk(tmp_path):
    from tharkun.avengers.backends import _project_and_module

    write(tmp_path, "pkg/__init__.py")
    mod = write(tmp_path, "pkg/m.py")
    assert _project_and_module(str(mod)) == project_root_and_module(str(mod))


# -- the real import ----------------------------------------------------------


def make_pkg(root: Path, name: str = "pkg", *, limit: int = 50, init: str = "") -> Path:
    write(root, f"{name}/__init__.py", init)
    write(root, f"{name}/_c.py", f"LIMIT = {limit}\n")
    return write(root, f"{name}/mod.py", """
        from . import _c

        def f(x):
            if x > _c.LIMIT:
                return -1
            return x * 2
    """)


def test_relative_imports_resolve_and_the_function_runs(tmp_path):
    mod = make_pkg(tmp_path)
    loaded = load_project_module(str(mod), confine=nothing)
    assert loaded.dotted == "pkg.mod" and loaded.root == str(tmp_path.resolve())
    assert loaded.module.f(10) == 20 and loaded.module.f(100) == -1


def test_the_package_init_runs_first_and_its_effect_is_what_the_module_sees(tmp_path):
    # the r7 shape: __init__ lowers a sibling's constant; a synthetic parent would never run it
    mod = make_pkg(tmp_path, init="from . import _c\n_c.LIMIT = 5\n")
    loaded = load_project_module(str(mod), confine=nothing)
    assert loaded.module.f(10) == -1, "x=10 is above the LIMIT __init__ set"
    assert loaded.module.f(5) == 10


def test_init_and_submodules_run_in_their_real_order(tmp_path):
    write(tmp_path, "ordpkg/__init__.py", "import builtins\nbuiltins._ORDER = ['init']\nfrom . import a\n")
    write(tmp_path, "ordpkg/a.py", "import builtins\nbuiltins._ORDER.append('a')\nfrom . import b\n")
    write(tmp_path, "ordpkg/b.py", "import builtins\nbuiltins._ORDER.append('b')\n")
    mod = write(tmp_path, "ordpkg/entry.py", "import builtins\nbuiltins._ORDER.append('entry')\nfrom . import b\n")
    try:
        load_project_module(str(mod), confine=nothing)
        import builtins
        assert builtins._ORDER == ["init", "a", "b", "entry"]
    finally:
        import builtins
        if hasattr(builtins, "_ORDER"):
            del builtins._ORDER


def test_two_copies_of_one_package_do_not_collide(tmp_path):
    orig = make_pkg(tmp_path / "orig", limit=50)
    mutant = make_pkg(tmp_path / "mut", limit=5)
    a = load_project_module(str(orig), confine=nothing)
    b = load_project_module(str(mutant), confine=nothing)
    assert a.module.f(10) == 20 and b.module.f(10) == -1
    assert a.raw is not b.raw and a.raw.__file__ != b.raw.__file__
    # loading the second displaced the first's cache entries; the first still works through its own globals
    assert a.module.f(10) == 20


def test_reloading_the_same_directory_gives_a_fresh_module_instance(tmp_path):
    mod = write(tmp_path / "p" if False else tmp_path, "cnt/mod.py", "COUNT = []\n\ndef bump():\n    COUNT.append(1)\n    return len(COUNT)\n")
    write(tmp_path, "cnt/__init__.py")
    first = load_project_module(str(mod), confine=nothing)
    assert first.module.bump() == 1 and first.module.bump() == 2
    second = load_project_module(str(mod), confine=nothing)
    assert second.module.bump() == 1, "a new load is a new process-like state, not the cached module"
    assert second.evicted, "the cached copy was displaced"


def test_a_sibling_named_like_the_standard_library_does_not_shadow_it(tmp_path):
    write(tmp_path, "pk/__init__.py")
    write(tmp_path, "pk/fractions.py", "LIMIT = 50\n")
    mod = write(tmp_path, "pk/mod.py", "from .fractions import LIMIT\n\ndef f(x):\n    return x if x <= LIMIT else -1\n")
    loaded = load_project_module(str(mod), confine=nothing)
    import fractions
    assert hasattr(fractions, "Fraction"), "the standard library module is untouched"
    assert loaded.module.f(10) == 10
    assert sys.modules["pk.fractions"].LIMIT == 50


def test_a_sibling_named_like_the_inverters_helper_modules_is_just_a_sibling(tmp_path):
    write(tmp_path, "pk2/__init__.py")
    write(tmp_path, "pk2/bw_runtime.py", "LIMIT = 50\n")
    write(tmp_path, "pk2/inverter_helpers.py", "LIMIT = 60\n")
    mod = write(tmp_path, "pk2/mod.py", "from .bw_runtime import LIMIT as A\nfrom .inverter_helpers import LIMIT as B\n")
    loaded = load_project_module(str(mod), confine=nothing)
    assert (loaded.raw.A, loaded.raw.B) == (50, 60)
    assert "bw_runtime" not in sys.modules or sys.modules["bw_runtime"].__name__ == "bw_runtime"


def test_sys_path_is_restored_and_the_root_is_not_left_behind(tmp_path):
    mod = make_pkg(tmp_path)
    before = list(sys.path)
    load_project_module(str(mod), confine=nothing)
    assert sys.path == before


def test_extra_roots_are_importable_during_the_import_and_not_left_behind(tmp_path):
    write(tmp_path, "libs/helperlib.py", "V = 7\n")
    write(tmp_path, "app/__init__.py")
    mod = write(tmp_path, "app/m.py", "import helperlib\nV = helperlib.V\n")
    loaded = load_project_module(str(mod), roots=[str(tmp_path / "libs")], confine=nothing)
    assert loaded.raw.V == 7
    assert str((tmp_path / "libs").resolve()) not in sys.path


def test_the_roots_are_importable_during_every_call_and_the_confine_context_and_only_then(tmp_path):
    """A function that imports a project module by absolute name at call time finds it, as it would natively
    (the project root is on ``sys.path`` while project code runs); the anchored replay re-executes the module
    through ``module._confine()`` and needs the same."""
    write(tmp_path, "libs/helperlib.py", "V = 7\n")
    write(tmp_path, "toplevel_helper_lz.py", "K = 3\n")
    write(tmp_path, "app2/__init__.py")
    mod = write(tmp_path, "app2/m.py",
                "def f():\n    import helperlib, toplevel_helper_lz\n    return helperlib.V + toplevel_helper_lz.K\n")
    root, libs = str(tmp_path.resolve()), str((tmp_path / "libs").resolve())
    before = list(sys.path)
    loaded = load_project_module(str(mod), roots=[libs], confine=nothing)
    assert loaded.module.f() == 10
    assert sys.path == before, "neither root is left on sys.path after the call"
    with loaded.module._confine():
        assert root in sys.path and libs in sys.path
    assert sys.path == before


def test_a_failing_import_raises_project_import_error_and_leaves_the_process_clean(tmp_path):
    write(tmp_path, "bad/__init__.py")
    mod = write(tmp_path, "bad/mod.py", "raise RuntimeError('boom')\n")
    prior = write(tmp_path, "keep/__init__.py")
    del prior
    before = set(sys.modules)
    with pytest.raises(ProjectImportError, match="boom") as info:
        load_project_module(str(mod), confine=nothing)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert not [m for m in set(sys.modules) - before if m == "bad" or m.startswith("bad.")]


def test_a_failed_reload_puts_the_previous_copy_back(tmp_path):
    mod = make_pkg(tmp_path)
    good = load_project_module(str(mod), confine=nothing)
    (tmp_path / "pkg" / "_c.py").write_text("raise RuntimeError('now broken')\n")
    with pytest.raises(ProjectImportError):
        load_project_module(str(mod), confine=nothing)
    assert sys.modules["pkg.mod"] is good.raw


def test_a_file_with_no_import_name_is_refused(tmp_path):
    f = write(tmp_path, "my-dir/__init__.py")
    inner = write(tmp_path, "my-dir/m.py")
    assert f.exists()
    with pytest.raises(ProjectImportError, match="no import name"):
        load_project_module(str(inner), confine=nothing)


def test_loading_is_refused_inside_no_targets(tmp_path):
    mod = make_pkg(tmp_path)
    with no_targets(), pytest.raises(ProjectImportError, match="switched off"):
        load_project_module(str(mod), confine=nothing)


def test_confine_is_required():
    with pytest.raises(TypeError):
        load_project_module("x.py")  # type: ignore[call-arg]


# -- the confining facade -----------------------------------------------------


def test_the_import_and_every_function_call_run_inside_confine(tmp_path):
    mod = make_pkg(tmp_path)
    rec = Recorder()
    loaded = load_project_module(str(mod), confine=rec)
    assert rec.events == ["enter", "exit"], "the import is confined"
    assert loaded.module.f(3) == 6
    assert rec.events == ["enter", "exit", "enter", "exit"], "one confined call"


def test_project_code_sees_confine_active_during_a_call(tmp_path):
    write(tmp_path, "pc/__init__.py")
    mod = write(tmp_path, "pc/mod.py", "import builtins\n\ndef probe():\n    return builtins._SEEN()\n")
    rec = Recorder()
    import builtins
    builtins._SEEN = lambda: rec.depth
    try:
        loaded = load_project_module(str(mod), confine=rec)
        assert loaded.module.probe() == 1
    finally:
        del builtins._SEEN


def test_a_call_made_inside_a_confined_call_does_not_re_enter(tmp_path):
    write(tmp_path, "nest/__init__.py")
    mod = write(tmp_path, "nest/mod.py", "def g():\n    return 1\n\ndef f():\n    return g() + 1\n")
    rec = Recorder()
    loaded = load_project_module(str(mod), confine=rec)
    rec.events.clear()
    assert loaded.module.f() == 2
    assert rec.events == ["enter", "exit"]


def test_the_facade_writes_through_and_wraps_only_functions(tmp_path):
    mod = make_pkg(tmp_path)
    loaded = load_project_module(str(mod), confine=nothing)
    assert isinstance(loaded.module, ConfinedModule)
    loaded.module.INJECTED = 41
    assert loaded.raw.INJECTED == 41
    assert loaded.module._c is loaded.raw._c, "a module attribute passes through untouched"
    assert loaded.module.f.__name__ == "f" and loaded.module.f.__wrapped__ is loaded.raw.f
    del loaded.module.INJECTED
    assert not hasattr(loaded.raw, "INJECTED")
    assert "f" in dir(loaded.module)


def test_facade_calls_on_two_threads_are_each_confined(tmp_path):
    mod = make_pkg(tmp_path)
    rec = Recorder()
    loaded = load_project_module(str(mod), confine=rec)
    rec.events.clear()
    results = []

    def run():
        results.append(loaded.module.f(2))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [4] * 4 and rec.events.count("enter") == 4


# -- the isolated import ------------------------------------------------------


def test_the_isolated_import_reports_and_never_loads_the_project_here(tmp_path):
    mod = make_pkg(tmp_path, init="from . import _c\n_c.LIMIT = 5\n")
    report = import_project_isolated(str(mod))
    assert report["ok"] is True and report["dotted"] == "pkg.mod"
    assert "f" in report["names"]
    assert "pkg" not in sys.modules and "pkg.mod" not in sys.modules


def test_the_isolated_import_reports_a_failure_instead_of_raising(tmp_path):
    write(tmp_path, "bad/__init__.py")
    mod = write(tmp_path, "bad/mod.py", "raise RuntimeError('boom')\n")
    report = import_project_isolated(str(mod))
    assert report["ok"] is False and "RuntimeError: boom" in report["error"]


def test_the_isolated_import_kills_a_hang(tmp_path):
    write(tmp_path, "slow/__init__.py")
    mod = write(tmp_path, "slow/mod.py", "import time\ntime.sleep(60)\n")
    report = import_project_isolated(str(mod), timeout=1.5)
    assert report["ok"] is False and report["error"] == "timeout"


def test_the_isolated_import_runs_under_the_environment_model(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    write(tmp_path, "esc/__init__.py")
    mod = write(tmp_path, "esc/mod.py", f"""
        import os
        try:
            os.remove({str(victim)!r})
        except OSError:
            pass
        with open({str(victim)!r}, "w") as fh:
            fh.write("clobbered")
    """)
    report = import_project_isolated(str(mod))
    assert victim.read_text() == "keep"
    assert "file_effect" in report["flags"] or "file_unmodelled" in report["flags"]


def test_isolated_import_of_a_non_module_and_with_targets_off(tmp_path):
    txt = write(tmp_path, "x.txt", "1")
    assert import_project_isolated(str(txt))["ok"] is False
    mod = make_pkg(tmp_path)
    with no_targets():
        assert "switched off" in import_project_isolated(str(mod))["error"]


# -- in-process confinement: the replay of a package that tries to escape --------


def test_package_code_cannot_write_delete_or_rename_outside_scratch_during_an_in_process_import_and_replay(tmp_path):
    """MAIN's requirement for #1132b: the in-process replay of the user's package runs under the environment model
    and file-system confinement (``solve/environment.py::confined``), so what it tries at import time and what it
    tries on every call leaves the files it does not own intact."""
    from tharkun.solve.environment import confined

    victim = tmp_path / "victim.txt"
    other = tmp_path / "other.txt"
    keep = tmp_path / "keepdir"
    victim.write_text("victim")
    other.write_text("other")
    keep.mkdir()
    (keep / "inner.txt").write_text("inner")
    before = {p.relative_to(tmp_path): (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
              for p in tmp_path.rglob("*") if "pkg" not in p.parts}
    write(tmp_path, "pkg/__init__.py", f"""
        import os, shutil
        for op in (lambda: os.remove({str(victim)!r}), lambda: os.rename({str(other)!r}, {str(other) + '.moved'!r}),
                   lambda: shutil.rmtree({str(keep)!r}), lambda: open({str(victim)!r}, "w").write("clobbered")):
            try:
                op()
            except OSError:
                pass
    """)
    mod = write(tmp_path, "pkg/mod.py", f"""
        import os, pathlib, shutil

        def f(x):
            for op in (lambda: os.remove({str(victim)!r}), lambda: os.replace({str(other)!r}, {str(other) + '.moved'!r}),
                       lambda: pathlib.Path({str(victim)!r}).unlink(), lambda: shutil.rmtree({str(keep)!r}),
                       lambda: open({str(other)!r}, "w").write("clobbered"), lambda: os.truncate({str(victim)!r}, 0)):
                try:
                    op()
                except OSError:
                    pass
            return x * 2
    """)
    flags: set = set()
    loaded = load_project_module(str(mod), confine=lambda: confined((str(mod),), None, flags))
    for x in range(5):                                      # many replays, as an inverter does
        assert loaded.module.f(x) == x * 2
    after = {p.relative_to(tmp_path): (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
             for p in tmp_path.rglob("*") if "pkg" not in p.parts}
    assert after == before, "nothing outside the scratch changed"
    assert {"file_effect", "file_unmodelled"} <= flags
    # and the window is closed again: this process can mutate files normally
    os.remove(victim)
    assert not victim.exists()
