"""[internal ref]: what a program may ask of ITS OWN ``__main__`` namespace and of the import system, and still be certified deterministic.

The child runs the program as a native ``__main__`` (docs/SV.md "How it decides"): ``__builtins__`` is the builtins MODULE, ``sys.path[0]`` is the
program's directory and never ``tharkun/sv``, so ``import spec`` is the program's own module or a ``ModuleNotFoundError``.  Four MIN pairs (TRUE,
natively verified) were UNKNOWN only because the determinism census refused their reads (``imports types``, ``uses __builtins__``, ``attribute __name__
resolves to nothing known``, ``imports spec``):

* ``types.ModuleType`` alone resolves from ``types`` (``pyresolve.RESTRICTED_MODULE_ATTRS``);
* ``__builtins__`` resolves ONLY as the argument of ``isinstance(...)``/``type(...)`` in the top-level program (``Resolver._builtins_type_probe``);
* ``.__name__`` / ``.__qualname__`` / ``.__doc__`` / ``.__module__`` / ``.__init__`` (``DUNDER_OK``) need no further name in the known attributes;
* a literal's text (``assert False, "msg"``) shows no address;
* ``import X`` resolves when a run OBSERVED X absent (``_child.py::watch_imports``, ``Outcome.absent_imports``,
  ``Resolver.absent_imports``): it raises ``ModuleNotFoundError`` on every run in this environment.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tharkun.solve.determinism import determinism_census
from tharkun.sv import _child, verify
from tharkun.sv.engine import run_once

PAIRS = {
    "ch_main_builtins_module_true": "import types\n\nassert isinstance(__builtins__, types.ModuleType)\n",
    "cmp_m05_builtins_module": "assert type(__builtins__).__name__ == 'module'\n",
    "ch_main_no_foreign_modules_true": (
        "try:\n    import spec\nexcept ImportError:\n    pass\nelse:\n    assert False, \"found a module named spec\"\n"
        "try:\n    import witness\nexcept ImportError:\n    pass\nelse:\n    assert False, \"found a module named witness\"\n"),
    "cmp_m13_import_spec": "try:\n    import spec\nexcept ImportError:\n    pass\nelse:\n    assert False\n",
}
ABSENT = frozenset({"spec", "witness"})


def census(src, absent=frozenset()):
    return determinism_census(textwrap.dedent(src), "<program>", absent_imports=absent)


def native(tmp_path, src, name="prog.py"):
    """The truth: ``python prog.py``, exit status 0 or 1 and the last line of the traceback."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(src))
    p = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, cwd=str(tmp_path), timeout=60)
    return p.returncode, (p.stderr.strip().splitlines() or [""])[-1]


# ---------------------------------------------------------------------------
# the census
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(PAIRS))
def test_the_four_pair_programs_are_certified_when_the_absent_imports_are_observed(tmp_path, name):
    assert native(tmp_path, PAIRS[name])[0] == 0, name                      # the control: natively they hold
    assert census(PAIRS[name], ABSENT).ok, (name, census(PAIRS[name], ABSENT).reason)


def test_an_import_that_no_run_observed_absent_is_still_refused():
    for name in ("ch_main_no_foreign_modules_true", "cmp_m13_import_spec"):
        got = census(PAIRS[name])
        assert not got.ok and "imports spec" in got.reason, got
    # observed absent is per NAME: one observed name does not excuse another
    assert not census(PAIRS["ch_main_no_foreign_modules_true"], frozenset({"spec"})).ok


@pytest.mark.parametrize("src", [
    "import spec.sub\n",                                  # a dotted name: not reported absent, not resolved
    "from spec import x\n",                               # an ImportFrom binds names the census would have to resolve
    "import spec\nspec.run()\n",                          # a use of the failed import's name is a refusal as before
])
def test_only_a_plain_import_of_an_observed_absent_name_resolves(src):
    got = census(src, frozenset({"spec", "spec.sub"}))
    assert not got.ok, (src, got.reason)


def test_an_aliased_absent_import_resolves():
    assert census("try:\n    import spec as s\nexcept ImportError:\n    pass\n", frozenset({"spec"})).ok


@pytest.mark.parametrize("src, ok", [
    ("import types\nassert isinstance(__builtins__, types.ModuleType)\n", True),
    ("assert type(__builtins__).__name__ == 'module'\n", True),
    ("assert isinstance(__builtins__, dict)\n", True),                                 # a probe of WHAT it is, whatever T
    ("import types\nx = types.ModuleType('m')\nassert isinstance(x, types.ModuleType)\n", True),
    ("from types import ModuleType\nassert isinstance(__builtins__, ModuleType)\n", True),
    # anything else done with __builtins__ reaches eval and the like
    ("__builtins__.eval('1')\n", False),
    ("x = __builtins__\n", False),
    ("__builtins__['eval']('1')\n", False),
    ("assert len(vars(__builtins__)) > 0\n", False),
    ("assert isinstance(__builtins__, dict, 1)\n", False),
    ("assert type(__builtins__, 'x', ()) is not None\n", False),
    ("assert isinstance(x=__builtins__, y=dict)\n", False),
    ("__builtins__ = {}\nassert isinstance(__builtins__, dict)\n", False),             # the program rebinds it
    ("isinstance = lambda a, b: True\nassert isinstance(__builtins__, dict)\n", False),   # not the builtin
    ("def f(a=__builtins__):\n    return a\n", False),
    # the rest of types is not resolved
    ("import types\nassert types.FunctionType\n", False),
    ("import types\nassert types.MappingProxyType\n", False),
    ("import types\nx = types.ModuleType('m')\nprint(x)\n", False),                   # a module object has no address model
    # dunders: the value of the name, the docstring and the module are text; the machinery is not
    ("class A:\n    pass\nassert A.__name__ == 'A'\nassert type(A()).__qualname__ == 'A'\n", True),
    ("class A:\n    pass\nassert A.__dict__\n", False),
    ("assert (1).__class__\n", False),
])
def test_the_main_namespace_probes_resolve_and_nothing_else_does(src, ok):
    got = census(src)
    assert got.ok is ok, (src, got.reason)


def test_a_helper_module_sees_a_dict_for_builtins_so_the_probe_is_not_resolved_there(tmp_path):
    """``__builtins__`` is the module only in ``__main__``: ``isinstance(__builtins__, types.ModuleType)`` in a module the program imports is
    False natively (a dict), and the child's ``native_main_module`` makes no promise about it, so the probe is resolved in the top-level program
    only: a ``Resolver`` built for a recursively-read file (``_pkg_dir`` set) refuses it."""
    import ast

    from tharkun.solve.determinism import DET_MODULES
    from tharkun.solve.pyresolve import Resolver, Top

    src = "import types\nassert not isinstance(__builtins__, types.ModuleType)\n"
    helper = tmp_path / "helper.py"
    helper.write_text(src)
    main = tmp_path / "main.py"
    main.write_text("import helper\n")
    assert native(tmp_path, "import sys\nsys.path.insert(0, %r)\nimport helper\n" % str(tmp_path), "main2.py")[0] == 0   # natively a dict there
    Resolver(ast.parse(src), DET_MODULES, filename=str(main)).check_all()                    # the top-level program: resolved
    with pytest.raises(Top, match="__builtins__"):
        Resolver(ast.parse(src), DET_MODULES, filename=str(helper), _pkg_dir=str(tmp_path)).check_all()


def test_a_literals_text_shows_no_address_whatever_else_the_file_does():
    src = "try:\n    x = 1\nexcept ValueError:\n    pass\nassert x == 1, 'a message'\n"
    assert census(src).ok
    # but the text of a value that may be an object still does
    assert not census("class A:\n    pass\ntry:\n    x = A()\nexcept ValueError:\n    pass\nprint(x)\n").ok
    assert not census("class A:\n    pass\ntry:\n    x = A()\nexcept ValueError:\n    pass\nassert False, str(x)\n").ok


# ---------------------------------------------------------------------------
# the child records what no finder had
# ---------------------------------------------------------------------------

def program(tmp_path, body, name="prog.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return str(path)


def test_the_child_reports_the_names_no_finder_had(tmp_path):
    out = run_once(sys.executable, program(tmp_path, PAIRS["ch_main_no_foreign_modules_true"]), [], timeout=30)
    assert out.status == "clean" and out.flags == [], (out.status, out.exc, out.flags)
    assert out.absent_imports == ["spec", "witness"]
    out = run_once(sys.executable, program(tmp_path, "import json, os, math\ntry:\n    import nothing_here_1147\nexcept ImportError:\n    pass\n"),
                   [], timeout=30)
    assert out.absent_imports == ["nothing_here_1147"]                       # a module that exists is not reported


def test_a_module_in_the_programs_directory_is_found_not_absent(tmp_path):
    (tmp_path / "sibling.py").write_text("X = 1\n")
    out = run_once(sys.executable, program(tmp_path, "import sibling\nassert sibling.X == 1\n"), [], timeout=30)
    assert out.status == "clean" and out.absent_imports == []


def test_a_module_the_program_creates_and_imports_is_not_reported_absent(tmp_path):
    """The census never read the code behind a second ``import``: a name that became importable is not an environment fact."""
    body = """
        try:
            import created_1147
        except ImportError:
            pass
        with open("created_1147.py", "w") as fh:
            fh.write("VALUE = 1\\n")
        import importlib
        importlib.invalidate_caches()
        import created_1147
        assert created_1147.VALUE == 1
    """
    sub = tmp_path / "w"
    sub.mkdir()
    path = sub / "prog.py"
    path.write_text(textwrap.dedent(body).replace('"created_1147.py"', repr(str(sub / "created_1147.py"))))
    out = run_once(sys.executable, str(path), [], timeout=30)
    assert out.status == "clean", (out.status, out.exc, out.msg)
    assert "created_1147" not in out.absent_imports, out.absent_imports


def test_a_dotted_name_is_not_reported(tmp_path):
    out = run_once(sys.executable, program(tmp_path, "try:\n    import os.nothing\nexcept ImportError:\n    pass\ntry:\n    import json.nothing\nexcept ImportError:\n    pass\n"),
                   [], timeout=30)
    assert "os.nothing" not in out.absent_imports and "json.nothing" not in out.absent_imports


def test_the_recorder_is_the_last_finder_and_never_finds_anything():
    saved = list(sys.meta_path)
    try:
        _child._State.absent.clear()
        _child.watch_imports()
        _child.watch_imports()
        assert sys.meta_path.count(_child._AbsentRecorder) == 1 and sys.meta_path[-1] is _child._AbsentRecorder
        assert _child._AbsentRecorder.find_spec("anything_1147_not_there") is None
        with pytest.raises(ModuleNotFoundError):
            __import__("another_1147_not_there")
        assert "another_1147_not_there" in _child._State.absent
    finally:
        sys.meta_path[:] = saved
        _child._State.absent.clear()


# ---------------------------------------------------------------------------
# end to end: the pairs, and the programs that must not be certified
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(PAIRS))
def test_the_four_pairs_are_true_and_deterministic(tmp_path, name):
    path = program(tmp_path, PAIRS[name])
    v = verify(path, ("AssertionError",), budget_seconds=20, jobs=1)
    assert v.verdict == "true" and v.component == "deterministic", (name, v.verdict, v.component, v.reason)


@pytest.mark.parametrize("label, body, expect", [
    # natively violated through the same namespace: FALSE, the child reads what python reads
    ("builtins is not a dict", "assert isinstance(__builtins__, dict)\n", "false"),
    ("builtins is not a function", "import types\nassert not isinstance(__builtins__, types.ModuleType)\n", "false"),
    ("an absent import taken as present", "try:\n    import spec\nexcept ImportError:\n    x = 1\nassert x == 2\n", "false"),
    ("the module name is wrong", "assert type(__builtins__).__name__ == 'dict'\n", "false"),
])
def test_the_same_reads_still_find_a_native_violation(tmp_path, label, body, expect):
    assert native(tmp_path, body)[0] == 1, label                             # natively it fails
    v = verify(program(tmp_path, body), ("AssertionError",), budget_seconds=20, jobs=1)
    assert v.verdict == expect, (label, v.verdict, v.component, v.reason)


def test_a_module_that_exists_is_never_certified_as_absent(tmp_path):
    (tmp_path / "spec.py").write_text("raise ValueError('mine')\n")
    body = "try:\n    import spec\nexcept ImportError:\n    pass\nelse:\n    assert False\n"
    # natively `import spec` runs the program's own spec.py: a ValueError, not an ImportError
    assert native(tmp_path, body)[1].startswith("ValueError")
    v = verify(program(tmp_path, body), ("AssertionError",), budget_seconds=15, jobs=1)
    assert v.verdict != "true", (v.verdict, v.component, v.reason)


def test_a_program_that_writes_a_module_and_imports_it_is_not_certified(tmp_path):
    sub = tmp_path / "w"
    sub.mkdir()
    target = sub / "created_1147b.py"
    path = sub / "prog.py"
    path.write_text(textwrap.dedent(f"""
        try:
            import created_1147b
        except ImportError:
            pass
        with open({str(target)!r}, "w") as fh:
            fh.write("import time\\nVALUE = 1\\n")
        import importlib
        importlib.invalidate_caches()
        import created_1147b
        assert created_1147b.VALUE == 1
    """))
    v = verify(str(path), ("AssertionError",), budget_seconds=15, jobs=1)
    assert v.verdict != "true", (v.verdict, v.component, v.reason)


# ---------------------------------------------------------------------------
# negative controls: each rule undone, the tests above must fail
# ---------------------------------------------------------------------------

def test_the_probe_tests_catch_a_resolver_that_lets_any_use_of_builtins_through(monkeypatch):
    from tharkun.solve import pyresolve

    monkeypatch.setattr(pyresolve.Resolver, "_builtins_type_probe", lambda self, node: node.id == "__builtins__")
    with pytest.raises(AssertionError):
        test_the_main_namespace_probes_resolve_and_nothing_else_does("x = __builtins__\n", False)
    with pytest.raises(pytest.fail.Exception):                          # its pytest.raises(Top): DID NOT RAISE
        test_a_helper_module_sees_a_dict_for_builtins_so_the_probe_is_not_resolved_there(__import__("pathlib").Path(__import__("tempfile").mkdtemp()))


def test_the_absent_tests_catch_a_resolver_that_excuses_every_import(monkeypatch):
    from tharkun.solve import pyresolve

    monkeypatch.setattr(pyresolve.Resolver, "__init__", (lambda real: lambda self, *a, **k: real(self, *a, **{**k, "absent_imports": {"spec", "witness", "spec.sub", "x"}}))(pyresolve.Resolver.__init__))
    with pytest.raises(AssertionError):
        test_an_import_that_no_run_observed_absent_is_still_refused()


def test_the_created_module_test_catches_a_child_that_reports_every_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(_child, "still_absent_imports", lambda: sorted(_child._State.absent))
    _child._State.absent.clear()
    _child._State.absent.add("created_1147")
    assert _child.still_absent_imports() == ["created_1147"]            # the mutant keeps a name the real one drops once it is importable
    (tmp_path / "created_1147.py").write_text("X = 1\n")
    monkeypatch.undo()
    sys.path.insert(0, str(tmp_path))
    try:
        _child._State.absent.clear()
        _child._State.absent.add("created_1147")
        assert _child.still_absent_imports() == []
    finally:
        sys.path.remove(str(tmp_path))
        _child._State.absent.clear()
        sys.modules.pop("created_1147", None)
