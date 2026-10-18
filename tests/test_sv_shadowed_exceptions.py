"""A program that binds a property exception name itself gets no TRUE (owner 2026-10-18).

Whether an uncaught exception of a program-defined class merely *named* like the property's class
(``class IndexError(Exception)``) violates ``uncaught(IndexError)`` is unresolved for SV-COMP 2027, so a
TRUE on such a program is withheld (UNKNOWN). FALSE needs an instance of the built-in class, which is a
violation under either reading, and stays definitive.
"""
import os
import subprocess
import sys

import tharkun

import pytest

from tharkun.sv.engine import shadowed_property_names, verify

CLASSES = ("IndexError", "KeyError")

BINDINGS = [
    ("class IndexError(Exception):\n    pass\n", ("IndexError",)),
    ("IndexError = ValueError\n", ("IndexError",)),
    ("import json as KeyError\n", ("KeyError",)),
    ("from os import path as IndexError\n", ("IndexError",)),
    ("def f(KeyError):\n    pass\n", ("KeyError",)),
    ("def f(*, KeyError=1):\n    pass\n", ("KeyError",)),
    ("try:\n    pass\nexcept Exception as IndexError:\n    pass\n", ("IndexError",)),
    ("import builtins\nbuiltins.KeyError = int\n", ("KeyError",)),
    ("globals()['IndexError'] = 1\n", ("IndexError",)),
    ("setattr(__builtins__, 'KeyError', 1)\n", ("KeyError",)),
    ("for IndexError in []:\n    pass\n", ("IndexError",)),
    ("match 1:\n    case IndexError:\n        pass\n", ("IndexError",)),
    ("(KeyError := 3)\n", ("KeyError",)),
    ("KeyError = 1\ndel KeyError\n", ("KeyError",)),
    ("type IndexError = int\n", ("IndexError",)),
    ("def g[KeyError]():\n    pass\n", ("KeyError",)),
    ("class C:\n    IndexError = 1\n", ("IndexError",)),
    ("def f():\n    global KeyError\n", ("KeyError",)),
    ("class IndexError(Exception):\n    pass\nKeyError = 2\n", ("IndexError", "KeyError")),
    # a class named so dynamically
    ("class M(Exception):\n    pass\nM.__name__ = 'IndexError'\n", ("IndexError",)),
    ("class M(Exception):\n    pass\nsetattr(M, '__qualname__', 'Key' + 'Error')\n", ("KeyError",)),
    ("class M(Exception):\n    pass\nM.__name__ = name\n", ("IndexError", "KeyError")),
    ("E = type('IndexError', (Exception,), {})\n", ("IndexError",)),
    ("E = type(name, (Exception,), {})\n", ("IndexError", "KeyError")),
    ("import types\nE = types.new_class('KeyError', (Exception,))\n", ("KeyError",)),
    # namespace updates
    ("globals().update(IndexError=ValueError)\n", ("IndexError",)),
    ("vars().update({'KeyError': ValueError})\n", ("KeyError",)),
    ("globals().setdefault('IndexError', ValueError)\n", ("IndexError",)),
    # unparsable source fails closed
    ("def broken(:\n", ("IndexError", "KeyError")),
]

NOT_BINDINGS = [
    "raise IndexError()\n",
    "try:\n    [][0]\nexcept IndexError:\n    pass\n",
    "x = {}\nprint(x.get('KeyError'))\n",
    "print(IndexError.__name__, KeyError)\n",
    "class TypeError(Exception):\n    pass\n",       # not a class of this property
    "print(type(3))\n",                              # one-argument type() names nothing
    "d = {}\nd.update({'a': 1})\n",
    "def f():\n    pass\nf.__doc__ = 'IndexError'\n",
]


@pytest.mark.parametrize("source,expected", BINDINGS)
def test_every_binding_form_is_found(source, expected):
    assert shadowed_property_names(source, CLASSES) == expected


@pytest.mark.parametrize("source", NOT_BINDINGS)
def test_uses_and_other_names_are_not_bindings(source):
    assert shadowed_property_names(source, CLASSES) == ()


SHADOWED_TRUE_PROGRAMS = [
    ("assert_named", "class AssertionError(Exception):\n    pass\n\nraise AssertionError('mine')\n", ("AssertionError",)),
    ("index_named", "class IndexError(Exception):\n    pass\n\nraise IndexError()\n", ("KeyError", "IndexError")),
    ("type_named", "class TypeError(Exception):\n    pass\n\nraise TypeError()\n", ("TypeError", "AttributeError")),
    ("zerodiv_named", "class ZeroDivisionError(Exception):\n    pass\n\nraise ZeroDivisionError()\n",
     ("ZeroDivisionError", "FloatingPointError")),
    ("safe_but_shadowing", "KeyError = 1\n\ndef main():\n    return [1, 2][1]\n\nmain()\n", ("KeyError", "IndexError")),
]


def _verify(tmp_path, name, source, classes):
    path = tmp_path / (name + ".py")
    path.write_text(source)
    return path, verify(str(path), classes, python=sys.executable, budget_seconds=30, run_timeout=20,
                        canary_seconds=0, jobs=1, stages=(), witness_dir=str(tmp_path / "proofs"),
                        store_witness=False)


@pytest.mark.parametrize("name,source,classes", SHADOWED_TRUE_PROGRAMS)
def test_a_shadowed_property_name_withholds_true(tmp_path, name, source, classes):
    path, v = _verify(tmp_path, name, source, classes)
    probe = ("import builtins, runpy, sys\ntry:\n    runpy.run_path(sys.argv[1], run_name='__main__')\n"
             "except BaseException as e:\n    print(type(e).__name__, type(e) is getattr(builtins, type(e).__name__, None))\n")
    native = subprocess.run([sys.executable, "-c", probe, str(path)], text=True, capture_output=True, timeout=30)
    if native.stdout.strip():
        escaped, is_builtin = native.stdout.split()
        assert escaped in classes and is_builtin == "False"   # the program's own class, never the built-in one
    assert v.verdict == "unknown", v.line()
    assert "property exception name" in v.reason
    assert v.component == "none"
    assert set(v.detail["shadowed_property_names"]) <= set(classes)
    for word in ("TRUE", "FALSE", "UNKNOWN", "ERROR"):       # no verdict word inside the reason
        assert word not in v.reason.upper().split()
    assert not (tmp_path / "proofs").exists() or not any((tmp_path / "proofs").rglob("witness.yml"))


def test_a_builtin_violation_stays_false_despite_shadowing(tmp_path):
    source = "class IndexError(Exception):\n    pass\n\nx = []\nx[0]\n"   # the built-in IndexError escapes
    _, v = _verify(tmp_path, "shadow_and_builtin", source, ("KeyError", "IndexError"))
    assert v.verdict == "false" and v.exception == "IndexError", v.line()


@pytest.mark.parametrize("name,source", [
    ("plain_safe", "def main():\n    return [1, 2][1]\n\nmain()\n"),
    ("caught_builtin", "def main():\n    try:\n        [][0]\n    except IndexError:\n        return 0\n\nmain()\n"),
])
def test_programs_that_only_use_the_name_keep_their_true(tmp_path, name, source):
    _, v = _verify(tmp_path, name, source, ("KeyError", "IndexError"))
    assert v.verdict == "true", v.line()


def test_a_class_renamed_to_a_property_name_withholds_true(tmp_path):
    source = "class Missing(Exception):\n    pass\n\nMissing.__name__ = 'IndexError'\nMissing.__qualname__ = 'IndexError'\nraise Missing('x')\n"
    _, v = _verify(tmp_path, "renamed", source, ("KeyError", "IndexError"))
    assert v.verdict == "unknown" and v.detail["shadowed_property_names"] == ["IndexError"], v.line()


def test_local_import_files_follows_absolute_local_imports(tmp_path):
    from tharkun.sv.engine import local_import_files
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "inner.py").write_text("import leaf\n")
    (tmp_path / "leaf.py").write_text("x = 1\n")
    (tmp_path / "helper.py").write_text("from pkg import inner\n")
    main = tmp_path / "main.py"
    main.write_text("import helper\nimport json\n")
    assert sorted(local_import_files(str(main))) == sorted(str(tmp_path / f) for f in
                                                          ("helper.py", "pkg/__init__.py", "pkg/inner.py", "leaf.py"))
    (tmp_path / "leaf.py").write_text("def broken(:\n")
    assert local_import_files(str(main)) is None              # unreadable graph: fail closed


def test_an_undeclared_local_module_that_shadows_withholds_true(tmp_path):
    (tmp_path / "helper.py").write_text("IndexError = ValueError\n")
    source = "import helper\n\ndef main():\n    return [1, 2][1]\n\nmain()\n"
    _, v = _verify(tmp_path, "imports_helper", source, ("KeyError", "IndexError"))
    assert v.verdict != "true", v.line()
    if v.detail.get("shadowed_property_names") is not None:
        assert v.detail["shadowed_property_names"] == ["IndexError"]
    (tmp_path / "helper.py").write_text("SIZE = 2\n")                 # the same program without the shadow
    _, plain = _verify(tmp_path, "imports_plain_helper", source, ("KeyError", "IndexError"))
    assert plain.verdict == "true" or "shadowed_property_names" not in plain.detail, plain.line()


def test_a_declared_sibling_file_that_shadows_withholds_true(tmp_path):
    (tmp_path / "props").mkdir()
    (tmp_path / "props" / "data-lookup.prp").write_text(
        "CHECK( init(main()), ! uncaught(KeyError))\nCHECK( init(main()), ! uncaught(IndexError))\n")
    (tmp_path / "side.py").write_text("KeyError = ValueError\n")
    (tmp_path / "main.py").write_text("def main():\n    return [1, 2][1]\n\nmain()\n")
    task = tmp_path / "main.yml"
    task.write_text("format_version: '2.0'\ninput_files: 'main.py'\nrequired_files:\n  - 'side.py'\n"
                    "properties:\n  - property_file: props/data-lookup.prp\n    expected_verdict: true\n"
                    "options:\n  language: Python\n")
    v = verify(str(tmp_path / "main.py"), ("KeyError", "IndexError"), python=sys.executable, budget_seconds=30,
               run_timeout=20, canary_seconds=0, jobs=1, stages=(), task=str(task),
               witness_dir=str(tmp_path / "proofs"), store_witness=False)
    assert v.verdict == "unknown" and v.detail["shadowed_property_names"] == ["KeyError"], v.line()


def test_the_safety_net_removes_a_witness_written_for_a_withheld_true(tmp_path, monkeypatch):
    import tharkun.sv.scheduler as scheduler
    from tharkun.sv.engine import Verdict
    run_dir = tmp_path / "proofs" / "p-1"
    run_dir.mkdir(parents=True)
    witness, obligations = run_dir / "witness.yml", run_dir / "obligations.yml"
    witness.write_text("x\n")
    obligations.write_text("y\n")

    def bypassing_run(*args, **kwargs):     # a route to TRUE that did not pass done()
        return Verdict("true", component="proof:test",
                       detail={"witness_path": str(witness), "obligations_path": str(obligations)})

    monkeypatch.setattr(scheduler, "run", bypassing_run)
    path = tmp_path / "shadow.py"
    path.write_text("class KeyError(Exception):\n    pass\n")
    v = verify(str(path), ("KeyError", "IndexError"), python=sys.executable, budget_seconds=10)
    assert v.verdict == "unknown" and v.component == "none" and v.detail["withheld_component"] == "proof:test"
    assert not witness.exists() and not obligations.exists() and not run_dir.exists()


def test_the_command_line_leaves_no_correctness_witness_after_withholding(tmp_path):
    program = tmp_path / "named.py"
    program.write_text("class IndexError(Exception):\n    pass\n\nraise IndexError()\n")
    prp = tmp_path / "data-lookup.prp"
    prp.write_text("CHECK( init(main()), ! uncaught(KeyError))\nCHECK( init(main()), ! uncaught(IndexError))\n")
    proofs = tmp_path / "proofs"
    out = subprocess.run([sys.executable, "-B", "-m", "tharkun.sv", str(program), "--property", str(prp),
                          "--budget-seconds", "20", "--jobs", "1", "--no-stage", "hybrid", "--proof-dir", str(proofs)],
                         capture_output=True, text=True, timeout=180, cwd=tmp_path,
                         env=dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(tharkun.__file__))),
                                  PYTHONDONTWRITEBYTECODE="1"))
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("UNKNOWN(the program binds the property exception name IndexError"), out.stdout[-2000:]
    assert not list(proofs.rglob("witness.yml")) and not list(proofs.rglob("store-witness.yml"))
