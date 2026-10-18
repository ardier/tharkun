"""[internal ref]: a property class is the BUILTIN of that name, not a class the program defines under it.

``class IndexError(Exception): pass`` then ``raise IndexError()`` is not an ``IndexError``, and Tharkun reported FALSE for
it (component ``deterministic``) because the child recorded the exception's MRO by NAME and ``Outcome.violated`` matched by
name.  Every record now carries the names of the classes in the MRO that ARE builtin classes (``sv/_child.py::
exception_mro``: the very object, snapshotted before the program runs), in the engine's child, the stationary probe, the
in-process drive and the Atheris worker; the grader's ``eregion_sv.replay`` does the same (its own tests).

Per family (the four property families of the track): a user class with the builtin's name does NOT count; a subclass of
the builtin DOES (``class E(IndexError)``, also ``class IndexError(IndexError)``); a caught one never counts; a class
that merely claims ``__module__ = "builtins"`` is no builtin.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from tharkun.sv import _child, nondet_harness
from tharkun.sv.engine import run_once
from tharkun.sv.prove import run_probe

FAMILIES = {
    "IndexError": "data-lookup",
    "AssertionError": "assertion-safety",
    "TypeError": "dynamic-typing",
    "ZeroDivisionError": "arithmetic",
}


def program(tmp_path, body: str):
    path = tmp_path / "prog.py"
    path.write_text(textwrap.dedent(body))
    return str(path)


SHADOW = """
    class {name}(Exception):
        pass
    raise {name}()
"""
SUBCLASS = """
    class E({name}):
        pass
    raise E()
"""
SAME_NAME_SUBCLASS = """
    import builtins
    class {name}(builtins.{name}):
        pass
    raise {name}()
"""
BUILTIN = "raise {name}()\n"
SPOOF = """
    class {name}(Exception):
        __module__ = "builtins"
    raise {name}()
"""

#: (id, source, counts as the property class?)
CASES = [
    ("a user class with the builtin's name", SHADOW, False),
    ("a user class that claims __module__ == builtins", SPOOF, False),
    ("the builtin itself", BUILTIN, True),
    ("a subclass of the builtin", SUBCLASS, True),
    ("a user class with the builtin's name that subclasses the builtin", SAME_NAME_SUBCLASS, True),
]


@pytest.mark.parametrize("name", sorted(FAMILIES))
@pytest.mark.parametrize("label, source, counts", CASES, ids=[c[0] for c in CASES])
def test_the_engine_child_matches_the_builtin_not_the_name(tmp_path, name, label, source, counts):
    out = run_once(sys.executable, program(tmp_path, source.format(name=name)), [], timeout=30)
    assert out.status == "raised"
    assert out.violated((name,)) == (name if counts else None), (label, out.exc, out.mro)


@pytest.mark.parametrize("name", sorted(FAMILIES))
@pytest.mark.parametrize("label, source, counts", CASES, ids=[c[0] for c in CASES])
def test_the_in_process_drive_matches_the_builtin_not_the_name(tmp_path, name, label, source, counts):
    out = nondet_harness.drive(program(tmp_path, source.format(name=name)), [])
    assert out.status == "raised"
    assert out.violated((name,)) == (name if counts else None), (label, out.exc, out.mro)


@pytest.mark.parametrize("name", sorted(FAMILIES))
@pytest.mark.parametrize("label, source, counts", CASES, ids=[c[0] for c in CASES])
def test_the_stationary_probe_records_only_builtin_classes(tmp_path, name, label, source, counts):
    res = run_probe(sys.executable, program(tmp_path, source.format(name=name)), [], [], [], timeout=30)
    assert res["status"] == "raised"
    assert (name in res["mro"]) is counts, (label, res["exc"], res["mro"])


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_caught_exception_never_counts(tmp_path, name):
    out = run_once(sys.executable, program(tmp_path, f"""
        try:
            raise {name}()
        except {name}:
            pass
        class {name}(Exception):
            pass
        try:
            raise {name}()
        except {name}:
            pass
    """), [], timeout=30)
    assert out.status == "clean" and out.violated((name,)) is None


def test_the_recorded_mro_is_the_builtin_chain_only():
    class IndexError(Exception):          # noqa: A001 -- the shadow under test
        pass

    class E(LookupError):
        pass

    assert _child.exception_mro(IndexError()) == ["Exception", "BaseException", "object"]
    assert _child.exception_mro(E()) == ["LookupError", "Exception", "BaseException", "object"]
    assert _child.exception_mro(KeyError()) == ["KeyError", "LookupError", "Exception", "BaseException", "object"]


def test_the_snapshot_is_taken_before_the_program_can_rebind_a_builtin(tmp_path):
    out = run_once(sys.executable, program(tmp_path, """
        import builtins
        real = builtins.IndexError
        class Fake(Exception):
            pass
        builtins.IndexError = Fake
        raise real()
    """), [], timeout=30)
    assert out.violated(("IndexError",)) == "IndexError"        # the real IndexError is still the builtin object


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_end_to_end_a_user_class_with_a_property_name_is_not_a_false_verdict(tmp_path, name):
    from tharkun.sv import verify

    path = program(tmp_path, SHADOW.format(name=name))
    verdict = verify(path, (name,), budget_seconds=15, jobs=1)
    assert verdict.verdict != "false", (verdict.verdict, verdict.reason)
    real = verify(program(tmp_path, BUILTIN.format(name=name)), (name,), budget_seconds=15, jobs=1)
    assert real.verdict == "false"


@pytest.mark.parametrize("bases", ["AssertionError, ValueError", "ValueError, AssertionError",
                                   "Parent, ValueError", "ValueError, Parent"])
def test_multiple_exception_bases_cannot_drop_a_property_class(tmp_path, bases):
    import subprocess

    from tharkun.solve.absint import prove
    from tharkun.sv import verify

    source = f"class Parent(AssertionError): pass\nclass E({bases}): pass\nraise E()\n"
    path = program(tmp_path, source)
    native = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
    assert native.returncode == 1 and native.stderr.rstrip().endswith("E")
    proof = prove(source, ("AssertionError",), filename=path)
    assert proof.verdict == "UNKNOWN" and "multiple inheritance" in proof.reason
    out = run_once(sys.executable, path, [], timeout=30)
    assert out.violated(("AssertionError",)) == "AssertionError"
    verdict = verify(path, ("AssertionError",), budget_seconds=10, jobs=1, canary_seconds=0)
    assert verdict.verdict == "false", (verdict.verdict, verdict.reason)


def test_single_exception_inheritance_keeps_its_proof():
    from tharkun.solve.absint import prove

    source = "class E(ValueError): pass\nraise E()\n"
    assert prove(source, ("AssertionError",)).verdict == "TRUE"
