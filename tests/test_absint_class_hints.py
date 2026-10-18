"""``check_type(value, C)`` for a class ``C`` of the program (docs/ABSINT.md "check_type with a class of the program").

The benchmark's runtime check is ``isinstance(value, C)``.  The programs under ``tests/fixtures/absint/class_hint`` are
run natively over EVERY boolean draw vector (``isinstance`` is the check here), and a TRUE for a property class is a
failure whenever some vector makes an exception of that class escape.
"""
from __future__ import annotations

import contextlib
import io
import pathlib
import sys
import types

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.environment import TaskEnvironment

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "absint" / "class_hint"
PROGRAMS = sorted(p.stem for p in FIXTURES.glob("cu*.py"))
CLASSES = {
    "dynamic-typing": ("TypeError", "AttributeError"),
    "arithmetic": ("ZeroDivisionError", "FloatingPointError"),
    "data-lookup": ("IndexError", "KeyError"),
    "assertion-safety": ("AssertionError",),
}


def _stub(vector, used):
    mod = types.ModuleType("_sv_verifier")

    def nondet_bool():
        i = len(used)
        used.append(True)
        return vector[i] if i < len(vector) else False

    def check_type(value, hint):
        if not isinstance(value, hint):
            raise TypeError(f"expected {hint!r}, got {value!r}")

    mod.nondet_bool, mod.check_type = nondet_bool, check_type
    return mod


def native_escapes(source: str) -> set:
    escapes, seen, todo = set(), set(), [()]
    while todo:
        prefix = todo.pop()
        if prefix in seen:
            continue
        seen.add(prefix)
        used: list = []
        saved = sys.modules.get("_sv_verifier")
        sys.modules["_sv_verifier"] = _stub(list(prefix), used)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, "<class_hint>", "exec"), {"__name__": "__main__"})
        except BaseException as exc:  # noqa: BLE001 -- how the run ended is the point
            escapes.add(type(exc).__name__)
        finally:
            if saved is None:
                sys.modules.pop("_sv_verifier", None)
            else:
                sys.modules["_sv_verifier"] = saved
        for j in range(len(prefix), min(len(used), 12)):
            todo.append(prefix + (False,) * (j - len(prefix)) + (True,))
    return escapes


def _prove(name, classes, **kw):
    path = FIXTURES / f"{name}.py"
    return prove(path.read_text(), classes, filename=str(path), time_budget=30.0,
                 environment=TaskEnvironment((str(path),)), **kw)


@pytest.mark.parametrize("name", PROGRAMS)
def test_never_true_where_a_run_escapes(name):
    escapes = native_escapes((FIXTURES / f"{name}.py").read_text())
    for prop, classes in CLASSES.items():
        for mode in (None, False):
            result = _prove(name, classes, split_draws=mode)
            if set(classes) & escapes:
                assert result.verdict != "TRUE", (name, prop, mode, escapes, result.explain())


@pytest.mark.parametrize("name", ["cu01_instance_of_own_class", "cu02_subclass_instance_base_hint",
                                  "cu15_exception_class_hint", "cu16_list_of_instances_hint_list",
                                  "cu21_instance_after_mutation_ok"])
def test_an_instance_of_the_class_or_a_subclass_is_proven(name):
    assert native_escapes((FIXTURES / f"{name}.py").read_text()) == set()
    for prop, classes in CLASSES.items():
        result = _prove(name, classes)
        assert result.verdict == "TRUE", (name, prop, result.explain())


def test_each_outcome_of_a_boolean_draw_is_an_instance_of_the_common_base():
    name = "cu04_flag_selects_subclass_ok"
    assert native_escapes((FIXTURES / f"{name}.py").read_text()) == set()
    for prop, classes in CLASSES.items():
        assert _prove(name, classes, split_draws=False).verdict != "TRUE", prop     # joined: a mixed result
        assert _prove(name, classes).verdict == "TRUE", prop


@pytest.mark.parametrize("name", ["cu03_base_instance_subclass_hint", "cu06_scalar_for_class_hint",
                                  "cu18_none_for_class_hint", "cu19_class_object_not_instance",
                                  "cu20_function_for_class_hint", "cu22_unrelated_classes_same_shape"])
def test_a_certain_mismatch_is_never_a_proof_of_dynamic_typing(name):
    assert "TypeError" in native_escapes((FIXTURES / f"{name}.py").read_text())
    assert _prove(name, CLASSES["dynamic-typing"]).verdict != "TRUE"
    assert _prove(name, CLASSES["arithmetic"]).verdict in ("TRUE", "UNKNOWN")


@pytest.mark.parametrize("name", ["cu08_dunder_class_property", "cu09_metaclass_instancecheck_false",
                                  "cu10_metaclass_instancecheck_true", "cu12_class_reassigned_attr",
                                  "cu13_getattribute_class"])
def test_a_class_whose_instance_check_is_not_the_type_chain_is_refused(name):
    """A ``__class__`` of its own, a metaclass with ``__instancecheck__``, a reassigned ``__class__``: the chain walk is
    not the check, so no proof is claimed for TypeError where the native run raises one."""
    escapes = native_escapes((FIXTURES / f"{name}.py").read_text())
    result = _prove(name, CLASSES["dynamic-typing"])
    if "TypeError" in escapes:
        assert result.verdict != "TRUE", result.explain()


def test_a_mismatch_of_an_object_is_refused_because_its_repr_is_formatted():
    name = "cu11_repr_raises_on_mismatch"
    assert "KeyError" in native_escapes((FIXTURES / f"{name}.py").read_text())
    for prop in ("dynamic-typing", "data-lookup"):
        result = _prove(name, CLASSES[prop])
        assert result.verdict != "TRUE", (prop, result.explain())
