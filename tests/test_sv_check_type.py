"""Independent runtime model, pinned to recorded MR1793 behavior, never skipped."""
import json
from pathlib import Path
import sys
from typing import Literal

import pytest

from tests.fixtures.sv_check_type_cases import CASES
from tharkun.solve.census import site_census
from tharkun.solve.determinism import determinism_census
from tharkun.solve.drawflow import plan_draws
from tharkun.sv import _child, engine, product_proof, prove


FIXTURES = Path(__file__).parent / "fixtures"
OBSERVED = json.loads((FIXTURES / "sv_check_type_outcomes.json").read_text())
EXPECTED = {row["case"]: row["outcome"] for row in OBSERVED}


def test_recorded_table_has_exact_case_membership():
    labels = [case[0] for case in CASES]
    assert len(labels) == len(set(labels)) == len(OBSERVED)
    assert labels == [row["case"] for row in OBSERVED]


@pytest.mark.parametrize("label,value,hint", CASES, ids=[case[0] for case in CASES])
def test_recorded_upstream_outcome_without_model_error(monkeypatch, label, value, hint):
    monkeypatch.setattr(_child._State, "flags", set())
    try:
        result = _child._make_sv_verifier().check_type(value, hint)
    except BaseException as error:
        actual = {"exception": type(error).__name__, "message": str(error)}
    else:
        actual = {"return": result}
        assert result is None
    assert actual == EXPECTED[label]
    assert not _child._State.flags


def test_literal_preserves_both_truth_tests_of_a_false_equality_result(monkeypatch):
    monkeypatch.setattr(_child._State, "flags", set())
    events = []

    class Truth:
        def __bool__(self):
            events.append("bool")
            return events.count("bool") == 2

    class Value:
        def __eq__(self, other):
            events.append(("eq", other))
            return Truth()

    assert _child._make_sv_verifier().check_type(Value(), Literal[1]) is None
    assert events == [("eq", 1), "bool", "bool"]
    assert not _child._State.flags


def test_element_callback_stopiteration_keeps_generator_exception_semantics(monkeypatch):
    monkeypatch.setattr(_child._State, "flags", set())

    class Meta(type):
        def __instancecheck__(cls, value):
            raise StopIteration("callback stop")

    class Hint(metaclass=Meta):
        pass

    with pytest.raises(RuntimeError, match="^generator raised StopIteration$"):
        _child._make_sv_verifier().check_type([1], list[Hint])
    assert not _child._State.flags


def test_iterators_stop_at_the_first_bad_element(monkeypatch):
    from collections.abc import Iterable

    monkeypatch.setattr(_child._State, "flags", set())
    values = iter([1, "bad", 3])
    with pytest.raises(TypeError):
        _child._make_sv_verifier().check_type(values, Iterable[int])
    assert list(values) == [3]
    assert not _child._State.flags


@pytest.mark.parametrize("value,hint", [([], list[int, str]), ({}, dict[int])])
def test_malformed_arity_is_checked_even_for_empty_containers(monkeypatch, value, hint):
    monkeypatch.setattr(_child._State, "flags", set())
    with pytest.raises(ValueError):
        _child._make_sv_verifier().check_type(value, hint)
    assert not _child._State.flags


SPARSE = ("import _sv_verifier\n"
          "n = _sv_verifier.nondet_int()\n"
          "_sv_verifier.check_type('bad' if n == 1000003 else 1, int)\n")
LOOP = ("import _sv_verifier\n"
        "for i in range(abs(_sv_verifier.nondet_int())):\n"
        "    _sv_verifier.check_type('bad' if i == 1000003 else 1, int)\n")


def test_primitive_check_type_is_deterministic_but_typeerror_stays_unmodeled():
    assert not site_census(SPARSE, ("TypeError",)).no_site
    assert determinism_census(SPARSE).ok
    address_message = ("import _sv_verifier\n"
                       "try:\n    _sv_verifier.check_type(object(), int)\n"
                       "except TypeError as error:\n"
                       "    assert 'a' not in error.args[0]\n")
    assert not determinism_census(address_message).ok


def test_product_proof_does_not_ignore_a_check_type_call(tmp_path):
    source = SPARSE + "assert True\n"
    path = tmp_path / "product.py"
    path.write_text(source)
    assert product_proof.prove(source, ("AssertionError",), program=str(path), python=sys.executable) is None


def test_drawflow_and_stationary_refuse_an_unmodeled_check_type(tmp_path, monkeypatch):
    assert not plan_draws(LOOP).ok
    path = tmp_path / "loop.py"
    path.write_text(LOOP)
    monkeypatch.setattr(prove, "_stationary", lambda *a, **k: pytest.fail("unmodeled call cannot start a probe"))
    assert prove.prove_dynamic(str(path), ("TypeError",), python=sys.executable,
                               remaining=lambda: 10, jobs=1) is None


def test_sparse_type_error_never_becomes_true_and_replays_without_model_error(tmp_path):
    path = tmp_path / "sparse.py"
    path.write_text(SPARSE)
    verdict = engine.verify(str(path), ("TypeError",), budget_seconds=120, max_runs=1,
                            canary_seconds=0, stages=[], concolic=False, jobs=1)
    assert verdict.verdict != "true"
    confirmed, out = engine.replay(str(path), {"vector": [{"kind": "int", "value": 1000003}]},
                                   ("TypeError",), timeout=10)
    assert confirmed and out.exc == "TypeError" and "model_error" not in out.flags


@pytest.mark.parametrize("name,vector,exception", [
    ("args_assigned_call", [{"kind": "bool", "value": False}], None),
    ("args_assigned_call", [{"kind": "bool", "value": True}], None),
    ("args_call", [{"kind": "int", "value": 7}, {"kind": "bool", "value": False}], None),
    ("args_call", [{"kind": "int", "value": 7}, {"kind": "bool", "value": True}], None),
    ("args_assigned_call_unsafe", [{"kind": "bool", "value": False}], "AttributeError"),
    ("args_assigned_call_unsafe", [{"kind": "bool", "value": True}], None),
    ("args_nested_call_unsafe", [{"kind": "int", "value": 0}, {"kind": "bool", "value": False}], "TypeError"),
    ("args_nested_call_unsafe", [{"kind": "int", "value": 0}, {"kind": "bool", "value": True}], None),
])
def test_unchanged_pending_fixtures_have_expected_native_behavior(name, vector, exception):
    path = FIXTURES / "sv_missing_verifier" / (name + ".py")
    out = engine.run_once(sys.executable, str(path), vector, timeout=10)
    assert out.status == ("clean" if exception is None else "raised")
    assert out.exc == exception and "model_error" not in out.flags
    assert engine._replayed(vector, out)


@pytest.mark.parametrize("name,unsafe", [("args_assigned_call", False), ("args_call", False),
                                      ("args_assigned_call_unsafe", True), ("args_nested_call_unsafe", True)])
def test_pending_full_pipeline_restores_genuine_false_without_false_on_safe(name, unsafe):
    path = FIXTURES / "sv_missing_verifier" / (name + ".py")
    verdict = engine.verify(str(path), ("TypeError", "AttributeError"), budget_seconds=120,
                            max_runs=1, canary_seconds=0, stages=[], concolic=False, jobs=1)
    # Safe TRUE requires the peer abstract-interpreter model; UNKNOWN stays honest.
    assert (verdict.verdict == "false") if unsafe else (verdict.verdict != "false")
