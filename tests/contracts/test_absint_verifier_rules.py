"""Scoped evidence for the official SV verifier's nondeterministic containers."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from tests.contracts._native import peer_runtime_check_type, run_native
from tests.fixtures import sv_verifier_nondet
from tests.fixtures.sv_check_type_probes import check_type_probe_calls
from tests.contracts.test_engine_audit_consumed import untested
from tharkun.solve.absint import prove
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.safety import Binding, cases
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES
from tharkun.solve.terms import Refused

ALL = tuple(ALL_PROPERTY_CLASSES)
_CHECK_TYPE_OUTCOMES = Path(__file__).resolve().parents[1] / "fixtures" / "sv_check_type_mr1793_outcomes.json"


def _official_stub():
    return sv_verifier_nondet


def _recorded_check_type_outcomes():
    data = json.loads(_CHECK_TYPE_OUTCOMES.read_text())
    assert data["upstream"] == "SV-Benchmarks python/_sv_verifier.py (MR1793)::check_type"
    return data["outcomes"]


def _outcome_cell(call):
    try:
        return {"return": call()}
    except BaseException as exc:  # noqa: BLE001 -- outcome fixture records exception classes
        return {"exception": type(exc).__name__}


def _native_outcome(call):
    try:
        call()
    except BaseException as exc:  # the contract covers the seven property exceptions
        return type(exc).__name__
    return ""


@pytest.fixture(scope="module")
def _peer_check_type():
    return peer_runtime_check_type()


def _one_iteration(stub):
    draws = iter((True, False))
    stub.nondet_bool = lambda: next(draws)


@pytest.mark.parametrize("exception", ALL)
@pytest.mark.parametrize("where", ("list element", "dict key callback", "dict value callback", "dict key hash"))
def test_official_stub_propagates_callback_and_hash_property_exceptions(exception, where):
    stub = _official_stub()
    _one_iteration(stub)
    error = getattr(builtins, exception)

    def explode():
        raise error("official stub callback")

    class BadHash:
        def __hash__(self):
            raise error("official stub key hash")

    if where == "list element":
        outcome = _native_outcome(lambda: stub.nondet_list(explode))
    elif where == "dict key callback":
        outcome = _native_outcome(lambda: stub.nondet_dict(explode, lambda: 1))
    elif where == "dict value callback":
        outcome = _native_outcome(lambda: stub.nondet_dict(lambda: 1, explode))
    else:
        outcome = _native_outcome(lambda: stub.nondet_dict(lambda: BadHash(), lambda: 1))
    assert outcome == exception, (where, exception, outcome)


@pytest.mark.parametrize("exception", ALL)
def test_official_dict_stub_propagates_equal_key_comparison_exceptions(exception):
    stub = _official_stub()
    draws = iter((True, True, False))
    stub.nondet_bool = lambda: next(draws)
    error = getattr(builtins, exception)

    class BadEqual:
        def __hash__(self):
            return 1

        def __eq__(self, other):
            raise error("official stub key equality")

    keys = iter((BadEqual(), BadEqual()))
    outcome = _native_outcome(lambda: stub.nondet_dict(lambda: next(keys), lambda: 1))
    assert outcome == exception, outcome


def test_official_stub_independent_draws_empty_case_and_equal_key_overwrite():
    stub = _official_stub()
    draws = iter((True, True, False))
    stub.nondet_bool = lambda: next(draws)
    values = iter((3, 4))
    assert stub.nondet_list(lambda: next(values)) == [3, 4]

    draws = iter((True, True, False))
    stub.nondet_bool = lambda: next(draws)
    keys = iter((4, 4))
    values = iter((10, 20))
    assert stub.nondet_dict(lambda: next(keys), lambda: next(values)) == {4: 20}

    draws = iter((True, True, True, False))
    stub.nondet_bool = lambda: next(draws)
    keys = iter((1, True, 1.0))
    values = iter((10, 20, 30))
    aliased = stub.nondet_dict(lambda: next(keys), lambda: next(values))
    assert aliased == {1: 30} and len(aliased) == 1

    draws = iter((False,))
    stub.nondet_bool = lambda: next(draws)
    assert stub.nondet_list(lambda: pytest.fail("empty draw called its callback")) == []


@pytest.mark.parametrize(
    "qualname,call",
    [
        ("_sv_verifier.nondet_list", "xs = sv.nondet_list(sv.nondet_int)"),
        ("_sv_verifier.nondet_list", "xs = sv.nondet_list(nondet_elem=sv.nondet_bool)"),
        ("_sv_verifier.nondet_dict", "d = sv.nondet_dict(sv.nondet_int, sv.nondet_float)"),
        ("_sv_verifier.nondet_dict", "d = sv.nondet_dict(nondet_key=sv.nondet_str, nondet_value=sv.nondet_bool)"),
    ],
)
def test_exact_stub_scalar_callbacks_are_admitted_and_all_property_checks_are_proven_safe(qualname, call):
    src = "import _sv_verifier as sv\n" + call + "\n"
    result = prove(src, ALL, time_budget=60.0)
    assert result.verdict == "TRUE", result.explain()
    rows = [site for site in result.sites if site[2].startswith(qualname + ":")]
    assert {site[2] for site in rows} == {f"{qualname}:{exc}" for exc in ALL}, rows
    assert all("UNPROVEN" not in site[4] for site in rows), rows


@pytest.mark.parametrize(
    "qualname,call,reason",
    [
        ("_sv_verifier.nondet_list", "xs = sv.nondet_list(user_callback)", "exact verifier scalar callback"),
        ("_sv_verifier.nondet_list", "xs = sv.nondet_list(lambda: 1)", "exact verifier scalar callback"),
        ("_sv_verifier.nondet_dict", "d = sv.nondet_dict(user_callback, sv.nondet_int)", "exact verifier scalar callback"),
        ("_sv_verifier.nondet_dict", "d = sv.nondet_dict(sv.nondet_int, sv.nondet_list)", "unsupported verifier generator: nondet_list"),
    ],
)
def test_nonstub_or_nested_callbacks_are_named_refusals(qualname, call, reason):
    src = "import _sv_verifier as sv\ndef user_callback(): return 1\n" + call + "\n"
    result = prove(src, ALL, time_budget=60.0)
    assert result.verdict == "UNKNOWN", result.explain()
    assert reason in result.reason, result.reason
    assert not any(site[2].startswith(qualname + ":") for site in result.sites)


def test_rebound_callback_is_not_treated_as_the_original_stub_scalar():
    src = (
        "import _sv_verifier as sv\n"
        "def user_callback(): return 1\n"
        "draw = sv.nondet_int\n"
        "draw = user_callback\n"
        "xs = sv.nondet_list(draw)\n"
    )
    result = prove(src, ALL, time_budget=60.0)
    assert result.verdict == "UNKNOWN" and "exact verifier scalar callback" in result.reason


def test_supported_mixed_exact_scalar_callbacks_prove_the_call_safe():
    src = (
        "import _sv_verifier as sv\n"
        "draw = sv.nondet_int\n"
        "if sv.nondet_bool(): draw = sv.nondet_bool\n"
        "xs = sv.nondet_list(draw)\n"
    )
    result = prove(src, ALL, time_budget=60.0)
    assert result.verdict == "TRUE", result.explain()
    assert any(site[2] == "_sv_verifier.nondet_list:TypeError" for site in result.sites)


@pytest.mark.parametrize(
    "src",
    [
        "import _sv_verifier as sv\nxs = sv.nondet_list(sv.nondet_int)\nassert len(xs) == 0\n",
        "import _sv_verifier as sv\nd = sv.nondet_dict(sv.nondet_int, sv.nondet_int)\nassert len(d) == 0\n",
        "import _sv_verifier as sv\nd = sv.nondet_dict(sv.nondet_int, sv.nondet_int)\nassert 0 not in d\n",
        "import _sv_verifier as sv\nd = sv.nondet_dict(sv.nondet_int, sv.nondet_int)\nx = d[0]\n",
        "import _sv_verifier as sv\na = sv.nondet_list(sv.nondet_int)\nb = sv.nondet_list(sv.nondet_int)\nassert a is b\n",
        "import _sv_verifier as sv\nxs = sv.nondet_list(sv.nondet_int)\nalias = xs\nalias.append(7)\nassert len(xs) == 0\n",
    ],
)
def test_unknown_container_draws_do_not_create_false_definite_facts(src):
    assert prove(src, ("AssertionError", "KeyError"), time_budget=60.0).verdict != "TRUE"


@pytest.mark.parametrize("qualname", ("_sv_verifier.nondet_list", "_sv_verifier.nondet_dict"))
def test_contract_withdrawal_and_untested_contract_disable_the_transfer(qualname, monkeypatch):
    load_all()
    src = (
        "import _sv_verifier as sv\n"
        + ("value = sv.nondet_list(sv.nondet_int)\n" if qualname.endswith("list")
           else "value = sv.nondet_dict(sv.nondet_int, sv.nondet_int)\n")
    )
    contract = REGISTRY[qualname]
    monkeypatch.delitem(REGISTRY, qualname)
    withdrawn = prove(src, ALL, time_budget=60.0)
    assert withdrawn.verdict == "UNKNOWN" and "no tested contract" in withdrawn.reason
    monkeypatch.setitem(REGISTRY, qualname, untested(contract))
    untested_result = prove(src, ALL, time_budget=60.0)
    assert untested_result.verdict == "UNKNOWN" and "no tested contract" in untested_result.reason


def test_alias_mutation_remains_visible_and_valid_call_shapes_are_native_clean():
    source = (
        "import _sv_verifier as sv\n"
        "xs = sv.nondet_list(sv.nondet_int)\n"
        "alias = xs\n"
        "alias.append(7)\n"
        "assert xs is alias and len(xs) == len(alias)\n"
    )
    assert run_native(source) == ""
    # The unknown list length keeps the assertion-safety result conservative,
    # while valid append binding itself cannot raise TypeError.
    assert prove(source, ("TypeError",), time_budget=60.0).verdict == "TRUE"


def test_repeated_loop_allocation_is_fresh_and_each_draw_has_nonnegative_length():
    source = (
        "import _sv_verifier as sv\n"
        "items = []\n"
        "for _ in range(2):\n"
        "    current = sv.nondet_list(sv.nondet_int)\n"
        "    assert len(current) >= 0\n"
        "    items.append(current)\n"
        "assert items[0] is items[1]\n"
    )
    # The final identity assertion fails for every native draw; the per-iteration
    # length property is true for every draw and exercises the admitted loop call.
    assert run_native(source) == "AssertionError"
    assert prove(source, ("AssertionError",), time_budget=60.0).verdict != "TRUE"


def test_supported_list_draw_in_loop_proves_its_universal_length_bound():
    source = (
        "import _sv_verifier as sv\n"
        "for _ in range(2):\n"
        "    current = sv.nondet_list(sv.nondet_int)\n"
        "    assert len(current) >= 0\n"
    )
    assert run_native(source) == ""
    result = prove(source, ("AssertionError",), time_budget=60.0)
    assert result.verdict == "TRUE", result.explain()


@pytest.mark.parametrize(
    "call,expected",
    [
        ("sv.nondet_list()", "TypeError"),
        ("sv.nondet_list(sv.nondet_int, sv.nondet_bool)", "TypeError"),
        ("sv.nondet_dict()", "TypeError"),
        ("sv.nondet_dict(sv.nondet_int)", "TypeError"),
    ],
)
def test_invalid_official_stub_argument_shapes_raise_typeerror(call, expected):
    stub = _official_stub()
    source = f"value = {call}\n"
    assert _native_outcome(lambda: exec(source, {"sv": stub})) == expected
    assert run_native("import _sv_verifier as sv\n" + source) == expected
    proof_source = "import _sv_verifier as sv\n" + source
    result = prove(proof_source, ("TypeError",), time_budget=60.0)
    assert result.verdict != "TRUE", result.explain()


@pytest.mark.parametrize("qualname", ("_sv_verifier.nondet_list", "_sv_verifier.nondet_dict"))
@pytest.mark.parametrize("exception", ALL)
def test_registered_safety_cases_name_the_real_m1_gap(qualname, exception):
    refusal = cases(qualname, exception, Binding(None, (), None))
    assert isinstance(refusal, Refused)
    assert "M1 terms cannot represent a fresh nondeterministic container" in refusal.reason


@pytest.mark.parametrize(
    "source",
    [
        "import _sv_verifier as sv\nsv.check_type(1, int)\n",
        "import _sv_verifier as sv\nsv.check_type(True, int)\n",
        "from _sv_verifier import check_type as check\nhint = bool\ncheck(True, hint)\n",
        "import _sv_verifier as sv\nsv.check_type(1.5, float)\n",
        "import _sv_verifier as sv\nsv.check_type('x', str)\n",
        "import _sv_verifier as sv\nsv.check_type([], list)\n",
        "import _sv_verifier as sv\nsv.check_type({}, dict)\n",
        "from _sv_verifier import check_type\ncheck_type(value='x', hint=str)\n",
    ],
)
def test_exact_builtin_check_type_calls_are_proven_safe(source):
    result = prove(source, ALL, time_budget=60.0)
    assert result.verdict == "TRUE", result.explain()
    assert any(site[2] == "_sv_verifier.check_type:TypeError" for site in result.sites)


@pytest.mark.parametrize(
    "label",
    (
        "match:int",
        "match:bool-as-int",
        "match:bool",
        "match:float",
        "match:str",
        "match:list",
        "match:dict",
    ),
)
def test_exact_builtin_check_type_matches_have_no_return_or_exception(label, _peer_check_type):
    expected = _recorded_check_type_outcomes()[label]
    assert expected == {"return": None}
    call = check_type_probe_calls()[label]
    assert _outcome_cell(lambda: call(_peer_check_type)) == expected


def test_peer_check_type_matches_recorded_upstream_cells(_peer_check_type):
    outcomes = _recorded_check_type_outcomes()
    calls = check_type_probe_calls()
    assert set(outcomes) == set(calls) and len(calls) == 27
    for label, call in calls.items():
        assert _outcome_cell(lambda: call(_peer_check_type)) == outcomes[label], label


def test_exact_builtin_check_type_mismatch_is_a_proven_typeerror(_peer_check_type):
    source = "import _sv_verifier as sv\nsv.check_type(1, str)\n"
    expected = _recorded_check_type_outcomes()["mismatch:int-as-str"]
    assert expected == {"exception": "TypeError"}
    call = check_type_probe_calls()["mismatch:int-as-str"]
    assert _outcome_cell(lambda: call(_peer_check_type)) == expected
    result = prove(source, ("TypeError",), time_budget=60.0)
    assert result.verdict != "TRUE", result.explain()


@pytest.mark.parametrize(
    "label,source,invoke",
    [
        (
            "binding:extra-positional",
            "import _sv_verifier as sv\nsv.check_type(1, int, str)\n",
            lambda check: check(1, int, str),
        ),
        (
            "binding:duplicate-value",
            "import _sv_verifier as sv\nsv.check_type(1, int, value=2)\n",
            lambda check: check(1, int, value=2),
        ),
        (
            "binding:missing-value",
            "import _sv_verifier as sv\nsv.check_type(hint=int)\n",
            lambda check: check(hint=int),
        ),
    ],
)
def test_check_type_invalid_argument_binding_is_native_typeerror(label, source, invoke, _peer_check_type):
    expected = _recorded_check_type_outcomes()[label]
    assert expected == {"exception": "TypeError"}
    assert _outcome_cell(lambda: invoke(_peer_check_type)) == expected
    result = prove(source, ("TypeError",), time_budget=60.0)
    assert result.verdict != "TRUE", result.explain()


@pytest.mark.parametrize("exception", ALL)
@pytest.mark.parametrize("hook", ("metaclass", "class attribute"))
def test_check_type_global_contract_propagates_user_hook_exceptions(exception, hook, _peer_check_type):
    label = f"hook:{'metaclass' if hook == 'metaclass' else 'class-attribute'}:{exception}"
    expected = _recorded_check_type_outcomes()[label]
    expected_class = "TypeError" if hook == "class attribute" and exception == "AttributeError" else exception
    assert expected == {"exception": expected_class}
    call = check_type_probe_calls()[label]
    assert _outcome_cell(lambda: call(_peer_check_type)) == expected


def test_tuple_mismatch_repr_callback_is_not_assumed_safe(_peer_check_type):
    label = "repr:tuple-mismatch"
    expected = _recorded_check_type_outcomes()[label]
    assert expected == {"exception": "AssertionError"}
    call = check_type_probe_calls()[label]
    assert _outcome_cell(lambda: call(_peer_check_type)) == expected
    source = (
        "import _sv_verifier as sv\n"
        "class BadRepr:\n"
        "    def __repr__(self): raise AssertionError('tuple mismatch diagnostic repr')\n"
        "sv.check_type((BadRepr(),), int)\n"
    )
    result = prove(source, ("AssertionError",), time_budget=60.0)
    assert result.verdict == "UNKNOWN", result.explain()
    assert "custom or unsupported object" in result.reason


def test_container_mismatch_repr_callback_is_not_assumed_safe(_peer_check_type):
    label = "repr:list-mismatch"
    expected = _recorded_check_type_outcomes()[label]
    assert expected == {"exception": "AssertionError"}
    call = check_type_probe_calls()[label]
    assert _outcome_cell(lambda: call(_peer_check_type)) == expected
    source = (
        "import _sv_verifier as sv\n"
        "class BadRepr:\n"
        "    def __repr__(self): raise AssertionError('mismatch diagnostic repr')\n"
        "sv.check_type([BadRepr()], str)\n"
    )
    result = prove(source, ("AssertionError",), time_budget=60.0)
    assert result.verdict == "UNKNOWN", result.explain()
    assert "nested container __repr__" in result.reason


@pytest.mark.parametrize(
    "source,reason",
    [
        ("import _sv_verifier as sv\nsv.check_type([], complex)\n", "exact builtin class"),
        ("import _sv_verifier as sv\nsv.check_type([], (int, str))\n", "exact builtin class"),
        (
            "import _sv_verifier as sv\nclass C: pass\nsv.check_type(C(), int)\n",
            "custom or unsupported object",
        ),
        (
            "import _sv_verifier as sv\nvalue = 1 if sv.nondet_bool() else 'x'\nsv.check_type(value, int)\n",
            "mixed or unsupported possible types",
        ),
    ],
)
def test_check_type_outside_exact_builtin_subset_is_named_unknown(source, reason):
    result = prove(source, ALL, time_budget=60.0)
    assert result.verdict == "UNKNOWN", result.explain()
    assert reason in result.reason, result.reason


@pytest.mark.parametrize("exception", ALL)
def test_check_type_safety_template_names_hint_identity_gap(exception):
    refusal = cases("_sv_verifier.check_type", exception, Binding(None, (), None))
    assert isinstance(refusal, Refused)
    assert "M1 terms cannot express exact builtin class identity" in refusal.reason


def test_check_type_contract_withdrawal_disables_the_absint_transfer(monkeypatch):
    load_all()
    source = "import _sv_verifier as sv\nsv.check_type(1, int)\n"
    contract = REGISTRY["_sv_verifier.check_type"]
    monkeypatch.delitem(REGISTRY, "_sv_verifier.check_type")
    withdrawn = prove(source, ALL, time_budget=60.0)
    assert withdrawn.verdict == "UNKNOWN" and "no tested contract" in withdrawn.reason
    monkeypatch.setitem(REGISTRY, "_sv_verifier.check_type", untested(contract))
    untested_result = prove(source, ALL, time_budget=60.0)
    assert untested_result.verdict == "UNKNOWN" and "no tested contract" in untested_result.reason
