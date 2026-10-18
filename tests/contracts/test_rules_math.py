"""The ``math`` contracts as rules ([internal ref], [internal ref]): every function and constant is a registered rule of the right kind,
its native semantics are CPython's, and its call shape is the one CPython accepts."""

from __future__ import annotations

import math

import pytest

from tests.contracts.test_math_module import VALID
from tharkun.solve.contracts import all_contracts, get
from tharkun.solve.contracts.rules import all_rules, get_rule
from tharkun.solve.contracts.rules_math import MathConstant, MathFunction

CONSTANTS = ("pi", "e", "tau", "inf", "nan")
FUNCTIONS = sorted(VALID)


def test_every_math_contract_is_a_rule_and_every_math_rule_a_contract():
    contracts = {c.qualname for c in all_contracts() if c.qualname.startswith("math.")}
    rules = {r.qualname for r in all_rules() if r.qualname.startswith("math.")}
    assert contracts == rules == {f"math.{n}" for n in (*FUNCTIONS, *CONSTANTS)}
    assert len(FUNCTIONS) == 57


@pytest.mark.parametrize("name", FUNCTIONS)
def test_a_function_is_a_math_function_rule_holding_the_registry_contract(name):
    rule = get_rule(f"math.{name}")
    assert isinstance(rule, MathFunction) and rule.kind == "math_function"
    assert rule.contract is get(f"math.{name}") and rule.tested


@pytest.mark.parametrize("name", CONSTANTS)
def test_a_constant_is_a_math_constant_rule(name):
    rule = get_rule(f"math.{name}")
    assert isinstance(rule, MathConstant) and rule.kind == "math_constant"
    assert rule.contract is get(f"math.{name}")


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b and type(a) is type(b)


@pytest.mark.parametrize("name", FUNCTIONS)
def test_native_is_cpythons_own_function(name):
    assert _same(get_rule(f"math.{name}").native(*VALID[name]), getattr(math, name)(*VALID[name]))


@pytest.mark.parametrize("name", CONSTANTS)
def test_a_constants_native_is_its_value_and_a_call_is_refused(name):
    rule = get_rule(f"math.{name}")
    assert _same(rule.native(), getattr(math, name))
    with pytest.raises(TypeError):
        rule.native(1.0)


def _args(name, n):
    """``n`` arguments of the kinds ``name`` takes: the valid ones, cut or extended with the first."""
    valid = list(VALID[name])
    return (valid + [valid[0]] * n)[:n]


def _cpython_accepts(name, n):
    try:
        getattr(math, name)(*_args(name, n))
    except TypeError:
        return False
    except (ValueError, OverflowError):
        return True       # the call was accepted and then failed on its values: the shape is fine
    return True


@pytest.mark.parametrize("name", FUNCTIONS)
def test_the_shape_is_the_one_cpython_accepts(name):
    rule = get_rule(f"math.{name}")
    for n in range(0, 6):
        assert rule.applicable(n, False) == _cpython_accepts(name, n), (name, n)


@pytest.mark.parametrize("name", ["isclose", "nextafter", "prod"])
def test_the_functions_with_keyword_only_options_allow_keywords(name):
    assert get_rule(f"math.{name}").shape.keywords_ok


@pytest.mark.parametrize("name", [n for n in FUNCTIONS if n not in ("isclose", "nextafter", "prod")])
def test_every_other_function_takes_no_keyword(name):
    assert not get_rule(f"math.{name}").applicable(len(VALID[name]), True)
