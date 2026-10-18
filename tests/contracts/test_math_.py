"""Tests backing every claim in ``tharkun.solve.contracts.rules_math`` ([internal ref])."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


def _real_number_cases():
    """(name, callable) pairs for the single-argument "real number" functions."""
    return [
        ("sqrt", lambda x: math.sqrt(x)),
        ("log", lambda x: math.log(x)),
        ("isnan", lambda x: math.isnan(x)),
        ("isinf", lambda x: math.isinf(x)),
        ("isfinite", lambda x: math.isfinite(x)),
        ("nextafter", lambda x: math.nextafter(x, 1.0)),
    ]


@pytest.mark.parametrize("name,fn", _real_number_cases())
class TestRealNumberFamily:
    """sqrt/log/isnan/isinf/isfinite/nextafter share one propagation shape."""

    def test_contract_is_tested(self, name, fn):
        assert get(f"math.{name}").tested

    def test_accepts_float(self, name, fn):
        fn(4.0)  # must not raise

    def test_accepts_index_like(self, name, fn):
        class HasIndex:
            def __index__(self):
                return 4

        fn(HasIndex())  # must not raise

    def test_no_protocol_raises_typeerror(self, name, fn):
        with pytest.raises(TypeError):
            fn(None)
        with pytest.raises(TypeError):
            fn("x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_float_dunder_propagation(self, name, fn, exc_name):
        Bad = raiser("__float__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            fn(Bad())


class TestSqrt:
    def test_domain_error_is_valueerror(self):
        with pytest.raises(ValueError):
            math.sqrt(-1)

    @given(st.floats(min_value=0, allow_nan=False, allow_infinity=False))
    def test_nonnegative_result(self, x):
        assert math.sqrt(x) >= 0.0


class TestLog:
    def test_domain_error_is_valueerror(self):
        with pytest.raises(ValueError):
            math.log(0)
        with pytest.raises(ValueError):
            math.log(-1)

    def test_base_argument_propagation(self):
        Bad = raiser("__float__", "KeyError")
        with pytest.raises(KeyError):
            math.log(8, Bad())


class TestFsum:
    contract = get("math.fsum")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)))
    def test_returns_float(self, xs):
        assert isinstance(math.fsum(xs), float)

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            math.fsum(5)

    def test_bad_element_raises_typeerror(self):
        with pytest.raises(TypeError):
            math.fsum(["x"])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iteration_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            math.fsum(BadIter())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_element_dunder_propagation(self, exc_name):
        Bad = raiser("__float__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            math.fsum([Bad()])


class TestFloorCeil:
    def test_contracts_are_tested(self):
        assert get("math.floor").tested
        assert get("math.ceil").tested

    def test_basic(self):
        assert math.floor(5.7) == 5
        assert math.ceil(5.2) == 6

    def test_tries_dunder_first(self):
        class FloorObj:
            def __floor__(self):
                return 3

        assert math.floor(FloorObj()) == 3

    def test_falls_back_to_index_or_float(self):
        class HasFloatOnly:
            def __float__(self):
                return 5.7

        assert math.floor(HasFloatOnly()) == 5

    def test_no_protocol_raises_typeerror(self):
        class Neither:
            pass

        with pytest.raises(TypeError):
            math.floor(Neither())
        with pytest.raises(TypeError):
            math.ceil(Neither())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_own_dunder_propagation(self, exc_name):
        BadFloor = raiser("__floor__", exc_name)
        BadCeil = raiser("__ceil__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            math.floor(BadFloor())
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            math.ceil(BadCeil())


class TestFactorial:
    contract = get("math.factorial")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.integers(min_value=0, max_value=20))
    def test_matches_manual_product(self, n):
        expected = 1
        for i in range(2, n + 1):
            expected *= i
        assert math.factorial(n) == expected

    def test_negative_raises_valueerror(self):
        with pytest.raises(ValueError):
            math.factorial(-1)

    def test_whole_valued_float_raises_typeerror(self):
        # SURPRISING, confirmed on 3.14.4: 5.0 is rejected outright, unlike sqrt/log
        # which happily accept a float.
        with pytest.raises(TypeError):
            math.factorial(5.0)

    def test_non_int_raises_typeerror(self):
        with pytest.raises(TypeError):
            math.factorial("x")

    def test_index_like_is_accepted(self):
        class HasIndex:
            def __index__(self):
                return 4

        assert math.factorial(HasIndex()) == 24

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            math.factorial(Bad())


class TestGcd:
    contract = get("math.gcd")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_no_args_returns_zero(self):
        assert math.gcd() == 0

    @given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000))
    def test_result_divides_both(self, a, b):
        g = math.gcd(a, b)
        assert a % g == 0
        assert b % g == 0

    def test_float_raises_typeerror(self):
        with pytest.raises(TypeError):
            math.gcd(5.0)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            math.gcd(Bad())


class TestPow:
    contract = get("math.pow")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_always_returns_float(self):
        assert math.pow(2, 3) == 8.0
        assert isinstance(math.pow(2, 3), float)

    def test_domain_error_is_valueerror(self):
        with pytest.raises(ValueError):
            math.pow(-1, 0.5)

    def test_no_protocol_raises_typeerror(self):
        with pytest.raises(TypeError):
            math.pow("x", 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_first_arg_propagation(self, exc_name):
        Bad = raiser("__float__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            math.pow(Bad(), 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_second_arg_propagation(self, exc_name):
        Bad = raiser("__float__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            math.pow(1, Bad())


class TestNextafter:
    contract = get("math.nextafter")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_moves_towards_target(self):
        assert math.nextafter(1.0, 2.0) > 1.0
        assert math.nextafter(1.0, 0.0) < 1.0

    def test_second_arg_propagation(self):
        Bad = raiser("__float__", "KeyError")
        with pytest.raises(KeyError):
            math.nextafter(1.0, Bad())
