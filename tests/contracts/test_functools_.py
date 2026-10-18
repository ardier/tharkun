"""Tests backing every claim in ``tharkun.solve.contracts.rules_functools`` ([internal ref] batch 3)."""

from __future__ import annotations

import functools
import operator

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES
from tharkun.solve.contracts import get


class TestReduce:
    contract = get("functools.reduce")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert functools.reduce(operator.add, [1, 2, 3]) == 6

    def test_with_initial(self):
        assert functools.reduce(operator.add, [1, 2, 3], 10) == 16

    def test_empty_with_initial_returns_initial_unchanged(self):
        assert functools.reduce(operator.add, [], 5) == 5

    def test_empty_without_initial_raises_typeerror(self):
        with pytest.raises(TypeError):
            functools.reduce(operator.add, [])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_function_propagation(self, exc_name):
        def bad_fn(a, b):
            raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            functools.reduce(bad_fn, [1, 2, 3])

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: function/the iterable's __iter__ is arbitrary user code -- it
        # can raise ANY exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_function_propagates_non_property_exception(self):
        def bad_fn(a, b):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            functools.reduce(bad_fn, [1, 2, 3])


class TestPartial:
    contract = get("functools.partial")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_binds_leading_args(self):
        add = lambda a, b: a + b
        add1 = functools.partial(add, 1)
        assert add1(2) == 3

    def test_binds_keyword_args(self):
        def f(a, b, c=0):
            return (a, b, c)

        p = functools.partial(f, c=99)
        assert p(1, 2) == (1, 2, 99)

    def test_construction_never_validates_signature(self):
        # Building the partial doesn't call int() -- must not raise here.
        functools.partial(int, "x", "y")

    def test_call_propagates_underlying_function_errors(self):
        p = functools.partial(int, "x", "y")
        with pytest.raises(TypeError):
            p()

    def test_too_many_args_raises_typeerror(self):
        def f(a, b):
            return a + b

        p = functools.partial(f, 1)
        with pytest.raises(TypeError):
            p(2, 3)

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: func is arbitrary user code -- calling the partial can raise
        # ANY exception func itself raises, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_call_propagates_property_exception(self, exc_name):
        def bad_fn(a):
            raise PROPERTY_CLASSES[exc_name]("boom")

        p = functools.partial(bad_fn)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            p(1)

    def test_call_propagates_non_property_exception(self):
        def bad_fn(a):
            raise ValueError("boom")

        p = functools.partial(bad_fn)
        with pytest.raises(ValueError):
            p(1)


class TestLruCache:
    contract = get("functools.lru_cache")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_caches_results(self):
        calls = []

        @functools.lru_cache(maxsize=None)
        def f(x):
            calls.append(x)
            return x * 2

        assert f(5) == 10
        assert f(5) == 10
        assert calls == [5]  # only called once

    def test_cache_is_exactly_lru_cache_maxsize_none(self):
        calls = []

        @functools.cache
        def g(x):
            calls.append(x)
            return x

        g(1)
        g(1)
        assert calls == [1]

    def test_a_raising_call_is_not_cached(self):
        calls = []

        @functools.lru_cache(maxsize=None)
        def f(x):
            calls.append(x)
            raise KeyError("boom")

        with pytest.raises(KeyError):
            f(1)
        with pytest.raises(KeyError):
            f(1)
        assert calls == [1, 1]  # called again -- the failure was not cached

    def test_unhashable_args_raise_typeerror(self):
        @functools.cache
        def f(x):
            return x

        with pytest.raises(TypeError):
            f([1, 2])

    def test_bad_maxsize_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            functools.lru_cache(maxsize="x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_wrapped_function_propagation(self, exc_name):
        @functools.lru_cache(maxsize=None)
        def f(x):
            raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            f(1)

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: the wrapped function and the arguments' __hash__/__eq__ are
        # arbitrary user code -- either can raise ANY exception, not just one of
        # the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_wrapped_function_propagates_non_property_exception(self):
        @functools.lru_cache(maxsize=None)
        def f(x):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            f(1)

    def test_argument_hash_propagates_non_property_exception(self):
        class BadHash:
            def __hash__(self):
                raise ValueError("boom")

        @functools.lru_cache(maxsize=None)
        def f(x):
            return x

        with pytest.raises(ValueError):
            f(BadHash())
