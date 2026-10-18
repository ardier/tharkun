"""Tests backing every claim in ``tharkun.solve.contracts.rules_itertools`` ([internal ref] batch 3)."""

from __future__ import annotations

import itertools as it

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


class _BadIter:
    """Raises from __iter__ itself."""

    def __init__(self, exc):
        self._exc = exc

    def __iter__(self):
        raise self._exc


class _BadNext:
    """__iter__ succeeds; __next__ raises on first pull."""

    def __init__(self, exc):
        self._exc = exc

    def __iter__(self):
        return self

    def __next__(self):
        raise self._exc


class TestCount:
    contract = get("itertools.count")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic_sequence(self):
        c = it.count(5, 2)
        assert [next(c) for _ in range(3)] == [5, 7, 9]

    def test_bad_start_raises_typeerror_eagerly(self):
        with pytest.raises(TypeError):
            it.count("a")

    def test_bad_step_raises_typeerror_eagerly(self):
        with pytest.raises(TypeError):
            it.count(1, "a")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_step_add_propagation_from_first_next(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): an object that passes count's
        # number-like check (has __index__, so construction succeeds) but whose
        # __radd__ raises propagates that exception -- not "never", and not
        # limited to TypeError. Confirmed empirically: this fires on the FIRST
        # next() call already, not deferred to a later one.
        Step = raiser("__radd__", exc_name, nargs=1)
        Step.__index__ = lambda self: 1
        c = it.count(0, Step())
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            next(c)

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: start's/step's __add__/__radd__ is arbitrary user code -- it
        # can raise ANY exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_step_add_propagates_non_property_exception(self):
        class Step:
            def __index__(self):
                return 1

            def __radd__(self, other):
                raise ValueError("boom")

        c = it.count(0, Step())
        with pytest.raises(ValueError):
            next(c)


class TestChain:
    contract = get("itertools.chain")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert list(it.chain([1, 2], [3, 4])) == [1, 2, 3, 4]

    def test_construction_never_raises_for_bad_argument(self):
        it.chain(5)  # must not raise

    def test_consumption_raises_typeerror_for_non_iterable(self):
        with pytest.raises(TypeError):
            list(it.chain(5))

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lazy_iter_propagation(self, exc_name):
        c = it.chain(_BadIter(PROPERTY_CLASSES[exc_name]("boom")))
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            list(c)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lazy_next_propagation(self, exc_name):
        c = it.chain(_BadNext(PROPERTY_CLASSES[exc_name]("boom")))
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            list(c)

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_lazy_next_propagates_non_property_exception(self):
        c = it.chain(_BadNext(ValueError("boom")))
        with pytest.raises(ValueError):
            list(c)


class TestProduct:
    contract = get("itertools.product")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert list(it.product([1, 2], [3, 4])) == [(1, 3), (1, 4), (2, 3), (2, 4)]

    def test_repeat(self):
        assert list(it.product([1, 2], repeat=2)) == [(1, 1), (1, 2), (2, 1), (2, 2)]

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            it.product(5)

    def test_negative_repeat_raises_valueerror(self):
        with pytest.raises(ValueError):
            it.product([1, 2], repeat=-1)

    def test_materializes_eagerly_at_construction(self):
        # A hostile __next__ (not just __iter__) fails at the product() call itself.
        with pytest.raises(KeyError):
            it.product(_BadNext(KeyError("boom")))

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iter_propagation_at_construction(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            it.product(_BadIter(PROPERTY_CLASSES[exc_name]("boom")))

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True


class TestCombinations:
    contract = get("itertools.combinations")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert list(it.combinations([1, 2, 3], 2)) == [(1, 2), (1, 3), (2, 3)]

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            it.combinations(5, 1)

    def test_negative_r_raises_valueerror(self):
        with pytest.raises(ValueError):
            it.combinations([1, 2], -1)

    def test_materializes_eagerly_at_construction(self):
        with pytest.raises(KeyError):
            it.combinations(_BadNext(KeyError("boom")), 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iter_propagation_at_construction(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            it.combinations(_BadIter(PROPERTY_CLASSES[exc_name]("boom")), 1)

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True


class TestPermutations:
    contract = get("itertools.permutations")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert sorted(it.permutations([1, 2])) == [(1, 2), (2, 1)]

    def test_r_defaults_to_length(self):
        assert len(list(it.permutations([1, 2, 3]))) == 6

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            it.permutations(5, 1)

    def test_negative_r_raises_valueerror(self):
        with pytest.raises(ValueError):
            it.permutations([1, 2], -1)

    def test_materializes_eagerly_at_construction(self):
        with pytest.raises(KeyError):
            it.permutations(_BadNext(KeyError("boom")), 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iter_propagation_at_construction(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            it.permutations(_BadIter(PROPERTY_CLASSES[exc_name]("boom")), 1)

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True


class TestIslice:
    contract = get("itertools.islice")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_stop_only_form(self):
        assert list(it.islice([1, 2, 3, 4, 5], 3)) == [1, 2, 3]

    def test_start_stop_step_form(self):
        assert list(it.islice(range(10), 1, 8, 2)) == [1, 3, 5, 7]

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            it.islice(5, 3)

    def test_bad_stop_raises_valueerror_eagerly(self):
        with pytest.raises(ValueError):
            it.islice([1, 2, 3], "x")

    def test_negative_stop_raises_valueerror_eagerly(self):
        with pytest.raises(ValueError):
            it.islice([1, 2, 3], -1)

    def test_construction_only_calls_iter_not_next(self):
        # __iter__ succeeding but doing nothing means construction should not pull
        # any elements -- confirmed by a hostile __next__ NOT firing until consumed.
        s = it.islice(_BadNext(KeyError("boom")), 5)
        with pytest.raises(KeyError):
            list(s)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iter_propagation_at_construction(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            it.islice(_BadIter(PROPERTY_CLASSES[exc_name]("boom")), 5)

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True


class TestGroupby:
    contract = get("itertools.groupby")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_groups_consecutive_runs_not_full_grouping(self):
        groups = [(k, list(g)) for k, g in it.groupby([1, 1, 2, 1])]
        assert [k for k, _ in groups] == [1, 2, 1]  # THREE runs, not two groups

    def test_with_key_function(self):
        groups = [(k, list(g)) for k, g in it.groupby([1, 2, 3, 4], key=lambda x: x % 2)]
        assert [k for k, _ in groups] == [1, 0, 1, 0]

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            it.groupby(5)

    def test_noncallable_key_raises_typeerror_lazily_not_at_construction(self):
        g = it.groupby([1, 2], key=5)  # must NOT raise here
        with pytest.raises(TypeError):
            list(g)  # raises only once an element is actually read

    def test_construction_only_calls_iter_not_next(self):
        g = it.groupby(_BadNext(KeyError("boom")))
        with pytest.raises(KeyError):
            list(g)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iter_propagation_at_construction(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            it.groupby(_BadIter(PROPERTY_CLASSES[exc_name]("boom")))

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_key_function_propagation(self, exc_name):
        def bad_key(x):
            raise PROPERTY_CLASSES[exc_name]("boom")

        g = it.groupby([1, 2], key=bad_key)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            list(g)

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: the iterable's __iter__/__next__ and key are arbitrary user
        # code -- either can raise ANY exception, not just one of the 7 property
        # classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_key_function_propagates_non_property_exception(self):
        def bad_key(x):
            raise ValueError("boom")

        g = it.groupby([1, 2], key=bad_key)
        with pytest.raises(ValueError):
            list(g)
