"""Tests backing every claim in ``tharkun.solve.contracts.rules_random`` ([internal ref])."""

from __future__ import annotations

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


class TestRandom:
    contract = get("random.random")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_in_unit_interval(self):
        for _ in range(200):
            x = random.random()
            assert 0.0 <= x < 1.0


class TestRandint:
    contract = get("random.randint")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.integers(min_value=-1000, max_value=1000), st.integers(min_value=-1000, max_value=1000))
    def test_in_inclusive_range(self, a, b):
        lo, hi = min(a, b), max(a, b)
        x = random.randint(lo, hi)
        assert lo <= x <= hi

    def test_empty_range_raises_valueerror(self):
        with pytest.raises(ValueError):
            random.randint(5, 1)

    def test_non_index_raises_typeerror(self):
        with pytest.raises(TypeError):
            random.randint("a", "b")
        with pytest.raises(TypeError):
            random.randint(1.5, 5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            random.randint(Bad(), 5)


class TestRandrange:
    contract = get("random.randrange")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.integers(min_value=1, max_value=1000))
    def test_excludes_stop(self, n):
        for _ in range(20):
            assert 0 <= random.randrange(n) < n

    def test_empty_range_raises_valueerror(self):
        with pytest.raises(ValueError):
            random.randrange(0)
        with pytest.raises(ValueError):
            random.randrange(5, 1)

    def test_zero_step_raises_valueerror(self):
        with pytest.raises(ValueError):
            random.randrange(1, 10, 0)

    def test_non_index_raises_typeerror(self):
        with pytest.raises(TypeError):
            random.randrange("x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            random.randrange(Bad())


class TestChoice:
    contract = get("random.choice")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers(), min_size=1))
    def test_returns_a_member(self, xs):
        assert random.choice(xs) in xs

    def test_empty_raises_indexerror(self):
        with pytest.raises(IndexError):
            random.choice([])

    def test_non_sequence_raises_typeerror(self):
        with pytest.raises(TypeError):
            random.choice(5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_len_propagation(self, exc_name):
        class BadLen:
            def __len__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            random.choice(BadLen())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_getitem_propagation(self, exc_name):
        class BadGetitem:
            def __len__(self):
                return 3

            def __getitem__(self, i):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            random.choice(BadGetitem())


class TestUniform:
    contract = get("random.uniform")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.floats(min_value=-1000, max_value=1000, allow_nan=False), st.floats(min_value=-1000, max_value=1000, allow_nan=False))
    def test_in_range_either_order(self, a, b):
        x = random.uniform(a, b)
        lo, hi = min(a, b), max(a, b)
        assert lo <= x <= hi

    def test_a_greater_than_b_never_raises(self):
        random.uniform(5, 1)  # must not raise

    def test_no_protocol_raises_typeerror(self):
        with pytest.raises(TypeError):
            random.uniform("a", "b")

    def test_float_only_stub_is_not_enough(self):
        # uniform() is pure Python (a + (b - a) * random()), NOT a C-level
        # __float__/__index__ conversion like math.sqrt/log -- a stub with only
        # __float__ set does not satisfy its arithmetic and still raises TypeError.
        FloatOnly = raiser("__float__", "TypeError")
        with pytest.raises(TypeError):
            random.uniform(FloatOnly(), 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_rsub_propagation(self, exc_name):
        # b - a calls a.__rsub__(b) when a has no __sub__.
        Bad = raiser("__rsub__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            random.uniform(Bad(), 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_sub_propagation(self, exc_name):
        # b - a calls b.__sub__(a) when b defines __sub__.
        Bad = raiser("__sub__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            random.uniform(1, Bad())


class TestSeed:
    contract = get("random.seed")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @pytest.mark.parametrize("value", [None, 42, 3.14, "x", b"bytes", bytearray(b"x")])
    def test_accepts_documented_types(self, value):
        assert random.seed(value) is None

    def test_reseeding_with_same_int_reproduces_sequence(self):
        random.seed(123)
        a = [random.random() for _ in range(5)]
        random.seed(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_unsupported_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            random.seed([])

    def test_never_raises_a_property_class_for_unsupported_type(self):
        # seed() type-checks directly (no dunder call), so a hostile object with
        # raising dunders still just gets the plain "unsupported seed type" TypeError.
        Bad = raiser("__index__", "KeyError")
        with pytest.raises(TypeError):
            random.seed(Bad())


class TestSample:
    contract = get("random.sample")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers(), min_size=1, max_size=20))
    def test_returns_k_distinct_positions(self, xs):
        k = len(xs) // 2 or 1
        result = random.sample(xs, k)
        assert len(result) == k
        assert all(x in xs for x in result)

    def test_k_greater_than_population_raises_valueerror(self):
        with pytest.raises(ValueError):
            random.sample([1, 2], 5)

    def test_negative_k_raises_valueerror(self):
        with pytest.raises(ValueError):
            random.sample([1, 2], -1)

    def test_non_sequence_raises_typeerror(self):
        with pytest.raises(TypeError):
            random.sample(5, 1)

    def test_set_raises_typeerror(self):
        # sets are explicitly rejected by name, not just "not a sequence" duck typing
        with pytest.raises(TypeError):
            random.sample({1, 2}, 1)

    def test_does_not_mutate_population(self):
        xs = [1, 2, 3]
        random.sample(xs, 2)
        assert xs == [1, 2, 3]


class TestShuffle:
    contract = get("random.shuffle")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_mutates_in_place_returns_none(self):
        xs = [1, 2, 3, 4, 5]
        original = list(xs)
        assert random.shuffle(xs) is None
        assert sorted(xs) == sorted(original)

    def test_immutable_sequence_raises_typeerror(self):
        with pytest.raises(TypeError):
            random.shuffle("abc")
        with pytest.raises(TypeError):
            random.shuffle((1, 2, 3))

    def test_non_len_raises_typeerror(self):
        with pytest.raises(TypeError):
            random.shuffle(5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_len_propagation(self, exc_name):
        class BadLen:
            def __len__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            random.shuffle(BadLen())
