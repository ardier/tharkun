"""Tests backing every claim in ``tharkun.solve.contracts.rules_list`` ([internal ref])."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


def _eq_raiser(exc_name):
    return raiser("__eq__", exc_name, nargs=1)


class TestAppend:
    contract = get("list.append")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers()), st.integers())
    def test_appends_at_end_returns_none(self, xs, x):
        original_len = len(xs)
        assert xs.append(x) is None
        assert xs[-1] == x
        assert len(xs) == original_len + 1

    def test_never_raises_for_any_receiver_element_combo(self):
        xs = [object(), None, "x"]
        xs.append(object())  # must not raise


class TestExtend:
    contract = get("list.extend")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers()), st.lists(st.integers()))
    def test_matches_concatenation(self, a, b):
        a2 = list(a)
        a2.extend(b)
        assert a2 == a + b

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            [].extend(5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iteration_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            [].extend(BadIter())


class TestInsert:
    contract = get("list.insert")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_inserts_at_index(self):
        xs = [1, 2, 3]
        xs.insert(1, "x")
        assert xs == [1, "x", 2, 3]

    def test_out_of_range_index_clamps_never_raises_indexerror(self):
        xs = [1, 2, 3]
        xs.insert(-100, "lo")
        assert xs[0] == "lo"
        ys = [1, 2, 3]
        ys.insert(100, "hi")
        assert ys[-1] == "hi"

    def test_non_index_raises_typeerror(self):
        with pytest.raises(TypeError):
            [1, 2].insert("x", 3)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            [1, 2].insert(Bad(), "x")


class TestPop:
    contract = get("list.pop")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_default_pops_last(self):
        xs = [1, 2, 3]
        assert xs.pop() == 3
        assert xs == [1, 2]

    def test_negative_index(self):
        xs = [1, 2, 3]
        assert xs.pop(-1) == 3

    def test_empty_raises_indexerror(self):
        with pytest.raises(IndexError):
            [].pop()

    def test_out_of_range_raises_indexerror(self):
        with pytest.raises(IndexError):
            [1].pop(5)
        with pytest.raises(IndexError):
            [1, 2, 3].pop(-5)

    def test_non_index_raises_typeerror(self):
        with pytest.raises(TypeError):
            [1].pop("x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            [1, 2].pop(Bad())


class TestRemove:
    contract = get("list.remove")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_removes_first_match_returns_none(self):
        xs = [1, 2, 1]
        assert xs.remove(1) is None
        assert xs == [2, 1]

    def test_not_found_raises_valueerror(self):
        with pytest.raises(ValueError):
            [1, 2].remove(5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_eq_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            [1, 2].remove(_eq_raiser(exc_name)())


class TestIndex:
    contract = get("list.index")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers(), min_size=1))
    def test_returns_position_of_first_occurrence(self, xs):
        x = xs[0]
        idx = xs.index(x)
        assert 0 <= idx < len(xs)
        assert xs[idx] == x

    def test_not_found_raises_valueerror(self):
        with pytest.raises(ValueError):
            [1, 2].index(5)

    def test_bad_start_stop_raises_typeerror(self):
        with pytest.raises(TypeError):
            [1, 2, 3].index(1, "x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_eq_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            [1, 2].index(_eq_raiser(exc_name)())


class TestCount:
    contract = get("list.count")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers(min_value=0, max_value=3)))
    def test_matches_manual_count(self, xs):
        for v in range(4):
            assert xs.count(v) == sum(1 for x in xs if x == v)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_eq_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            [1, 2].count(_eq_raiser(exc_name)())


class TestClear:
    contract = get("list.clear")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_empties_the_list(self):
        xs = [1, 2, 3]
        assert xs.clear() is None
        assert xs == []


class TestCopy:
    contract = get("list.copy")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_shallow_copy(self):
        inner = [1]
        xs = [inner, 2]
        ys = xs.copy()
        assert ys == xs
        assert ys is not xs
        assert ys[0] is inner  # shallow: shared, not copied


class TestReverse:
    contract = get("list.reverse")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers()))
    def test_reverses_in_place(self, xs):
        original = list(xs)
        assert xs.reverse() is None
        assert xs == list(reversed(original))


class TestSort:
    contract = get("list.sort")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers()))
    def test_sorts_in_place_returns_none(self, xs):
        expected = sorted(xs)
        assert xs.sort() is None
        assert xs == expected

    def test_reverse_true(self):
        xs = [3, 1, 2]
        xs.sort(reverse=True)
        assert xs == [3, 2, 1]

    def test_noncallable_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            [1, 2].sort(key=5)

    def test_incomparable_elements_raise_typeerror(self):
        with pytest.raises(TypeError):
            [1, "a"].sort()

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_key_propagation(self, exc_name):
        def bad_key(_x):
            raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            [1, 2].sort(key=bad_key)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lt_propagation(self, exc_name):
        Bad = raiser("__lt__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            [Bad(), Bad()].sort()


class TestGetItem:
    contract = get("list.__getitem__")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers(), min_size=1), st.integers())
    def test_positive_and_negative_index_agree(self, xs, i):
        idx = i % len(xs)
        assert xs[idx] == xs[idx - len(xs)]  # e.g. xs[0] == xs[-len(xs)]

    def test_out_of_range_raises_indexerror(self):
        with pytest.raises(IndexError):
            [1, 2][5]
        with pytest.raises(IndexError):
            [1, 2][-5]

    def test_bad_key_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            [1, 2]["x"]

    def test_slice_out_of_range_never_raises(self):
        xs = [1, 2, 3]
        assert xs[10:20] == []
        assert xs[-100:100] == [1, 2, 3]

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            [1, 2][Bad()]


class TestSetItem:
    contract = get("list.__setitem__")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_sets_at_index(self):
        xs = [1, 2, 3]
        xs[1] = "x"
        assert xs == [1, "x", 3]

    def test_out_of_range_raises_indexerror(self):
        xs = [1, 2]
        with pytest.raises(IndexError):
            xs[5] = 0

    def test_bad_key_type_raises_typeerror(self):
        xs = [1, 2]
        with pytest.raises(TypeError):
            xs["x"] = 0

    def test_slice_assignment_needs_iterable(self):
        xs = [1, 2, 3]
        with pytest.raises(TypeError):
            xs[1:2] = 5

    def test_extended_slice_length_mismatch_raises_valueerror(self):
        xs = [1, 2, 3]
        with pytest.raises(ValueError):
            xs[::2] = [9]

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        xs = [1, 2]
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            xs[Bad()] = 0
