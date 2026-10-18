"""Tests backing every claim in ``tharkun.solve.contracts.rules_builtins`` ([internal ref]).

Each test corresponds to one ``RaiseClaim``/``ReturnSpec`` marked ``tested=True`` in
``builtins.py``; a claim with no test here is a bug in that module, not a gap the
reviewer should assume is covered. Run on CPython 3.14.4 only (``docs/contracts/
FORMAT.md``); a different interpreter is not this batch's claim.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, attr_raiser, raiser
from tharkun.solve.contracts import get

# ---------------------------------------------------------------------------
# len
# ---------------------------------------------------------------------------


class TestLen:
    contract = get("len")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers()))
    def test_returns_int_ge_0(self, xs):
        assert isinstance(len(xs), int)
        assert len(xs) >= 0

    @pytest.mark.parametrize("obj", [None, 5, 5.0, object(), True])
    def test_no_dunder_raises_typeerror(self, obj):
        with pytest.raises(TypeError):
            len(obj)

    @pytest.mark.parametrize("bad", ["5", 5.5, None, [1, 2]])
    def test_len_return_value_without_index_raises_typeerror(self, bad):
        class Bad:
            def __len__(self):
                return bad

        with pytest.raises(TypeError):
            len(Bad())

    def test_len_return_value_with_index_is_used(self):
        class Idx:
            def __index__(self):
                return 3

        class Ok:
            def __len__(self):
                return Idx()

        assert len(Ok()) == 3

    def test_dunder_raising_typeerror_propagates(self):
        with pytest.raises(TypeError):
            len(raiser("__len__", "TypeError")())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_dunder_propagation_all_classes(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            len(raiser("__len__", exc_name)())

    def test_empty_containers_have_len_zero(self):
        assert len([]) == len("") == len({}) == len(()) == len(set()) == len(b"") == 0


# ---------------------------------------------------------------------------
# int
# ---------------------------------------------------------------------------


class TestInt:
    contract = get("int")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.integers())
    def test_int_of_int_is_identity(self, n):
        assert int(n) == n

    @pytest.mark.parametrize("bad", [None, [], {}, object(), 1 + 2j])
    def test_single_arg_no_protocol_raises_typeerror(self, bad):
        with pytest.raises(TypeError):
            int(bad)

    def test_int_prefers___int___over___index__(self):
        class Both:
            def __index__(self):
                return 5

            def __int__(self):
                return 7

        assert int(Both()) == 7

    def test_int_falls_back_to___index__(self):
        class OnlyIndex:
            def __index__(self):
                return 9

        assert int(OnlyIndex()) == 9

    def test_int_does_not_consult___trunc__(self):
        class OnlyTrunc:
            def __trunc__(self):
                return 42

        with pytest.raises(TypeError):
            int(OnlyTrunc())

    def test_int_returning_non_int_from___int___raises_typeerror(self):
        class BadInt:
            def __int__(self):
                return "5"

        with pytest.raises(TypeError):
            int(BadInt())

    def test_two_arg_form_requires_str_bytes_bytearray_regardless_of_dunders(self):
        class HasEverything:
            def __int__(self):
                return 1

            def __index__(self):
                return 1

        with pytest.raises(TypeError):
            int(HasEverything(), 2)
        assert int("101", 2) == 5
        assert int(b"101", 2) == 5
        assert int(bytearray(b"101"), 2) == 5

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_single_arg_dunder_propagation(self, exc_name):
        # __int__ raising propagates directly.
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            int(raiser("__int__", exc_name)())
        # __index__ propagates too, when __int__ is absent.
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            int(raiser("__index__", exc_name)())

    def test_two_arg_form_never_invokes_dunders(self):
        # A receiver whose __int__ would raise is irrelevant: the 2-arg form never
        # calls it because x is not str/bytes/bytearray -> plain TypeError, not the
        # dunder's exception.
        obj = raiser("__int__", "KeyError")()
        with pytest.raises(TypeError):
            int(obj, 2)


# ---------------------------------------------------------------------------
# float
# ---------------------------------------------------------------------------


class TestFloat:
    contract = get("float")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.floats(allow_nan=True, allow_infinity=True))
    def test_float_of_float_is_identity_or_nan(self, x):
        result = float(x)
        assert isinstance(result, float)
        assert result == x or (math.isnan(result) and math.isnan(x))

    @pytest.mark.parametrize("bad", [None, [], {}, object()])
    def test_no_protocol_raises_typeerror(self, bad):
        with pytest.raises(TypeError):
            float(bad)

    def test_float_prefers___float___over___index__(self):
        class Both:
            def __index__(self):
                return 1

            def __float__(self):
                return 2.5

        assert float(Both()) == 2.5

    def test_float_falls_back_to___index__(self):
        class OnlyIndex:
            def __index__(self):
                return 7

        assert float(OnlyIndex()) == 7.0

    def test_float_does_not_consult___trunc__(self):
        class OnlyTrunc:
            def __trunc__(self):
                return 42

        with pytest.raises(TypeError):
            float(OnlyTrunc())

    def test_float_returning_non_float_raises_typeerror(self):
        class BadFloat:
            def __float__(self):
                return "5"

        with pytest.raises(TypeError):
            float(BadFloat())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_dunder_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            float(raiser("__float__", exc_name)())


# ---------------------------------------------------------------------------
# str
# ---------------------------------------------------------------------------


class TestStr:
    contract = get("str")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.integers() | st.text() | st.floats(allow_nan=False))
    def test_returns_str(self, x):
        assert isinstance(str(x), str)

    def test_object_with_no_dunders_uses_default_repr(self):
        class Plain:
            pass

        assert str(Plain()).startswith("<")

    def test_falls_back_to___repr___when_no___str__(self):
        class ReprOnly:
            def __repr__(self):
                return "R!"

        assert str(ReprOnly()) == "R!"

    def test_str_returning_non_str_raises_typeerror(self):
        class BadStr:
            def __str__(self):
                return 5

        with pytest.raises(TypeError):
            str(BadStr())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_dunder_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            str(raiser("__str__", exc_name)())


# ---------------------------------------------------------------------------
# abs
# ---------------------------------------------------------------------------


class TestAbs:
    contract = get("abs")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.integers())
    def test_returns_nonnegative_int(self, n):
        assert abs(n) >= 0

    @given(st.floats(allow_nan=False))
    def test_returns_nonnegative_float(self, x):
        assert abs(x) >= 0.0

    def test_complex_abs_is_modulus(self):
        assert abs(3 + 4j) == 5.0

    @pytest.mark.parametrize("bad", [None, "x", [], {}, object()])
    def test_no___abs___raises_typeerror(self, bad):
        with pytest.raises(TypeError):
            abs(bad)

    def test_only___index___is_not_enough(self):
        class OnlyIndex:
            def __index__(self):
                return 5

        with pytest.raises(TypeError):
            abs(OnlyIndex())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_dunder_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            abs(raiser("__abs__", exc_name)())


# ---------------------------------------------------------------------------
# round
# ---------------------------------------------------------------------------


class TestRound:
    contract = get("round")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_native_typeerror_is_separate_from_dunder_propagation(self):
        from tharkun.solve.contracts.adapters.common import native_raise_condition, never_raises

        assert native_raise_condition(self.contract, "TypeError") == (
            "wrong call shape, argument has no usable __round__, ndigits is not indexable, or __round__/ndigits.__index__ raises TypeError"
        )
        assert native_raise_condition(self.contract, "IndexError") is None
        assert not never_raises(self.contract, "TypeError")
        assert "__round__" in self.contract.raise_claim("TypeError").condition
        assert "ndigits.__index__" in self.contract.raise_claim("TypeError").condition
        assert "__round__" in self.contract.raise_claim("IndexError").condition
        assert "ndigits.__index__" in self.contract.raise_claim("IndexError").condition
        with pytest.raises(TypeError):
            round()
        with pytest.raises(TypeError):
            round(object())
        with pytest.raises(TypeError):
            round(1, object())

    @pytest.mark.parametrize(("value", "expected"), [
        (0.5, 0), (1.5, 2), (2.5, 2), (-0.5, 0), (-1.5, -2),
    ])
    def test_one_argument_rounds_ties_to_even_and_returns_int(self, value, expected):
        result = round(value)
        assert result == expected
        assert type(result) is int

    def test_int_bool_and_large_finite_float_return_ints(self):
        assert type(round(123)) is int and round(123) == 123
        assert type(round(True)) is int and round(True) == 1
        assert type(round(1e100)) is int

    @pytest.mark.parametrize(("value", "exception"), [
        (float("nan"), ValueError), (float("inf"), OverflowError), (float("-inf"), OverflowError),
    ])
    def test_nan_and_infinities_raise_their_native_nonproperty_class(self, value, exception):
        with pytest.raises(exception):
            round(value)

    def test_native_keyword_and_ndigits_forms_are_valid(self):
        assert round(1.25, 0) == 1.0
        assert type(round(1.25, 0)) is float
        assert type(round(1.25, None)) is int
        assert type(round(17, 0)) is int
        assert type(round(17, None)) is int
        assert round(number=1.5) == 2
        assert type(round(number=1.5, ndigits=0)) is float
        assert math.isnan(round(math.nan, 0))
        assert math.isinf(round(math.inf, 0))

    def test_custom_round_result_is_returned_as_is(self):
        class Custom:
            def __round__(self):
                return 3.5

        result = round(Custom())
        assert result == 3.5 and type(result) is float

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_custom_dunder_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            round(raiser("__round__", exc_name)())

    @pytest.mark.parametrize("value", [17, 1.75])
    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_ndigits_index_dunder_propagation(self, value, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            round(value, raiser("__index__", exc_name)())


# ---------------------------------------------------------------------------
# divmod
# ---------------------------------------------------------------------------


class TestDivmod:
    contract = get("divmod")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.integers(), st.integers().filter(lambda n: n != 0))
    def test_returns_pair_matching_floordiv_mod(self, a, b):
        q, r = divmod(a, b)
        assert (q, r) == (a // b, a % b)

    @pytest.mark.parametrize("a,b", [(1, 0), (1.0, 0), (1, 0.0), (-1, 0)])
    def test_zero_divisor_raises_zerodivisionerror(self, a, b):
        with pytest.raises(ZeroDivisionError):
            divmod(a, b)

    @pytest.mark.parametrize("a,b", [("a", "b"), (1 + 2j, 1), (None, 1), ([], 1)])
    def test_unsupported_types_raise_typeerror(self, a, b):
        with pytest.raises(TypeError):
            divmod(a, b)

    def test_floordiv_and_mod_alone_are_not_enough(self):
        class HasFloorMod:
            def __floordiv__(self, o):
                return 1

            def __mod__(self, o):
                return 0

        with pytest.raises(TypeError):
            divmod(HasFloorMod(), 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_dunder_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            divmod(raiser("__divmod__", exc_name, nargs=1)(), 1)


# ---------------------------------------------------------------------------
# pow
# ---------------------------------------------------------------------------


class TestPow:
    contract = get("pow")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.integers(min_value=-6, max_value=6), st.integers(min_value=0, max_value=6))
    def test_two_arg_matches_operator(self, a, b):
        assert pow(a, b) == a**b

    @pytest.mark.parametrize("base", [0, 0.0, 0j])
    def test_zero_base_negative_exponent_raises_zerodivisionerror(self, base):
        with pytest.raises(ZeroDivisionError):
            pow(base, -1)

    def test_three_arg_form_never_raises_zerodivisionerror_for_zero_base(self):
        # Same "zero to a negative power" shape, but the 3-arg form reports it as
        # ValueError ("base is not invertible"), not ZeroDivisionError.
        with pytest.raises(ValueError):
            pow(0, -1, 5)

    def test_three_arg_form_modulus_zero_is_valueerror_not_zerodivisionerror(self):
        with pytest.raises(ValueError):
            pow(2, 3, 0)

    @pytest.mark.parametrize("a,b", [("a", "b"), (None, 1), ([], 1)])
    def test_unsupported_types_raise_typeerror(self, a, b):
        with pytest.raises(TypeError):
            pow(a, b)

    def test_three_arg_form_requires___pow___to_accept_modulus(self):
        class TwoArgPowOnly:
            def __pow__(self, other):
                return 1

        with pytest.raises(TypeError):
            pow(TwoArgPowOnly(), 1, 5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_dunder_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            pow(raiser("__pow__", exc_name, nargs=1)(), 1)


# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------


class TestMinMax:
    min_contract = get("min")
    max_contract = get("max")

    def test_contracts_are_tested(self):
        assert self.min_contract.tested and self.max_contract.tested

    @given(st.lists(st.integers(), min_size=1))
    def test_min_le_all_max_ge_all(self, xs):
        lo, hi = min(xs), max(xs)
        assert all(lo <= x for x in xs)
        assert all(hi >= x for x in xs)

    def test_empty_with_default_returns_default(self):
        assert min([], default=9) == 9
        assert max([], default=9) == 9

    def test_empty_without_default_raises_valueerror(self):
        with pytest.raises(ValueError):
            min([])
        with pytest.raises(ValueError):
            max([])

    def test_noncallable_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            min([1, 2], key=5)
        with pytest.raises(TypeError):
            max([1, 2], key=5)

    def test_incomparable_elements_raise_typeerror(self):
        with pytest.raises(TypeError):
            min([1, "a"])
        with pytest.raises(TypeError):
            max([1, "a"])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_key_function_propagation(self, exc_name):
        def bad_key(_x):
            raise PROPERTY_CLASSES[exc_name]("boom-key")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            min([1, 2], key=bad_key)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            max([1, 2], key=bad_key)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lt_propagation(self, exc_name):
        Bad = raiser("__lt__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            min([Bad(), Bad()])
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            max([Bad(), Bad()])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iterable_iter_propagation(self, exc_name):
        # The raises table already marked these as propagating (dunder_propagation_
        # claims covers every non-native class regardless of exact wording), but
        # this specific path -- the ITERABLE's own __iter__, as opposed to key= or
        # __lt__ -- was not actually exercised until now.
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            min(BadIter())
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            max(BadIter())


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


class TestSum:
    contract = get("sum")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers()))
    def test_matches_manual_accumulation(self, xs):
        total = 0
        for x in xs:
            total += x
        assert sum(xs) == total

    def test_empty_returns_start(self):
        assert sum([]) == 0
        assert sum([], 10) == 10

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            sum(5)

    def test_strings_are_rejected_even_with_string_start(self):
        with pytest.raises(TypeError):
            sum(["a", "b"], "")

    def test_incompatible_element_raises_typeerror(self):
        with pytest.raises(TypeError):
            sum([1, None])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_radd_propagation(self, exc_name):
        Bad = raiser("__radd__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            sum([Bad()])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iterable_iter_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            sum(BadIter())


# ---------------------------------------------------------------------------
# any / all
# ---------------------------------------------------------------------------


class TestAnyAll:
    any_contract = get("any")
    all_contract = get("all")

    def test_contracts_are_tested(self):
        assert self.any_contract.tested and self.all_contract.tested

    @given(st.lists(st.integers()))
    def test_match_the_definition(self, xs):
        assert any(xs) == (len([x for x in xs if x]) > 0)
        assert all(xs) == (len([x for x in xs if not x]) == 0)

    def test_empty_iterable(self):
        assert any([]) is False
        assert all([]) is True

    def test_return_is_a_bool_not_an_element(self):
        assert any([0, 5]) is True
        assert all([3, 5]) is True
        assert any(["", None]) is False
        assert all([1, ""]) is False

    @pytest.mark.parametrize("fn", [any, all])
    def test_non_iterable_and_wrong_arity_raise_typeerror(self, fn):
        with pytest.raises(TypeError):
            fn(5)
        with pytest.raises(TypeError):
            fn()
        with pytest.raises(TypeError):
            fn([1], [2])

    def test_any_stops_at_the_first_truthy_element(self):
        Bad = raiser("__bool__", "AssertionError")
        assert any([1, Bad()]) is True  # the second element is never read for its truth value

    def test_all_stops_at_the_first_falsy_element(self):
        Bad = raiser("__bool__", "AssertionError")
        assert all([0, Bad()]) is False

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_element_truth_propagation(self, exc_name):
        Bad = raiser("__bool__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            any([Bad()])
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            all([Bad()])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_element_len_propagation(self, exc_name):
        Bad = raiser("__len__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            any([Bad()])
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            all([Bad()])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iterable_iter_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            any(BadIter())
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            all(BadIter())


# ---------------------------------------------------------------------------
# range
# ---------------------------------------------------------------------------


class TestRange:
    contract = get("range")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.integers(min_value=-50, max_value=50), st.integers(min_value=-50, max_value=50))
    def test_matches_manual_enumeration(self, a, b):
        expected = []
        i = a
        while i < b:
            expected.append(i)
            i += 1
        assert list(range(a, b)) == expected

    def test_is_a_range_object_not_a_list(self):
        r = range(5)
        assert isinstance(r, range)
        assert list(r) == [0, 1, 2, 3, 4]

    def test_zero_or_too_many_args_raise_typeerror(self):
        with pytest.raises(TypeError):
            range()
        with pytest.raises(TypeError):
            range(1, 2, 3, 4)

    @pytest.mark.parametrize("bad", [1.5, "1", None, []])
    def test_non_index_argument_raises_typeerror(self, bad):
        with pytest.raises(TypeError):
            range(bad)

    def test_step_zero_raises_valueerror(self):
        with pytest.raises(ValueError):
            range(1, 2, 0)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            range(Bad())
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            range(0, Bad())


# ---------------------------------------------------------------------------
# sorted
# ---------------------------------------------------------------------------


class TestSorted:
    contract = get("sorted")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers()))
    def test_result_is_sorted_and_input_unmutated(self, xs):
        original = list(xs)
        result = sorted(xs)
        assert result == sorted(original)
        assert result == list(sorted(original))
        assert xs == original  # sorted() does not mutate its argument
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    def test_reverse_true(self):
        assert sorted([3, 1, 2], reverse=True) == [3, 2, 1]

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            sorted(None)

    def test_noncallable_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            sorted([1, 2], key=5)

    def test_incomparable_elements_raise_typeerror(self):
        with pytest.raises(TypeError):
            sorted([1, "a"])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_key_propagation(self, exc_name):
        def bad_key(_x):
            raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            sorted([1, 2], key=bad_key)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lt_propagation(self, exc_name):
        Bad = raiser("__lt__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            sorted([Bad(), Bad()])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iterable_iter_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            sorted(BadIter())


# ---------------------------------------------------------------------------
# isinstance
# ---------------------------------------------------------------------------


class TestIsinstance:
    contract = get("isinstance")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.integers())
    def test_returns_bool(self, n):
        assert isinstance(isinstance(n, int), bool)

    def test_bad_classinfo_raises_typeerror(self):
        with pytest.raises(TypeError):
            isinstance(1, "not-a-type")

    def test_tuple_classinfo_short_circuits_before_raising(self):
        # int matches first; the bad 'x' entry is never reached.
        assert isinstance(1, (int, "x")) is True

    def test_tuple_classinfo_with_no_match_before_bad_entry_raises(self):
        with pytest.raises(TypeError):
            isinstance(1, (str, "x"))

    def test_union_type_classinfo(self):
        assert isinstance(1, int | str) is True
        assert isinstance(1.5, int | str) is False

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_instancecheck_propagation(self, exc_name):
        class Meta(type):
            pass

        Meta.__instancecheck__ = raiser("__instancecheck__", exc_name, nargs=1).__dict__["__instancecheck__"]

        class WithMeta(metaclass=Meta):
            pass

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            isinstance(5, WithMeta)


# ---------------------------------------------------------------------------
# open
# ---------------------------------------------------------------------------


class TestOpen:
    contract = get("open")

    def test_contract_is_tested_is_false_pending_return_shape(self):
        assert self.contract.tested is False  # raises are tested; return shape isn't

    def test_opens_and_writes_a_real_file(self, tmp_path):
        p = tmp_path / "f.txt"
        with open(p, "w") as fh:
            fh.write("hi")
        assert p.read_text() == "hi"

    def test_path_without_fspath_or_str_bytes_raises_typeerror(self):
        with pytest.raises(TypeError):
            open(None)
        with pytest.raises(TypeError):
            open(object())

    def test_fspath_object_is_accepted(self, tmp_path):
        class PathLike:
            def __fspath__(self):
                return str(tmp_path / "via_fspath.txt")

        with open(PathLike(), "w") as fh:
            fh.write("x")
        assert (tmp_path / "via_fspath.txt").read_text() == "x"

    def test_fspath_returning_non_str_bytes_raises_typeerror(self):
        class BadFspath:
            def __fspath__(self):
                return 5

        with pytest.raises(TypeError):
            open(BadFspath())

    def test_non_str_mode_raises_typeerror(self, tmp_path):
        with pytest.raises(TypeError):
            open(str(tmp_path / "f.txt"), 5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_fspath_propagation(self, exc_name):
        Bad = raiser("__fspath__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            open(Bad())


# ---------------------------------------------------------------------------
# print
# ---------------------------------------------------------------------------


class TestPrint:
    contract = get("print")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_returns_none(self, capsys):
        assert print("x") is None

    def test_writes_to_given_file(self):
        import io

        buf = io.StringIO()
        print("hi", file=buf)
        assert buf.getvalue() == "hi\n"

    @pytest.mark.parametrize("kwargs", [{"sep": 5}, {"end": 5}])
    def test_non_str_sep_end_raises_typeerror(self, kwargs):
        with pytest.raises(TypeError):
            print("a", "b", **kwargs)

    def test_file_without_write_raises_attributeerror(self):
        class NoWrite:
            pass

        with pytest.raises(AttributeError):
            print("x", file=NoWrite())

    def test_flush_true_requires_flush_attribute(self):
        class WriteOnly:
            def write(self, s):
                pass

        with pytest.raises(AttributeError):
            print("x", file=WriteOnly(), flush=True)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_write_propagation(self, exc_name):
        obj = attr_raiser("write", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            print("x", file=obj)
