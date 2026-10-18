"""Tests backing every claim in ``tharkun.solve.contracts.rules_builtins_more`` ([internal ref] batch 4).

One test group per contract; each ``RaiseClaim``/``ReturnSpec``/effect the module marks
``tested=True`` has a test here.  CPython 3.14.4 only (``docs/contracts/FORMAT.md``).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, nonproperty_raiser, raiser
from tharkun.solve.contracts import ALL_PROPERTY_CLASSES, get

EXC_NAMES = sorted(PROPERTY_CLASSES)


def _every_class_claim_is_covered(contract):
    """A contract in this file claims all seven classes (TOTAL), each explicitly."""
    assert {c.exc for c in contract.raises} == set(ALL_PROPERTY_CLASSES)
    assert contract.tested


class _Sentinel:
    """An element a native constructor must never look at: every protocol raises."""

    def __eq__(self, other):
        raise AssertionError("element __eq__ was called")

    __hash__ = None

    def __lt__(self, other):
        raise AssertionError("element __lt__ was called")

    def __bool__(self):
        raise AssertionError("element __bool__ was called")

    def __iter__(self):
        raise AssertionError("element __iter__ was called")

    def __len__(self):
        raise AssertionError("element __len__ was called")


# ---------------------------------------------------------------------------
# ord
# ---------------------------------------------------------------------------


class TestOrd:
    contract = get("ord")

    def test_contract_is_total_and_tested(self):
        _every_class_claim_is_covered(self.contract)
        assert self.contract.may_raise_other is False
        assert self.contract.effects.may_execute_callback is False

    @given(st.characters(codec=None, min_codepoint=0, max_codepoint=0x10FFFF, exclude_categories=()))
    def test_every_single_character_str_returns_its_code_point(self, ch):
        n = ord(ch)
        assert isinstance(n, int) and 0 <= n <= 0x10FFFF

    @given(st.binary(min_size=1, max_size=1))
    def test_single_byte_returns_0_to_255(self, b):
        assert 0 <= ord(b) <= 255
        assert ord(bytearray(b)) == ord(b)

    def test_boundary_code_points_including_lone_surrogates(self):
        assert ord("\x00") == 0
        assert ord("\ud800") == 0xD800
        assert ord("\U0010ffff") == 0x10FFFF

    @pytest.mark.parametrize("bad", ["", "ab", "\U0010ffff\U0010ffff", b"", b"ab", bytearray(), bytearray(b"ab")])
    def test_wrong_length_is_typeerror_never_valueerror(self, bad):
        with pytest.raises(TypeError):
            ord(bad)

    @pytest.mark.parametrize("bad", [None, 5, 5.5, True, [], (), {}, object(), memoryview(b"a"), 0j])
    def test_non_text_argument_is_typeerror(self, bad):
        with pytest.raises(TypeError):
            ord(bad)

    def test_arity_and_keyword_are_typeerror(self):
        with pytest.raises(TypeError):
            ord()
        with pytest.raises(TypeError):
            ord("a", "b")
        with pytest.raises(TypeError):
            ord(c="a")

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_subclass_hooks_are_never_called(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        def boom(self, *a, **k):
            raise exc("hook ran")

        hooks = {"__len__": boom, "__index__": boom, "__getitem__": boom, "__iter__": boom, "__int__": boom}
        S = type("S", (str,), hooks)
        B = type("B", (bytes,), hooks)
        A = type("A", (bytearray,), hooks)
        assert ord(S("a")) == 97
        assert ord(B(b"a")) == 97
        assert ord(A(b"a")) == 97

    @given(st.one_of(st.none(), st.integers(), st.floats(), st.text(max_size=3), st.binary(max_size=3),
                     st.lists(st.integers(), max_size=3), st.booleans()))
    def test_no_exception_other_than_typeerror_for_any_builtin_value(self, v):
        try:
            ord(v)
        except TypeError:
            pass


# ---------------------------------------------------------------------------
# chr
# ---------------------------------------------------------------------------


class TestChr:
    contract = get("chr")

    def test_contract_is_total_and_tested(self):
        _every_class_claim_is_covered(self.contract)
        assert self.contract.may_raise_other and self.contract.effects.may_execute_callback

    @given(st.integers(min_value=0, max_value=0x10FFFF))
    def test_returns_str_of_length_one(self, i):
        s = chr(i)
        assert type(s) is str and len(s) == 1 and ord(s) == i

    @pytest.mark.parametrize("bad", [-1, -(10**30), 0x110000, 10**30])
    def test_out_of_range_is_valueerror_not_a_property_class(self, bad):
        with pytest.raises(ValueError):
            chr(bad)

    @pytest.mark.parametrize("bad", [65.0, "a", None, [], (1,), 1j, b"a"])
    def test_non_integer_is_typeerror(self, bad):
        with pytest.raises(TypeError):
            chr(bad)

    def test_arity_and_keyword_are_typeerror(self):
        with pytest.raises(TypeError):
            chr()
        with pytest.raises(TypeError):
            chr(1, 2)
        with pytest.raises(TypeError):
            chr(i=65)

    def test_bool_is_an_int(self):
        assert chr(True) == "\x01"

    def test_index_returning_non_int_is_typeerror(self):
        class Bad:
            def __index__(self):
                return "x"

        with pytest.raises(TypeError):
            chr(Bad())

    def test_index_is_used_for_a_non_int_object(self):
        class Idx:
            def __index__(self):
                return 66

        assert chr(Idx()) == "B"

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_index_propagation_all_classes(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            chr(raiser("__index__", exc_name)())

    def test_index_raising_something_else_propagates(self):
        with pytest.raises(RuntimeError):
            chr(nonproperty_raiser("__index__", RuntimeError)())

    def test_int_subclass_index_override_is_never_called(self):
        class Evil(int):
            def __index__(self):
                raise KeyError("must not run")

        assert chr(Evil(65)) == "A"


# ---------------------------------------------------------------------------
# type
# ---------------------------------------------------------------------------


class TestType:
    contract = get("type")

    def test_contract_is_total_and_tested(self):
        _every_class_claim_is_covered(self.contract)
        assert self.contract.may_raise_other and self.contract.effects.may_execute_callback

    def test_the_one_argument_shape_is_named_callback_free(self):
        from tharkun.solve.contracts.adapters.common import NO_CALLBACK_SHAPES

        assert NO_CALLBACK_SHAPES["type"] == frozenset({1})

    @pytest.mark.parametrize("x,cls", [
        (1, int), (True, bool), (1.5, float), ("s", str), (b"b", bytes), (None, type(None)), ([], list),
        ((), tuple), ({}, dict), (set(), set), (int, type), (type, type), (len, type(len)),
    ])
    def test_one_argument_returns_the_exact_class(self, x, cls):
        assert type(x) is cls

    def test_one_argument_ignores_a_hostile___class___property(self):
        class Evil:
            @property
            def __class__(self):
                raise KeyError("must not run")

        assert type(Evil()) is Evil

    def test_one_argument_ignores_a_metaclass___call__(self):
        class Meta(type):
            def __call__(cls, *a, **k):
                raise ZeroDivisionError("must not run")

        class C(metaclass=Meta):
            pass

        assert type(C) is Meta

    def test_one_argument_ignores_a_hostile___getattribute__(self):
        class Evil:
            def __getattribute__(self, name):
                raise AttributeError("must not run")

        assert type(Evil()) is Evil

    def test_arity_zero_two_and_keywords_are_typeerror(self):
        with pytest.raises(TypeError):
            type()
        with pytest.raises(TypeError):
            type(1, 2)
        with pytest.raises(TypeError):
            type(object=1)
        with pytest.raises(TypeError):
            type(1, y=1)

    @pytest.mark.parametrize("args", [
        (1, (), {}), ("A", 1, {}), ("A", (), 1), ("A", (int, str), {}), ("A", (), {"__slots__": 1}),
        ("A", (1,), {}),
    ])
    def test_three_argument_native_typeerror(self, args):
        with pytest.raises(TypeError):
            type(*args)

    def test_three_argument_null_in_name_is_valueerror(self):
        with pytest.raises(ValueError):
            type("A\x00", (), {})

    def test_three_argument_returns_a_new_class(self):
        C = type("C", (), {"x": 1})
        assert isinstance(C, type) and C.x == 1 and C.__name__ == "C"

    def test_three_argument_can_return_a_non_type_through_a_metaclass_new(self):
        class MN(type):
            def __new__(m, name, bases, ns, **k):
                return 5 if name == "Derived" else super().__new__(m, name, bases, ns, **k)

        class Base(metaclass=MN):
            pass

        assert type("Derived", (Base,), {}) == 5

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_three_argument_propagation_all_classes(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class Sub:
            def __init_subclass__(cls, **k):
                raise exc("init_subclass")

        with pytest.raises(exc):
            type("D", (Sub,), {})

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_three_argument_set_name_and_metaclass_hooks_propagate(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class SN:
            def __set_name__(self, owner, name):
                raise exc("set_name")

        with pytest.raises(exc):
            type("D", (), {"x": SN()})

        class MN(type):
            def __new__(m, name, bases, ns, **k):
                if name == "Derived":
                    raise exc("meta new")
                return super().__new__(m, name, bases, ns, **k)

        class MI(type):
            def __init__(cls, name, bases, ns, **k):
                if name == "Derived":
                    raise exc("meta init")
                super().__init__(name, bases, ns, **k)

        class BaseN(metaclass=MN):
            pass

        class BaseI(metaclass=MI):
            pass

        with pytest.raises(exc):
            type("Derived", (BaseN,), {})
        with pytest.raises(exc):
            type("Derived", (BaseI,), {})

    def test_three_argument_hook_may_raise_outside_the_seven(self):
        class Sub:
            def __init_subclass__(cls, **k):
                raise RuntimeError("user")

        with pytest.raises(RuntimeError):
            type("D", (Sub,), {})


# ---------------------------------------------------------------------------
# list / tuple / set -- the shared iterable-consuming shape
# ---------------------------------------------------------------------------

CTORS = [list, tuple, set]


@pytest.mark.parametrize("ctor", CTORS, ids=lambda c: c.__name__)
class TestIterableConstructors:
    def test_contracts_are_total_and_tested(self, ctor):
        c = get(ctor.__name__)
        _every_class_claim_is_covered(c)
        assert c.may_raise_other and c.effects.may_execute_callback
        assert c.effects.mutates_args == ("iterable",)

    def test_no_argument_is_the_empty_container(self, ctor):
        assert ctor() == ctor([])
        assert len(ctor()) == 0

    @pytest.mark.parametrize("bad", [None, 1, 1.5, True, object()])
    def test_non_iterable_is_typeerror(self, ctor, bad):
        with pytest.raises(TypeError):
            ctor(bad)

    def test_arity_and_keyword_are_typeerror(self, ctor):
        with pytest.raises(TypeError):
            ctor([1], [2])
        with pytest.raises(TypeError):
            ctor(iterable=[1])

    def test_iter_returning_a_non_iterator_is_typeerror(self, ctor):
        class Bad:
            def __iter__(self):
                return 5

        with pytest.raises(TypeError):
            ctor(Bad())

    def test_iter_set_to_none_is_typeerror(self, ctor):
        class NoIter:
            __iter__ = None

        with pytest.raises(TypeError):
            ctor(NoIter())

    @pytest.mark.parametrize("make,expect", [
        (lambda: "abc", ["a", "b", "c"]), (lambda: {1: 2, 3: 4}, [1, 3]), (lambda: range(4), [0, 1, 2, 3]),
        (lambda: (1, 2), [1, 2]), (lambda: [1, 2], [1, 2]), (lambda: iter([5, 6]), [5, 6]),
        (lambda: b"ab", [97, 98]),
    ])
    def test_accepts_every_builtin_iterable(self, ctor, make, expect):
        assert sorted(ctor(make())) == expect

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_iter_propagation_all_classes(self, ctor, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            ctor(raiser("__iter__", exc_name)())

    @pytest.mark.parametrize("exc_name", ["TypeError", "AttributeError", "KeyError", "ZeroDivisionError",
                                          "FloatingPointError", "AssertionError", "IndexError"])
    def test_next_propagation_all_classes(self, ctor, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class It:
            def __iter__(self):
                return self

            def __next__(self):
                raise exc("next")

        with pytest.raises(exc):
            ctor(It())

    def test_stopiteration_from_next_ends_the_iteration(self, ctor):
        class It:
            def __iter__(self):
                return self

            def __next__(self):
                raise StopIteration

        assert len(ctor(It())) == 0

    @pytest.mark.parametrize("exc_name", [n for n in EXC_NAMES if n != "IndexError"])
    def test_getitem_protocol_propagation(self, ctor, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class Seq:
            def __getitem__(self, i):
                if i < 2:
                    return i
                raise exc("getitem")

        with pytest.raises(exc):
            ctor(Seq())

    def test_getitem_protocol_indexerror_ends_the_iteration_silently(self, ctor):
        class Seq:
            def __getitem__(self, i):
                if i < 2:
                    return i
                raise IndexError("end")

        assert sorted(ctor(Seq())) == [0, 1]

    def test_an_iterator_argument_is_advanced_to_exhaustion(self, ctor):
        it = iter([1, 2, 3])
        ctor(it)
        assert list(it) == []

    def test_a_list_argument_is_not_modified(self, ctor):
        src = [3, 1, 2]
        ctor(src)
        assert src == [3, 1, 2]

    def test_hook_may_raise_outside_the_seven(self, ctor):
        with pytest.raises(RuntimeError):
            ctor(nonproperty_raiser("__iter__", RuntimeError)())


class TestList:
    contract = get("list")

    def test_return_is_a_new_list(self):
        src = [1, 2]
        out = list(src)
        assert out == src and out is not src and type(out) is list

    def test_list_of_a_list_subclass_is_a_plain_list(self):
        class L(list):
            pass

        assert type(list(L([1]))) is list

    @pytest.mark.parametrize("exc_name", [n for n in EXC_NAMES if n != "TypeError"])
    def test_len_probe_propagates_every_class_but_typeerror(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class L:
            def __iter__(self):
                return iter([1, 2])

            def __len__(self):
                raise exc("len")

        with pytest.raises(exc):
            list(L())

    @pytest.mark.parametrize("exc_name", [n for n in EXC_NAMES if n != "TypeError"])
    def test_length_hint_probe_propagates_every_class_but_typeerror(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class L:
            def __iter__(self):
                return iter([1, 2])

            def __length_hint__(self):
                raise exc("hint")

        with pytest.raises(exc):
            list(L())

    def test_len_and_length_hint_typeerror_is_swallowed(self):
        class L:
            def __iter__(self):
                return iter([1, 2])

            def __len__(self):
                raise TypeError("len")

            def __length_hint__(self):
                raise TypeError("hint")

        assert list(L()) == [1, 2]

    def test_negative_length_hint_is_valueerror(self):
        class L:
            def __iter__(self):
                return iter([1])

            def __length_hint__(self):
                return -1

        with pytest.raises(ValueError):
            list(L())

    def test_elements_are_never_inspected(self):
        s = _Sentinel()
        assert list([s, s])[0] is s


class TestTuple:
    contract = get("tuple")

    def test_exact_tuple_is_returned_as_the_same_object(self):
        t = (1, 2)
        assert tuple(t) is t

    def test_tuple_subclass_is_copied_to_a_plain_tuple(self):
        class T(tuple):
            pass

        t = T((1, 2))
        assert tuple(t) is not t and type(tuple(t)) is tuple

    @pytest.mark.parametrize("hook", ["__len__", "__length_hint__"])
    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_no_length_probe(self, hook, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class L:
            def __iter__(self):
                return iter([1, 2])

        def boom(self):
            raise exc("probe")

        setattr(L, hook, boom)
        assert tuple(L()) == (1, 2)

    def test_elements_are_never_inspected(self):
        s = _Sentinel()
        assert tuple([s, s])[0] is s


class TestSet:
    contract = get("set")

    def test_return_is_a_new_set_of_distinct_elements(self):
        src = [1, 1, True, 1.0, 2]
        out = set(src)
        assert out == {1, 2} and type(out) is set

    @pytest.mark.parametrize("bad", [[[1]], [{}], [{1}], [[]]])
    def test_unhashable_element_is_typeerror(self, bad):
        with pytest.raises(TypeError):
            set(bad)

    def test_hash_returning_non_int_is_typeerror(self):
        class H:
            def __hash__(self):
                return "x"

        with pytest.raises(TypeError):
            set([H()])

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_element_hash_propagation_all_classes(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            set([raiser("__hash__", exc_name)()])

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_element_eq_propagation_on_hash_collision_all_classes(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class E:
            def __hash__(self):
                return 1

            def __eq__(self, other):
                raise exc("eq")

        with pytest.raises(exc):
            set([E(), E()])

    def test_eq_is_not_called_without_a_hash_collision(self):
        class E:
            def __init__(self, h):
                self.h = h

            def __hash__(self):
                return self.h

            def __eq__(self, other):
                raise AssertionError("eq called")

        assert len(set([E(1), E(2)])) == 2

    def test_iteration_order_of_str_elements_depends_on_the_hash_seed(self):
        code = "print(list(set(['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'])))"
        outs = set()
        for seed in range(8):
            env = {**os.environ, "PYTHONHASHSEED": str(seed)}
            outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                    check=True).stdout)
        assert len(outs) > 1
        effects = self.contract.effects
        assert effects.clock_or_env and not effects.deterministic

    def test_iteration_order_of_int_elements_does_not(self):
        code = "print(list(set([5, 3, 9, 1, 7, 100, 2])))"
        outs = set()
        for seed in range(4):
            env = {**os.environ, "PYTHONHASHSEED": str(seed)}
            outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                    check=True).stdout)
        assert len(outs) == 1

    def test_membership_len_and_equality_do_not_depend_on_the_seed(self):
        code = "s = set(['a', 'b', 'c']); print('a' in s, len(s), s == {'c', 'b', 'a'})"
        outs = set()
        for seed in range(4):
            env = {**os.environ, "PYTHONHASHSEED": str(seed)}
            outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                    check=True).stdout)
        assert outs == {"True 3 True\n"}


# ---------------------------------------------------------------------------
# zip / enumerate / reversed -- lazy: the call only builds the iterator
# ---------------------------------------------------------------------------


class TestZip:
    contract = get("zip")

    def test_contract_is_total_and_tested(self):
        _every_class_claim_is_covered(self.contract)
        assert self.contract.may_raise_other and self.contract.effects.may_execute_callback

    def test_no_arguments_is_an_empty_iterator(self):
        assert list(zip()) == []

    @pytest.mark.parametrize("args", [(1,), ([1], 2), (None,)])
    def test_non_iterable_is_typeerror_at_the_call(self, args):
        with pytest.raises(TypeError):
            zip(*args)

    def test_unknown_keyword_is_typeerror(self):
        with pytest.raises(TypeError):
            zip([1], foo=1)

    def test_iter_returning_a_non_iterator_is_typeerror(self):
        class Bad:
            def __iter__(self):
                return 5

        with pytest.raises(TypeError):
            zip(Bad())

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_iter_propagation_at_the_call_all_classes(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            zip([1], raiser("__iter__", exc_name)())

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_strict_value_bool_propagation_all_classes(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            zip([1], strict=raiser("__bool__", exc_name)())

    def test_the_call_consumes_nothing_and_next_runs_later(self):
        class It:
            def __iter__(self):
                return self

            def __next__(self):
                raise KeyError("next")

        z = zip(It())  # no exception: __next__ has not run
        with pytest.raises(KeyError):
            list(z)

    def test_a_getitem_protocol_argument_is_only_indexed_at_consumption(self):
        class Seq:
            def __getitem__(self, i):
                raise KeyError("getitem")

        z = zip(Seq())
        with pytest.raises(KeyError):
            list(z)

    def test_strict_valueerror_is_raised_at_consumption_not_at_the_call(self):
        z = zip([1], [1, 2], strict=True)
        with pytest.raises(ValueError):
            list(z)

    def test_any_strict_value_is_accepted(self):
        assert list(zip([1], strict=1)) == [(1,)]

    def test_return_is_a_lazy_tuple_iterator_stopping_at_the_shortest(self):
        z = zip([1, 2, 3], "ab")
        assert type(z) is zip and iter(z) is z
        assert list(z) == [(1, "a"), (2, "b")]

    def test_elements_are_passed_through_uninspected(self):
        s = _Sentinel()
        assert list(zip([s], [s]))[0][0] is s


class TestEnumerate:
    contract = get("enumerate")

    def test_contract_is_total_and_tested(self):
        _every_class_claim_is_covered(self.contract)
        assert self.contract.may_raise_other and self.contract.effects.may_execute_callback

    @pytest.mark.parametrize("args", [(), (1,), ([1], "a"), ([1], 1.5), ([1], None), ([1], 2, 3)])
    def test_native_typeerror(self, args):
        with pytest.raises(TypeError):
            enumerate(*args)

    def test_both_keywords_are_accepted(self):
        assert list(enumerate(iterable=[7])) == [(0, 7)]
        assert list(enumerate([7], start=2)) == [(2, 7)]

    def test_unknown_keyword_is_typeerror(self):
        with pytest.raises(TypeError):
            enumerate([1], foo=1)

    @given(st.integers())
    def test_start_is_any_int_unbounded(self, n):
        assert list(enumerate(["x"], n)) == [(n, "x")]

    def test_bool_start_is_an_int(self):
        assert list(enumerate([1], True)) == [(1, 1)]

    def test_int_subclass_start_ignores_its___index___override(self):
        class Evil(int):
            def __index__(self):
                raise KeyError("must not run")

        assert list(enumerate([1], Evil(3))) == [(3, 1)]

    def test_index_returning_non_int_is_typeerror(self):
        class Bad:
            def __index__(self):
                return "x"

        with pytest.raises(TypeError):
            enumerate([1], Bad())

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_iter_and_start_index_propagation_all_classes(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]
        with pytest.raises(exc):
            enumerate(raiser("__iter__", exc_name)())
        with pytest.raises(exc):
            enumerate([1], raiser("__index__", exc_name)())

    def test_the_call_consumes_nothing_and_next_runs_later(self):
        class It:
            def __iter__(self):
                return self

            def __next__(self):
                raise IndexError("next")

        e = enumerate(It())
        with pytest.raises(IndexError):
            list(e)

    def test_return_is_a_lazy_pair_iterator(self):
        e = enumerate("ab", 5)
        assert type(e) is enumerate and iter(e) is e
        assert list(e) == [(5, "a"), (6, "b")]

    def test_elements_are_passed_through_uninspected(self):
        s = _Sentinel()
        assert list(enumerate([s]))[0][1] is s


class TestReversed:
    contract = get("reversed")

    def test_contract_is_total_and_tested(self):
        _every_class_claim_is_covered(self.contract)
        assert self.contract.may_raise_other and self.contract.effects.may_execute_callback

    @pytest.mark.parametrize("bad", [1, None, set(), iter([]), (i for i in ()), frozenset()])
    def test_non_reversible_is_typeerror(self, bad):
        with pytest.raises(TypeError):
            reversed(bad)

    def test_arity_and_keyword_are_typeerror(self):
        with pytest.raises(TypeError):
            reversed()
        with pytest.raises(TypeError):
            reversed([1], [2])
        with pytest.raises(TypeError):
            reversed(seq=[1])

    def test_a_sequence_needs_both_len_and_getitem(self):
        class LenOnly:
            def __len__(self):
                return 2

        class GetOnly:
            def __getitem__(self, i):
                return i

        with pytest.raises(TypeError):
            reversed(LenOnly())
        with pytest.raises(TypeError):
            reversed(GetOnly())

    def test_reversed_none_disables_reversal(self):
        class No:
            __reversed__ = None

        with pytest.raises(TypeError):
            reversed(No())

    @pytest.mark.parametrize("src,expect", [
        ([1, 2, 3], [3, 2, 1]), ((1, 2), [2, 1]), ("ab", ["b", "a"]), (range(3), [2, 1, 0]),
        ({1: 0, 2: 0}, [2, 1]), (b"ab", [98, 97]),
    ])
    def test_builtin_sequences_dicts_and_ranges_reverse(self, src, expect):
        assert list(reversed(src)) == expect

    def test_dict_views_reverse(self):
        d = {1: "a", 2: "b"}
        assert list(reversed(d.keys())) == [2, 1]
        assert list(reversed(d.values())) == ["b", "a"]
        assert list(reversed(d.items())) == [(2, "b"), (1, "a")]

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_reversed_hook_propagation_all_classes(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            reversed(raiser("__reversed__", exc_name)())

    @pytest.mark.parametrize("exc_name", EXC_NAMES)
    def test_len_probe_propagation_at_the_call_all_classes(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class S:
            def __len__(self):
                raise exc("len")

            def __getitem__(self, i):
                return i

        with pytest.raises(exc):
            reversed(S())

    @pytest.mark.parametrize("exc_name", [n for n in EXC_NAMES if n != "IndexError"])
    def test_getitem_runs_at_consumption_not_at_the_call(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class S:
            def __len__(self):
                return 2

            def __getitem__(self, i):
                raise exc("getitem")

        r = reversed(S())  # no exception yet
        with pytest.raises(exc):
            list(r)

    def test_an_own___reversed___is_preferred_even_for_a_list_subclass(self):
        class L(list):
            def __reversed__(self):
                raise ZeroDivisionError("own")

        with pytest.raises(ZeroDivisionError):
            reversed(L([1]))

    def test_an_own___reversed___may_return_any_object(self):
        class R:
            def __reversed__(self):
                return 5

        assert reversed(R()) == 5

    def test_the_result_reads_elements_when_consumed_but_fixes_the_start_at_the_call(self):
        xs = [1, 2]
        r = reversed(xs)
        xs[0] = 99          # an element assignment after the call IS visible
        xs.append(3)        # an append is not: the start index was len(xs) - 1 at the call
        assert list(r) == [2, 99]

    def test_elements_are_passed_through_uninspected(self):
        s = _Sentinel()
        assert list(reversed([s]))[0] is s


# ---------------------------------------------------------------------------
# no dead entries (the batch's rule: every contract is consumed by an adapter)
# ---------------------------------------------------------------------------


def test_every_contract_of_this_batch_is_consumed_by_an_adapter():
    """Mechanical form of "no dead entries": each qualname must appear in at least one table or
    model an engine reads. The per-engine tests (tests/contracts/test_adapters_*_builtins.py,
    test_adapters_draws.py, test_adapters_avengers.py, tests/test_sv_prove.py) pin what each
    consumer DOES with it; this fails the moment a contract is added with none."""
    from tharkun.solve.absint.interp import Analyzer
    from tharkun.solve.contracts import get, rules_builtins_more
    from tharkun.solve.contracts.adapters import common
    from tharkun.solve.contracts.adapters.concolic import CONCRETE_FOLD_ARITY

    consumers = {
        "absint model (Analyzer.c_<name>)": {n[2:] for n in dir(Analyzer) if n.startswith("c_")},
        "concolic native fold": set(CONCRETE_FOLD_ARITY),
        "concolic/drawflow one-argument shape": set(common.NO_CALLBACK_SHAPES) | set(common.CLASS_ONLY_SHAPES),
        "draws/avengers constructor value": set(common.ZERO_ARGUMENT_CONSTRUCTORS),
    }
    qualnames = [q for name in rules_builtins_more.__all__ for q in getattr(rules_builtins_more, name).names]
    unconsumed = [q for q in qualnames if not any(q in names for names in consumers.values())]
    assert qualnames and all(get(q) is not None for q in qualnames)
    assert unconsumed == []
