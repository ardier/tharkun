"""Tests backing every claim in ``tharkun.solve.contracts.rules_set`` ([internal ref])."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


def _hash_raiser(exc_name):
    return raiser("__hash__", exc_name)


class TestAdd:
    contract = get("set.add")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.sets(st.integers()), st.integers())
    def test_adds_element_returns_none(self, s, x):
        assert s.add(x) is None
        assert x in s

    def test_unhashable_raises_typeerror(self):
        with pytest.raises(TypeError):
            {1, 2}.add([])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {1, 2}.add(_hash_raiser(exc_name)())


class TestRemove:
    contract = get("set.remove")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_removes_present_element(self):
        s = {1, 2}
        assert s.remove(1) is None
        assert s == {2}

    def test_missing_element_raises_keyerror(self):
        with pytest.raises(KeyError):
            {1, 2}.remove(5)

    def test_unhashable_raises_typeerror(self):
        with pytest.raises(TypeError):
            {1, 2}.remove([])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {1, 2}.remove(_hash_raiser(exc_name)())


class TestDiscard:
    contract = get("set.discard")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_removes_present_element(self):
        s = {1, 2}
        assert s.discard(1) is None
        assert s == {2}

    def test_missing_element_does_not_raise(self):
        s = {1, 2}
        assert s.discard(5) is None
        assert s == {1, 2}

    def test_unhashable_raises_typeerror(self):
        with pytest.raises(TypeError):
            {1, 2}.discard([])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {1, 2}.discard(_hash_raiser(exc_name)())


class TestPop:
    contract = get("set.pop")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_pops_a_member(self):
        s = {1, 2, 3}
        x = s.pop()
        assert x in {1, 2, 3}
        assert x not in s
        assert len(s) == 2

    def test_empty_raises_keyerror(self):
        with pytest.raises(KeyError):
            set().pop()


class TestUnion:
    contract = get("set.union")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_does_not_mutate_receiver(self):
        s = {1, 2}
        result = s.union([3, 4])
        assert result == {1, 2, 3, 4}
        assert s == {1, 2}

    def test_accepts_any_iterable(self):
        assert {1, 2}.union((3, 4)) == {1, 2, 3, 4}

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            {1, 2}.union(5)

    def test_unhashable_element_raises_typeerror(self):
        with pytest.raises(TypeError):
            {1, 2}.union([[]])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iteration_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {1, 2}.union(BadIter())


class TestIntersection:
    contract = get("set.intersection")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_does_not_mutate_receiver(self):
        s = {1, 2, 3}
        result = s.intersection([2, 3, 4])
        assert result == {2, 3}
        assert s == {1, 2, 3}

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            {1, 2}.intersection(5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iteration_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {1, 2}.intersection(BadIter())


class TestUpdate:
    contract = get("set.update")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_mutates_receiver(self):
        s = {1, 2}
        assert s.update([3, 4]) is None
        assert s == {1, 2, 3, 4}

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            {1, 2}.update(5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iteration_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {1, 2}.update(BadIter())


class TestClearCopy:
    def test_contracts_are_tested(self):
        assert get("set.clear").tested
        assert get("set.copy").tested

    def test_clear_empties(self):
        s = {1, 2}
        assert s.clear() is None
        assert s == set()

    def test_copy_is_independent(self):
        s = {1, 2}
        s2 = s.copy()
        assert s2 == s
        assert s2 is not s
        s2.add(3)
        assert s == {1, 2}


# --------------------------------------------------------------------------------------------------
# The rest of the family ([internal ref]): difference, symmetric_difference, issubset, issuperset, isdisjoint, the
# three *_update methods, the operators | & - ^ and their in-place forms, the comparisons, __contains__ and
# the frozenset constructor.  Every claim below is run on CPython; the non-mutating members run on BOTH types
# (a frozenset shares the C function), the mutating ones raise AttributeError on a frozenset.
# --------------------------------------------------------------------------------------------------
import operator

from tharkun.solve.contracts.registry import REGISTRY

#: method -> (accepts several arguments, mutates the receiver)
_ITER_METHODS = {
    "difference": (True, False),
    "symmetric_difference": (False, False),
    "issubset": (False, False),
    "issuperset": (False, False),
    "isdisjoint": (False, False),
    "difference_update": (True, True),
    "intersection_update": (True, True),
    "symmetric_difference_update": (False, True),
}
_NON_MUTATING = sorted(m for m, (_many, mut) in _ITER_METHODS.items() if not mut)
_MUTATING = sorted(m for m, (_many, mut) in _ITER_METHODS.items() if mut)
_SINGLE_ARG = sorted(m for m, (many, _mut) in _ITER_METHODS.items() if not many)
_SET_TYPES = [set, frozenset]
_OPERATORS = {"or": operator.or_, "and": operator.and_, "sub": operator.sub, "xor": operator.xor}
_INPLACE = {"ior": operator.ior, "iand": operator.iand, "isub": operator.isub, "ixor": operator.ixor}
_COMPARISONS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}


def test_the_whole_family_is_registered_and_tested():
    names = (
        [f"set.{m}" for m in _ITER_METHODS]
        + [f"set.__{o}__" for o in (*_OPERATORS, *_INPLACE, *_COMPARISONS, "eq", "ne", "contains")]
        + ["frozenset"]
    )
    for q in names:
        c = get(q)
        assert c is not None and c.tested, q


def _colliding(exc_name):
    """A class whose instances all hash alike and whose ``__eq__`` raises ``exc_name``."""

    class Hostile:
        def __hash__(self):
            return 1

        def __eq__(self, other):
            raise PROPERTY_CLASSES[exc_name]("boom-eq")

    return Hostile


class _BadIter:
    def __init__(self, exc_name):
        self.exc = PROPERTY_CLASSES[exc_name]

    def __iter__(self):
        raise self.exc("boom-iter")


@pytest.mark.parametrize("name", sorted(_ITER_METHODS))
class TestIterableMethods:
    def test_a_non_iterable_argument_raises_typeerror(self, name):
        with pytest.raises(TypeError):
            getattr({1, 2}, name)(5)

    def test_an_unhashable_element_raises_typeerror_and_a_set_element_is_not_converted(self, name):
        with pytest.raises(TypeError):
            getattr({1, 2}, name)([[1]])
        with pytest.raises(TypeError):
            getattr({1, 2}, name)([{1}])

    def test_every_other_iterable_is_accepted(self, name):
        for arg in ([1, 3], (1, 3), "ab", {1: 2}, iter([1, 3]), range(3), frozenset({1}), {1, 5}):
            getattr({1, 2}, name)(arg)

    def test_a_keyword_argument_raises_typeerror(self, name):
        with pytest.raises(TypeError):
            getattr({1, 2}, name)(other=[1])

    def test_the_argument_count(self, name):
        many, _mut = _ITER_METHODS[name]
        if many:
            getattr({1, 2}, name)()                      # no argument is allowed
            getattr({1, 2}, name)([1], [2])               # and so are several
        else:
            with pytest.raises(TypeError):
                getattr({1, 2}, name)()
            with pytest.raises(TypeError):
                getattr({1, 2}, name)([1], [2])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_a_hostile_iterator_propagates(self, name, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            getattr({1, 2}, name)(_BadIter(exc_name))

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_a_hostile_hash_propagates(self, name, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            getattr({1, 2}, name)([_hash_raiser(exc_name)()])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_a_hostile_eq_on_a_collision_propagates(self, name, exc_name):
        hostile = _colliding(exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            getattr({hostile()}, name)([hostile()])


@pytest.mark.parametrize("name", _NON_MUTATING)
class TestNonMutatingMethods:
    def test_neither_operand_changes_and_a_frozenset_has_the_method(self, name):
        for typ in _SET_TYPES:
            s, arg = typ({1, 2}), [2, 3]
            getattr(s, name)(arg)
            assert s == typ({1, 2}) and arg == [2, 3]


@pytest.mark.parametrize("name", _MUTATING)
class TestMutatingMethods:
    def test_a_frozenset_has_no_such_method(self, name):
        with pytest.raises(AttributeError):
            getattr(frozenset({1}), name)([1])

    def test_returns_none_and_mutates_the_receiver(self, name):
        s = {1, 2, 3}
        assert getattr(s, name)([2, 9]) is None
        assert s != {1, 2, 3}


class TestResults:
    def test_difference(self):
        s = {1, 2, 3}
        assert s.difference([2], (3,)) == {1} and s == {1, 2, 3}
        assert s.difference() == s and s.difference() is not s

    def test_symmetric_difference(self):
        assert {1, 2}.symmetric_difference([2, 3]) == {1, 3}

    def test_predicates(self):
        assert {1}.issubset([1, 2]) and not {1, 5}.issubset([1, 2])
        assert {1, 2}.issuperset([1]) and not {1}.issuperset([1, 2])
        assert {1}.isdisjoint([2]) and not {1}.isdisjoint([1, 2])
        assert set().issubset([]) and set().isdisjoint([]) and {1}.issuperset([])

    def test_update_methods(self):
        s = {1, 2, 3}
        s.difference_update([1], [2])
        assert s == {3}
        s = {1, 2, 3}
        s.intersection_update([2, 3], [3, 4])
        assert s == {3}
        s = {1, 2}
        s.symmetric_difference_update([2, 3])
        assert s == {1, 3}

    def test_the_receiver_as_its_own_argument(self):
        for name, expected in (("update", {1, 2}), ("difference_update", set()), ("intersection_update", {1, 2}),
                               ("symmetric_difference_update", set())):
            s = {1, 2}
            getattr(s, name)(s)
            assert s == expected, name

    def test_the_retained_element_of_equal_values_comes_from_a_known_side(self):
        # the element a result keeps is the receiver's for union/difference, the argument's for intersection
        assert type(next(iter({1} | {1.0}))) is int
        assert type(next(iter({1}.intersection({1.0})))) is float
        assert type(next(iter({1, 2}.difference({1.0})))) is int
        assert type(next(iter({1}.intersection([1.0])))) is float


@pytest.mark.parametrize("name", sorted(_OPERATORS))
class TestBinaryOperators:
    def test_two_sets_or_frozensets(self, name):
        op = _OPERATORS[name]
        for left in _SET_TYPES:
            for right in _SET_TYPES:
                result = op(left({1, 2}), right({2, 3}))
                assert type(result) is left            # the LEFT operand's type
        assert op({1, 2}, {2, 3}) == {"or": {1, 2, 3}, "and": {2}, "sub": {1}, "xor": {1, 3}}[name]

    def test_neither_operand_changes(self, name):
        a, b = {1, 2}, {2, 3}
        _ = _OPERATORS[name](a, b)
        assert a == {1, 2} and b == {2, 3}

    @pytest.mark.parametrize("other", [[1], (1,), 5, None, "a", {1: 2}, 1.5])
    def test_anything_else_is_a_typeerror_on_either_side(self, name, other):
        op = _OPERATORS[name]
        for typ in _SET_TYPES:
            with pytest.raises(TypeError):
                op(typ({1}), other)
            with pytest.raises(TypeError):
                op(other, typ({1}))

    def test_a_dict_keys_view_implements_the_reflected_operator(self, name):
        op = _OPERATORS[name]
        op({1}, {1: 2}.keys())
        op({1: 2}.keys(), {1})

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_a_hostile_eq_on_a_collision_propagates(self, name, exc_name):
        hostile = _colliding(exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            _OPERATORS[name]({hostile()}, {hostile()})


@pytest.mark.parametrize("name", sorted(_INPLACE))
class TestInplaceOperators:
    @staticmethod
    def _run(name, left, right):
        ns = {"a": left, "b": right, "keep": left}
        exec(f"a {dict(ior='|', iand='&', isub='-', ixor='^')[name]}= b", ns)
        return ns["a"], ns["keep"]

    def test_a_set_is_changed_in_place_and_returned(self, name):
        result, keep = self._run(name, {1, 2}, {2, 3})
        assert result is keep
        assert result == {"ior": {1, 2, 3}, "iand": {2}, "isub": {1}, "ixor": {1, 3}}[name]

    def test_a_set_accepts_a_frozenset_operand(self, name):
        result, keep = self._run(name, {1, 2}, frozenset({2, 3}))
        assert result is keep

    def test_a_frozenset_has_no_in_place_form_and_is_rebound_to_a_new_object(self, name):
        result, keep = self._run(name, frozenset({1, 2}), {2, 3})
        assert type(result) is frozenset and result is not keep and keep == frozenset({1, 2})

    @pytest.mark.parametrize("other", [[1], (1,), 5, None, {1: 2}])
    def test_anything_else_is_a_typeerror(self, name, other):
        with pytest.raises(TypeError):
            self._run(name, {1, 2}, other)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_a_hostile_eq_on_a_collision_propagates(self, name, exc_name):
        hostile = _colliding(exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            self._run(name, {hostile()}, {hostile()})

    def test_the_receiver_as_its_own_operand(self, name):
        s = {1, 2}
        result, _keep = self._run(name, s, s)
        assert result == {"ior": {1, 2}, "iand": {1, 2}, "isub": set(), "ixor": set()}[name]


@pytest.mark.parametrize("name", sorted(_COMPARISONS))
class TestOrderingComparisons:
    def test_subset_semantics_between_sets_and_frozensets(self, name):
        op = _COMPARISONS[name]
        for left in _SET_TYPES:
            for right in _SET_TYPES:
                assert isinstance(op(left({1}), right({1, 2})), bool)
        small, big = {1}, {1, 2}
        assert {"lt": True, "le": True, "gt": False, "ge": False}[name] == _COMPARISONS[name](small, big)
        assert {"lt": False, "le": True, "gt": False, "ge": True}[name] == _COMPARISONS[name](big.copy(), big)

    @pytest.mark.parametrize("other", [[1], (1,), 5, None, "a", {1: 2}, 1.5])
    def test_anything_else_is_a_typeerror_on_either_side(self, name, other):
        op = _COMPARISONS[name]
        with pytest.raises(TypeError):
            op({1}, other)
        with pytest.raises(TypeError):
            op(other, {1})

    def test_a_dict_keys_view_implements_the_reflected_comparison(self, name):
        _COMPARISONS[name]({1}, {1: 2}.keys())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_a_hostile_eq_on_a_collision_propagates(self, name, exc_name):
        hostile = _colliding(exc_name)
        small, big = {hostile()}, {hostile(), 5}
        a, b = (small, big) if name in ("lt", "le") else (big, small)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            _COMPARISONS[name](a, b)


class TestEquality:
    @pytest.mark.parametrize("other", [[1], (1,), 5, None, "a", {1: 2}, 1.5, object()])
    def test_a_non_set_is_never_equal_and_never_an_error(self, other):
        assert ({1} == other) is False and ({1} != other) is True
        assert (other == {1}) is False and (other != {1}) is True

    def test_a_set_equals_a_frozenset_with_the_same_elements(self):
        assert {1, 2} == frozenset({1, 2}) and frozenset({1}) != {1, 2}

    def test_different_lengths_are_unequal_without_comparing_elements(self):
        hostile = _colliding("TypeError")
        assert ({hostile()} == {hostile(), 5}) is False

    def test_nan_is_equal_to_itself_only_as_the_same_object(self):
        nan = float("nan")
        assert {nan} == {nan}
        assert {nan} != {float("nan")}
        assert (0.0 in {-0.0}) and ({0.0} == {-0.0})

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    @pytest.mark.parametrize("op", [operator.eq, operator.ne])
    def test_a_hostile_eq_on_a_collision_propagates(self, op, exc_name):
        hostile = _colliding(exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            op({hostile()}, {hostile()})


class TestContains:
    def test_an_unhashable_element_is_a_typeerror(self):
        for x in ([], {}, [[]], bytearray(b"a")):
            with pytest.raises(TypeError):
                _ = x in {1}

    def test_a_set_element_is_converted_to_a_frozenset(self):
        assert ({1} in {frozenset({1})}) is True
        assert ({2} in {frozenset({1})}) is False

    def test_membership_is_by_hash_then_equality(self):
        assert 1.0 in {1} and True in {1} and ("a" not in {1})
        assert (3 in {1, 2}) is False and (3 not in {1, 2}) is True

    def test_nan_is_a_member_only_as_the_same_object(self):
        nan = float("nan")
        assert nan in {nan}
        assert float("nan") not in {nan}

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_a_hostile_hash_propagates(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            _ = _hash_raiser(exc_name)() in {1}

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_a_hostile_eq_on_a_collision_propagates(self, exc_name):
        hostile = _colliding(exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            _ = hostile() in {hostile()}


class TestSetKeyConversion:
    """Only remove, discard and `in` convert an unhashable SET to a frozenset; no other method does."""

    def test_remove_and_discard_convert(self):
        s = {frozenset({1}), 5}
        s.discard({1})
        assert s == {5}
        s = {frozenset({1}), 5}
        s.remove({1})
        assert s == {5}
        with pytest.raises(KeyError):
            {5}.remove({1})
        {5}.discard({1})

    def test_no_other_element_taking_method_converts(self):
        for call in (lambda: {5}.add({1}), lambda: {5}.union([{1}]), lambda: {5}.update([{1}]),
                     lambda: {5}.difference([{1}]), lambda: {5}.issubset([{1}])):
            with pytest.raises(TypeError):
                call()


class TestFrozensetConstructor:
    contract = get("frozenset")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_builds_the_distinct_elements(self):
        assert frozenset([1, 1.0, True, 2]) == frozenset({1, 2})
        assert frozenset() == frozenset([])
        assert type(frozenset("abc")) is frozenset

    def test_an_exact_frozenset_comes_back_as_the_same_object(self):
        f = frozenset({1})
        assert frozenset(f) is f
        assert frozenset({1}) is not f

    @pytest.mark.parametrize("call", [lambda: frozenset(1), lambda: frozenset(None), lambda: frozenset([[1]]),
                                      lambda: frozenset([{1}]), lambda: frozenset([1], [2]),
                                      lambda: frozenset(iterable=[1])])
    def test_typeerror(self, call):
        with pytest.raises(TypeError):
            call()

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hostile_iterator_and_hash_propagate(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            frozenset(_BadIter(exc_name))
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            frozenset([_hash_raiser(exc_name)()])

    def test_a_hostile_eq_on_a_collision_propagates(self):
        hostile = _colliding("AssertionError")
        with pytest.raises(AssertionError):
            frozenset([hostile(), hostile()])


class TestMethodsOfFrozenset:
    """A frozenset has exactly the non-mutating methods (the contracts of `set.<m>` are shared)."""

    def test_the_method_names(self):
        non_dunder = lambda t: {n for n in dir(t) if not n.startswith("__")}
        assert non_dunder(frozenset) == {"copy", "difference", "intersection", "isdisjoint", "issubset",
                                         "issuperset", "symmetric_difference", "union"}
        assert non_dunder(set) == non_dunder(frozenset) | {
            "add", "clear", "difference_update", "discard", "intersection_update", "pop", "remove",
            "symmetric_difference_update", "update"}

    def test_a_frozenset_copy_is_the_same_object_and_a_set_copy_is_not(self):
        f, s = frozenset({1}), {1}
        assert f.copy() is f and s.copy() is not s
