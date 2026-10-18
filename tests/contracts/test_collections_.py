"""Tests backing every claim in ``tharkun.solve.contracts.rules_collections`` ([internal ref] batch 3)."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict, deque, namedtuple

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get
from tharkun.solve.contracts.safety import Binding, cases
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES
from tharkun.solve.terms import Refused


class TestDefaultdict:
    contract = get("collections.defaultdict")

    def test_constructor_contract_is_tested(self):
        assert self.contract.tested
        assert self.contract.qualname == "collections.defaultdict"
        assert self.contract.may_raise_other is True
        assert self.contract.effects.may_execute_callback is True

    def test_constructor_safety_cases_name_the_term_gap(self):
        binding = Binding(None, (), None)
        for exc in ALL_PROPERTY_CLASSES:
            result = cases("collections.defaultdict", exc, binding)
            assert isinstance(result, Refused), (exc, result)
            assert "M1 terms cannot represent" in result.reason, (exc, result)

    def test_list_factory_inserts_a_fresh_mutable_list(self):
        d = defaultdict(list)
        first = d["x"]
        first.append(1)
        assert d["x"] is first and d["x"] == [1]
        assert d["y"] is not first

    def test_none_factory_raises_keyerror_without_insertion(self):
        d = defaultdict()
        with pytest.raises(KeyError):
            d["x"]
        assert not d

    def test_initializer_and_keyword_mapping_are_supported_by_cpython(self):
        d = defaultdict(list, {"x": [1]}, y=2)
        assert d == {"x": [1], "y": 2}

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_initializer_iteration_exceptions_propagate(self, exc_name):
        exc_type = PROPERTY_CLASSES[exc_name]

        class BadIterable:
            def __iter__(self):
                raise exc_type("boom")

        with pytest.raises(exc_type):
            defaultdict(list, BadIterable())

    def test_invalid_factory_and_too_many_positional_arguments_raise_typeerror(self):
        with pytest.raises(TypeError):
            defaultdict(1)
        with pytest.raises(TypeError):
            defaultdict(list, {}, {})

    def test_factory_exceptions_propagate(self):
        class FactoryError(Exception):
            pass

        def factory():
            raise FactoryError

        with pytest.raises(FactoryError):
            defaultdict(factory)["x"]

    def test_is_a_dict_subclass_but_not_an_exact_dict(self):
        d = defaultdict(list)
        assert isinstance(d, dict)
        assert type(d) is not dict


class TestDequeAppend:
    def test_contracts_are_tested(self):
        assert get("collections.deque.append").tested
        assert get("collections.deque.appendleft").tested

    def test_basic(self):
        d = deque([1, 2])
        assert d.append(3) is None
        assert d == deque([1, 2, 3])
        assert d.appendleft(0) is None
        assert d == deque([0, 1, 2, 3])

    def test_maxlen_evicts_silently_on_append(self):
        d = deque([1, 2, 3], maxlen=3)
        d.append(4)
        assert list(d) == [2, 3, 4]  # 1 evicted from the left

    def test_maxlen_evicts_silently_on_appendleft(self):
        d = deque([1, 2, 3], maxlen=3)
        d.appendleft(0)
        assert list(d) == [0, 1, 2]  # 3 evicted from the right


class TestDequePop:
    contract = get("collections.deque.pop")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_pops_rightmost(self):
        d = deque([1, 2, 3])
        assert d.pop() == 3
        assert d == deque([1, 2])

    def test_empty_raises_indexerror(self):
        with pytest.raises(IndexError):
            deque().pop()


class TestDequePopleft:
    contract = get("collections.deque.popleft")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_pops_leftmost(self):
        d = deque([1, 2, 3])
        assert d.popleft() == 1
        assert d == deque([2, 3])

    def test_empty_raises_indexerror(self):
        with pytest.raises(IndexError):
            deque().popleft()


class TestDequeExtend:
    contract = get("collections.deque.extend")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        d = deque([1])
        d.extend([2, 3])
        assert d == deque([1, 2, 3])

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            deque().extend(5)

    def test_maxlen_evicts_during_extend(self):
        d = deque([1], maxlen=2)
        d.extend([2, 3, 4])
        assert list(d) == [3, 4]

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iteration_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            deque().extend(BadIter())

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: the argument's __iter__/__next__ is arbitrary user code -- it
        # can raise ANY exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_iteration_propagates_non_property_exception(self):
        class BadIter:
            def __iter__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            deque().extend(BadIter())


class TestDequeRotate:
    contract = get("collections.deque.rotate")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_rotate_right(self):
        d = deque([1, 2, 3])
        d.rotate(1)
        assert d == deque([3, 1, 2])

    def test_rotate_left(self):
        d = deque([1, 2, 3])
        d.rotate(-1)
        assert d == deque([2, 3, 1])

    def test_default_is_one(self):
        d = deque([1, 2, 3])
        d.rotate()
        assert d == deque([3, 1, 2])

    def test_non_index_raises_typeerror(self):
        with pytest.raises(TypeError):
            deque([1, 2]).rotate("x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): n's __index__ propagates any exception
        # it raises, including IndexError -- an earlier draft claimed "never".
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            deque([1, 2]).rotate(Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_index_propagates_non_property_exception(self):
        class BadIndex:
            def __index__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            deque([1, 2]).rotate(BadIndex())


class TestDefaultdictGetitem:
    contract = get("collections.defaultdict.__getitem__")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_missing_key_calls_factory_and_inserts(self):
        dd = defaultdict(list)
        dd["x"].append(1)
        assert dict(dd) == {"x": [1]}

    def test_no_factory_raises_keyerror(self):
        dd = defaultdict(None)
        with pytest.raises(KeyError):
            dd["missing"]

    def test_noncallable_factory_rejected_at_construction(self):
        with pytest.raises(TypeError):
            defaultdict(5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_factory_propagation(self, exc_name):
        def bad_factory():
            raise PROPERTY_CLASSES[exc_name]("boom")

        dd = defaultdict(bad_factory)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            dd["missing"]

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: default_factory is arbitrary user code -- it can raise ANY
        # exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_factory_propagates_non_property_exception(self):
        def bad_factory():
            raise ValueError("boom")

        dd = defaultdict(bad_factory)
        with pytest.raises(ValueError):
            dd["missing"]


class TestCounterGetitem:
    contract = get("collections.Counter.__getitem__")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_present_key(self):
        c = Counter("abca")
        assert c["a"] == 2

    def test_missing_key_returns_zero_never_raises(self):
        c = Counter("abc")
        assert c["z"] == 0

    def test_missing_key_does_not_insert(self):
        c = Counter("abc")
        c["z"]
        assert "z" not in c

    def test_unhashable_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            Counter("abc")[[]]

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): looking up whether a key is present
        # still hashes it, even though a miss returns 0 rather than raising --
        # a hostile __hash__ propagates just like dict.__getitem__'s does.
        Bad = raiser("__hash__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            Counter("abc")[Bad()]

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: the key's __hash__/__eq__ is arbitrary user code -- it can
        # raise ANY exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_hash_propagates_non_property_exception(self):
        class BadHash:
            def __hash__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            Counter("abc")[BadHash()]


class TestCounterMostCommon:
    contract = get("collections.Counter.most_common")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_sorted_descending(self):
        c = Counter("abca")
        assert c.most_common() == [("a", 2), ("b", 1), ("c", 1)]

    def test_n_limits_results(self):
        c = Counter("abca")
        assert c.most_common(1) == [("a", 2)]

    def test_bad_n_raises_typeerror(self):
        with pytest.raises(TypeError):
            Counter("abc").most_common("x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_n_comparison_propagation(self, exc_name):
        # REVIEW FIX: n is compared (== 1, then >= len(self), via heapq.nlargest)
        # before any native TypeError check fires -- a hostile __eq__ propagates
        # whatever it raises. Confirmed empirically (not just claimed): __eq__ is
        # the first comparison heapq.nlargest's Python implementation reaches.
        class BadN:
            def __eq__(self, other):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            Counter("abca").most_common(BadN())

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: n's comparison (via heapq.nlargest) is arbitrary user code --
        # it can raise ANY exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_n_comparison_propagates_non_property_exception(self):
        class BadN:
            def __eq__(self, other):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            Counter("abca").most_common(BadN())


class TestOrderedDictMoveToEnd:
    contract = get("collections.OrderedDict.move_to_end")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_default_moves_to_end(self):
        od = OrderedDict(a=1, b=2, c=3)
        od.move_to_end("a")
        assert list(od) == ["b", "c", "a"]

    def test_last_false_moves_to_beginning(self):
        od = OrderedDict(a=1, b=2, c=3)
        od.move_to_end("c", last=False)
        assert list(od) == ["c", "a", "b"]

    def test_missing_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            OrderedDict(a=1).move_to_end("nope")

    def test_unhashable_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            OrderedDict(a=1).move_to_end([])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): an earlier draft claimed only KeyError
        # could happen, missing that the lookup hashes the key first.
        Bad = raiser("__hash__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            OrderedDict(a=1).move_to_end(Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_hash_propagates_non_property_exception(self):
        class BadHash:
            def __hash__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            OrderedDict(a=1).move_to_end(BadHash())


class TestNamedtuple:
    contract = get("collections.namedtuple")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_creates_a_tuple_subclass(self):
        Pt = namedtuple("Pt", ["x", "y"])
        assert issubclass(Pt, tuple)
        p = Pt(1, 2)
        assert p.x == 1 and p.y == 2
        assert p == (1, 2)

    def test_bad_identifier_field_raises_valueerror(self):
        with pytest.raises(ValueError):
            namedtuple("Pt", ["x", "1invalid"])

    def test_duplicate_field_raises_valueerror(self):
        with pytest.raises(ValueError):
            namedtuple("Pt", ["x", "x"])

    def test_wrong_arg_count_raises_typeerror(self):
        Pt = namedtuple("Pt", ["x", "y"])
        with pytest.raises(TypeError):
            Pt(1)
        with pytest.raises(TypeError):
            Pt(1, 2, 3)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_field_names_iteration_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): field_names, when a custom iterable
        # (not the space/comma-separated str shorthand), is iterated with the
        # object's own __iter__ -- a hostile one propagates unchanged.
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            namedtuple("Pt", BadIter())

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True


class TestNamedtupleReplace:
    contract = get("collections.namedtuple._replace")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_replace_returns_new_instance(self):
        Pt = namedtuple("Pt", ["x", "y"])
        p = Pt(1, 2)
        p2 = p._replace(y=99)
        assert p2 == (1, 99)
        assert p == (1, 2)  # original unchanged (immutable)

    def test_unknown_field_raises_typeerror(self):
        Pt = namedtuple("Pt", ["x", "y"])
        p = Pt(1, 2)
        with pytest.raises(TypeError):
            p._replace(z=5)
