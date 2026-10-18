"""Tests backing every claim in ``tharkun.solve.contracts.rules_protocols`` ([internal ref] batch 2).

Every ordering/dispatch claim is re-derived here independently of the CPython source
reading in ``protocols.py`` -- via instrumented classes and a call-order log -- so a
misreading of the C source and a test typo are unlikely to agree by coincidence.
"""

from __future__ import annotations

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES, nonproperty_raiser, raiser
from tharkun.solve.contracts import get
from tharkun.solve.contracts.rules import get_rule

_ALL_PROTOCOL_QUALNAMES = [
    "protocol.binary_operator",
    "protocol.unary_operator",
    "protocol.inplace_operator",
    "protocol.comparison",
    "protocol.call",
    "protocol.attribute_access",
    "protocol.container_len",
    "protocol.length_hint",
    "protocol.container_getitem",
    "protocol.contains_fallback",
    "protocol.iteration",
    "protocol.hashing",
    "protocol.truthiness",
    "protocol.numeric_conversion",
    "protocol.context_manager",
]


@pytest.mark.parametrize("qualname", _ALL_PROTOCOL_QUALNAMES)
def test_every_protocol_contract_flags_may_execute_callback(qualname):
    # Review finding: every protocol contract dispatches to an arbitrary user dunder,
    # so effects.may_execute_callback must be True -- the other Effects fields
    # (deterministic, io, mutates_receiver, ...) describe only the native dispatch
    # mechanism, never the whole operation, whenever this is True.
    contract = get(qualname)
    assert contract.effects.may_execute_callback is True


def test_env_dependent_add_breaks_the_contracts_own_determinism_claim():
    # The exact review repro: an __add__ that reads os.environ makes the whole
    # operation's result depend on process environment, even though
    # protocol.binary_operator's own dispatch effects (deterministic=True, io=False,
    # clock_or_env=False) describe none of that -- which is precisely why
    # may_execute_callback=True gates those fields to "native mechanism only".
    import os

    class C:
        def __add__(self, other):
            return os.environ["THARKUN_REVIEW_VALUE"]

    x = C()
    old = os.environ.get("THARKUN_REVIEW_VALUE")
    try:
        os.environ["THARKUN_REVIEW_VALUE"] = "first"
        assert x + 1 == "first"
        os.environ["THARKUN_REVIEW_VALUE"] = "second"
        assert x + 1 == "second"
    finally:
        if old is None:
            os.environ.pop("THARKUN_REVIEW_VALUE", None)
        else:
            os.environ["THARKUN_REVIEW_VALUE"] = old

    effects = get("protocol.binary_operator").effects
    assert effects.may_execute_callback is True
    assert effects.deterministic is True  # true of the DISPATCH mechanism only


def test_call_protocol_does_not_claim_callback_determinism():
    effects = get("protocol.call").effects
    assert effects.may_execute_callback is True
    assert effects.deterministic is False


# ---------------------------------------------------------------------------
# protocol.binary_operator
# ---------------------------------------------------------------------------


class TestBinaryOperator:
    contract = get("protocol.binary_operator")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_same_type_tries_add_once_no_radd_retry(self):
        log = []

        class A:
            def __add__(self, other):
                log.append("add")
                return NotImplemented

            def __radd__(self, other):
                log.append("radd")
                return "should not be called"

        with pytest.raises(TypeError):
            A() + A()
        assert log == ["add"]

    def test_unrelated_types_left_then_right(self):
        log = []

        class A:
            def __add__(self, other):
                log.append("A.add")
                return NotImplemented

        class B:
            def __radd__(self, other):
                log.append("B.radd")
                return "B.radd-result"

        assert A() + B() == "B.radd-result"
        assert log == ["A.add", "B.radd"]

    def test_both_notimplemented_raises_typeerror(self):
        class A:
            def __add__(self, other):
                return NotImplemented

        class B:
            def __radd__(self, other):
                return NotImplemented

        with pytest.raises(TypeError):
            A() + B()

    def test_subclass_overriding_reflected_op_goes_first_and_skips_left(self):
        log = []

        class A:
            def __add__(self, other):
                log.append("A.add")
                return "A.add-result"

        class Sub(A):
            def __radd__(self, other):
                log.append("Sub.radd")
                return "Sub.radd-result"

        assert A() + Sub() == "Sub.radd-result"
        assert log == ["Sub.radd"]  # A.add never even tried

    def test_subclass_not_overriding_uses_normal_order(self):
        log = []

        class A:
            def __add__(self, other):
                log.append("A.add")
                return NotImplemented  # forces a fall-through to __radd__

            def __radd__(self, other):
                log.append("A.radd")
                return "A.radd-result"

        class SubNoOverride(A):
            pass

        # SubNoOverride inherits A's __radd__ UNCHANGED (same slot pointer), so this
        # is NOT the subclass-priority case -- normal left-then-right order applies.
        assert A() + SubNoOverride() == "A.radd-result"
        assert log == ["A.add", "A.radd"]

    def test_subclass_priority_falls_back_when_reflected_notimplemented(self):
        log = []

        class A:
            def __add__(self, other):
                log.append("A.add")
                return "A.add-result"

            def __radd__(self, other):
                log.append("A.radd")
                return NotImplemented

        class Sub(A):
            def __radd__(self, other):
                log.append("Sub.radd")
                return NotImplemented

        assert A() + Sub() == "A.add-result"
        assert log == ["Sub.radd", "A.add"]

    def test_no_protocol_raises_typeerror(self):
        with pytest.raises(TypeError):
            object() + object()

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_add_dunder_propagation(self, exc_name):
        Bad = raiser("__add__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            Bad() + 1

    def test_may_raise_other_is_true(self):
        # __add__/__radd__ is arbitrary user code -- it can raise ANY exception, not
        # just one of the 7 property classes. Declaring may_raise_other=False here
        # would be a false closed-world claim.
        assert self.contract.may_raise_other is True

    def test_add_dunder_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__add__", ValueError, nargs=1)
        with pytest.raises(ValueError):
            Bad() + 1


# ---------------------------------------------------------------------------
# protocol.unary_operator
# ---------------------------------------------------------------------------


class TestUnaryOperator:
    contract = get("protocol.unary_operator")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_neg_pos_invert(self):
        class Ops:
            def __neg__(self):
                return "neg"

            def __pos__(self):
                return "pos"

            def __invert__(self):
                return "invert"

        o = Ops()
        assert -o == "neg"
        assert +o == "pos"
        assert ~o == "invert"

    def test_no_dunder_raises_typeerror(self):
        with pytest.raises(TypeError):
            -object()
        with pytest.raises(TypeError):
            +object()
        with pytest.raises(TypeError):
            ~object()

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_neg_propagation(self, exc_name):
        Bad = raiser("__neg__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            -Bad()

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_neg_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__neg__", ValueError)
        with pytest.raises(ValueError):
            -Bad()


# ---------------------------------------------------------------------------
# protocol.inplace_operator
# ---------------------------------------------------------------------------


class TestInplaceOperator:
    contract = get("protocol.inplace_operator")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_uses_iadd_when_present(self):
        log = []

        class HasIadd:
            def __iadd__(self, other):
                log.append("iadd")
                return "iadd-result"

        x = HasIadd()
        x += 1
        assert x == "iadd-result"
        assert log == ["iadd"]

    def test_falls_back_to_add_when_iadd_absent(self):
        log = []

        class NoIadd:
            def __add__(self, other):
                log.append("add")
                return "add-result"

        y = NoIadd()
        y += 1
        assert y == "add-result"
        assert log == ["add"]

    def test_falls_back_to_add_when_iadd_returns_notimplemented(self):
        log = []

        class IaddNI:
            def __iadd__(self, other):
                log.append("iadd")
                return NotImplemented

            def __add__(self, other):
                log.append("add")
                return "add-after-notimplemented"

        z = IaddNI()
        z += 1
        assert z == "add-after-notimplemented"
        assert log == ["iadd", "add"]

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iadd_propagation(self, exc_name):
        Bad = raiser("__iadd__", exc_name, nargs=1)
        b = Bad()
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            b += 1

    def test_no_iadd_and_no_binary_support_raises_typeerror(self):
        with pytest.raises(TypeError):
            x = object()
            x += 1

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_iadd_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__iadd__", ValueError, nargs=1)
        b = Bad()
        with pytest.raises(ValueError):
            b += 1


# ---------------------------------------------------------------------------
# protocol.comparison
# ---------------------------------------------------------------------------


class TestComparison:
    contract = get("protocol.comparison")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_default_eq_is_identity(self):
        class Plain:
            pass

        a, b = Plain(), Plain()
        assert (a == b) is False
        assert (a == a) is True
        assert (a != b) is True

    def test_default_ordering_raises_typeerror(self):
        class Plain:
            pass

        with pytest.raises(TypeError):
            Plain() < Plain()

    def test_lt_reflects_to_gt(self):
        log = []

        class Lt:
            def __lt__(self, other):
                log.append("lt")
                return NotImplemented

        class Gt:
            def __gt__(self, other):
                log.append("gt")
                return True

        assert (Lt() < Gt()) is True
        assert log == ["lt", "gt"]

    def test_subclass_priority_is_unconditional(self):
        log = []

        class Lt:
            def __lt__(self, other):
                log.append("Lt.lt")
                return False

        class LtSub(Lt):
            def __gt__(self, other):
                log.append("LtSub.gt")
                return "priority-result"

        assert (Lt() < LtSub()) == "priority-result"
        assert log == ["LtSub.gt"]  # Lt.lt never tried

    def test_eq_never_raises_typeerror_for_unsupported(self):
        class Plain:
            pass

        # Even wildly mismatched types compare False via identity, never TypeError.
        assert (Plain() == 5) is False
        assert (5 == Plain()) is False

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lt_propagation(self, exc_name):
        Bad = raiser("__lt__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            Bad() < 1

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_lt_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__lt__", ValueError, nargs=1)
        with pytest.raises(ValueError):
            Bad() < 1


# ---------------------------------------------------------------------------
# protocol.attribute_access
# ---------------------------------------------------------------------------


class TestAttributeAccess:
    contract = get("protocol.attribute_access")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_getattr_only_called_on_lookup_failure(self):
        log = []

        class WithGetattr:
            x = 5

            def __getattr__(self, name):
                log.append(name)
                if name == "missing":
                    return "fallback"
                raise AttributeError(name)

        o = WithGetattr()
        assert o.x == 5
        assert log == []  # __getattr__ not invoked for a normal hit
        assert o.missing == "fallback"
        assert log == ["missing"]

    def test_missing_attribute_raises_attributeerror(self):
        class Plain:
            pass

        with pytest.raises(AttributeError):
            Plain().nope

    def test_data_descriptor_beats_instance_dict(self):
        class DataDesc:
            def __get__(self, obj, owner):
                return "from-descriptor"

            def __set__(self, obj, value):
                pass

        class HasDataDesc:
            attr = DataDesc()

        h = HasDataDesc()
        h.__dict__["attr"] = "instance-value"
        assert h.attr == "from-descriptor"

    def test_non_data_descriptor_loses_to_instance_dict(self):
        class NonDataDesc:
            def __get__(self, obj, owner):
                return "from-descriptor"

        class HasNonDataDesc:
            attr = NonDataDesc()

        n = HasNonDataDesc()
        assert n.attr == "from-descriptor"
        n.__dict__["attr"] = "instance-wins"
        assert n.attr == "instance-wins"

    def test_slots_class_has_no_dict_and_raises_on_unset_slot(self):
        class Slotted:
            __slots__ = ("a",)

        s = Slotted()
        with pytest.raises(AttributeError):
            s.a  # unset slot
        with pytest.raises(AttributeError):
            s.b = 5  # not in __slots__, no __dict__ to fall back to
        s.a = 10
        assert s.a == 10

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_getattr_fallback_propagation(self, exc_name):
        class BadGetattr:
            def __getattr__(self, name):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            BadGetattr().anything

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_getattribute_propagation(self, exc_name):
        # __getattribute__ runs on EVERY access, before __getattr__ is even
        # considered -- its own exceptions propagate the same way.
        class BadGetattribute:
            def __getattribute__(self, name):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            BadGetattribute().anything

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_getattr_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__getattr__", ValueError, nargs=1)
        with pytest.raises(ValueError):
            Bad().anything


# ---------------------------------------------------------------------------
# protocol.container_len
# ---------------------------------------------------------------------------


class TestContainerLen:
    contract = get("protocol.container_len")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_negative_len_raises_valueerror(self):
        class NegLen:
            def __len__(self):
                return -1

        with pytest.raises(ValueError):
            len(NegLen())

    def test_len_via_index_like_return(self):
        class Idx:
            def __index__(self):
                return 3

        class HasIt:
            def __len__(self):
                return Idx()

        assert len(HasIt()) == 3

    @pytest.mark.parametrize("bad", ["5", 5.5, None])
    def test_len_return_without_index_raises_typeerror(self, bad):
        class BadLen:
            def __len__(self):
                return bad

        with pytest.raises(TypeError):
            len(BadLen())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_len_dunder_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): a user __len__ that raises IndexError
        # (or any of the other 6 classes) propagates uncaught -- it is NOT "never".
        Bad = raiser("__len__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            len(Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_len_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__len__", ValueError)
        with pytest.raises(ValueError):
            len(Bad())


# ---------------------------------------------------------------------------
# protocol.container_getitem
# ---------------------------------------------------------------------------


class TestContainerGetitem:
    contract = get("protocol.container_getitem")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_negative_index_passed_verbatim_no_wraparound(self):
        log = []

        class UserSeq:
            def __getitem__(self, key):
                log.append(key)
                return key

        u = UserSeq()
        assert u[-1] == -1  # NOT wrapped to len-1 the way list would
        assert log == [-1]

    def test_slice_passed_as_slice_object_verbatim(self):
        received = []

        class UserSeq:
            def __getitem__(self, key):
                received.append(key)
                return None

        u = UserSeq()
        u[1:5]
        u[1:5:2]
        u[...]
        assert received == [slice(1, 5, None), slice(1, 5, 2), Ellipsis]

    def test_no_getitem_raises_typeerror(self):
        with pytest.raises(TypeError):
            object()[0]

    def test_no_setitem_raises_typeerror(self):
        class GetOnly:
            def __getitem__(self, k):
                return k

        with pytest.raises(TypeError):
            GetOnly()[0] = 5

    def test_no_delitem_raises_typeerror(self):
        class GetOnly:
            def __getitem__(self, k):
                return k

        with pytest.raises(TypeError):
            del GetOnly()[0]

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_getitem_propagation(self, exc_name):
        Bad = raiser("__getitem__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            Bad()[0]

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_getitem_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__getitem__", ValueError, nargs=1)
        with pytest.raises(ValueError):
            Bad()[0]


# ---------------------------------------------------------------------------
# protocol.contains_fallback
# ---------------------------------------------------------------------------


class TestContainsFallback:
    contract = get("protocol.contains_fallback")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_uses_contains_when_present(self):
        log = []

        class HasContains:
            def __contains__(self, x):
                log.append("contains")
                return x == 5

        assert 5 in HasContains()
        assert log == ["contains"]

    def test_falls_back_to_iter_when_no_contains(self):
        log = []

        class HasIterOnly:
            def __iter__(self):
                log.append("iter")
                return iter([1, 2, 3])

        assert 2 in HasIterOnly()
        assert log == ["iter"]

    def test_falls_back_to_getitem_when_no_contains_or_iter(self):
        log = []

        class HasGetitemOnly:
            def __getitem__(self, i):
                log.append(i)
                if i > 3:
                    raise IndexError
                return i * 10

        assert 20 in HasGetitemOnly()
        assert log == [0, 1, 2]

    def test_not_found_via_getitem_fallback(self):
        class Empty:
            def __getitem__(self, i):
                raise IndexError

        assert 5 not in Empty()

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_contains_propagation(self, exc_name):
        Bad = raiser("__contains__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            5 in Bad()

    def test_none_of_the_three_protocols_raises_typeerror(self):
        with pytest.raises(TypeError):
            5 in object()

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_contains_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__contains__", ValueError, nargs=1)
        with pytest.raises(ValueError):
            5 in Bad()


# ---------------------------------------------------------------------------
# protocol.iteration
# ---------------------------------------------------------------------------


class TestIteration:
    contract = get("protocol.iteration")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_iter_next_protocol(self):
        class MyIter:
            def __init__(self):
                self.n = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.n >= 3:
                    raise StopIteration
                self.n += 1
                return self.n

        assert list(MyIter()) == [1, 2, 3]

    def test_old_style_getitem_iteration(self):
        class OldStyle:
            def __getitem__(self, i):
                if i >= 3:
                    raise IndexError
                return i * 100

        assert list(OldStyle()) == [0, 100, 200]

    def test_getitem_iteration_other_exception_propagates(self):
        class RaisesKeyError:
            def __getitem__(self, i):
                if i == 1:
                    raise KeyError("not an end-of-iteration signal")
                return i

        with pytest.raises(KeyError):
            list(RaisesKeyError())

    def test_old_style_getitem_iteration_also_ends_cleanly_on_stopiteration(self):
        # REVIEW FIX: the notes previously claimed IndexError was the ONLY signal
        # that ends old-style __getitem__ iteration cleanly -- a __getitem__ that
        # raises StopIteration directly ends it exactly the same way (the general
        # for-loop/comprehension StopIteration rule applies here too).
        class Stops:
            def __getitem__(self, index):
                raise StopIteration("done")

        assert list(Stops()) == []
        assert (1 in Stops()) is False

    @pytest.mark.parametrize("exc_name", sorted(set(PROPERTY_CLASSES) - {"IndexError"}))
    def test_old_style_getitem_iteration_propagation_non_indexerror_classes(self, exc_name):
        # Every class EXCEPT IndexError propagates uncaught -- IndexError is the
        # end-of-iteration signal itself (tested separately below), not a case
        # this "never happens for other reasons" list should include.
        Bad = raiser("__getitem__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            list(Bad())

    def test_old_style_getitem_iteration_indexerror_ends_cleanly_not_propagated(self):
        # An __getitem__ that raises IndexError on the FIRST call means an empty
        # sequence -- it is consumed as StopIteration, never seen by the caller.
        Bad = raiser("__getitem__", "IndexError", nargs=1)
        assert list(Bad()) == []

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_next_dunder_propagation(self, exc_name):
        Bad = raiser("__next__", exc_name)
        Bad.__iter__ = lambda self: self
        b = Bad()
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            next(b)

    def test_not_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            iter(object())

    def test_next_with_default_suppresses_stopiteration(self):
        it = iter([])
        assert next(it, "default") == "default"

    def test_next_without_default_raises_stopiteration(self):
        it = iter([])
        with pytest.raises(StopIteration):
            next(it)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iter_dunder_propagation(self, exc_name):
        Bad = raiser("__iter__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            iter(Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_iter_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__iter__", ValueError)
        with pytest.raises(ValueError):
            iter(Bad())


# ---------------------------------------------------------------------------
# protocol.hashing
# ---------------------------------------------------------------------------


class TestHashing:
    contract = get("protocol.hashing")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_defining_eq_without_hash_disables_hash(self):
        class HasEq:
            def __eq__(self, other):
                return True

        assert HasEq.__hash__ is None
        with pytest.raises(TypeError):
            hash(HasEq())

    def test_defining_eq_and_hash_stays_hashable(self):
        class HasEqAndHash:
            def __eq__(self, other):
                return True

            def __hash__(self):
                return 42

        assert hash(HasEqAndHash()) == 42

    def test_subclass_overriding_eq_becomes_unhashable_again(self):
        class Base:
            def __hash__(self):
                return 1

        class SubEq(Base):
            def __eq__(self, other):
                return True

        assert SubEq.__hash__ is None
        with pytest.raises(TypeError):
            hash(SubEq())

    def test_plain_object_is_hashable_by_default(self):
        hash(object())  # must not raise

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_dunder_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): a REAL (non-None) __hash__ that raises
        # KeyError (or any of the other 6 classes) propagates uncaught.
        Bad = raiser("__hash__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            hash(Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_hash_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__hash__", ValueError)
        with pytest.raises(ValueError):
            hash(Bad())


# ---------------------------------------------------------------------------
# protocol.truthiness
# ---------------------------------------------------------------------------


class TestTruthiness:
    contract = get("protocol.truthiness")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_bool_takes_priority_over_len(self):
        class HasBoth:
            def __bool__(self):
                return False

            def __len__(self):
                return 100  # would be truthy if consulted

        assert bool(HasBoth()) is False

    def test_len_used_when_no_bool(self):
        class LenOnly:
            def __len__(self):
                return 0

        assert bool(LenOnly()) is False

        class LenNonzero:
            def __len__(self):
                return 1

        assert bool(LenNonzero()) is True

    def test_default_true_with_neither(self):
        class Neither:
            pass

        assert bool(Neither()) is True

    def test_bool_returning_non_bool_raises_typeerror(self):
        class BadBool:
            def __bool__(self):
                return 5

        with pytest.raises(TypeError):
            bool(BadBool())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_bool_dunder_propagation(self, exc_name):
        Bad = raiser("__bool__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            bool(Bad())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_len_only_path_propagation(self, exc_name):
        # With no __bool__, bool(x) falls back to __len__ -- that dunder's own
        # exceptions propagate too, not just __bool__'s.
        Bad = raiser("__len__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            bool(Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_bool_propagates_non_property_exception(self):
        Bad = nonproperty_raiser("__bool__", ValueError)
        with pytest.raises(ValueError):
            bool(Bad())


# ---------------------------------------------------------------------------
# protocol.numeric_conversion
# ---------------------------------------------------------------------------


class TestNumericConversion:
    contract = get("protocol.numeric_conversion")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_hex_requires_index_not_int(self):
        class HasIndex:
            def __index__(self):
                return 42

        class HasIntOnly:
            def __int__(self):
                return 42

        assert hex(HasIndex()) == "0x2a"
        with pytest.raises(TypeError):
            hex(HasIntOnly())

    def test_operator_index_requires_index_not_int(self):
        import operator

        class HasIntOnly:
            def __int__(self):
                return 42

        with pytest.raises(TypeError):
            operator.index(HasIntOnly())

    def test_index_returning_non_int_raises_typeerror(self):
        class BadIndex:
            def __index__(self):
                return "not-an-int"

        with pytest.raises(TypeError):
            import operator

            operator.index(BadIndex())

    def test_index_returning_int_subclass_is_accepted_with_deprecation_warning(self):
        # REVIEW FIX: the notes previously claimed __index__ shares __bool__'s exact-
        # type requirement -- it does not. A strict int SUBCLASS return is accepted,
        # normalized to exact int, with a DeprecationWarning (not a TypeError).
        import operator

        class I(int):
            pass

        class HasSubclassIndex:
            def __index__(self):
                return I(1)

        with pytest.deprecated_call():
            result = operator.index(HasSubclassIndex())
        assert result == 1
        assert type(result) is int

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_propagation(self, exc_name):
        import operator

        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            operator.index(Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_index_propagates_non_property_exception(self):
        import operator

        Bad = nonproperty_raiser("__index__", ValueError)
        with pytest.raises(ValueError):
            operator.index(Bad())


# ---------------------------------------------------------------------------
# protocol.context_manager
# ---------------------------------------------------------------------------


class TestContextManager:
    contract = get("protocol.context_manager")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_enter_return_value_binds_to_as_name(self):
        class CM:
            def __enter__(self):
                return "entered-value"

            def __exit__(self, *exc_info):
                return False

        with CM() as v:
            assert v == "entered-value"

    def test_exit_receives_none_on_clean_exit(self):
        received = []

        class CM:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                received.append((exc_type, exc, tb))
                return False

        with CM():
            pass
        assert received == [(None, None, None)]

    def test_exit_receives_exception_info_on_propagation(self):
        received = []

        class CM:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                received.append(exc_type)
                return False

        with pytest.raises(KeyError):
            with CM():
                raise KeyError("boom")
        assert received == [KeyError]

    def test_exit_returning_true_suppresses_exception(self):
        class Suppressing:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return True

        with Suppressing():
            raise KeyError("suppressed")
        # reaching here means the exception was swallowed

    def test_exit_raising_replaces_original_and_chains_it(self):
        class ExitRaises:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                raise IndexError("from-exit")

        with pytest.raises(IndexError) as excinfo:
            with ExitRaises():
                raise KeyError("original")
        assert isinstance(excinfo.value.__context__, KeyError)

    def test_missing_enter_raises_typeerror(self):
        class NoEnter:
            def __exit__(self, *a):
                return False

        with pytest.raises(TypeError):
            with NoEnter():
                pass

    def test_missing_exit_raises_typeerror(self):
        class NoExit:
            def __enter__(self):
                return self

        with pytest.raises(TypeError):
            with NoExit():
                pass

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_enter_propagation(self, exc_name):
        class BadEnter:
            def __enter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

            def __exit__(self, *a):
                return False

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            with BadEnter():
                pass

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_exit_propagation_on_clean_block(self, exc_name):
        class BadExit:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            with BadExit():
                pass

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_enter_propagates_non_property_exception(self):
        class BadEnter:
            def __enter__(self):
                raise ValueError("boom")

            def __exit__(self, *a):
                return False

        with pytest.raises(ValueError):
            with BadEnter():
                pass


class TestCallProtocol:
    contract = get("protocol.call")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_instance_call_propagates_every_property_exception(self, exc_name):
        Bad = raiser("__call__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            Bad()()

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_function_call_propagates_every_property_exception(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        def target():
            raise exc(f"function-{exc_name}")

        with pytest.raises(exc):
            target()

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_metaclass_call_propagates_every_property_exception(self, exc_name):
        exc = PROPERTY_CLASSES[exc_name]

        class Meta(type):
            def __call__(cls, *args, **kwargs):
                raise exc(f"metaclass-{exc_name}")

        class Target(metaclass=Meta):
            pass

        with pytest.raises(exc):
            Target()

    def test_instance_attribute_named_call_does_not_replace_type_slot(self):
        class Target:
            def __call__(self):
                return "class slot"

        target = Target()
        target.__call__ = lambda: (_ for _ in ()).throw(AssertionError("instance attr"))
        assert target() == "class slot"

    def test_instance_call_forwards_positional_and_keyword_arguments(self):
        class Target:
            def __call__(self, value, *, suffix):
                return value + suffix

        assert Target()("a", suffix="b") == "ab"

    def test_class_construction_propagates_new_and_init_exceptions(self):
        class NewFails:
            def __new__(cls):
                raise KeyError("new")

        class InitFails:
            def __init__(self):
                raise FloatingPointError("init")

        with pytest.raises(KeyError, match="new"):
            NewFails()
        with pytest.raises(FloatingPointError, match="init"):
            InitFails()

    def test_typeerror_covers_noncallables_and_constructor_errors(self):
        with pytest.raises(TypeError):
            object()()
        for value in (1, True, 1.0, "x", None):
            with pytest.raises(TypeError):
                value()

        class BadInit:
            def __init__(self):
                return 1

        with pytest.raises(TypeError, match=r"__init__\(\) should return None"):
            BadInit()

        class NoKeywords:
            def __init__(self):
                pass

        with pytest.raises(TypeError):
            NoKeywords(missing=True)

    def test_nonproperty_callback_exceptions_remain_open(self):
        Bad = nonproperty_raiser("__call__", LookupError)
        with pytest.raises(LookupError, match="boom-__call__-LookupError"):
            Bad()()
        assert self.contract.may_raise_other is True

    def test_contract_claims_are_tested_and_call_shape_is_open(self):
        assert all(claim.tested for claim in self.contract.raises)
        shape = get_rule("protocol.call").shape
        assert shape.accepts(0)
        assert shape.accepts(4, has_keywords=True)
