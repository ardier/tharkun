"""Tests backing every claim in ``tharkun.solve.contracts.rules_bytes`` ([internal ref])."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


class TestDecode:
    contract = get("bytes.decode")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_default_utf8(self):
        assert b"hi".decode() == "hi"

    def test_bad_encoding_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            b"x".decode(5)

    def test_invalid_bytes_raises_unicodedecodeerror(self):
        with pytest.raises(UnicodeDecodeError):
            b"\xff".decode("utf-8")

    def test_unknown_encoding_raises_lookuperror(self):
        with pytest.raises(LookupError):
            b"x".decode("no-such-encoding")

    def test_never_raises_a_non_typeerror_property_class(self):
        # decode() only ever raises TypeError among the 7 property classes (bad
        # encoding/errors type); LookupError/UnicodeDecodeError are expected and fine.
        non_typeerror = tuple(e for name, e in PROPERTY_CLASSES.items() if name != "TypeError")
        for args in [("no-such-encoding",), ("utf-8", "no-such-handler")]:
            try:
                b"\xff".decode(*args)
            except non_typeerror as e:
                pytest.fail(f"decode{args} raised {type(e).__name__}")
            except Exception:
                pass


class TestHex:
    contract = get("bytes.hex")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert b"\xab\xcd".hex() == "abcd"

    @given(st.binary())
    def test_length_is_twice(self, b):
        assert len(b.hex()) == 2 * len(b)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_bytes_per_sep_index_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): an earlier draft used never_claims
        # with no native at all -- bytes_per_sep's __index__ propagates any
        # exception it raises.
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            b"x".hex("-", Bad())


class TestFromhex:
    contract = get("bytes.fromhex")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert bytes.fromhex("ab") == b"\xab"

    def test_bad_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            bytes.fromhex(5)

    def test_non_hex_raises_valueerror(self):
        with pytest.raises(ValueError):
            bytes.fromhex("zz")


class TestIndex:
    contract = get("bytes.index")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_found_by_int(self):
        assert b"abc".index(97) == 0  # ord('a')

    def test_found_by_bytes(self):
        assert b"abc".index(b"b") == 1

    def test_not_found_raises_valueerror(self):
        with pytest.raises(ValueError):
            b"abc".index(b"z")

    def test_int_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError):
            b"abc".index(256)
        with pytest.raises(ValueError):
            b"abc".index(-1)

    def test_arbitrary_object_raises_typeerror_without_calling_eq(self):
        calls = []

        class Spy:
            def __eq__(self, other):
                calls.append(other)
                return False

        with pytest.raises(TypeError):
            b"abc".index(Spy())
        assert calls == []  # __eq__ was never reached

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_start_end_index_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): start/end's __index__ propagates
        # independently of sub's own type-checked-not-duck-typed behavior.
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            b"abc".index(b"b", Bad())


class TestJoin:
    contract = get("bytes.join")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert b"-".join([b"a", b"b"]) == b"a-b"

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            b"-".join(5)

    def test_str_element_raises_typeerror(self):
        with pytest.raises(TypeError):
            b"-".join([b"a", "b"])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iteration_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            b"-".join(BadIter())

    def test_may_raise_other_is_true(self):
        # INHERITED REVIEW FIX: __iter__/__next__ is arbitrary user code -- it can
        # raise ANY exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_iteration_propagates_non_property_exception(self):
        class BadIter:
            def __iter__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            b"-".join(BadIter())


class TestGetItem:
    contract = get("bytes.__getitem__")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_int_key_returns_int_not_bytes(self):
        result = b"abc"[0]
        assert result == 97
        assert isinstance(result, int)

    def test_out_of_range_raises_indexerror(self):
        with pytest.raises(IndexError):
            b"abc"[5]

    def test_bad_key_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            b"abc"["x"]

    def test_slice_out_of_range_never_raises(self):
        assert b"abc"[10:20] == b""
        assert b"abc"[-100:100] == b"abc"

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            b"abc"[Bad()]
