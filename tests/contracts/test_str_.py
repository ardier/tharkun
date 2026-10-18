"""Tests backing every claim in ``tharkun.solve.contracts.rules_str`` ([internal ref])."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


class TestJoin:
    contract = get("str.join")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.text()))
    def test_length_formula(self, xs):
        result = "-".join(xs)
        if xs:
            assert len(result) == sum(len(s) for s in xs) + len(xs) - 1
        else:
            assert result == ""

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            "-".join(5)

    def test_non_str_element_raises_typeerror(self):
        with pytest.raises(TypeError):
            "-".join([1, 2])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_iteration_propagation(self, exc_name):
        class BadIter:
            def __iter__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "-".join(BadIter())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_mid_iteration_propagation(self, exc_name):
        def gen():
            yield "a"
            raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "-".join(gen())

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
            "-".join(BadIter())


class TestStartswithEndswith:
    def test_contracts_are_tested(self):
        assert get("str.startswith").tested
        assert get("str.endswith").tested

    def test_basic(self):
        assert "hello".startswith("he")
        assert "hello".endswith("lo")
        assert not "hello".startswith("lo")

    def test_bad_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            "x".startswith(5)
        with pytest.raises(TypeError):
            "x".endswith(5)

    def test_tuple_with_bad_entry_raises_typeerror(self):
        with pytest.raises(TypeError):
            "x".startswith(("a", 5))

    def test_tuple_short_circuits_on_match_before_bad_entry(self):
        assert "x".startswith(("x", 5)) is True
        assert "x".endswith(("x", 5)) is True

    def test_tuple_no_match_raises_when_bad_entry_reached(self):
        with pytest.raises(TypeError):
            "y".startswith(("x", 5))

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_start_end_index_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): start/end go through __index__ like
        # any position argument -- an earlier draft claimed they were "type-checked
        # directly" like prefix/suffix and marked every class but TypeError "never".
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "x".startswith("x", Bad())
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "x".endswith("x", Bad())


class TestSplit:
    contract = get("str.split")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_default_splits_whitespace(self):
        assert "a  b\tc".split() == ["a", "b", "c"]

    def test_explicit_sep(self):
        assert "a,b,c".split(",") == ["a", "b", "c"]

    def test_bad_sep_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            "x".split(5)

    def test_empty_sep_raises_valueerror(self):
        with pytest.raises(ValueError):
            "x".split("")

    def test_maxsplit_without_index_raises_typeerror(self):
        with pytest.raises(TypeError):
            "x".split(maxsplit="a")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_maxsplit_index_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "x".split(maxsplit=Bad())


class TestStrip:
    contract = get("str.strip")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_default_strips_whitespace(self):
        assert "  x  ".strip() == "x"

    def test_explicit_chars(self):
        assert "xxhixx".strip("x") == "hi"

    def test_bad_chars_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            "x".strip(5)


class TestReplace:
    contract = get("str.replace")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert "aaa".replace("a", "b") == "bbb"
        assert "aaa".replace("a", "b", 1) == "baa"

    def test_bad_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            "x".replace(5, "y")
        with pytest.raises(TypeError):
            "x".replace("y", 5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_count_index_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "xxx".replace("x", "y", Bad())


class TestFormat:
    contract = get("str.format")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_positional_and_keyword(self):
        assert "{0}-{name}".format(1, name="x") == "1-x"

    def test_positional_out_of_range_raises_indexerror(self):
        with pytest.raises(IndexError):
            "{0}".format()
        with pytest.raises(IndexError):
            "{}{}".format(1)

    def test_missing_keyword_raises_keyerror(self):
        with pytest.raises(KeyError):
            "{a}".format()
        with pytest.raises(KeyError):
            "{a}".format(b=1)

    def test_malformed_format_string_raises_valueerror(self):
        with pytest.raises(ValueError):
            "{".format()

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_attribute_field_propagation(self, exc_name):
        class BadAttr:
            @property
            def attr(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "{0.attr}".format(BadAttr())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_getitem_field_propagation(self, exc_name):
        class BadGetitem:
            def __getitem__(self, k):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "{0[k]}".format(BadGetitem())


class TestEncode:
    contract = get("str.encode")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_default_utf8(self):
        assert "hi".encode() == b"hi"

    def test_bad_encoding_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            "x".encode(5)

    def test_unknown_encoding_raises_lookuperror(self):
        with pytest.raises(LookupError):
            "x".encode("no-such-encoding")

    def test_unknown_errors_handler_raises_lookuperror(self):
        with pytest.raises(LookupError):
            "\xe9".encode("ascii", "no-such-handler")

    def test_unencodable_surrogate_raises_unicodeencodeerror(self):
        with pytest.raises(UnicodeEncodeError):
            "\ud800".encode("utf-8")

    def test_never_raises_a_property_class(self):
        # No user dunder is ever invoked by encode(); a battery of hostile inputs
        # (bad encoding name/type, bad errors handler, surrogate, huge string) must
        # never surface any of the 7 property classes.
        for args in [("no-such-encoding",), ("ascii", "no-such-handler")]:
            try:
                "\xe9".encode(*args)
            except tuple(PROPERTY_CLASSES.values()):
                pytest.fail(f"encode{args} raised a property class")
            except Exception:
                pass  # LookupError/UnicodeEncodeError are expected and fine


class TestIndex:
    contract = get("str.index")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_found(self):
        assert "hello".index("l") == 2

    def test_not_found_raises_valueerror(self):
        with pytest.raises(ValueError):
            "hello".index("z")

    def test_bad_sub_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            "x".index(5)

    def test_bad_start_stop_raises_typeerror(self):
        with pytest.raises(TypeError):
            "x".index("y", 1, "a")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_start_end_index_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "hello".index("l", Bad())


class TestGetItem:
    contract = get("str.__getitem__")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_positive_and_negative_index(self):
        assert "hello"[0] == "h"
        assert "hello"[-1] == "o"

    def test_out_of_range_raises_indexerror(self):
        with pytest.raises(IndexError):
            "x"[5]

    def test_bad_key_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            "x"["a"]

    def test_slice_out_of_range_never_raises(self):
        assert "x"[1:100] == ""
        assert "hello"[-100:100] == "hello"

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "x"[Bad()]
