"""Tests backing every claim in ``tharkun.solve.contracts.rules_re`` ([internal ref] batch 3)."""

from __future__ import annotations

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


def _index_raiser(exc_name):
    return raiser("__index__", exc_name)


def _and_raiser(exc_name):
    """A flags-like object whose __and__/__rand__ raise (re.compile/match/etc.
    combine flags via & rather than converting via __index__)."""
    Bad = raiser("__and__", exc_name, nargs=1)
    Bad.__rand__ = raiser("__rand__", exc_name, nargs=1).__dict__["__rand__"]
    return Bad


class TestCompile:
    contract = get("re.compile")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_returns_pattern(self):
        assert isinstance(re.compile("a"), re.Pattern)

    def test_bad_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.compile(5)

    def test_malformed_pattern_raises_re_error_not_property_class(self):
        with pytest.raises(re.error):
            re.compile("[")

    def test_flag_pattern_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError):
            re.compile(b"x", re.UNICODE)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_flags_and_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): flags is combined via & with the
        # internal flag bits, not converted via __index__ -- a flags object whose
        # __and__/__rand__ raises propagates that exception unchanged.
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            re.compile("a", _and_raiser(exc_name)())

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_flags_and_propagates_non_property_exception(self):
        class BadFlags:
            def __and__(self, other):
                raise ValueError("boom")

            __rand__ = __and__

        with pytest.raises(ValueError):
            re.compile("a", BadFlags())


class TestMatchSearchFullmatch:
    def test_contracts_are_tested(self):
        assert get("re.match").tested
        assert get("re.search").tested
        assert get("re.fullmatch").tested

    def test_match_is_start_anchored_only(self):
        assert re.match("a", "ab") is not None
        assert re.match("b", "ab") is None

    def test_search_scans_whole_string(self):
        assert re.search("b", "ab") is not None

    def test_fullmatch_requires_both_ends(self):
        assert re.fullmatch("ab", "ab") is not None
        assert re.fullmatch("a", "ab") is None

    def test_no_match_returns_none(self):
        assert re.match("z", "ab") is None

    def test_accessing_group_on_none_raises_attributeerror(self):
        m = re.match("z", "ab")
        with pytest.raises(AttributeError):
            m.group()

    def test_bad_string_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.match("a", 5)

    def test_bad_pattern_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.match(5, "a")

    def test_str_bytes_kind_mismatch_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.match("a", b"a")
        with pytest.raises(TypeError):
            re.match(b"a", "a")

    def test_flags_with_compiled_pattern_raises_valueerror(self):
        with pytest.raises(ValueError):
            re.match(re.compile("a"), "a", re.I)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_flags_and_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            re.match("a", "a", _and_raiser(exc_name)())
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            re.search("a", "a", _and_raiser(exc_name)())
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            re.fullmatch("a", "a", _and_raiser(exc_name)())

    def test_effects_flag_may_execute_callback(self):
        assert get("re.match").effects.may_execute_callback is True
        assert get("re.search").effects.may_execute_callback is True
        assert get("re.fullmatch").effects.may_execute_callback is True

    def test_flags_and_propagates_non_property_exception(self):
        class BadFlags:
            def __and__(self, other):
                raise ValueError("boom")

            __rand__ = __and__

        with pytest.raises(ValueError):
            re.match("a", "a", BadFlags())
        with pytest.raises(ValueError):
            re.search("a", "a", BadFlags())
        with pytest.raises(ValueError):
            re.fullmatch("a", "a", BadFlags())


class TestSub:
    contract = get("re.sub")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic_replacement(self):
        assert re.sub("a", "b", "aaa") == "bbb"

    def test_callable_repl(self):
        assert re.sub(r"\d", lambda m: str(int(m.group()) + 1), "1 2 3") == "2 3 4"

    def test_returns_same_type_as_input(self):
        assert isinstance(re.sub("a", "b", "aaa"), str)
        assert isinstance(re.sub(b"a", b"b", b"aaa"), bytes)

    def test_bad_string_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.sub("a", "b", 5)

    def test_bad_repl_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.sub("a", 5, "aaa")

    def test_bad_count_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.sub("a", "b", "aaa", count="x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_callable_repl_propagation(self, exc_name):
        def bad_repl(m):
            raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            re.sub("a", bad_repl, "aaa")

    def test_group_indexerror_propagates_through_callable_repl(self):
        # re.Match.group's own IndexError, reached via a callback -- ties the two
        # contracts together the way the docstring says.
        with pytest.raises(IndexError):
            re.sub(r"(a)", lambda m: m.group(5), "a")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_count_index_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            re.sub("a", "b", "aaa", count=_index_raiser(exc_name)())

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_callable_repl_propagates_non_property_exception(self):
        def bad_repl(m):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            re.sub("a", bad_repl, "aaa")


class TestSplit:
    contract = get("re.split")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert re.split(",", "a,b,c") == ["a", "b", "c"]

    def test_capturing_groups_are_interleaved(self):
        assert re.split(r"(,)", "a,b") == ["a", ",", "b"]

    def test_bad_pattern_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.split(None, "a")

    def test_bad_maxsplit_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.split("a", "abc", maxsplit="x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_maxsplit_index_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): maxsplit's __index__ propagates any
        # exception it raises, including IndexError -- the class an earlier draft
        # incorrectly claimed could "never" happen.
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            re.split("a", "aaa", maxsplit=_index_raiser(exc_name)())

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_maxsplit_index_propagates_non_property_exception(self):
        class BadIndex:
            def __index__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            re.split("a", "aaa", maxsplit=BadIndex())


class TestFindall:
    contract = get("re.findall")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_no_group_returns_whole_matches(self):
        assert re.findall(r"\d", "a1b2") == ["1", "2"]

    def test_one_group_returns_just_that_group(self):
        assert re.findall(r"(\d)", "a1b2") == ["1", "2"]

    def test_multiple_groups_returns_tuples(self):
        assert re.findall(r"(\d)(\w)", "1a2b") == [("1", "a"), ("2", "b")]

    def test_bad_string_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.findall("a", 5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_flags_and_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            re.findall("a", "a", _and_raiser(exc_name)())

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_flags_and_propagates_non_property_exception(self):
        class BadFlags:
            def __and__(self, other):
                raise ValueError("boom")

            __rand__ = __and__

        with pytest.raises(ValueError):
            re.findall("a", "a", BadFlags())


class TestEscape:
    contract = get("re.escape")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_escapes_special_chars(self):
        assert re.escape("a.b") == r"a\.b"

    def test_bytes_input_returns_bytes(self):
        assert re.escape(b"a.b") == rb"a\.b"

    def test_bad_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            re.escape(5)

    @given(st.text())
    def test_escaped_pattern_matches_literally(self, s):
        assert re.fullmatch(re.escape(s), s) is not None


# ---------------------------------------------------------------------------
# re.Match methods
# ---------------------------------------------------------------------------


@pytest.fixture
def named_match():
    return re.match(r"(?P<first>\w+)\s(?P<second>\w+)?", "hello ")


class TestMatchGroup:
    contract = get("re.Match.group")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_group_zero_is_whole_match(self, named_match):
        assert named_match.group() == "hello "
        assert named_match.group(0) == named_match.group()

    def test_group_by_index(self, named_match):
        assert named_match.group(1) == "hello"

    def test_group_by_name(self, named_match):
        assert named_match.group("first") == "hello"

    def test_unmatched_optional_group_is_none(self, named_match):
        assert named_match.group(2) is None

    def test_out_of_range_index_raises_indexerror(self, named_match):
        with pytest.raises(IndexError):
            named_match.group(5)

    def test_unknown_name_raises_indexerror_not_keyerror(self, named_match):
        with pytest.raises(IndexError):
            named_match.group("nope")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_propagation(self, exc_name, named_match):
        # SOUNDNESS FIX repro (gatekeeper): an object with a raising __index__
        # propagates that exact exception, not IndexError -- an earlier draft
        # claimed every class but IndexError was "never".
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            named_match.group(_index_raiser(exc_name)())

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: the argument's __index__ is arbitrary user code -- it can
        # raise ANY exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_index_propagates_non_property_exception(self, named_match):
        class BadIndex:
            def __index__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            named_match.group(BadIndex())


class TestMatchGroups:
    contract = get("re.Match.groups")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_returns_tuple_in_order(self, named_match):
        assert named_match.groups() == ("hello", None)

    def test_default_substitutes_for_unmatched(self, named_match):
        assert named_match.groups(default="X") == ("hello", "X")


class TestMatchGroupdict:
    contract = get("re.Match.groupdict")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_only_named_groups(self, named_match):
        assert named_match.groupdict() == {"first": "hello", "second": None}

    def test_default_substitutes_for_unmatched(self, named_match):
        assert named_match.groupdict(default="X") == {"first": "hello", "second": "X"}


class TestMatchStartEndSpan:
    def test_contracts_are_tested(self):
        assert get("re.Match.start").tested
        assert get("re.Match.end").tested
        assert get("re.Match.span").tested

    def test_whole_match(self, named_match):
        assert named_match.start() == 0
        assert named_match.end() == 6
        assert named_match.span() == (0, 6)

    def test_unmatched_group_is_negative_one(self, named_match):
        assert named_match.start(2) == -1
        assert named_match.end(2) == -1
        assert named_match.span(2) == (-1, -1)

    def test_out_of_range_group_raises_indexerror(self, named_match):
        with pytest.raises(IndexError):
            named_match.start(5)
        with pytest.raises(IndexError):
            named_match.end(5)
        with pytest.raises(IndexError):
            named_match.span(5)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_propagation(self, exc_name, named_match):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            named_match.start(_index_raiser(exc_name)())
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            named_match.end(_index_raiser(exc_name)())
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            named_match.span(_index_raiser(exc_name)())

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: the argument's __index__ is arbitrary user code -- it can
        # raise ANY exception, not just one of the 7 property classes.
        assert get("re.Match.start").may_raise_other is True
        assert get("re.Match.end").may_raise_other is True
        assert get("re.Match.span").may_raise_other is True

    def test_effects_flag_may_execute_callback(self):
        assert get("re.Match.start").effects.may_execute_callback is True
        assert get("re.Match.end").effects.may_execute_callback is True
        assert get("re.Match.span").effects.may_execute_callback is True

    def test_index_propagates_non_property_exception(self, named_match):
        class BadIndex:
            def __index__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            named_match.start(BadIndex())
        with pytest.raises(ValueError):
            named_match.end(BadIndex())
        with pytest.raises(ValueError):
            named_match.span(BadIndex())
