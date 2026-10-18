"""Tests backing every claim in ``tharkun.solve.contracts.strformat_`` ([internal ref] batch 3)."""

from __future__ import annotations

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


class TestPercentFormat:
    contract = get("str.__mod__")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert "%s-%d" % ("x", 5) == "x-5"

    def test_mapping_form(self):
        assert "%(name)s" % {"name": "x"} == "x"

    def test_missing_mapping_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            "%(name)s" % {}

    def test_too_few_args_raises_typeerror(self):
        with pytest.raises(TypeError):
            "%s %s" % (1,)

    def test_too_many_args_raises_typeerror(self):
        with pytest.raises(TypeError):
            "%s" % (1, 2)

    def test_single_non_tuple_value_not_unpacked(self):
        assert "%s" % [1, 2] == "[1, 2]"  # must NOT raise "too many arguments"

    def test_numeric_specifier_with_non_number_raises_typeerror(self):
        with pytest.raises(TypeError):
            "%d" % "x"

    def test_unsupported_format_char_raises_valueerror(self):
        with pytest.raises(ValueError):
            "%y" % 5

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_str_conversion_propagation(self, exc_name):
        Bad = raiser("__str__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            "%s" % Bad()

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_str_conversion_propagates_non_property_exception(self):
        class Bad:
            def __str__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            "%s" % Bad()


class TestFormat:
    contract = get("format")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert format(5, "d") == "5"

    def test_empty_spec_falls_back_to_str(self):
        class Plain:
            def __str__(self):
                return "plain-str"

        assert format(Plain(), "") == "plain-str"

    def test_default_format_rejects_nonempty_spec(self):
        class NoFormat:
            pass

        with pytest.raises(TypeError):
            format(NoFormat(), "x")

    def test_bad_spec_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            format(5, 5)

    def test_type_specific_invalid_spec_raises_valueerror(self):
        with pytest.raises(ValueError):
            format(5, "zzz")
        with pytest.raises(ValueError):
            format("x", "d")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_dunder_format_propagation(self, exc_name):
        Bad = raiser("__format__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            format(Bad(), "x")

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_dunder_format_propagates_non_property_exception(self):
        class Bad:
            def __format__(self, spec):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            format(Bad(), "x")


class TestRepr:
    contract = get("repr")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_default_repr_never_raises(self):
        class Plain:
            pass

        repr(Plain())  # must not raise

    def test_custom_repr(self):
        class R:
            def __repr__(self):
                return "R!"

        assert repr(R()) == "R!"

    def test_non_str_return_raises_typeerror(self):
        class BadRepr:
            def __repr__(self):
                return 5

        with pytest.raises(TypeError):
            repr(BadRepr())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_repr_propagation(self, exc_name):
        Bad = raiser("__repr__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            repr(Bad())

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: __repr__ is arbitrary user code -- it can raise ANY
        # exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_repr_propagates_non_property_exception(self):
        class Bad:
            def __repr__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            repr(Bad())


class TestAscii:
    contract = get("ascii")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_escapes_non_ascii(self):
        assert ascii("h\xe9llo") == "'h\\xe9llo'"

    def test_matches_repr_for_ascii_only(self):
        assert ascii("hello") == repr("hello")

    def test_non_str_repr_return_raises_typeerror(self):
        class BadRepr:
            def __repr__(self):
                return 5

        with pytest.raises(TypeError):
            ascii(BadRepr())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_repr_propagation(self, exc_name):
        Bad = raiser("__repr__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            ascii(Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_repr_propagates_non_property_exception(self):
        class Bad:
            def __repr__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            ascii(Bad())


class TestFStringConversionSpecs:
    """f-strings are syntax, not a callable -- these confirm the compile-down claim
    the module docstring makes, rather than testing a Contract of their own."""

    def test_bang_r_is_repr(self):
        class R:
            def __repr__(self):
                return "R!"

            def __str__(self):
                return "S!"

        x = R()
        assert f"{x!r}" == repr(x) == "R!"

    def test_bang_s_is_str(self):
        class R:
            def __repr__(self):
                return "R!"

            def __str__(self):
                return "S!"

        x = R()
        assert f"{x!s}" == str(x) == "S!"

    def test_bang_a_is_ascii(self):
        assert f"{'héllo'!a}" == ascii("héllo")

    def test_format_spec_after_conversion_uses_format_on_the_conversion_result(self):
        # f"{x!r:>10}" applies the :>10 spec to repr(x), i.e. format(repr(x), ">10").
        x = 5
        assert f"{x!r:>10}" == format(repr(x), ">10")
