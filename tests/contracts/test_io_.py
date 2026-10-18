"""Tests backing every claim in ``tharkun.solve.contracts.rules_io`` ([internal ref] batch 3)."""

from __future__ import annotations

import io

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


class TestStringIOWrite:
    contract = get("io.StringIO.write")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_returns_char_count(self):
        s = io.StringIO()
        assert s.write("hello") == 5

    def test_advances_position(self):
        s = io.StringIO()
        s.write("ab")
        s.write("cd")
        assert s.getvalue() == "abcd"

    def test_non_str_raises_typeerror(self):
        with pytest.raises(TypeError):
            io.StringIO().write(5)

    def test_class_with_str_dunder_not_accepted(self):
        class FakeStr:
            def __str__(self):
                return "fake"

        with pytest.raises(TypeError):
            io.StringIO().write(FakeStr())

    def test_write_after_close_raises_valueerror(self):
        s = io.StringIO()
        s.close()
        with pytest.raises(ValueError):
            s.write("x")


class TestStringIORead:
    contract = get("io.StringIO.read")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_read_all(self):
        s = io.StringIO("hello")
        assert s.read() == "hello"

    def test_read_n_chars(self):
        s = io.StringIO("hello")
        assert s.read(2) == "he"
        assert s.read(2) == "ll"  # position advanced

    def test_negative_size_reads_all(self):
        s = io.StringIO("hello")
        assert s.read(-1) == "hello"

    def test_bad_size_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            io.StringIO("hello").read("x")

    def test_read_after_close_raises_valueerror(self):
        s = io.StringIO("hi")
        s.close()
        with pytest.raises(ValueError):
            s.read()

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_size_index_propagation(self, exc_name):
        # SOUNDNESS FIX repro (gatekeeper): size's __index__ propagates any
        # exception it raises -- an earlier draft claimed "never" for all but
        # TypeError.
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            io.StringIO("hello").read(Bad())

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_size_index_propagates_non_property_exception(self):
        class BadIndex:
            def __index__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            io.StringIO("hello").read(BadIndex())


class TestStringIOGetvalue:
    contract = get("io.StringIO.getvalue")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_returns_whole_buffer_regardless_of_position(self):
        s = io.StringIO("hello")
        s.read(3)  # advance position
        assert s.getvalue() == "hello"  # unaffected by position

    def test_after_close_raises_valueerror(self):
        s = io.StringIO("hi")
        s.close()
        with pytest.raises(ValueError):
            s.getvalue()


class TestStringIOSeek:
    contract = get("io.StringIO.seek")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_seek_returns_new_position(self):
        s = io.StringIO("hello")
        assert s.seek(2) == 2
        assert s.read() == "llo"

    def test_bad_pos_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            io.StringIO("hi").seek("x")

    def test_negative_seek_raises_valueerror(self):
        with pytest.raises(ValueError):
            io.StringIO("hi").seek(-1)

    def test_bad_whence_raises_valueerror(self):
        with pytest.raises(ValueError):
            io.StringIO("hi").seek(0, 5)

    def test_after_close_raises_valueerror(self):
        s = io.StringIO("hi")
        s.close()
        with pytest.raises(ValueError):
            s.seek(0)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_pos_index_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            io.StringIO("hi").seek(Bad())

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_pos_index_propagates_non_property_exception(self):
        class BadIndex:
            def __index__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            io.StringIO("hi").seek(BadIndex())


class TestStringIOClose:
    contract = get("io.StringIO.close")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_close_returns_none(self):
        assert io.StringIO().close() is None

    def test_idempotent(self):
        s = io.StringIO()
        s.close()
        s.close()  # must not raise

    def test_every_other_method_raises_valueerror_after_close(self):
        s = io.StringIO("hi")
        s.close()
        for op in (lambda: s.write("x"), lambda: s.read(), lambda: s.getvalue(), lambda: s.seek(0)):
            with pytest.raises(ValueError):
                op()


class TestBytesIOWrite:
    contract = get("io.BytesIO.write")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_accepts_bytes(self):
        b = io.BytesIO()
        assert b.write(b"hi") == 2

    def test_accepts_bytearray_and_memoryview(self):
        b = io.BytesIO()
        assert b.write(bytearray(b"x")) == 1
        assert b.write(memoryview(b"y")) == 1

    def test_str_rejected_with_typeerror(self):
        with pytest.raises(TypeError):
            io.BytesIO().write("not-bytes")

    def test_opposite_of_stringio_write(self):
        # StringIO wants exact str, rejects bytes; BytesIO wants buffer-protocol,
        # rejects str. Both raise TypeError, for opposite reasons.
        with pytest.raises(TypeError):
            io.StringIO().write(b"bytes")
        with pytest.raises(TypeError):
            io.BytesIO().write("string")

    def test_write_after_close_raises_valueerror(self):
        b = io.BytesIO()
        b.close()
        with pytest.raises(ValueError):
            b.write(b"x")
