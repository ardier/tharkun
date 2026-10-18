"""Engine 2 (static analyser) consumption of the batch-4 builtin contracts ([internal ref] batch 4).

``solve/absint/interp.py`` has no ``b_``/``t_`` transfer function for ``ord``, ``chr`` or
``tuple``; before this batch each was ``V(top=True)`` and the first call aborted the whole
proof ("call of an unknown value" -- the real refusal on bm_richards_nondet line 337, and on
bm_unpack_sequence_nondet line 421 for ``tuple(range(10))``).  ``builtin_name`` now tags a
builtin that has a contract-driven model (``c_<name>``) AND a TESTED contract; the contract
supplies the seven raise classes, the model proves the ones it can for the abstract
arguments (``_contract_builtin_call``).

What is pinned here, per the batch's rule "each new contract is consumed by an adapter, with
a test": the call stops aborting; all seven classes are recorded at it (TOTAL); a class the
model cannot argue stays unproven; and withdrawing or un-testing the contract withdraws the
model (the call aborts exactly as before).
"""

from __future__ import annotations

import dataclasses

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.contracts.adapters.absint import contract_for_builtin
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

ALL = tuple(ALL_PROPERTY_CLASSES)
MODELLED = ("ord", "chr", "tuple")


def _ops(r, name):
    return {s[2] for s in r.sites if s[2].startswith(f"{name}:")}


def _unproven(r, name):
    return {s[2].split(":")[1] for s in r.sites if s[2].startswith(f"{name}:") and "UNPROVEN" in s[4]}


class TestGate:
    @pytest.mark.parametrize("name", MODELLED)
    def test_a_tested_contract_exists_for_every_modelled_builtin(self, name):
        assert contract_for_builtin(name) is not None

    def test_an_unmodelled_or_unknown_name_has_no_gate_entry(self):
        assert contract_for_builtin("no_such_builtin") is None

    def test_the_call_no_longer_aborts_the_whole_proof(self):
        for src in ('x = ord("A")', "x = chr(65)", "x = tuple(range(3))"):
            r = prove(src, ALL)
            assert "unknown value" not in r.reason, (src, r.reason)
            assert r.verdict == "TRUE", (src, r.reason)

    @pytest.mark.parametrize("name,src", [
        ("ord", 'x = ord("A")'), ("chr", "x = chr(65)"), ("tuple", "x = tuple([1, 2])"),
    ])
    def test_every_property_class_is_recorded_at_the_call(self, name, src):
        r = prove(src, ALL)
        assert _ops(r, name) == {f"{name}:{exc}" for exc in ALL}

    @pytest.mark.parametrize("name,src", [
        ("ord", 'x = ord("A")'), ("chr", "x = chr(65)"), ("tuple", "x = tuple([1, 2])"),
    ])
    def test_withdrawing_the_contract_restores_the_abort(self, monkeypatch, name, src):
        load_all()
        monkeypatch.delitem(REGISTRY, name)
        r = prove(src, ALL)
        assert r.verdict == "UNKNOWN"
        assert "call of an unknown value" in r.reason

    @pytest.mark.parametrize("name,src", [
        ("ord", 'x = ord("A")'), ("chr", "x = chr(65)"), ("tuple", "x = tuple([1, 2])"),
    ])
    def test_an_untested_contract_withdraws_the_model_too(self, monkeypatch, name, src):
        load_all()
        c = REGISTRY[name]
        monkeypatch.setitem(REGISTRY, name, dataclasses.replace(c, tested=False))
        r = prove(src, ALL)
        assert r.verdict == "UNKNOWN"
        assert "call of an unknown value" in r.reason


class TestOrd:
    def test_a_one_character_literal_is_exact(self):
        assert prove('assert ord("A") == 65\n', ALL).verdict == "TRUE"

    def test_negative_control_a_wrong_value_is_not_proven(self):
        assert prove('assert ord("A") == 66\n', ALL).verdict == "UNKNOWN"

    @pytest.mark.parametrize("ch", ["\x00", "a", "Z", "é", "\u20ac", "\U0001f600", "\U0010ffff"])
    def test_the_model_agrees_with_cpython_on_every_literal(self, ch):
        assert prove(f"assert ord({ch!r}) == {ord(ch)}\n", ALL).verdict == "TRUE"
        assert prove(f"assert ord({ch!r}) == {ord(ch) + 1}\n", ALL).verdict == "UNKNOWN"

    def test_the_result_is_an_int_the_analysis_can_compute_with(self):
        # richards: A = ord('A'); later pkt.data[i] = A + w.count - 1
        assert prove('A = ord("A")\ny = A + 3 - 1\nassert y == 67\n', ALL).verdict == "TRUE"

    @pytest.mark.parametrize("src", ['x = ord("")', 'x = ord("ab")', "x = ord(5)", "x = ord(1.5)",
                                       "x = ord(None)", "x = ord()", 'x = ord("a", "b")'])
    def test_a_call_that_always_raises_typeerror_is_not_proven(self, src):
        r = prove(src, ("TypeError",))
        assert r.verdict == "UNKNOWN"

    def test_an_unknown_length_string_leaves_typeerror_unproven(self):
        r = prove("import _sv_verifier\ns = _sv_verifier.nondet_str()\nx = ord(s)\n", ALL)
        assert r.verdict == "UNKNOWN"
        assert _unproven(r, "ord") == {"TypeError"}

    def test_a_mixed_length_literal_set_leaves_typeerror_unproven(self):
        r = prove("import _sv_verifier\nn = _sv_verifier.nondet_int()\n"
                  's = "a" if n > 0 else "bc"\nx = ord(s)\n', ALL)
        assert r.verdict == "UNKNOWN"
        assert _unproven(r, "ord") == {"TypeError"}

    def test_an_object_argument_is_refused_not_guessed(self):
        r = prove("class C:\n    pass\nx = ord(C())\n", ALL)
        assert r.verdict == "UNKNOWN" and "ord of an unknown or object value" in r.reason


class TestChr:
    def test_an_int_result_is_a_str_the_analysis_can_compute_with(self):
        assert prove('c = chr(65)\ns = c + "x"\n', ALL).verdict == "TRUE"

    def test_an_unbounded_int_is_fine_because_valueerror_is_not_a_property_class(self):
        r = prove("import _sv_verifier\nn = _sv_verifier.nondet_int()\nc = chr(n)\n", ALL)
        assert r.verdict == "TRUE"

    def test_bool_is_an_int(self):
        assert prove("c = chr(True)\n", ALL).verdict == "TRUE"

    @pytest.mark.parametrize("src", ["x = chr(1.5)", 'x = chr("a")', "x = chr(None)", "x = chr()", "x = chr(1, 2)"])
    def test_a_non_int_argument_or_bad_arity_is_not_proven(self, src):
        assert prove(src, ("TypeError",)).verdict == "UNKNOWN"

    def test_a_maybe_float_argument_leaves_typeerror_unproven(self):
        r = prove("import _sv_verifier\nn = _sv_verifier.nondet_int()\nv = 1.5 if n > 0 else 65\nx = chr(v)\n", ALL)
        assert r.verdict == "UNKNOWN"
        assert _unproven(r, "chr") == {"TypeError"}

    def test_the_result_is_a_str_not_an_int(self):
        r = prove("c = chr(65)\ny = c + 1\n", ("TypeError",))
        assert r.verdict == "UNKNOWN"
        assert any("str" in s[4] and "int" in s[4] for s in r.sites if s[2] == "binop")

    def test_an_object_argument_is_refused_not_guessed(self):
        r = prove("class C:\n    pass\nx = chr(C())\n", ALL)
        assert r.verdict == "UNKNOWN" and "chr of an unknown or object value" in r.reason


class TestTuple:
    def test_tuple_of_range_unpacks(self):
        # bm_unpack_sequence: x = tuple(range(10)); a, b, ..., j = x
        src = "x = tuple(range(3))\na, b, c = x\n"
        assert prove(src, ALL).verdict == "TRUE"

    def test_the_length_is_tracked_so_a_valid_index_proves_and_an_invalid_one_does_not(self):
        assert prove("t = tuple([1, 2, 3])\nx = t[2]\n", ALL).verdict == "TRUE"
        r = prove("t = tuple([1, 2, 3])\nx = t[3]\n", ("IndexError",))
        assert r.verdict == "UNKNOWN"

    def test_the_element_facts_are_carried(self):
        r = prove('t = tuple(["a", "b"])\nx = t[0] + 1\n', ("TypeError",))
        assert r.verdict == "UNKNOWN"
        assert any("str" in s[4] and "int" in s[4] for s in r.sites if s[2] == "binop")

    def test_the_empty_tuple(self):
        assert prove("t = tuple()\nassert len(t) == 0\n", ALL).verdict == "TRUE"

    def test_a_tuple_comes_back_as_itself_with_its_items(self):
        # exact items survive: t[1] is 'b', so t[1] + 1 is a str + int TypeError
        r = prove('t = (1, "b")\nu = tuple(t)\nx = u[1] + 1\n', ("TypeError",))
        assert r.verdict == "UNKNOWN"
        assert prove("t = (1, 2)\nu = tuple(t)\nx = u[1] + 1\n", ALL).verdict == "TRUE"

    def test_a_str_is_iterated_to_chars(self):
        assert prove('t = tuple("abc")\n', ALL).verdict == "TRUE"

    @pytest.mark.parametrize("src", ["x = tuple(5)", "x = tuple(None)", "x = tuple(1.5)", "x = tuple([1], [2])",
                                      "x = tuple(iterable=[1])"])
    def test_a_non_iterable_or_bad_shape_is_not_proven(self, src):
        assert prove(src, ("TypeError",)).verdict == "UNKNOWN"

    def test_an_instance_iter_body_is_analysed_not_trusted(self):
        # the contract admits KeyError only through the argument's __iter__; absint runs that
        # body, so a KeyError raised there must still surface as an unproven check
        bad = ("class Bad:\n    def __iter__(self):\n        raise KeyError('boom')\n"
               "t = tuple(Bad())\n")
        assert prove(bad, ("KeyError",)).verdict == "UNKNOWN"
        good = ("class Good:\n    def __iter__(self):\n        return iter([1, 2, 3])\n"
                "t = tuple(Good())\n")
        assert prove(good, ALL).verdict == "TRUE"

    def test_dict_keys_and_dict_are_iterated(self):
        assert prove("d = {1: 2, 3: 4}\nt = tuple(d)\n", ALL).verdict == "TRUE"
