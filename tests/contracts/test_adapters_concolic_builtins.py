"""Engine 3 (concolic/z3) consumption of the batch-4 builtin contracts ([internal ref] batch 4).

``sv/_symexec.py``'s executor raised ``Unsupported`` for every call to ``ord``, ``chr``,
``tuple``, ``zip``, ``enumerate``, ``reversed``, ``type`` and for ``list()`` with no argument
(or any argument that was not a list/tuple): the whole statement, and everything assigned in
it, became unmodelled.  ``adapters/concolic.py::concrete_fold`` now lets the executor
evaluate a call natively when the contract is TESTED and pure and every argument is an exact
builtin value; ``type_of_one_argument`` answers ``type(x)``.  Pinned here: the folds are
right (checked against CPython), a symbolic or unmodelled argument is still refused, a
one-shot iterator is never read twice, and withdrawing or un-testing a contract withdraws
the fold.
"""

from __future__ import annotations

import dataclasses
import sys
import textwrap

import pytest

from tharkun.solve.contracts.adapters import concolic as _concolic
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim
from tharkun.sv.sites import build_harness

pytest.importorskip("z3")

PRE = "import _sv_verifier\n\nif __name__ == '__main__':\n    n = _sv_verifier.nondet_int()\n"


def _run(body, classes=("IndexError",)):
    from tharkun.sv import _symexec

    src = PRE + textwrap.indent(textwrap.dedent(body), "    ")
    h = build_harness(src, classes, None)
    mod = {"__name__": "sv_harness_test"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    return list(_symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites],
                                       h.params, h.kinds, {}).values())


def _sat(outs):
    return [m for o, m in outs if o == "sat"]


def _only_non_sat(outs):
    return not _sat(outs) and any(o in ("unreached", "unsupported", "unsat") for o, _ in outs)


class TestAdapter:
    @pytest.mark.parametrize("name,fn", [("ord", ord), ("chr", chr), ("tuple", tuple), ("list", list),
                                         ("zip", zip), ("enumerate", enumerate), ("reversed", reversed)])
    def test_a_tested_pure_contract_yields_the_native_builtin(self, name, fn):
        arity = _concolic.CONCRETE_FOLD_ARITY[name][0]
        assert _concolic.concrete_fold(name, arg_count=arity, has_keywords=False) is fn

    @pytest.mark.parametrize("name,count", [("ord", 0), ("ord", 2), ("chr", 2), ("tuple", 2), ("list", 2),
                                            ("enumerate", 0), ("enumerate", 3), ("reversed", 0), ("reversed", 2)])
    def test_a_wrong_arity_has_no_fold(self, name, count):
        assert _concolic.concrete_fold(name, arg_count=count, has_keywords=False) is None

    def test_zip_has_no_upper_arity(self):
        assert _concolic.concrete_fold("zip", arg_count=0, has_keywords=False) is zip
        assert _concolic.concrete_fold("zip", arg_count=9, has_keywords=False) is zip

    @pytest.mark.parametrize("name", sorted(_concolic.CONCRETE_FOLD_ARITY))
    def test_keywords_have_no_fold(self, name):
        assert _concolic.concrete_fold(name, arg_count=1, has_keywords=True) is None

    def test_set_is_never_folded_its_order_depends_on_the_hash_seed(self):
        assert "set" not in _concolic.CONCRETE_FOLD_ARITY
        assert _concolic.concrete_fold("set", arg_count=1, has_keywords=False) is None

    @pytest.mark.parametrize("name", sorted(_concolic.CONCRETE_FOLD_ARITY))
    def test_withdrawing_or_untesting_the_contract_withdraws_the_fold(self, monkeypatch, name):
        load_all()
        arity = _concolic.CONCRETE_FOLD_ARITY[name][0]
        c = REGISTRY[name]
        monkeypatch.setitem(REGISTRY, name, dataclasses.replace(c, tested=False))
        assert _concolic.concrete_fold(name, arg_count=arity, has_keywords=False) is None
        monkeypatch.delitem(REGISTRY, name)
        assert _concolic.concrete_fold(name, arg_count=arity, has_keywords=False) is None

    @pytest.mark.parametrize("effects", [
        Effects(io=True), Effects(randomness="a generator"), Effects(clock_or_env=True),
        Effects(deterministic=False), Effects(mutates_receiver=True),
    ])
    def test_an_impure_contract_has_no_fold(self, monkeypatch, effects):
        load_all()
        synthetic = Contract(
            qualname="ord", raises=tuple(RaiseClaim(e, "never", tested=True) for e in ALL_PROPERTY_CLASSES),
            may_raise_other=False, returns=None, effects=effects, source=("synthetic",), tested=True,
        )
        monkeypatch.setitem(REGISTRY, "ord", synthetic)
        assert _concolic.concrete_fold("ord", arg_count=1, has_keywords=False) is None

    def test_type_of_exactly_one_argument_only(self):
        assert _concolic.type_of_one_argument(arg_count=1, has_keywords=False)
        for count in (0, 2, 3):
            assert not _concolic.type_of_one_argument(arg_count=count, has_keywords=False)
        assert not _concolic.type_of_one_argument(arg_count=1, has_keywords=True)

    def test_withdrawing_the_type_contract_withdraws_the_one_argument_fact(self, monkeypatch):
        load_all()
        monkeypatch.delitem(REGISTRY, "type")
        assert not _concolic.type_of_one_argument(arg_count=1, has_keywords=False)


class TestOrdChr:
    def test_ord_of_a_literal_is_evaluated_and_the_equation_is_solved(self):
        assert _sat(_run("if ord('A') + n == 70:\n    raise IndexError('r')\n")) == [{"n": 5}]

    def test_negative_control_a_false_ord_comparison_is_unreached(self):
        assert _only_non_sat(_run("if ord('A') == 66:\n    raise IndexError('r')\n"))

    def test_chr_of_a_literal_is_a_str(self):
        assert _sat(_run("if chr(66) == 'B':\n    raise IndexError('r')\n"))
        assert _only_non_sat(_run("if chr(66) == 'C':\n    raise IndexError('r')\n"))

    def test_a_symbolic_argument_is_refused_not_guessed(self):
        outs = _run("x = chr(n)\nif x == 'B':\n    raise IndexError('r')\n")
        assert _only_non_sat(outs) and any(o == "unsupported" for o, _ in outs)

    @pytest.mark.parametrize("call", ["ord(5)", "ord('ab')", "chr('a')", "chr(1.5)", "chr(99999999)"])
    def test_a_call_that_raises_natively_is_refused_and_its_result_unmodelled(self, call):
        # the native call raises (TypeError/ValueError), which the fold turns into Unsupported:
        # the statement is skipped and `x` is poisoned, exactly as before this batch, so a site
        # that DEPENDS on x is never claimed reachable
        outs = _run(f"x = {call}\nif x == 1:\n    raise IndexError('r')\n")
        assert _only_non_sat(outs) and any(o == "unsupported" for o, _ in outs)


class TestSequences:
    def test_tuple_keeps_a_symbolic_element_untouched(self):
        assert {"n": 4} in _sat(_run("t = tuple([n, 2])\nif t[0] == 4:\n    raise IndexError('r')\n"))

    def test_tuple_and_list_with_no_argument_are_empty(self):
        assert _sat(_run("if len(tuple()) == 0:\n    raise IndexError('r')\n"))
        assert _sat(_run("if len(list()) == 0:\n    raise IndexError('r')\n"))

    def test_list_of_a_str_is_its_characters(self):
        assert _sat(_run("l = list('abc')\nif l[1] == 'b':\n    raise IndexError('r')\n"))
        assert _only_non_sat(_run("l = list('abc')\nif l[1] == 'c':\n    raise IndexError('r')\n"))

    def test_list_of_a_list_is_a_copy(self):
        body = "a = [1, n]\nb = list(a)\nb[0] = 7\nif a[0] == 1 and b[0] == 7:\n    raise IndexError('r')\n"
        assert _sat(_run(body))

    def test_a_symbolic_length_list_is_refused_not_guessed(self):
        outs = _run("xs = [0 for i in range(n)]\nt = tuple(xs)\nif len(t) == 3:\n    raise IndexError('r')\n")
        assert _only_non_sat(outs) and any(o == "unsupported" for o, _ in outs)

    def test_set_stays_unsupported(self):
        outs = _run("s = set([1, 2])\nif 1 in s:\n    raise IndexError('r')\n")
        assert _only_non_sat(outs) and any(o == "unsupported" for o, _ in outs)


class TestLazyIterators:
    def test_zip_consumed_by_list(self):
        outs = _run("p = list(zip([1, 2, 3], [n, 5, 6]))\nif p[0][1] == 7:\n    raise IndexError('r')\n")
        assert {"n": 7} in _sat(outs)

    def test_zip_consumed_by_a_for_loop_with_tuple_targets(self):
        body = ("s = 0\nfor a, b in zip([1, 2], [n, 9]):\n    s = s + b\nif s == 20:\n"
                "    raise IndexError('r')\n")
        assert _sat(_run(body)) == [{"n": 11}]

    def test_zip_stops_at_the_shortest(self):
        body = "p = list(zip([1, 2, 3], [n, 5]))\nif len(p) == 2:\n    raise IndexError('r')\n"
        assert _sat(_run(body))
        body = "p = list(zip([1, 2, 3], [n, 5]))\nif len(p) == 3:\n    raise IndexError('r')\n"
        assert _only_non_sat(_run(body))

    def test_enumerate_counts_from_zero_and_from_start(self):
        body = ("total = 0\nfor i, v in enumerate([10, 20, n]):\n    total = total + i * v\n"
                "if total == 30:\n    raise IndexError('r')\n")
        assert _sat(_run(body)) == [{"n": 5}]
        assert _sat(_run("if list(enumerate(['x'], 5))[0][0] == 5:\n    raise IndexError('r')\n"))
        assert _only_non_sat(_run("if list(enumerate(['x'], 5))[0][0] == 0:\n    raise IndexError('r')\n"))

    def test_enumerate_with_a_symbolic_start_is_refused(self):
        outs = _run("p = list(enumerate(['x'], n))\nif p[0][0] == 3:\n    raise IndexError('r')\n")
        assert _only_non_sat(outs)

    def test_reversed_reverses(self):
        assert {"n": 9} in _sat(_run("r = list(reversed([1, 2, n]))\nif r[0] == 9:\n    raise IndexError('r')\n"))
        assert _only_non_sat(_run("r = list(reversed([1, 2, 3]))\nif r[0] == 1:\n    raise IndexError('r')\n"))

    def test_sorted_and_sum_consume_an_iterator(self):
        assert _sat(_run("q = sorted(zip([3, 1, 2], [n, 0, 0]))\nif q[0][0] == 1:\n    raise IndexError('r')\n"))
        assert _sat(_run("s = sum(x for x in reversed([1, 2, n]))\nif s == 10:\n    raise IndexError('r')\n"))

    def test_an_iterator_read_twice_is_refused_because_the_second_read_is_empty(self):
        # in the real program the second list() is [], so `len(b) == 0` IS reachable; the executor
        # models neither exhaustion nor a second read, so it must refuse, never claim "unreached"
        outs = _run("e = enumerate([1, 2])\na = list(e)\nb = list(e)\nif len(b) == 0:\n    raise IndexError('r')\n")
        assert any(o == "unsupported" for o, _ in outs) and not any(o in ("unreached", "unsat") for o, _ in outs)

    def test_an_iterator_is_not_a_list(self):
        outs = _run("e = zip([1], [2])\nx = e[0]\nif x == (1, 2):\n    raise IndexError('r')\n")
        assert not _sat(outs)


class TestType:
    def test_type_of_an_iterator_is_not_a_list(self):
        outs = _run("if type(zip([1], [2])) == list:\n    raise IndexError('r')\n")
        assert not _sat(outs)

    def test_type_of_a_symbolic_int_is_int(self):
        assert _sat(_run("if type(n) == int:\n    raise IndexError('r')\n"))
        assert _only_non_sat(_run("if type(n) == str:\n    raise IndexError('r')\n"))

    def test_type_of_literals(self):
        for lit, cls in (("1", "int"), ("True", "bool"), ("1.5", "float"), ("'s'", "str"), ("[1]", "list"),
                         ("(1,)", "tuple"), ("None", "type(None)")):
            assert _sat(_run(f"if type({lit}) == {cls}:\n    raise IndexError('r')\n")), (lit, cls)
        assert _only_non_sat(_run("if type(True) == int:\n    raise IndexError('r')\n"))  # bool is not int
