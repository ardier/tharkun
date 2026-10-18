"""A bound method of an array is the array's method ([internal ref]).

``Analyzer._call1`` sent every bound builtin method to ``list_method``: ``m = a.append`` on an array ran the LIST's append, which
skips the typecode of an element write (``_array_store_value``).  On an ``'f'`` array ``m(0.1); assert a[0] == 0.1`` was proven on
sv_2027 4e129af and is false (CPython stores the float32 0.10000000149011612); the TypeError of ``m('x')`` on an ``'i'`` array was
not checked either.

The regressions fail on 4e129af and pass here; a native differential runs every combination of array, call and read (a TRUE for a
class the program raises fails the test); and the negative control undoes the fix and the differential must catch it.
"""

from __future__ import annotations

import itertools
import random
import sys
import textwrap
import types

import pytest

from tharkun.solve.absint import prove

HEAD = "import _sv_verifier\nfrom array import array\ndef nd():\n    return _sv_verifier.nondet_int()\n"
CLASSES = ("ZeroDivisionError", "AssertionError", "IndexError", "TypeError", "KeyError")


def verdict(body: str, cls: str) -> str:
    return prove(HEAD + textwrap.dedent(body), [cls], filename="<program>").verdict


def native(src: str):
    """The exception class name a native run of ``src`` raises (``None``: none), every draw 0."""
    mod = types.ModuleType("_sv_verifier")
    mod.nondet_int = lambda: 0
    saved = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = mod
    try:
        exec(compile(src, "<native>", "exec"), {"__name__": "__main__"})
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    finally:
        if saved is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = saved
    return None


# ---------------------------------------------------------------------------
# [internal ref]: a bound method of an array is the array's method
# ---------------------------------------------------------------------------


def test_a_bound_append_of_an_f_array_stores_the_rounded_float_not_the_exact_one():
    body = "a = array('f')\nm = a.append\nm(0.1)\nassert a[0] == 0.1\n"
    assert native(HEAD + body) == "AssertionError"
    assert verdict(body, "AssertionError") == "UNKNOWN"
    # an exactly representable value is stored as it is
    assert verdict(body.replace("0.1", "0.5"), "AssertionError") == "TRUE"


def test_a_bound_method_and_the_direct_call_agree():
    for call in ("a.append(0.1)", "m = a.append\nm(0.1)"):
        assert verdict(f"a = array('f')\n{call}\nassert a[0] == 0.1\n", "AssertionError") == "UNKNOWN"


def test_a_bound_extend_checks_the_typecode_too():
    assert verdict("a = array('i')\nm = a.extend\nm(['x'])\n", "TypeError") == "UNKNOWN"
    assert verdict("a = array('i')\nm = a.append\nm('x')\n", "TypeError") == "UNKNOWN"
    assert verdict("a = array('i')\nm = a.append\nm(3)\n", "TypeError") == "TRUE"
    assert verdict("a = array('B')\nm = a.append\nm(3)\nassert len(a) == 1\n", "AssertionError") == "TRUE"


def test_a_bound_method_of_a_list_is_still_the_lists():
    assert verdict("xs = [1]\nm = xs.append\nm(2)\nassert len(xs) == 2\n", "AssertionError") == "TRUE"
    assert verdict("xs = []\nm = xs.pop\nm()\n", "IndexError") == "UNKNOWN"


def test_a_bound_method_whose_receiver_may_be_a_list_or_an_array_refuses():
    body = "xs = [1]\na = array('i', [1])\nr = xs if nd() > 0 else a\nm = r.append\nm(2)\n"
    result = prove(HEAD + body, ["TypeError"], filename="<program>")
    assert result.verdict == "UNKNOWN"


# ---------------------------------------------------------------------------
# the differential: every combination, run natively over the draws that matter
# ---------------------------------------------------------------------------

ARRAY_SETUPS = ["a = array('f')", "a = array('d')", "a = array('i')", "a = array('B')", "a = array('f', [1.0])", "a = array('b', [1])"]
ARRAY_CALLS = [
    "a.append(0.1)", "m = a.append\nm(0.1)", "m = a.append\nm(1)", "m = a.append\nm(300)", "m = a.append\nm('x')", "m = a.append\nm(2.5)",
    "m = a.extend\nm([0.1, 2])", "m = a.extend\nm([1])", "m = a.extend\nm(['x'])", "m = a.extend\nm(a)", "a.extend([0.1])", "m = a.append\nm(nd())",
    "m = a.append\nm(True)",
]
ARRAY_READS = [
    "assert a[0] == 0.1", "assert a[0] == 1", "assert a[0] == 0.5", "c = 10 // a[0]", "assert len(a) == 1", "assert a[-1] >= 1",
    "assert a[0] != 0.1", "assert a[0] <= 2.5",
]
DRAWS = [-2, 0, 1, 2, 3]


def native_raises(src: str) -> set[str]:
    out: set[str] = set()
    code = compile(src, "<program>", "exec")
    for draws in itertools.product(DRAWS, DRAWS):
        it = itertools.cycle(draws)
        mod = types.ModuleType("_sv_verifier")
        mod.nondet_int = lambda: next(it)
        saved = sys.modules.get("_sv_verifier")
        sys.modules["_sv_verifier"] = mod
        try:
            exec(code, {"__name__": "__main__"})
        except Exception as e:  # noqa: BLE001
            out |= {c.__name__ for c in type(e).__mro__}
        finally:
            if saved is None:
                sys.modules.pop("_sv_verifier", None)
            else:
                sys.modules["_sv_verifier"] = saved
    return out


def array_programs():
    return [HEAD + f"{setup}\n{call}\n{read}\n" for setup in ARRAY_SETUPS for call in ARRAY_CALLS for read in ARRAY_READS]


def wrong_trues(sources):
    wrong, proved = [], 0
    for src in sources:
        raised = native_raises(src)
        for cls in CLASSES:
            if prove(src, [cls], filename="<program>").verdict == "TRUE":
                proved += 1
                if cls in raised:
                    wrong.append(f"{cls}:\n{src}")
    return wrong, proved


SHARDS = 8


def _shard(programs, shard, n):
    rng = random.Random(1138)
    every = list(programs)
    rng.shuffle(every)
    return every[shard::SHARDS][:n]


@pytest.mark.parametrize("shard", range(SHARDS))
def test_no_bound_array_method_gets_a_wrong_true(shard):
    wrong, proved = wrong_trues(_shard(array_programs(), shard, 60))
    assert not wrong, "WRONG TRUE for " + wrong[0]
    assert proved > 100


# ---------------------------------------------------------------------------
# negative controls: each fix undone is caught
# ---------------------------------------------------------------------------


def test_the_differential_catches_the_bound_method_fix_undone(monkeypatch):
    """A bound method of an array runs the list's method (the sv_2027 behaviour)."""
    from tharkun.solve.absint import interp

    monkeypatch.setattr(interp.Analyzer, "array_method", lambda self, st, addrs, name, args, kwargs, node:
                        self.list_method(st, addrs, name, args, kwargs, node))
    wrong, _ = wrong_trues([HEAD + "a = array('f')\nm = a.append\nm(0.1)\nassert a[0] == 0.1\n"])
    assert wrong and wrong[0].startswith("AssertionError")
