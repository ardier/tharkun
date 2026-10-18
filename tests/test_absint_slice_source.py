"""A slice assignment moves the elements of its source into the list ([internal ref]).

``Analyzer.slice_store`` skipped the elements of a source that is a range, a dict (its keys), a dict view of keys or values or an
array: such a source fell into the branch for a source that is NOT iterable (a recorded ``TypeError``), and the path went on with
the list's OLD elements.  ``xs = [1]; xs[0:1] = range(0, 1); 10 // xs[0]`` was proven safe on sv_2027 4e129af and raises
``ZeroDivisionError``.  Found while auditing the consumers of a new heap kind for the places that treat an unknown kind as "bad, but
go on".

The regressions fail on 4e129af and pass here; a native differential runs every combination of list, slice, source and read (a TRUE
for a class the program raises fails the test); and the negative control undoes the fix and the differential must catch it.
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
# [internal ref]: the elements of a slice assignment's source are in the list afterwards
# ---------------------------------------------------------------------------

SOURCES_WITH_A_ZERO = ["range(0, 1)", "{0: 5}", "{0: 5}.keys()", "{5: 0}.values()", "array('i', [0])", "(0,)", "[0]"]


@pytest.mark.parametrize("source", SOURCES_WITH_A_ZERO)
def test_a_slice_assignment_moves_the_elements_of_every_iterable_source_into_the_list(source):
    body = f"xs = [1]\nxs[0:1] = {source}\nc = 10 // xs[0]\n"
    assert native(HEAD + body) == "ZeroDivisionError"
    assert verdict(body, "ZeroDivisionError") == "UNKNOWN"


@pytest.mark.parametrize("source", ["range(2, 3)", "{2: 5}", "{2: 5}.keys()", "{5: 2}.values()", "array('i', [2])", "(2,)", "[2]"])
def test_and_proves_what_those_elements_allow(source):
    body = f"xs = [1]\nxs[0:1] = {source}\nc = 10 // xs[0]\n"
    assert native(HEAD + body) is None
    assert verdict(body, "ZeroDivisionError") == "TRUE"


def test_a_range_source_keeps_its_bounds():
    body = "xs = [1]\nxs[0:1] = range(5, 8)\nassert xs[0] >= 1\nassert xs[0] <= 7\n"
    assert verdict(body, "AssertionError") == "TRUE"
    assert verdict(body.replace("<= 7", "<= 6"), "AssertionError") == "UNKNOWN"


def test_dict_items_has_no_element_summary_and_is_refused():
    r = prove(HEAD + "xs = [1]\nxs[0:1] = {1: 2}.items()\n", ["TypeError"], filename="<program>")
    assert r.verdict == "UNKNOWN" and r.reason.startswith("unsupported:") and "items" in r.reason


def test_a_source_that_is_not_iterable_is_still_a_typeerror_check():
    assert verdict("xs = [1]\nxs[0:1] = 5\n", "TypeError") == "UNKNOWN"
    assert verdict("xs = [1]\nxs[0:1] = range(3)\n", "TypeError") == "TRUE"
    assert verdict("xs = [1]\nxs[0:1] = {1: 2}\n", "TypeError") == "TRUE"
    assert verdict("xs = [1]\nxs[0:1] = array('i', [3])\n", "TypeError") == "TRUE"


# ---------------------------------------------------------------------------
# the differential: every combination, run natively over the draws that matter
# ---------------------------------------------------------------------------

LISTS = ["xs = [1]", "xs = [1, 2]", "xs = [0, 1, 2]", "xs = []", "xs = [nd(), 3]"]
SLICES = ["xs[0:1]", "xs[:]", "xs[1:]", "xs[:1]", "xs[0:0]", "xs[1:1]", "xs[::1]", "xs[len(xs):]", "xs[-1:]"]
SOURCE_SETUPS = [
    "range(0, 1)", "range(2)", "range(nd() % 3)", "range(1, 3)", "{0: 5}", "{1: 0, 2: 3}", "{0: 5}.keys()", "{5: 0}.values()",
    "{1: 0, 2: 3}.values()", "array('i', [0])", "array('i', [1, 2])", "array('f', [0.5])", "array('i', [nd(), 1])", "array('i')",
    "(0,)", "(1, 0)", "[0]", "[]", "'ab'", "()",
]
READS = [
    "c = 10 // xs[0]", "c = 10 // xs[-1]", "c = 10 // (xs[0] - 1)", "assert xs[0] >= 1", "assert xs[0] != 0", "assert len(xs) == 2",
    "assert len(xs) >= 1", "c = 10 // len(xs)", "c = xs[len(xs)]", "assert xs[0] == 1", "assert xs[-1] <= 2", "c = 10 // sum(xs)",
    "c = 10 // min(xs)", "c = 10 // max(xs)",
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


def slice_programs():
    return [HEAD + f"{lst}\n{sl} = {src}\n{read}\n" for lst in LISTS for sl in SLICES for src in SOURCE_SETUPS for read in READS]


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
def test_no_slice_store_gets_a_wrong_true(shard):
    wrong, proved = wrong_trues(_shard(slice_programs(), shard, 60))
    assert not wrong, "WRONG TRUE for " + wrong[0]
    assert proved > 100


# ---------------------------------------------------------------------------
# negative controls: each fix undone is caught
# ---------------------------------------------------------------------------


def test_the_differential_catches_the_slice_fix_undone(monkeypatch):
    """The source's elements are not joined into the list (the sv_2027 behaviour for a range, a dict, a view, an array)."""
    from tharkun.solve.absint import interp

    monkeypatch.setattr(interp.Analyzer, "elem_summary", lambda self, st, v: None)
    wrong, _ = wrong_trues([HEAD + "xs = [1]\nxs[0:1] = range(0, 1)\nc = 10 // xs[0]\n"])
    assert wrong and wrong[0].startswith("ZeroDivisionError")
