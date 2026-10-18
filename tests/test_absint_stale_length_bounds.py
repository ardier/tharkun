"""A summary bound that names a list's length must move with the length (a wrong TRUE on sv_2027).

``Analyzer.set_len`` changes a list's length variable ``L`` and re-expresses every stored summary bound
that names ``L`` so it keeps its meaning.  The object it returns (the replacement the caller built from the
stored one, and is about to store) was not re-expressed, so an element recorded as "``len(xs)`` when it was
stored" named ``L`` by its OLD value and read one length too high or too low after an append::

    a = nd(); xs = [0, 1]; xs[a] = len(xs); xs.append(5); c = 1 // xs[nd()]

stores 2 at ``a``, so ``xs`` is ``[0, 2, 5]`` for ``a == 1`` and ``xs[0]`` is 0; absint said the divisor was
in ``[1, 6]``.  The fix rewrites the in-flight object with the stored ones (``State.shift_bounds`` and
``State.rewrite_bounds_away`` take it).

Every program here is also run natively over every draw that matters: a TRUE for a class the program raises is a
failure, whatever the program looks like.
"""

from __future__ import annotations

import itertools
import random
import sys
import textwrap
import types

import pytest

from tharkun.solve.absint import prove

HEAD = "import _sv_verifier\ndef nd():\n    return _sv_verifier.nondet_int()\n"

LISTS = ["[0, 1]", "[1, 2]", "[3]", "[5, 5, 5]"]
STORES = [
    "xs[a] = len(xs)",
    "xs[a] = len(xs) - 1",
    "xs[a] = len(xs) - 2",
    "n = len(xs)\nxs[a] = n",
    "xs[a] = 0",
    "xs[nd()] = len(xs)",
]
CHANGES = [
    "xs.append(5)",
    "xs.append(len(xs))",
    "xs.append(xs[0])",
    "xs.insert(0, 7)",
    "xs.insert(1, len(xs))",
    "xs.extend([4, 4])",
    "xs.extend([len(xs)])",
    "xs.pop()",
    "xs.pop(0)",
    "xs += [9]",
    "xs += [len(xs)]",
    "xs[0:1] = [8, 8]",
    "xs *= 2",
    "xs.append(5)\nxs.append(6)",
    "xs.pop()\nxs.append(len(xs))",
]
READS = [
    "c = 1 // xs[nd()]",
    "assert xs[nd()] != 0",
    "assert xs[nd()] >= 1",
    "c = 1 // (xs[nd()] - 1)",
]
CLASSES = ("ZeroDivisionError", "AssertionError", "IndexError")
DRAWS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def native_raises(src: str) -> set[str]:
    """Every exception class some pair of draws (the store's index, the read's index) raises."""
    out: set[str] = set()
    code = compile(src, "<program>", "exec")
    for draws in itertools.product(DRAWS, DRAWS):
        it = iter(draws * 4)
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


def program(init: str, store: str, change: str, read: str) -> str:
    return HEAD + "a = nd()\nxs = " + init + "\n" + store + "\n" + change + "\n" + read + "\n"


def test_the_reported_program_is_not_proven():
    src = program("[0, 1]", "xs[a] = len(xs)", "xs.append(5)", "c = 1 // xs[nd()]")
    assert "ZeroDivisionError" in native_raises(src)
    assert prove(src, ["ZeroDivisionError"]).verdict != "TRUE"


@pytest.mark.parametrize("change", CHANGES)
def test_a_length_change_after_a_store_never_makes_a_wrong_true(change):
    rng = random.Random(1121)
    cases = [(i, s, r) for i in LISTS for s in STORES for r in READS]
    rng.shuffle(cases)
    for init, store, read in cases[:36]:
        src = program(init, store, change, read)
        raised = native_raises(src)
        for cls in CLASSES:
            if prove(src, [cls]).verdict == "TRUE":
                assert cls not in raised, f"WRONG TRUE for {cls}:\n{src}"


def test_the_bound_still_proves_what_is_true():
    """Precision control: the stored length and the appended elements are all at least 1, so the division is
    safe, and the fix must not turn that proof into an UNKNOWN."""
    for change in ("xs.append(5)", "xs.append(len(xs))", "xs.insert(0, 7)", "xs.extend([4, 4])", "xs += [9]"):
        src = program("[1, 2]", "xs[a] = len(xs)", change, "c = 1 // xs[nd()]")
        assert "ZeroDivisionError" not in native_raises(src), change
        result = prove(src, ["ZeroDivisionError"])
        assert result.verdict == "TRUE", (change, result.reason)


def test_a_segmented_store_followed_by_a_length_change():
    # the prefix pointer P is the other variable summaries name: a store at xs[i] extends the prefix and
    # the stored element's bound must move with it
    src = textwrap.dedent("""
        import _sv_verifier
        n = _sv_verifier.nondet_int()
        xs = [0] * 3
        for i in range(3):
            xs[i] = i + 1
        xs.append(0)
        c = 1 // xs[n]
    """)
    assert "ZeroDivisionError" in native_raises(src)
    assert prove(src, ["ZeroDivisionError"]).verdict != "TRUE"
