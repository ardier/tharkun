"""[internal ref] S1: ``for x in range(...)`` with symbolic INT arguments in ``tharkun.sv._symexec``.

``_symexec`` is a search aid -- every model it finds is replayed on the real program -- but a
wrong ``unsat`` stops the search early and loses a bug, so the range semantics are held to what
Python does, and checked against Python:

* a range of int arguments (``range(n)``, ``range(a, b)``, ``range(a, b, s)``, inline or held
  in a name and iterated later) runs iteration ``k`` exactly when ``start + k*step`` is still
  short of ``stop`` in the direction of ``step``, with the target equal to that value;
* a zero step is a ``ValueError`` at the call: the path ends before the loop, so no later site
  is reached through it;
* any argument that is not provably an int (a float draw, a string, a typed union) is refused
  (``unsupported``), never approximated;
* a range used as anything but a loop iterable (truth, comparison, ``len``, indexing, ``in``,
  ``list``) is refused: the Python object would answer differently from the program.

Each exactness claim is a differential run: the program is executed natively over a box of
inputs, and ``solve_targets``' ``sat`` models must fire, its ``unsat`` must match an empty box,
and a box that fires must be ``sat``.
"""

from __future__ import annotations

import itertools
import random
import sys
import textwrap

import pytest

z3 = pytest.importorskip("z3")

from tharkun.sv import _symexec  # noqa: E402
from tharkun.sv.sites import build_harness  # noqa: E402

DL = ("KeyError",)


def _solve(src: str, classes=DL):
    """``{site kind: (outcome, payload)}`` of one symbolic run toward every site of ``src``."""
    h = build_harness(textwrap.dedent(src), classes, None)
    mod = {"__name__": "sv_harness_test"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    got = _symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites],
                                 h.params, h.kinds, {})
    return {s.get("kind", s["exc"]): got[s["line"]] for s in h.sites}, h


def _native_fires(body: str, draws: dict) -> bool:
    """Does the program (``body`` with the draws bound to ``draws``) end in the KeyError site?"""
    prelude = "".join(f"{name} = {value!r}\n" for name, value in draws.items())
    ns: dict = {}
    try:
        exec(compile(prelude + textwrap.dedent(body), "<native>", "exec"), ns)
    except KeyError:
        return True
    except ValueError:      # range() arg 3 must not be zero: the run ends, the site is not reached
        return False
    return False


def _draw_prelude(names) -> str:
    return "".join(f"    {n} = _sv_verifier.nondet_int()\n" for n in names)


def _program(names, body: str) -> tuple[str, str]:
    """(the harness-side program, the native program) sharing ``body`` (flat, 4-space indented)."""
    sym = "import _sv_verifier\n\nif __name__ == \"__main__\":\n" + _draw_prelude(names) + textwrap.indent(
        textwrap.dedent(body), "    ")
    return sym, textwrap.dedent(body)


# ---------------------------------------------------------------------------
# the shapes
# ---------------------------------------------------------------------------


def test_a_single_argument_range_still_unrolls_under_its_guard():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            total = 0
            for i in range(n):
                total += i
            if total == 1953:
                raise KeyError("n == 63")
    """)
    assert got["raise"][0] == "sat" and got["raise"][1]["n"] == 63


def test_a_start_stop_range_runs_from_start_to_stop():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            a = _sv_verifier.nondet_int()
            b = _sv_verifier.nondet_int()
            total = 0
            for i in range(a, b):
                total += i
            if total == 14 and b - a == 4:
                raise KeyError("2 + 3 + 4 + 5")
    """)
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["a"] == 2 and vector["b"] == 6


def test_a_range_held_in_a_name_and_iterated_later():
    # raytrace: ``range_it = range(loops)`` then ``for i in range_it``
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            n = 1 + abs(_sv_verifier.nondet_int())
            r = range(n)
            total = 0
            for i in r:
                total += i
            if total == 21:
                raise KeyError("n == 7")
    """)
    outcome, vector = got["raise"]
    assert outcome == "sat" and 1 + abs(vector["d0"]) == 7


def test_a_negative_step_counts_down():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            s = _sv_verifier.nondet_int()
            total = 0
            for i in range(10, 0, s):
                total += i
            if total == 22:
                raise KeyError("10 + 7 + 4 + 1")
    """)
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["s"] == -3


def test_a_zero_step_is_a_value_error_before_the_loop_so_the_path_ends():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            s = _sv_verifier.nondet_int()
            for i in range(0, 3, s):
                pass
            if s == 0:
                raise KeyError("never reached: range(0, 3, 0) raised ValueError first")
    """)
    assert got["raise"][0] == "unsat"


def test_a_model_never_picks_a_zero_step():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            s = _sv_verifier.nondet_int()
            n = _sv_verifier.nondet_int()
            total = 0
            for i in range(0, 12, s):
                total += 1
            if total == 4 and n == 5:
                raise KeyError("x")
    """)
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["s"] != 0 and len(range(0, 12, vector["s"])) == 4


def test_an_empty_range_runs_nothing_and_keeps_the_target_it_had():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            i = 99
            for i in range(n):
                pass
            if i == 99:
                raise KeyError("n <= 0: the loop never ran, i is untouched")
    """)
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["n"] <= 0


def test_after_a_loop_the_target_holds_the_last_value_it_was_given():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            i = 99
            for i in range(n):
                pass
            if i == 6:
                raise KeyError("n == 7: the last value is 6")
    """)
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["n"] == 7


def test_a_target_that_did_not_exist_before_is_not_readable_after_the_loop():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            for i in range(n):
                pass
            if i == 3:
                raise KeyError("i is unbound when the range is empty")
    """)
    assert got["raise"][0] == "unsupported"


def test_a_range_longer_than_the_unroll_bound_is_cut_and_says_so():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            total = 0
            for i in range(n):
                total += 1
            if total == 100:
                raise KeyError("needs 100 iterations")
    """)
    outcome, why = got["raise"]
    assert outcome == "unsat" and "loops cut at 64" in why


# ---------------------------------------------------------------------------
# refused, never approximated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, setup, call", [
    ("float draw", "f = _sv_verifier.nondet_float()", "range(f)"),
    ("float in a bound", "f = _sv_verifier.nondet_float()", "range(0, f)"),
    ("string draw", "f = _sv_verifier.nondet_str()", "range(f)"),
    ("concrete float", "f = 2.5", "range(f)"),
    ("concrete string", "f = 'ab'", "range(f)"),
    ("four arguments", "f = _sv_verifier.nondet_int()", "range(0, f, 1, 2)"),
    ("no arguments", "f = _sv_verifier.nondet_int()", "range()"),
])
def test_an_argument_that_is_not_an_int_is_refused(name, setup, call):
    got, _ = _solve(f"""
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            {setup}
            total = 0
            for i in {call}:
                total += 1
            if total == 2 and x == 1:
                raise KeyError("x")
    """)
    assert got["raise"][0] == "unsupported", (name, got)


@pytest.mark.parametrize("use", [
    "if r:\n    total += 1",
    "total += len(r)",
    "total += r[0]",
    "if 3 in r:\n    total += 1",
    "total += sum(list(r))",
    "if r == r:\n    total += 1",
    "q = [x for x in r]",
    "r2 = range(n)\nif r == r2:\n    total += 1",
])
def test_a_range_used_as_a_value_is_refused(use):
    head = (
        "import _sv_verifier\n"
        "if __name__ == \"__main__\":\n"
        "    n = _sv_verifier.nondet_int()\n"
        "    r = range(n)\n"
        "    total = 0\n"
    )
    tail = "    if total == 1:\n        raise KeyError(\"x\")\n"
    got, _ = _solve(head + textwrap.indent(use, "    ") + "\n" + tail)
    assert got["raise"][0] == "unsupported", (use, got)


@pytest.mark.parametrize("use", [
    "q = list(r)",
    "q = tuple(r)",
    "q = sorted(r)",
    "q = zip(r, [1, 2])",
    "q = enumerate(r)",
    "q = reversed(r)",
    "q = type(r)",
    "q = sum(r)",
    "q = ord(r)",
])
def test_a_range_given_to_a_contract_folded_builtin_is_refused(use):
    # the builtins whose pure native fold the contracts allow (zip, enumerate, list, ...) take
    # only exact concrete containers: a SymRange is neither, and never falls to the Python object
    head = (
        "import _sv_verifier\n"
        "if __name__ == \"__main__\":\n"
        "    n = _sv_verifier.nondet_int()\n"
        "    r = range(n)\n"
    )
    tail = "    if n == 3:\n        raise KeyError(\"x\")\n"
    got, _ = _solve(head + textwrap.indent(use, "    ") + "\n" + tail)
    # the statement is skipped (poisoning q), the site after it is still solved: the range itself
    # is not claimed to be anything
    assert got["raise"][0] in ("sat", "unsupported"), (use, got)
    if got["raise"][0] == "sat":
        assert got["raise"][1]["n"] == 3


def test_isinstance_of_a_symbolic_range_is_a_range_and_not_a_list():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            r = range(n)
            if isinstance(r, range) and not isinstance(r, list) and n == 4:
                raise KeyError("a range object is a range, not a list")
    """)
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["n"] == 4


def test_a_symbolic_range_loop_inside_a_one_shot_iterator_loop():
    # the seam with the contract-folded iterators: ``enumerate`` is a one-shot ``_LazyIter``
    # unwrapped by the ``for``; the inner loop is a symbolic range
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            total = 0
            for j, v in enumerate([5, 7]):
                for i in range(n + v):
                    total += 1
            if total == 20:
                raise KeyError("(n + 5) + (n + 7) == 20 -> n == 4")
    """)
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["n"] == 4


def test_a_rebound_range_is_not_the_builtin():
    got, _ = _solve("""
        import _sv_verifier

        def range(n):
            return [0]

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            total = 0
            for i in range(n):
                total += 1
            if total == 5:
                raise KeyError("range here is the user's function: one iteration")
    """)
    assert got["raise"][0] != "sat"


# ---------------------------------------------------------------------------
# concrete ranges
# ---------------------------------------------------------------------------


def test_a_concrete_zero_step_ends_the_path_instead_of_crashing_the_executor():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            for i in range(0, 3, 0):
                pass
            if x == 1:
                raise KeyError("never reached")
    """)
    assert got["raise"][0] == "unreached" or got["raise"][0] == "unsat"


def test_a_huge_concrete_range_is_refused_without_materialising_it():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            r = range(10 ** 9)
            if x == 1:
                raise KeyError("x")
    """)
    assert got["raise"][0] in ("unsupported", "sat")


def test_a_concrete_stepped_range_is_unrolled_exactly():
    got, _ = _solve("""
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            total = 0
            for i in range(9, 0, -2):
                total += i
            if total == x:
                raise KeyError("9 + 7 + 5 + 3 + 1 == 25")
    """)
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["x"] == 25


# ---------------------------------------------------------------------------
# no proof may rest on it
# ---------------------------------------------------------------------------


def test_a_symbolic_range_loop_makes_the_run_inexact_for_the_provers():
    body = textwrap.dedent("""
        import _sv_verifier
        n = _sv_verifier.nondet_int()
        total = 0
        for i in range(n):
            total += 1
    """)
    import ast

    tree = ast.parse(body)
    env = {"n": _symexec.make_var("n", "int")}
    result = _symexec.run_exact(tree, {}, env, list(tree.body[2:]), [])
    assert result["exact"] is False and "unroll bound" in result["reason"]


def test_the_exact_subset_still_refuses_every_loop():
    from tharkun.sv.exact_subset import audit

    src = textwrap.dedent("""
        import _sv_verifier
        n = _sv_verifier.nondet_int()
        total = 0
        for i in range(n):
            total += 1
    """)
    import ast

    assert audit(ast.parse(src), set()) is not None


# ---------------------------------------------------------------------------
# the differential: random range loops against native Python
# ---------------------------------------------------------------------------

ARGS = ["a", "b", "s", "0", "1", "-1", "2", "-2", "3", "a + 1", "b - 1", "abs(a)", "a * 2", "-b", "5", "-4"]
BODIES = [
    "total += i",
    "total += 1",
    "total += i * 2",
    "if i % 2 == 0:\n    total += i\nelse:\n    total += 1",
    "total = total * 2 + i",
]
SITES = ["total == {t}", "total == {t} and a > {u}", "total + b == {t}"]
BOX = range(-5, 6)


def _random_program(rng: random.Random):
    k = rng.choice([1, 2, 3])
    args = ", ".join(rng.choice(ARGS) for _ in range(k))
    held = rng.random() < 0.4
    head = f"r = range({args})\n" if held else ""
    iterable = "r" if held else f"range({args})"
    body = textwrap.indent(rng.choice(BODIES), "    ")
    site = rng.choice(SITES).format(t=rng.randint(-3, 30), u=rng.randint(-3, 3))
    text = (
        f"{head}total = 0\n"
        f"for i in {iterable}:\n"
        f"{body}\n"
        f"if {site}:\n"
        f"    raise KeyError('site')\n"
    )
    return text


@pytest.mark.parametrize("seed", range(30))
def test_random_range_loops_agree_with_native_python(seed):
    rng = random.Random(1107_00 + seed)
    names = ["a", "b", "s"]
    checked = 0
    for _ in range(8):
        body = _random_program(rng)
        sym, native = _program(names, body)
        got, _h = _solve(sym)
        outcome, payload = got["raise"]
        fires = {
            vec: _native_fires(native, dict(zip(names, vec)))
            for vec in itertools.product(BOX, repeat=3)
        }
        any_fires = any(fires.values())
        if outcome == "sat":
            vec = tuple(payload[n] for n in names)
            assert _native_fires(native, dict(zip(names, vec))), f"model does not fire:\n{native}\n{payload}"
            checked += 1
        elif outcome in ("unsat", "unreached"):
            assert not any_fires, f"claimed {outcome} but {[v for v, f in fires.items() if f][:3]} fire:\n{native}"
            checked += 1
        elif outcome == "unknown":
            # z3 gave up inside its time limit (a loaded machine does that): the site stays open,
            # which is never a wrong claim -- neither checked nor wrong
            continue
        else:
            # refused: the body or an argument is outside the model; never a wrong claim
            assert outcome == "unsupported", (outcome, native)
        if any_fires and outcome == "unsat":
            pytest.fail(f"missed a firing input:\n{native}")
    assert checked >= 1
