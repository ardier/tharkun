"""[internal ref]: the harness exposes the draws a program makes INSIDE the function it ends by calling
(``def main(): x = nondet_int(); ...`` then ``main()``) and AT MODULE LEVEL, so the symbolic stages apply.

One test per construct the lift models, and one per condition it REFUSES by name (``sites.py::harness_refusal``
is what ``concolic.py`` logs: "concolic: no harness or no site (<reason>)").  The harness is a search aid, so
what is pinned here is that its text is the program (the entry function's body IS the rest of the program) and
that the symbolic run through it is right on the early-return guard + assert shape the real tasks have: every
symbolic answer is compared with a native run of the unmodified program.
"""
from __future__ import annotations

import sys
import textwrap
import types

import pytest

pytest.importorskip("z3")

from tharkun.solve.call_targets import call_targets  # noqa: E402
from tharkun.solve.module_namespace import module_constants  # noqa: E402
from tharkun.sv import _symexec, verify  # noqa: E402
from tharkun.sv.sites import build_harness, harness_refusal  # noqa: E402

AR = ("AssertionError",)


def _task(tmp_path, src: str):
    path = tmp_path / "task.py"
    path.write_text(textwrap.dedent(src))
    return str(path), call_targets(str(path), module_constants(str(path)))


def _harness(tmp_path, src: str, records="auto"):
    """``(harness or None, the named reason when there is none)``; ``records``: the producer's, or what is given."""
    path, produced = _task(tmp_path, src)
    records = produced if records == "auto" else records
    text = open(path).read()
    return build_harness(text, AR, None, records), harness_refusal(text, AR, None, records)


def _solve(tmp_path, src: str):
    """``({site kind: outcome}, harness)`` from ONE symbolic run of the harness (the concolic worker's stage 2)."""
    h, why = _harness(tmp_path, src)
    assert h is not None, why
    sys.modules.setdefault("_sv_verifier", types.ModuleType("_sv_verifier"))
    mod = {"__name__": "sv_harness_test"}
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    got = _symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites], h.params, h.kinds, {})
    return [got[s["line"]] for s in sorted(h.sites, key=lambda s: s["line"])], h


def _native(src: str, draws: list):
    """Run the UNMODIFIED program with these draws: ``True`` when it ends in an ``AssertionError``."""
    stub = types.ModuleType("_sv_verifier")
    it = iter(draws)
    stub.nondet_int = lambda: next(it)
    stub.nondet_bool = lambda: bool(next(it))
    old = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = stub
    try:
        exec(compile(textwrap.dedent(src), "task.py", "exec"), {"__name__": "__main__"})
    except AssertionError:
        return True
    finally:
        if old is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old
    return False


# ---------------------------------------------------------------------------
# the entry function: what is lifted
# ---------------------------------------------------------------------------

ENTRY = """
    import _sv_verifier

    def main():
        x = _sv_verifier.nondet_int()
        if not (1 <= x <= 100):
            return
        assert x * x > x

    main()
"""


def test_the_body_of_the_function_the_program_ends_by_calling_is_the_harness(tmp_path):
    h, why = _harness(tmp_path, ENTRY)
    assert h is not None, why
    assert (h.params, h.kinds, h.entry) == (["x"], ["int"], "entry main()")
    # the body IS the program: the def and its call are gone, the early return stays a return
    assert "def main" not in h.source and "main()" not in h.source
    assert "def __sv_harness(x: int):" in h.source and "return" in h.source
    assert [s["kind"] for s in h.sites] == ["assert"]
    compile(h.source, "sv_harness.py", "exec")


def test_the_same_program_under_the_main_guard(tmp_path):
    h, why = _harness(tmp_path, ENTRY.replace("\n    main()\n", '\n    if __name__ == "__main__":\n        main()\n'))
    assert h is not None, why
    assert (h.params, h.entry) == (["x"], "entry main()")


def test_loop_draws_in_the_entry_function_are_streams(tmp_path):
    h, why = _harness(tmp_path, """
        import _sv_verifier

        def main():
            i = 0
            while i < 4:
                v = _sv_verifier.nondet_int()
                if not (0 <= v <= 3):
                    return
                i = i + 1
            assert i == 4

        main()
    """)
    assert h is not None, why
    assert h.kinds == ["list:int"] and h.params[0].startswith("s")
    assert "def __sv_harness(" + h.params[0] + ": list[int])" in h.source


def test_draws_in_a_function_main_calls_stay_with_the_stub(tmp_path):
    h, why = _harness(tmp_path, """
        import _sv_verifier

        def helper():
            return _sv_verifier.nondet_int()

        def main():
            x = _sv_verifier.nondet_int()
            assert x != helper()

        main()
    """)
    assert h is not None, why
    assert h.params == ["x"]                                     # main's own draw is lifted
    assert "def helper" in h.source and "_sv_verifier.nondet_int()" in h.source   # the callee's stays a stub call


def test_a_callee_of_main_is_inlined_with_the_lifted_draws(tmp_path):
    h, why = _harness(tmp_path, """
        import _sv_verifier

        def cdiv(a, b):
            return -(a // -b)

        def main():
            a = _sv_verifier.nondet_int()
            b = _sv_verifier.nondet_int()
            if not (0 <= a <= 1000 and 1 <= b <= 1000):
                return
            assert cdiv(a, b) * b >= a

        main()
    """)
    assert h is not None, why
    assert h.feeds == {"cdiv": {"a": "a", "b": "b"}}          # the def-use link of the call, as for a main block
    assert [s["kind"] for s in h.sites] == ["assert"]


# ---------------------------------------------------------------------------
# the symbolic run through the harness: the early-return guard, checked against the native program
# ---------------------------------------------------------------------------

def test_symbolic_run_solves_the_assert_behind_an_early_return_guard(tmp_path):
    src = """
        import _sv_verifier

        def main():
            x = _sv_verifier.nondet_int()
            if not (1 <= x <= 100):
                return
            assert x * x > x

        main()
    """
    outcomes, h = _solve(tmp_path, src)
    (outcome, inputs), = outcomes
    assert outcome == "sat" and inputs == {"x": 1}          # only x == 1 breaks x*x > x inside [1, 100]
    assert _native(src, [1]) and not _native(src, [2]) and not _native(src, [0])


def test_symbolic_run_proves_no_modelled_run_fires_a_safe_assert(tmp_path):
    outcomes, h = _solve(tmp_path, ENTRY.replace("x * x > x", "x * x >= x"))
    (outcome, _), = outcomes
    assert outcome == "unsat"          # a search hint only (the run is bounded): never a TRUE by itself


def test_each_site_is_answered_under_its_own_guard(tmp_path):
    src = """
        import _sv_verifier

        def main():
            x = _sv_verifier.nondet_int()
            y = _sv_verifier.nondet_int()
            if x > 5:
                if y > x:
                    return
                assert y != 2
            assert x != -3
            if x > 5 and y > x:
                assert False

        main()
    """
    outcomes, h = _solve(tmp_path, src)
    first, second, third = outcomes
    assert first[0] == "sat" and second[0] == "sat" and third[0] in ("unsat", "unreached")
    x, y = first[1]["x"], first[1]["y"]
    assert x > 5 and y <= x and y == 2 and _native(src, [x, y])
    x, y = second[1]["x"], second[1]["y"]
    assert x == -3 and _native(src, [x, y])
    assert not _native(src, [7, 100])                      # the early return really leaves the program


def test_a_value_returned_from_the_entry_function_is_still_evaluated(tmp_path):
    src = """
        import _sv_verifier

        def check(v):
            assert v != 9
            return v

        def main():
            x = _sv_verifier.nondet_int()
            return check(x)

        main()
    """
    outcomes, h = _solve(tmp_path, src)
    (outcome, inputs), = outcomes
    assert outcome == "sat" and inputs == {"x": 9} and _native(src, [9])


# ---------------------------------------------------------------------------
# the entry function: each condition that is refused, by name
# ---------------------------------------------------------------------------

BASE = "import _sv_verifier\n\n{before}\ndef main({params}):\n    {body}\n\n{after}\n"
BODY = "x = _sv_verifier.nondet_int()\n    assert x != 3"


def _entry(before="", params="", body=BODY, after="main()"):
    return BASE.format(before=before, params=params, body=body, after=after)


@pytest.mark.parametrize("name,src,reason", [
    ("arguments", _entry(params="a", after="main(1)"), "has arguments"),
    ("two calls", _entry(after="main()\nmain()"), "called more than once"),
    ("recursion", _entry(body=BODY + "\n    main()"), "called more than once"),
    ("used as a value", _entry(before="f = None", after="f = main\nmain()"), "referenced besides its one call"),
    ("parameters with defaults", _entry(params="a=1"), "takes parameters or is decorated"),
    ("decorated", _entry(before="def deco(f):\n    return f\n@deco"), "takes parameters or is decorated"),
    ("generator", _entry(body="x = _sv_verifier.nondet_int()\n    yield x"), "is a generator"),
    ("global statement", _entry(body="global G\n    x = _sv_verifier.nondet_int()\n    G = x"),
     "declares global"),
    ("defined twice", _entry(before="def main():\n    pass"), "defined more than once"),
    ("rebound after a call", _entry(after="main()\nmain = None\nmain()"), "does not prove main a user function"),
    ("value used", _entry(after="r = main()"), "does not END with the call main()"),
    ("statements follow", _entry(after="main()\nprint(1)"), "does not END with the call main()"),
    ("exit code", _entry(after="import sys\nsys.exit(main())"), "does not END with the call main()"),
])
def test_an_entry_function_that_cannot_be_inlined_exactly_is_refused_by_name(tmp_path, name, src, reason):
    h, why = _harness(tmp_path, src)
    assert h is None, name
    assert reason in why, (name, why)


def test_without_the_call_target_fact_nothing_says_what_main_is(tmp_path):
    h, why = _harness(tmp_path, ENTRY, records=None)
    assert h is None and "no call_target fact" in why
    # the NEGATIVE CONTROL of the lift: the same program with the fact is lifted
    h, why = _harness(tmp_path, ENTRY)
    assert h is not None and h.params == ["x"]


def test_a_name_the_fact_cannot_prove_is_not_inlined(tmp_path):
    # ``main`` is rebound after its def: the call may reach another function
    h, why = _harness(tmp_path, """
        import _sv_verifier

        def main():
            x = _sv_verifier.nondet_int()
            assert x != 3

        def other():
            pass

        main = other
        main()
    """)
    assert h is None and "does not prove main a user function" in why


def test_draws_only_in_callees_are_named(tmp_path):
    h, why = _harness(tmp_path, """
        import _sv_verifier

        def helper():
            assert _sv_verifier.nondet_int() != 3

        def main():
            helper()

        main()
    """)
    assert h is None and "inside functions the entry function calls" in why


def test_a_class_built_in_the_entry_function_is_modelled_through_the_rescoped_fact(tmp_path):
    # the call ``Counter(x)`` is keyed by the scope ``main`` in the fact; in the harness it runs in the module's scope
    src = """
        import _sv_verifier

        class Counter:
            def __init__(self, v):
                self.v = v

        def main():
            x = _sv_verifier.nondet_int()
            c = Counter(x)
            assert c.v != 7

        main()
    """
    path, records = _task(tmp_path, src)
    h, why = _harness(tmp_path, src)
    assert h is not None, why
    assert h.entry_fn == "main" and any(r["scope"] == "main" for r in records)
    sys.modules.setdefault("_sv_verifier", types.ModuleType("_sv_verifier"))
    mod = {"__name__": "sv_harness_test"}
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    lines = [s["line"] for s in h.sites]
    proven = [c["name"] for c in module_constants(path) if c["proven"] and c["origin"] == "class"]
    got = _symexec.solve_targets(h.source, mod, "__sv_harness", lines, h.params, h.kinds, {},
                                 call_targets=h.scoped(records), proven_classes=proven)
    assert got[lines[0]] == ("sat", {"x": 7})
    # the control: the records as they were produced do not resolve the call in the harness
    again = _symexec.solve_targets(h.source, mod, "__sv_harness", lines, h.params, h.kinds, {},
                                   call_targets=records, proven_classes=proven)
    assert again[lines[0]][0] == "unsupported"


# ---------------------------------------------------------------------------
# module-level draws
# ---------------------------------------------------------------------------

MODULE = """
    import _sv_verifier

    SIZE = 1 + abs(_sv_verifier.nondet_int())
    LIMIT = 10

    def to_pos(x, y):
        return y * SIZE + x

    def to_xy(pos):
        y = pos // SIZE
        x = pos % SIZE
        return x, y

    if __name__ == "__main__":
        pos = _sv_verifier.nondet_int()
        assert pos < LIMIT or pos % SIZE != 3
"""


def test_a_draw_at_module_level_is_a_parameter_and_its_statement_moves_into_the_harness(tmp_path):
    h, why = _harness(tmp_path, MODULE)
    assert h is not None, why
    # ``pos`` is also a parameter name of to_xy: a draw assigned to a name a definition reads stays a global
    assert (h.params, h.kinds, h.entry) == (["d0", "d1"], ["int", "int"], "module level")
    body = h.source[h.source.index("def __sv_harness"):]
    assert "SIZE = 1 + abs(__sv_draw('int', d0))" in body and "global" in body and "LIMIT = 10" in body
    # the definitions stay at module level, before the harness
    assert h.source.index("def to_pos") < h.source.index("def __sv_harness")


def test_the_module_level_draw_is_logged_first_and_is_a_symbolic_term(tmp_path):
    src = """
        import _sv_verifier

        SIZE = 1 + abs(_sv_verifier.nondet_int())

        def at(v):
            return v % SIZE

        if __name__ == "__main__":
            v = _sv_verifier.nondet_int()
            assert at(v) != 3
    """
    outcomes, h = _solve(tmp_path, src)
    (outcome, inputs), = outcomes
    assert outcome == "sat"
    # the callee reads SIZE, a module global the harness assigns: both draws are in the model, in program order
    assert _native(src, [inputs[p] for p in h.params])


def test_module_level_draws_with_no_main_block_and_an_entry_function(tmp_path):
    h, why = _harness(tmp_path, """
        import _sv_verifier

        N = _sv_verifier.nondet_int()

        def main():
            x = _sv_verifier.nondet_int()
            if x != N:
                return
            assert x != 7

        main()
    """)
    assert h is not None, why
    assert (h.params, h.entry) == (["N", "x"], "entry main() with module-level draws")
    assert "def main" not in h.source


@pytest.mark.parametrize("name,src,reason", [
    ("a class body reads the moved name", """
        import _sv_verifier
        N = _sv_verifier.nondet_int()
        class C:
            K = N
        assert C.K != 3
    """, "C reads N while it is being defined"),
    ("a default reads it", """
        import _sv_verifier
        N = _sv_verifier.nondet_int()
        def f(a=N):
            return a
        assert f() != 3
    """, "f reads N while it is being defined"),
    ("the module namespace is read", """
        import _sv_verifier
        N = _sv_verifier.nondet_int()
        M = len(globals())
        assert N != M
    """, "reads the module namespace (globals)"),
    ("a star import under a compound statement", """
        import _sv_verifier
        N = _sv_verifier.nondet_int()
        try:
            from math import *
        except ImportError:
            pass
        assert N != 3
    """, "a star import under a compound statement"),
    ("the entry function assigns a moved global", """
        import _sv_verifier
        N = _sv_verifier.nondet_int()
        def main():
            N = 3
            assert N != 3
        main()
    """, "assigns N"),
])
def test_a_module_level_move_that_would_change_the_program_is_refused_by_name(tmp_path, name, src, reason):
    h, why = _harness(tmp_path, src)
    assert h is None, name
    assert reason in why, (name, why)


def test_a_refused_module_level_move_falls_back_to_the_main_block(tmp_path):
    h, why = _harness(tmp_path, """
        import _sv_verifier

        N = _sv_verifier.nondet_int()

        class C:
            K = N

        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            assert x != C.K
    """)
    # the block is lifted as it always was; the module-level draw is left to the stub
    assert h is not None, why
    assert (h.entry, h.params) == ("main block", ["x"])


def test_a_program_with_no_draw_outside_a_main_block_keeps_the_main_block_harness(tmp_path):
    h, why = _harness(tmp_path, """
        import _sv_verifier
        LIMIT = 5

        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            assert x < LIMIT
    """)
    assert h is not None and h.entry == "main block" and h.params == ["x"]
    assert "LIMIT = 5" in h.source[:h.source.index("def __sv_harness")]       # not moved: it holds no draw


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_a_violation_behind_an_early_return_in_main_is_solved_symbolically(tmp_path):
    path, _ = _task(tmp_path, """
        import _sv_verifier

        def main():
            x = _sv_verifier.nondet_int()
            if not (0 <= x <= 10 ** 7):
                return
            assert x * x - 3 * x != 1519054

        main()
    """)
    v = verify(path, AR, budget_seconds=60, jobs=1, canary_seconds=3, early_stop_seconds=3,
               witness_dir=tmp_path / "proofs")
    assert v.line() == "FALSE(AssertionError)"
    assert v.component == "concolic:symexec"
    x = v.witness["vector"][0]["value"]
    assert x * x - 3 * x == 1519054 and 0 <= x <= 10 ** 7


def test_a_violation_behind_a_module_level_draw_is_solved_symbolically(tmp_path):
    path, _ = _task(tmp_path, """
        import _sv_verifier

        K = 1 + abs(_sv_verifier.nondet_int())

        def mix(v):
            return (v + K) % 999983

        if __name__ == "__main__":
            v = _sv_verifier.nondet_int()
            assert mix(v) != 1234
    """)
    v = verify(path, AR, budget_seconds=60, jobs=1, canary_seconds=3, early_stop_seconds=3,
               witness_dir=tmp_path / "proofs")
    assert v.line() == "FALSE(AssertionError)"
    k, x = (e["value"] for e in v.witness["vector"])
    assert (x + 1 + abs(k)) % 999983 == 1234


def test_the_safe_twin_is_proved_not_found(tmp_path):
    path, _ = _task(tmp_path, ENTRY.replace("x * x > x", "x * x >= x"))
    v = verify(path, AR, budget_seconds=60, jobs=1, canary_seconds=3, early_stop_seconds=3,
               witness_dir=tmp_path / "proofs")
    assert v.line() == "TRUE" and v.component == "symbolic:product"
