"""tharkun.sv solves for violating draws ([internal ref] slices 2-3): sites, harness, concolic stage,
and the order of the verifier's stages."""

from __future__ import annotations

import sys
import textwrap

import pytest

from tharkun.sv import verify
from tharkun.sv.concolic import component_of
from tharkun.sv.sites import build_harness, violation_sites

DL = ("KeyError", "IndexError")


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


TWO = """
    import _sv_verifier

    if __name__ == "__main__":
        a = _sv_verifier.nondet_int()
        b = _sv_verifier.nondet_int()
        if a > 100000 and b == 3 * a + 17:
            raise KeyError("guarded site")
"""

POLY = """
    import _sv_verifier


    def check(x):
        if x * x - 3 * x == 1519054:
            raise IndexError("guarded site")
        return x


    if __name__ == "__main__":
        x = _sv_verifier.nondet_int()
        check(x)
"""


def test_sites_are_the_raises_and_asserts_of_the_property_classes():
    src = textwrap.dedent("""
        def f(x):
            if x > 3:
                raise IndexError("a")
            raise ValueError("outside every property")

        if __name__ == "__main__":
            assert f(1)
            raise KeyError
    """)
    sites = violation_sites(src, DL)
    assert [(s["exc"], s["function"], s["kind"]) for s in sites] == [
        ("IndexError", "f", "raise"), ("KeyError", "<main>", "raise")]
    assert sites[0]["guards"] == [("x > 3", "then")]
    assert [s["kind"] for s in violation_sites(src, ("AssertionError",))] == ["assert"]


def test_a_subclass_of_a_property_class_is_a_site():
    src = "if __name__ == '__main__':\n    raise ModuleNotFoundError('x')\n"
    assert [s["exc"] for s in violation_sites(src, ("ImportError",))] == ["ImportError"]


def test_the_harness_lifts_straight_line_draws_and_leaves_loops_to_the_stub():
    h = build_harness(textwrap.dedent("""
        import _sv_verifier
        from _sv_verifier import nondet_bool as nb

        def g(u, v):
            return u + v

        if __name__ == "__main__":
            a = _sv_verifier.nondet_int()
            c = nb()
            for _ in range(3):
                _sv_verifier.nondet_int()
            g(a, _sv_verifier.nondet_float_range(0.0, 1.0))
    """), DL)
    # ``a = nondet()`` makes ``a`` itself the parameter; a draw in a loop is a stream
    # ``sK`` (a list, one value per iteration); other draws get ``dK``
    assert h.params == ["a", "c", "s2", "d3"]
    assert h.kinds == ["int", "bool", "list:int", "float"]
    assert h.unlifted == 0
    assert h.feeds == {"g": {"u": "a", "v": "d3"}}
    assert h.seed() == {"a": 0, "c": False, "s2": [], "d3": 0.0}
    assert "def __sv_harness(a: int, c: bool, s2: list[int], d3: float)" in h.source
    compile(h.source, "sv_harness.py", "exec")


def test_no_main_block_or_no_draw_means_no_harness():
    assert build_harness("x = 1\n", DL) is None
    assert build_harness("if __name__ == '__main__':\n    print(1)\n", DL) is None


def test_components_name_the_strongest_technique():
    assert component_of(["symexec"]) == "concolic:symexec"
    assert component_of(["reach-search", "static"]) == "concolic:guard_solve"
    assert component_of([]) == "concolic:drive"


def test_a_guarded_violation_is_solved_not_enumerated(tmp_path):
    p = _prog(tmp_path, "two.py", TWO)
    v = verify(p, DL, budget_seconds=60, jobs=1)
    assert v.line() == "FALSE(KeyError)"
    # the one-shot symbolic run solves it before anything executes (slice 3)
    assert v.component == "concolic:symexec"
    a, b = (e["value"] for e in v.witness["vector"])
    assert a > 100000 and b == 3 * a + 17
    off = verify(p, DL, budget_seconds=6, jobs=1, stages=())
    assert off.verdict == "unknown"                    # enumeration alone never gets there


def test_a_quadratic_guard_in_a_callee_is_solved_natively(tmp_path):
    p = _prog(tmp_path, "poly.py", POLY)
    v = verify(p, DL, budget_seconds=60, jobs=1)
    assert v.line() == "FALSE(IndexError)"
    assert v.component in ("concolic:symexec", "concolic:guard_solve")   # the callee is inlined (slice 3)
    x = v.witness["vector"][0]["value"]
    assert x * x - 3 * x == 1519054


def test_enumeration_answers_say_where_the_value_came_from(tmp_path):
    p = _prog(tmp_path, "magic.py", """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            if x == 48611:
                [][0]
    """)
    v = verify(p, DL, budget_seconds=60, jobs=1, stages=())
    assert v.line() == "FALSE(IndexError)" and v.component == "enum:constants"
    d = _prog(tmp_path, "det.py", "if __name__ == '__main__':\n    print(1)\n")
    # (with the proofs on, the abstract interpreter decides it before the program runs)
    assert verify(d, DL, budget_seconds=60, jobs=1, prove=False).component == "deterministic"


PICK = """
    import _sv_verifier

    N = 16

    if __name__ == "__main__":
        slots = [0] * N
        for i in range(N):
            if _sv_verifier.nondet_bool():
                slots[i] = 1
        key = sum(slots[i] << i for i in range(N))
        needle = sum(1 << i for i in range(0, N, 2))
        print(slots[N if key == needle else 0])
"""


def test_implicit_sites_carry_the_condition_they_fire_under():
    from tharkun.sv.sites import implicit_sites
    src = textwrap.dedent("""
        def f(xs, n, d):
            try:
                y = 10 // d
            except ZeroDivisionError:
                y = 0
            return xs[n] if n > 3 else y % d
    """)
    sites = implicit_sites(src, ("IndexError", "ZeroDivisionError"))
    kinds = sorted((s["exc"], s["line"]) for s in sites)
    # 10 // d is caught; xs[n] fires only when n > 3, y % d only when not
    assert kinds == [("IndexError", 7), ("ZeroDivisionError", 7)]
    by = {s["exc"]: s["condition"] for s in sites}
    assert by["IndexError"].startswith("(n > 3) and ")
    assert by["ZeroDivisionError"].startswith("not (n > 3) and ")


def test_a_needle_among_2_to_the_16_draw_paths_is_one_formula(tmp_path):
    """``pick`` at N = 16: the key is assembled from sixteen loop draws, so no single
    guard can be flipped; the symbolic executor unrolls the loop, merges the ifs and
    solves ``key == needle`` in one z3 query."""
    pytest.importorskip("z3")
    p = _prog(tmp_path, "pick.py", PICK)
    v = verify(p, DL, budget_seconds=60, jobs=1)
    assert v.line() == "FALSE(IndexError)"
    assert v.component == "concolic:symexec"
    bits = [e["value"] for e in v.witness["vector"]]
    assert bits == [i % 2 == 0 for i in range(16)]
    assert verify(p, DL, budget_seconds=8, jobs=1, concolic=False).verdict == "unknown"


def test_the_float_tail_never_overflows():
    from tharkun.sv.engine import Candidates
    c = Candidates()
    vals = [c.value("float", r) for r in range(6000)]     # used to raise OverflowError
    assert vals[3000] is not None and vals[-1] is None    # the tail ends, it does not crash


def test_a_partial_run_record_is_a_crashed_run_not_a_crash(tmp_path, monkeypatch):
    import json
    import subprocess
    from tharkun.sv import engine

    def partial(cmd, **kw):
        req = json.load(open(cmd[-1]))
        with open(req["out"], "w") as fh:
            fh.write('{"status": "raised", "draws": [{"kind": "str", "value": "aaaa')
        return subprocess.CompletedProcess(cmd, -9, "", "")

    monkeypatch.setattr(engine.subprocess, "run", partial)
    out = engine.run_once(sys.executable, str(tmp_path / "p.py"), [], timeout=30)
    assert out.status == "crashed"


def test_the_str_tail_is_bounded():
    from tharkun.sv.engine import Candidates
    vals = [Candidates().value("str", r) for r in range(200)]
    assert max(len(v) for v in vals if v is not None) <= 2**20


def test_a_truncated_concolic_result_is_no_candidate(tmp_path, monkeypatch):
    import json
    import subprocess
    from pathlib import Path
    from tharkun.sv import concolic

    def partial(cmd, **kw):
        req = json.loads(Path(cmd[-1]).read_text())
        Path(req["out"]).write_text('{"reached":')
        return subprocess.CompletedProcess(cmd, -9, "", "")

    monkeypatch.setattr(concolic.subprocess, "run", partial)
    p = _prog(tmp_path, "partial.py", TWO)
    assert concolic.solve(p, DL, budget_seconds=60).candidates == []


# --- slice 3: callees, draw-dependent loops, typing sites, stage order ---------------------

def _solve_all(src, classes, extra=None):
    """``{kind or exc: outcome}`` of one symbolic run of the harness toward every site."""
    pytest.importorskip("z3")
    from tharkun.sv import _symexec
    h = build_harness(textwrap.dedent(src), classes, extra)
    mod = {"__name__": "sv_harness_test"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    got = _symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites],
                                 h.params, h.kinds, {})
    return {s.get("kind", s["exc"]): got[s["line"]] for s in h.sites}, h


def test_a_site_in_a_callee_is_solved_through_the_call():
    got, _h = _solve_all("""
        import _sv_verifier

        def check(a, b):
            t = a * 7 + b
            if t == 12345 and b > 1000:
                raise KeyError("in the callee")

        if __name__ == "__main__":
            a = _sv_verifier.nondet_int()
            b = _sv_verifier.nondet_int()
            check(a, b)
    """, DL)
    outcome, inputs = got["raise"]
    assert outcome == "sat"
    assert inputs["a"] * 7 + inputs["b"] == 12345 and inputs["b"] > 1000


def test_a_loop_whose_trip_count_is_a_draw_is_unrolled_under_its_guard():
    got, _h = _solve_all("""
        import _sv_verifier

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            total = 0
            for i in range(n):
                total += i
            if total == 1953:
                raise KeyError("n == 63")
    """, DL)
    outcome, inputs = got["raise"]
    assert outcome == "sat" and inputs["n"] == 63


def test_a_value_whose_type_depends_on_the_path_is_a_typing_site():
    got, h = _solve_all("""
        import _sv_verifier

        if __name__ == "__main__":
            flag = _sv_verifier.nondet_bool()
            v = 0
            if flag:
                v = "tag"
            print(v + 1)
    """, ("TypeError",))
    assert [s["kind"] for s in h.sites] == ["binop"]
    outcome, inputs = got["binop"]
    assert outcome == "sat" and inputs["flag"] is True


def test_numbers_only_operations_are_no_typing_sites():
    h = build_harness(textwrap.dedent("""
        import _sv_verifier
        if __name__ == "__main__":
            a = _sv_verifier.nondet_int()
            print(a + 1)
    """), ("TypeError",))
    assert h.sites == []


def test_a_site_no_run_can_fire_is_unsat():
    got, _h = _solve_all("""
        import _sv_verifier

        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            if x > 5 and x < 3:
                raise KeyError("dead")
    """, DL)
    assert got["raise"][0] == "unsat"


def test_an_unmodelled_statement_leaves_its_site_unsupported_not_unsat():
    got, _h = _solve_all("""
        import _sv_verifier

        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            y = hash(str(x))
            if y == 12:
                raise KeyError("behind a hash")
    """, DL)
    assert got["raise"][0] == "unsupported"


def test_extra_records_add_targets_and_safe_records_remove_markers():
    src = textwrap.dedent("""
        import _sv_verifier

        if __name__ == "__main__":
            xs = [1, 2, 3]
            i = _sv_verifier.nondet_int()
            print(xs[i])
            print(xs[0])
    """)
    lines = src.splitlines()
    first = next(n for n, t in enumerate(lines, 1) if "xs[i]" in t)
    second = next(n for n, t in enumerate(lines, 1) if "xs[0]" in t)
    plain = build_harness(src, ("IndexError",))
    assert len(plain.sites) == 2
    # the abstract interpreter proved xs[0] safe: no marker; and it names a region the checks do not know
    records = [{"line": second, "col": 10, "exc": "IndexError", "kind": "index", "status": "safe"},
               {"line": first, "col": 4, "exc": "IndexError", "kind": "region", "status": "may_raise",
                "residual": "i == 7", "scope": ["i > 0"]}]
    h = build_harness(src, ("IndexError",), records)
    assert sorted(s["kind"] for s in h.sites) == ["index", "region"]
    region = next(s for s in h.sites if s["kind"] == "region")
    assert region["residual"] == "(i > 0) and (i == 7)"


def test_every_site_unsat_stops_the_search_early(tmp_path):
    pytest.importorskip("z3")
    p = _prog(tmp_path, "dead.py", """
        import _sv_verifier

        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            if x > 5 and x < 3:
                raise KeyError("dead")
    """)
    # the abstract interpreter proves this one TRUE; without it the symbolic stage can only
    # say "no modelled run fires it", which caps the search
    assert verify(p, DL, budget_seconds=100, jobs=1).component.startswith("proof:")
    v = verify(p, DL, budget_seconds=100, jobs=1, prove=False)
    assert v.verdict == "unknown" and "stopped early" in v.reason
    assert v.seconds < 40                                   # capped at a tenth of the budget


def _fake_prover(monkeypatch, verdict_line):
    """Install a stand-in ``tharkun.sv.prove`` whose static proof says TRUE."""
    import types
    from tharkun.sv.engine import Verdict
    mod = types.ModuleType("tharkun.sv.prove")
    mod.prove_static = lambda program, classes, remaining=None, log=None, files=None: Verdict(
        "true", reason="stand-in proof", component="proof:stand-in")
    monkeypatch.setitem(sys.modules, "tharkun.sv.prove", mod)


def test_a_static_proof_decides_before_the_program_runs_long(tmp_path, monkeypatch):
    _fake_prover(monkeypatch, "TRUE")
    p = _prog(tmp_path, "slow.py", """
        import time
        if __name__ == "__main__":
            time.sleep(600)
    """)
    v = verify(p, DL, budget_seconds=30, jobs=1)
    # the deterministic run would need ten minutes: the proof answers after the canary cap
    assert v.line() == "TRUE" and v.component == "proof:stand-in"
    assert v.seconds < 20


def test_a_canary_violation_beats_a_proof_and_names_it(tmp_path, monkeypatch):
    _fake_prover(monkeypatch, "TRUE")
    p = _prog(tmp_path, "bad.py", """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            if x == 3:
                raise KeyError("the proof was wrong")
    """)
    v = verify(p, DL, budget_seconds=60, jobs=1)
    assert v.line() == "FALSE(KeyError)"
    assert v.detail["unsound_proof"] == "proof:stand-in"


# --- [internal ref]: random.choice(seq) over a fixed table, lifted as a "rindex" draw ---------------

CHOICE_TABLE = """
    import random

    TABLE = {"a": 1, "b": 2, "c": 3}

    if __name__ == "__main__":
        key = random.choice(list(TABLE.keys()))
        if key == "c":
            raise IndexError("hit c")
"""


def test_random_choice_over_a_literal_dict_keys_is_lifted_and_solved():
    """The index random.choice draws becomes an ordinary lifted ``rindex`` param, so the
    site guarded by *which* key it drew is solved like any other guard ([internal ref])."""
    got, h = _solve_all(CHOICE_TABLE, ("IndexError",))
    assert h.kinds[-1] == "rindex"
    assert h.unlifted == 0
    assert "__sv_index('rindex'" in h.source and "random.choice" not in h.source
    outcome, detail = got["raise"]
    assert outcome == "sat", detail


def test_random_choice_never_counts_its_own_bookkeeping_as_a_site():
    """__sv_index's internal subscript (an empty ``seq`` raises IndexError through the
    subscript itself, not a ``raise`` statement) must never show up as one of the
    program's OWN violation sites -- a regression guard, not just a shape check."""
    h = build_harness(textwrap.dedent(CHOICE_TABLE), ("IndexError",))
    assert len(h.sites) == 1
    assert all(s["function"] != "__sv_index" for s in h.sites)


def test_random_choice_is_left_unlifted_when_the_program_ever_seeds():
    """A program that seeds random ANYWHERE cannot be certified unseeded ([internal ref]):
    _child.py's own choice() would use real seeded randomness, not the drawn vector, so
    lifting it here would only waste solver rounds on candidates replay ignores."""
    src = """
        import _sv_verifier
        import random
        random.seed(1)

        TABLE = {"a": 1, "b": 2, "c": 3}

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            key = random.choice(list(TABLE.keys()))
            if key == "c" and n == 3:
                raise IndexError("hit c")
    """
    h = build_harness(textwrap.dedent(src), ("IndexError",))
    assert h is not None
    assert h.unlifted >= 1
    assert "random.choice(" in h.source
    assert "__sv_index('rindex'" not in h.source


def test_random_choice_inside_a_loop_is_left_unlifted():
    """A draw inside a loop needs index-stream support this slice does not add ([internal ref]) --
    left to the stub like any other draw there, never silently mis-modelled."""
    src = """
        import _sv_verifier
        import random

        TABLE = {"a": 1, "b": 2, "c": 3}

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            for _ in range(3):
                key = random.choice(list(TABLE.keys()))
                if key == "c" and n == 3:
                    raise IndexError("hit c")
    """
    h = build_harness(textwrap.dedent(src), ("IndexError",))
    assert h is not None
    assert h.unlifted >= 1
    assert "random.choice(" in h.source
    assert "__sv_index('rindex'" not in h.source


def test_random_choice_from_import_is_not_lifted():
    """Only the module-qualified ``random.choice`` form is recognised ([internal ref]) -- a
    ``from random import choice`` call is left completely untouched, never assumed."""
    src = """
        from random import choice

        TABLE = {"a": 1, "b": 2, "c": 3}

        if __name__ == "__main__":
            key = choice(list(TABLE.keys()))
            if key == "c":
                raise IndexError("hit c")
    """
    h = build_harness(textwrap.dedent(src), ("IndexError",))
    assert h is None or "__sv_index('rindex'" not in h.source


def test_random_choice_over_an_unmodelled_sequence_is_unsupported_not_wrong():
    """``random.choice`` of something that never evaluates to a concrete list/tuple (a
    set, here) must refuse cleanly -- unsupported, never a wrong verdict ([internal ref])."""
    got, h = _solve_all("""
        import random

        if __name__ == "__main__":
            key = random.choice({"a", "b", "c"})
            if key == "c":
                raise IndexError("hit c")
    """, ("IndexError",))
    outcome, detail = got["raise"]
    assert outcome == "unsupported", (outcome, detail)
# --- [internal ref]-followup: the unlifted-call stub logs an in-domain nondet_float_range value -

def test_concolic_worker_stub_float_range_logs_an_in_domain_value():
    """``_concolic_worker.py``'s stub answers every draw the harness could not lift with
    the kind's generic default (0.0 for a range) and logs it -- for
    nondet_float_range(a, b), 0.0 can fall outside the call's domain (e.g. a call like
    nondet_float_range(5, 10)), which _child.py's domain check would then refuse to
    confirm. The stub's nondet_float_range must snap into the call's own domain
    before logging, same as the hybrid/Atheris worker fix.  [internal ref]: the domain is what
    ``random.uniform(a, b)`` can return (``_child.float_range_producible``), and the
    stub's ``nondet_float`` default is ``-0.0`` (a value of ITS domain)."""
    import math

    from tharkun.sv import _child
    from tharkun.sv._concolic_worker import _stub
    m = _stub()
    v = m.nondet_float_range(5.0, 10.0)
    assert v == 5.0
    assert m.LOG[-1] == ("float", 5.0)
    # order-independent (a > b), and the ordinary in-domain case is unaffected
    low = m.nondet_float_range(10.0, 5.0)
    assert 5.0 <= low < 5.0 + 1e-14 and _child.float_range_producible(10.0, 5.0, low)
    assert m.nondet_float_range(-1.0, 1.0) == 0.0
    assert m.nondet_float_range(0.0, 1.0) == 0.0
    # a bound that is not producible is never logged: uniform(-5, -1) never returns 0.0, so the default snaps to its nearest value
    top = m.nondet_float_range(-5.0, -1.0)
    assert _child.float_range_producible(-5.0, -1.0, top) and top < -1.0 + 1e-14
    flt = m.nondet_float()
    assert flt == 0.0 and math.copysign(1.0, flt) < 0 and _child.float_producible(flt)


# --- [internal ref]: sorted()/list() of a dict (module-level table), a concrete-only builtin ------

def test_sorted_of_a_literal_dict_is_solved():
    """``sorted(TABLE)`` iterates (and sorts) the dict's keys, same as real Python --
    concrete int keys only, never a symbolic sort ([internal ref])."""
    got, h = _solve_all("""
        import _sv_verifier

        TABLE = {3: "c", 1: "a", 2: "b"}

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            keys = sorted(TABLE)
            if keys[0] == 1 and n == 7:
                raise IndexError("hit it")
    """, ("IndexError",))
    outcome, detail = got["raise"]
    assert outcome == "sat", detail


def test_sorted_of_a_symbolic_sequence_is_unsupported_not_wrong():
    """A sort over drawn (symbolic) values has no order relation this executor tracks --
    refuse cleanly, never guess an order ([internal ref])."""
    got, h = _solve_all("""
        import _sv_verifier

        if __name__ == "__main__":
            a = _sv_verifier.nondet_int()
            b = _sv_verifier.nondet_int()
            keys = sorted([a, b])
            if keys[0] == 3:
                raise IndexError("hit it")
    """, ("IndexError",))
    outcome, detail = got["raise"]
    assert outcome == "unsupported", (outcome, detail)


def test_sorted_of_an_unorderable_mix_is_unsupported_not_wrong():
    """Mixed, incomparable concrete types (str vs int) raise TypeError in real Python --
    refuse, never silently pick an order ([internal ref])."""
    got, h = _solve_all("""
        import _sv_verifier

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            keys = sorted(["a", 1])
            if keys[0] == "a" and n == 3:
                raise IndexError("hit it")
    """, ("IndexError",))
    outcome, detail = got["raise"]
    assert outcome == "unsupported", (outcome, detail)


# --- tuple/string merging for a symbolic dict index (bm_hexiom_nondet's ----------------
# ``LEVELS[level] = (board_text, solution_text)``, [internal ref]'s own follow-on note) ----------

def test_symbolic_dict_index_with_tuple_values_is_solved():
    """A dict whose VALUES are tuples (``LEVELS[level] = (board, solution)``, hexiom's
    exact shape) used to raise "not a number: tuple" the moment a SYMBOLIC index forced
    ``_ite`` to merge two different levels' tuples across the cascading ite chain --
    ``_ite`` had no tuple/tuple branch (only dict/dict and list/list were merged
    recursively). It now merges element by element, same as list/list."""
    got, _h = _solve_all("""
        import _sv_verifier

        TABLE = {1: ("a", "x"), 2: ("b", "y"), 3: ("c", "z")}

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            level = abs(n) % 3 + 1
            first, second = TABLE[level]
            if first == "b" and second == "y":
                raise IndexError("hit it")
    """, ("IndexError",))
    outcome, detail = got["raise"]
    assert outcome == "sat", detail


def test_symbolic_dict_index_with_plain_string_values_is_solved():
    """The same fix generalises past tuples: two DIFFERENT concrete strings as a dict's
    plain (non-tuple) values used to raise "not a number: str" for the identical
    reason -- ``_ite``'s string-sort branch only fired when a side was ALREADY a
    symbolic z3 string, never for two concrete strings on both sides."""
    got, _h = _solve_all("""
        import _sv_verifier

        TABLE = {1: "alpha", 2: "beta", 3: "gamma"}

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            level = abs(n) % 3 + 1
            v = TABLE[level]
            if v == "beta":
                raise IndexError("hit it")
    """, ("IndexError",))
    outcome, detail = got["raise"]
    assert outcome == "sat", detail


def test_symbolic_dict_index_with_mismatched_tuple_lengths_is_unsupported_not_wrong():
    """Negative control: tuples of DIFFERENT lengths across keys have no well-defined
    elementwise merge -- decline cleanly, never guess a shape (mirrors the existing
    list/list length check one branch up)."""
    got, _h = _solve_all("""
        import _sv_verifier

        TABLE = {1: ("a", "b"), 2: ("c",), 3: ("d", "e")}

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            level = abs(n) % 3 + 1
            pair = TABLE[level]
            if pair[0] == "a":
                raise IndexError("hit it")
    """, ("IndexError",))
    outcome, detail = got["raise"]
    assert outcome == "unsupported", (outcome, detail)


def test_symbolic_dict_index_merging_a_string_with_a_non_string_is_unsupported_not_wrong():
    """Negative control: merging a string with a non-string value would build a z3.If
    with mismatched sorts (a real crash inside z3, not a clean decline) unless BOTH
    sides are checked to be string-like first -- decline instead."""
    got, _h = _solve_all("""
        import _sv_verifier

        TABLE = {1: "alpha", 2: 42, 3: "gamma"}

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            level = abs(n) % 3 + 1
            v = TABLE[level]
            if v == "alpha":
                raise IndexError("hit it")
    """, ("IndexError",))
    outcome, detail = got["raise"]
    assert outcome == "unsupported", (outcome, detail)
