"""The ``tharkun.sv`` solving portfolio (``sv/stages.py``) and its static stage
``static:guard_values`` (``sv/guard_values.py``), [internal ref]."""

from __future__ import annotations

import textwrap

import pytest

from tharkun.sv import stages
from tharkun.sv.engine import Candidates, verify
from tharkun.sv.guard_values import seed_values

DL = ("IndexError", "KeyError")


def _prog(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src).lstrip("\n"))
    return str(p)


def test_select_keeps_registry_order_and_rejects_unknown_names():
    assert [s.name for s in stages.select()] == list(stages.DEFAULT)
    assert [s.name for s in stages.select(["concolic", "guard_values"])] == ["guard_values", "concolic"]
    assert [s.name for s in stages.select(off=["concolic"])] == ["guard_values", "comparison", "boundaries", "composition", "producer", "hybrid"]
    assert stages.select(()) == []
    with pytest.raises(ValueError):
        stages.select(["nope"])


def test_seeds_rank_after_the_head_and_carry_their_source():
    c = Candidates({"int": [77]}, seeded={"static:guard_values": {"int": [500]}})
    seq = [c.value("int", r) for r in range(8)]
    assert seq[:7] == [0, 1, -1, 2, -2, 500, 77]
    assert c.source({"kind": "int", "value": 500}) == "static:guard_values"
    assert c.source({"kind": "int", "value": 77}) == "constants"


def test_guard_values_solves_inlined_guards_over_draws(tmp_path):
    p = _prog(tmp_path, "g.py", """
        import _sv_verifier
        if __name__ == "__main__":
            a = _sv_verifier.nondet_int()
            b = _sv_verifier.nondet_int()
            n = 1 + abs(a)
            seen = {}
            if n > 5000 and n % 7 == 3:
                print(seen[n])
            seen[n] = a
            if 3 * a + 5 * b == 1234563 and a - b == 4321:
                pass
    """)
    ints = seed_values(p)["int"]
    assert any(1 + abs(x) > 5000 and (1 + abs(x)) % 7 == 3 for x in ints)
    assert any(3 * x + 5 * y == 1234563 and x - y == 4321 for x in ints for y in ints)


def test_guard_values_skips_names_that_are_not_straight_line(tmp_path):
    p = _prog(tmp_path, "loop.py", """
        import _sv_verifier
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            s = 0
            for i in range(n):
                s += i
            if s == 4950:
                pass
    """)
    assert seed_values(p) == {}                # s is loop-carried: nothing to solve for


def test_an_arithmetic_guard_is_found_by_the_static_stage_alone(tmp_path):
    p = _prog(tmp_path, "affine.py", """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            if x * 7 - 11 == 3489:
                [][0]
    """)
    v = verify(p, DL, budget_seconds=60, jobs=1, constants=False, deepen=False,
               stages=["guard_values"])
    assert v.line() == "FALSE(IndexError)" and v.component == "static:guard_values"
    assert v.witness["vector"][0]["value"] == 500
    off = verify(p, DL, budget_seconds=60, jobs=1, constants=False, deepen=False, stages=())
    assert off.verdict == "unknown"
