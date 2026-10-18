"""Absint loop candidates must survive independent induction and safety VCs."""
import ast
import time
from dataclasses import replace

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.census import obligation_census
from tharkun.solve.obligations import Invariant, source_points, witness
from tharkun.sv.validate import validate
from tharkun.sv.witness import dumps

SOURCE = "i = 1\nwhile i < 10:\n    7 // i\n    i += 1\nassert i == 10\n"
SPEC = "CHECK( init(main()), ! uncaught(ZeroDivisionError))\nCHECK( init(main()), ! uncaught(AssertionError))\n"
CLASSES = ("ZeroDivisionError", "AssertionError")


def candidates(source=SOURCE):
    census = obligation_census(source, CLASSES, "t.py")
    locs = {loc for node, (_, loc) in source_points(ast.parse(source), source, "t.py").items()
            if isinstance(node, ast.stmt)}
    result = prove(source, CLASSES, filename="t.py", invariant_locations=locs)
    assert result.verdict == "TRUE", result.reason
    return [o.invariant for o in census.obligations] + result.invariants


def check(facts, source=SOURCE):
    return validate(source, "t.py", SPEC, dumps(witness(source, "t.py", SPEC, facts)))


def test_absint_loop_invariant_is_independently_inductive():
    facts = candidates()
    assert any(i.type == "loop_invariant" for i in facts)
    assert check(facts).confirmed
    assert not check([i for i in facts if i.type != "loop_invariant"]).confirmed


def test_mutated_initiation_preservation_and_site_safety_are_rejected():
    facts = candidates()
    for value in ("False", "type(i) is int and i == 1", "type(i) is int and i >= 0"):
        mutated = [replace(i, value=value) if i.type == "loop_invariant" else i for i in facts]
        assert not check(mutated).confirmed, value
    # A source mutant remains invalid even after updating its witness hash.
    assert not check(facts, SOURCE.replace("i += 1", "i -= 1")).confirmed
    assert not check(facts, SOURCE.replace("i = 1", "i = 0")).confirmed


def test_violation_far_into_a_huge_loop_is_never_missed_by_unrolling():
    # [internal ref] negative control: the checker must reason about ONE arbitrary loop
    # step (initiation, preservation, per-step safety), never about a fixed
    # number of concretely-unrolled iterations. Put the actual violation at
    # iteration 999,999 of a million -- far past any plausible unrolling bound
    # -- and submit an invariant that is true but too WEAK to prove the site
    # safe. A checker that silently unrolls only a small bound would never
    # reach the bad iteration and would wrongly accept; real induction must
    # still fail to prove it, and must do so in roughly constant time.
    unsafe_src = "i = 0\nx = 0\nwhile i < 1000000:\n    x = 1 // (999999 - i)\n    i += 1\n"
    with pytest.raises(ZeroDivisionError):
        exec(unsafe_src, {})
    weak = [Invariant("t.py", 3, 1, "<module>", "type(i) is int and i >= 0", "loop_invariant")]
    t0 = time.monotonic()
    result = check(weak, unsafe_src)
    elapsed = time.monotonic() - t0
    assert not result.confirmed
    assert elapsed < 5.0  # independent of the 10**6 trip count: induction, not unrolling

    # Same huge trip count, but the loop stops one step short of the bad divisor,
    # proved with a genuinely inductive and sufficient invariant: CONFIRMED, and
    # just as fast -- the checker is deciding the specific program, not merely
    # timing out or bailing on anything with a large loop.
    safe_src = "i = 0\nx = 0\nwhile i < 999999:\n    x = 1 // (999999 - i)\n    i += 1\n"
    strong = [Invariant("t.py", 3, 1, "<module>", "type(i) is int and 0 <= i <= 999999", "loop_invariant")]
    t0 = time.monotonic()
    result = check(strong, safe_src)
    elapsed = time.monotonic() - t0
    assert result.confirmed
    assert elapsed < 5.0


def test_conditional_safety_in_unbounded_loop():
    src = "from _sv_verifier import nondet_int\nn = nondet_int()\ni = 1\nwhile i < n:\n    7 // i\n    i += 1\nassert i >= 1\n"
    assert check(candidates(src), src).confirmed


def test_loop_frame_and_control_gaps_cannot_be_hidden():
    # A user invariant does not excuse an unmodeled alias update or exit.
    for src in ("xs = [1]\ni = 0\nwhile i < 2:\n    7 // i\n    xs.pop()\n    i += 1\n", "i = 1\nwhile i < 10:\n    7 // i\n    break\n"):
        line = 3 if src.startswith("xs") else 2
        facts = [Invariant("t.py", line, 1, "<module>", "True", "loop_invariant")]
        assert not check(facts, src).confirmed


def test_conditional_source_safety_can_close_a_census_adapter_gap(tmp_path):
    from tharkun.sv.engine import verify
    src = "i = 1\nwhile i < 10:\n    7 // (i + 1)\n    i += 1\n"
    assert any(o.gap for o in obligation_census(src, ["ZeroDivisionError"]).obligations)
    program = tmp_path / "p.py"
    program.write_text(src)
    result = verify(str(program), ["ZeroDivisionError"],
                    budget_seconds=5, concolic=False, stages=[])
    assert result.verdict == "true" and result.detail["proof_checked"]
    # The source type rules also exclude KeyError from a builtin list access.
    program.write_text("xs = [1]\ni = 0\nx = xs[i]\n")
    result = verify(str(program), ["KeyError"],
                    budget_seconds=5, concolic=False, stages=[])
    assert result.verdict == "true" and result.detail["proof_checked"]
