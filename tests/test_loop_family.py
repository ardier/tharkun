"""Loop families propose invariants; only the fixpoint check may accept one."""

import ast
from unittest.mock import patch

from tharkun.solve.absint import prove
from tharkun.solve.loop_family import AFFINE, KERNEL, classify_loop_family


def test_source_families_and_lowered_branch_priority():
    samples = {
        "counter": "while i < n:\n i += 1",
        "accumulator": "while i < n:\n total += i\n i += 1",
        "search": "for x in xs:\n if x == target: break",
        "iterator": "for x in xs:\n print(x)",
        "scan": "for i in range(n):\n x = xs[i]",
        "affine": "while r < 30:\n acc = acc * 3 + 1\n r += 1",
    }
    for family, source in samples.items():
        assert classify_loop_family(ast.parse(source).body[0]) == family
    assert classify_loop_family(ast.parse("for x, y in pairs:\n print(x)").body[0]) == "iterator"
    lowered = ast.parse("def frame():\n if done: return 0\n return 1").body[0]
    assert classify_loop_family(lowered) == KERNEL


def test_two_programs_and_negative_controls():
    samples = (
        ("x=0\nwhile x<3:\n x+=1\nassert x==3", "x=0\nwhile x<3:\n x+=1\nassert x==2"),
        ("xs=[0]*4\nfor i in range(4):\n xs[i]=1\nprint(xs[3])",
         "xs=[0]*4\nfor i in range(4):\n xs[i]=1\nprint(xs[4])"),
    )
    for safe, bug in samples:
        assert prove(safe, ("AssertionError", "IndexError")).verdict == "TRUE"
        with patch("tharkun.solve.absint.interp.classify_loop_family", return_value=AFFINE):
            assert prove(bug, ("AssertionError", "IndexError")).verdict != "TRUE"


def test_relational_template_needs_induction_and_keeps_mutant_unknown():
    safe = """\
import _sv_verifier
x = y = 0
while _sv_verifier.nondet_bool():
    if x >= 4 and y == x:
        y += 2
    else:
        y += 1
    x += 1
assert y <= x + 1
"""
    assert prove(safe, ("AssertionError",)).verdict == "TRUE"
    with patch("tharkun.solve.absint.interp.classify_loop_family", return_value="counter"):
        assert prove(safe, ("AssertionError",)).verdict == "UNKNOWN"
    with patch("tharkun.solve.absint.interp.classify_loop_family", return_value=AFFINE):
        assert prove(safe.replace("y += 2", "y += 3"), ("AssertionError",)).verdict != "TRUE"


def test_induction_rejects_a_relation_false_only_after_the_first_iteration():
    """[internal ref] negative control: `y <= x` holds at loop entry (0 <= 0) and still
    holds after one iteration through the `if` branch (1 <= 1) -- so an entry-
    only check would wrongly accept it. A second iteration through the `else`
    branch breaks it (y jumps to 6 while x is only 2), so the assert is a
    reachable violation. The induction step check
    (`step(proposed.copy()).leq(proposed)` in `Analyzer._fixpoint`, not just
    `entry.leq(proposed)`) must reject the candidate on the general step, since
    nothing forces the `if` branch on every iteration.
    """
    src = """\
import _sv_verifier
x = y = 0
while _sv_verifier.nondet_bool():
    if x == 0:
        y += 1
    else:
        y += 5
    x += 1
assert y <= x
"""
    with patch("tharkun.solve.absint.interp.classify_loop_family", return_value=AFFINE):
        assert prove(src, ("AssertionError",)).verdict != "TRUE"
