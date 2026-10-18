"""[internal ref] raytrace: a call is named by its WHOLE node, not by where it starts.

``a + b + c`` with a user ``__add__`` is two calls whose ``BinOp`` nodes both start at ``a``.  The
call string used ``(line, col)`` of the start only, so the two calls shared one allocation site
inside ``__add__``; the second allocation found the first result still held by the outer
operation and absint refused ("allocation would invalidate an in-flight heap reference").  That
was a gap in the model's site identity, not a limit of the technique: the call string now carries
the node's end position too.  The site records' ``context`` still lists ``[line, col]`` pairs.
"""

from __future__ import annotations

import textwrap

from tharkun.solve.absint import prove

CLASSES = ["ZeroDivisionError"]

VECTOR = """
class V:
    def __init__(self, x):
        self.x = x
    def __add__(self, o):
        return V(self.x + o.x)
    def __mul__(self, o):
        return V(self.x * o.x)
"""


def _prove(body: str, classes=CLASSES):
    return prove(textwrap.dedent(VECTOR) + textwrap.dedent(body), classes, filename="<program>")


def test_a_chained_user_operator_is_proved_not_refused():
    result = _prove("""
        def main(a, b, c):
            v = V(a) + V(b) + V(c)
            return v.x // 1
        main(1, 2, 3)
    """)
    assert result.verdict == "TRUE", result.reason
    assert result.complete


def test_a_longer_chain_and_mixed_operators_are_proved():
    result = _prove("""
        def main(a, b, c, d):
            v = V(a) + V(b) + V(c) + V(d)
            w = V(a) * V(b) + V(c) * V(d)
            return v.x // 1 + w.x // 1
        main(1, 2, 3, 4)
    """)
    assert result.verdict == "TRUE", result.reason


def _program_records(result):
    """The site records of the program's own operations: not the ones of an object protocol's rule (``protocol.<name>:<class>``, [internal ref])."""
    return [r for r in result.records if not r["kind"].startswith("protocol.")]


def test_the_same_result_for_a_chain_written_with_temporaries():
    chained = _prove("""
        def main(a, b, c):
            v = V(a) + V(b) + V(c)
            return 10 // (v.x - 6)
        main(1, 2, 3)
    """)
    stepwise = _prove("""
        def main(a, b, c):
            t = V(a) + V(b)
            v = t + V(c)
            return 10 // (v.x - 6)
        main(1, 2, 3)
    """)
    assert chained.verdict == stepwise.verdict == "UNKNOWN"
    # the operator's own records (``protocol.binary_operator:<class>``, keyed by the position of the node) are not what is compared:
    # a chain on one line shares one position, the stepwise form has two
    assert ([r["kind"] for r in _program_records(chained)] == [r["kind"] for r in _program_records(stepwise)])


def test_a_real_division_by_zero_after_a_chain_is_still_found():
    result = _prove("""
        def main(a, b, c):
            v = V(a) + V(b) + V(c)
            return 10 // (v.x - 6)
        main(1, 2, 3)
    """)
    assert result.verdict == "UNKNOWN"
    divide = [r for r in _program_records(result) if r["exc"] == "ZeroDivisionError"]
    assert divide and divide[0]["status"] == "may_raise"


def test_the_site_records_context_keeps_line_and_column_pairs():
    result = _prove("""
        def helper(a):
            return 10 // a
        def main(a):
            return helper(a)
        main(0)
    """)
    contexts = [r["context"] for r in result.records if r["exc"] == "ZeroDivisionError"]
    assert contexts and all(len(pair) == 2 for ctx in contexts for pair in ctx), contexts
