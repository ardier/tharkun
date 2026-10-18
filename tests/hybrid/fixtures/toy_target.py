"""Four INDEPENDENT frontiers, deliberately trivial.

Each arm is gated by a different exact value, and no arm is nested inside another —
so the assist must unlock one, re-rank, and move to the next, rather than walking a
single chain. Each body is a different size so the dynamic ranking has something to
order by.
"""


def dispatch(op, x, y, z):
    hits = []
    if op == 1001:            # frontier A — biggest body, should rank first
        hits.append("A1")
        hits.append("A2")
        hits.append("A3")
        hits.append("A4")
    if x == 2002:             # frontier B
        hits.append("B1")
        hits.append("B2")
        hits.append("B3")
    if y == 3003:             # frontier C
        hits.append("C1")
        hits.append("C2")
    if z == 4004:             # frontier D — smallest, should rank last
        hits.append("D1")
    return hits
