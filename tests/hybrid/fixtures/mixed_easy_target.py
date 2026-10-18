"""EASY mixed: sibling frontiers, each with a shallow nested chain.

Independent params per arm, so the arms are genuinely siblings (depth 0) rather than a
chain — but each body then requires a second, nested guard, so the assist must pick a
frontier and the driver must solve depth inside it.
"""


def route(a, b, c):
    hits = []
    if a > 100:              # sibling A, depth 0
        if a < 110:          #   nested, depth 1
            hits.append("A")
    if b == 42:              # sibling B
        if b * 2 == 84:      #   nested (implied, still a level)
            hits.append("B")
    if len(c) >= 3:          # sibling C
        if c[0] == 120:      #   nested: first byte is 'x'
            hits.append("C")
    return hits
