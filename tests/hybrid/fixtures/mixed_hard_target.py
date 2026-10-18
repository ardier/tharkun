"""HARD mixed: sibling frontiers whose bodies need the *hard* machinery.

Each arm is independent (depth 0) but its interior needs a different mechanism:
  A  exact 32-bit magic, then a nested exact version   -> value copy
  B  nonlinear square, then a modular residue          -> SMT
  C  uninvertible modular hash                         -> SMT only
  D  string prefix, then a slice-aliased suffix        -> needs the input to GROW
"""


def parse(magic, ver, tag, mod, name):
    hits = []
    if magic == 3405691582:              # A: 0xCAFEBABE
        if ver == 7:
            hits.append("A")
    if tag * tag == 1369:                # B: tag == 37
        if tag % 5 == 2:
            hits.append("B")
    if (mod * 2654435761 + 1) % 4294967291 == 987654:   # C: uninvertible
        hits.append("C")
    if name.startswith("cfg://"):        # D: prefix ...
        rest = name[6:]
        if rest == "prod":               # ... then a suffix needing growth
            hits.append("D")
    return hits
