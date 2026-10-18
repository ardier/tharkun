"""Toy fixture for the constraint-tracking -> Hypothesis demo.

Each function has a guard built from *trivial, statically-trackable*
constraints (a comparison range + a modulo; a string prefix + a length
bound). Tharkun's ``condition_solver`` can read these constraints off the
AST and infer the variable's type, but its static value generation does
**not** fully crack either guard — the modulo and the length bound defeat
its first concrete pick. That gap is exactly what makes these good
Hypothesis targets: Hypothesis is the component that produces the value
that actually satisfies the guard.

See ``demos/constraint_to_hypothesis.py`` for the end-to-end demo that
tracks the constraints, generates the value, and verifies it by executing
the functions below.
"""

from __future__ import annotations


def classify(x: int) -> str:
    # Trivial, statically-trackable constraints: x in (50, 100) AND divisible by 7.
    # condition_solver tracks {x > 50, x < 100, x % 7 == 0} and infers x: int,
    # but its static pick (a range boundary like 51) fails the modulo.
    if x > 50 and x < 100 and x % 7 == 0:
        return "hit"
    return "miss"


def version_ok(s: str) -> str:
    # Trivial, statically-trackable string constraints: starts with "v" AND len >= 4.
    # condition_solver tracks {s.startswith("v"), len(s) >= 4} and infers s: str,
    # but its static pick ("vx") is too short and fails the length bound.
    if s.startswith("v") and len(s) >= 4:
        return "valid"
    return "invalid"
