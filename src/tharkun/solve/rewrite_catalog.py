"""The single source of truth for which guard expressions mean the same thing.

Two engines canonicalize clause text — the orientation normalizer
(:mod:`tharkun.canonicalize`, the authority) and the egglog e-graph
(:mod:`tharkun.egraph`, a shadow). They implement rules in different idioms: one walks the
AST, the other applies ``rewrite`` rules to a fixpoint. Left to themselves they would
drift, and a silent drift in a canonicalizer is the exact bug it exists to prevent — two
spellings of one constraint failing to merge.

So the rules do not live in either engine. They live here, as **tested pairs**: what must
canonicalize equal, and what must stay apart. A differential test holds *both* engines to
this catalog, so adding a rule means adding its pairs here and implementing it in both —
the test fails until they agree.

This is deliberately data, not a rule-compiler. A single DSL emitting both an AST transform
and an egglog rewrite is elegant but brittle, and the two engines are not the same shape:
orientation is a normalization *procedure* (flip, order, fold), egglog is local rewrites to
a fixpoint. Holding them to a shared *contract* rather than a shared *implementation* keeps
each idiomatic and still keeps them honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuleGroup:
    """One family of equivalences, with the pairs that pin it.

    ``equivalent`` — pairs that MUST canonicalize to the same string. Each is a spelling of
    one constraint; if a canonicalizer keeps them apart, a real fact is lost at a merge.

    ``distinct`` — pairs that MUST canonicalize apart. These guard against over-eager
    rewriting: ``x > 5`` and ``x >= 5`` are different constraints, and a canonicalizer that
    merged them would invent a fact — the dangerous direction.

    ``float_sensitive`` marks a group whose equivalences only hold for non-NaN operands, so
    the differential/soundness tests know to check the NaN case rather than assume it.
    """

    name: str
    equivalent: list[tuple[str, str]] = field(default_factory=list)
    distinct: list[tuple[str, str]] = field(default_factory=list)
    float_sensitive: bool = False
    # Which engines are REQUIRED to satisfy this group. Orientation (the authority) is
    # always held to every group. egglog is a shadow that catches up asynchronously, so a
    # new rule lands in orientation first with ``engines={"orientation"}``; once the egglog
    # rules are mirrored in and verified, the group is flipped to include "egglog" and the
    # differential test starts enforcing it there too. This is what lets the Python engine
    # move fast without a red differential test waiting on the shadow.
    engines: frozenset = field(default_factory=lambda: frozenset({"orientation", "egglog"}))


# The catalog grows one RuleGroup per increment; every engine is tested against all of it.
CATALOG: list[RuleGroup] = [
    RuleGroup(
        name="comparison-orientation",
        # A lone literal goes to the right and the operator turns to match, so the two
        # ways of writing one comparison collapse.
        equivalent=[
            ("5 < x", "x > 5"),
            ("5 <= x", "x >= 5"),
            ("x < 5", "5 > x"),
            ("x <= 5", "5 >= x"),
            ("len(key) < 7", "7 > len(key)"),
        ],
        distinct=[
            ("x > 5", "x >= 5"),        # strict vs non-strict — a real difference
            ("x > 5", "x > 3"),         # different bound
            ("x > 5", "x < 5"),         # opposite direction, not a rewrite of each other
        ],
    ),
    RuleGroup(
        name="equality-symmetry",
        # == and != are symmetric; their operands still get a deterministic order.
        equivalent=[
            ("x == 5", "5 == x"),
            ("x != 5", "5 != x"),
            ("a == b", "b == a"),
        ],
        distinct=[
            ("x == 5", "x != 5"),
        ],
    ),
    RuleGroup(
        name="boolean-commutativity",
        # and/or are commutative and associative; operands are ordered, chains flattened.
        equivalent=[
            ("a and b", "b and a"),
            ("a or b or c", "c or a or b"),
            ("a and b and c", "c and b and a"),
        ],
        distinct=[
            ("a and b", "a or b"),
        ],
    ),
    RuleGroup(
        name="constant-folding-and-linear",
        # Fold constant arithmetic, drop identities, move an additive constant across a
        # comparison so the literal side is canonical.
        equivalent=[
            ("len(x) - 1 > 3", "len(x) > 4"),
            ("a + 2 == 5", "a == 3"),
            ("x + 0 > 5", "x > 5"),
            ("x * 1 == y", "x == y"),
            ("5 + 3 < n", "n > 8"),
        ],
        distinct=[
            ("len(x) > 4", "len(x) > 5"),
        ],
    ),
    RuleGroup(
        name="binop-commutativity-idempotence",
        # Commutative operands are ordered so `a + b` ≡ `b + a`, `x * 2` ≡ `2 * x`; and a
        # BoolOp's duplicate operands collapse so `a and a` ≡ `a`. Measured as egglog's only
        # edge over orientation on real guards — absorbed into the authority so egglog can
        # stay a shadow rather than a load-bearing dependency.
        equivalent=[
            ("a + b == 10", "b + a == 10"),
            ("x * 2 == y", "2 * x == y"),
            ("a and a", "a"),
            ("a or a", "a"),
            ("a and a and b", "a and b"),
        ],
        distinct=[
            ("a - b == 0", "b - a == 0"),   # subtraction is NOT commutative
            ("a and b", "a"),               # dedup removes DUPLICATES, not distinct operands
            ("a + b == 1", "a * b == 1"),   # different operator
        ],
    ),
    RuleGroup(
        name="negation-nnf",
        # Negation is pushed to the leaves. The ORDERING comparisons here are float-safe
        # only for non-NaN operands — `not (x > 5)` is true for NaN while `x <= 5` is not —
        # so this group is float_sensitive and the NaN-safe form is added in I2.
        equivalent=[
            ("not (a and b)", "(not a) or (not b)"),
            ("not (a or b)", "(not a) and (not b)"),
            ("not (not a)", "a"),
            ("not (x == 5)", "x != 5"),        # ==/!= negation IS NaN-safe
            ("not (x != 5)", "x == 5"),
        ],
        distinct=[
            ("not a", "a"),
        ],
        float_sensitive=True,
    ),
    RuleGroup(
        name="string-slice-startswith",
        # A pure prefix-slice compared equal to a same-length literal IS `startswith`:
        # `X[:k] == LIT` with len(LIT) == k agrees with `X.startswith(LIT)` on every X.
        # `startswith` is the canonical form; the literal may start on either side.
        equivalent=[
            ("token[:2] == b'ab'", "token.startswith(b'ab')"),
            ("s[:3] == 'abc'", "s.startswith('abc')"),
            ("b'ab' == token[:2]", "token.startswith(b'ab')"),
        ],
        distinct=[
            ("token[:2] == b'ab'", "token[:3] == b'abc'"),          # different length
            ("token[:2] == b'ab'", "token[:2] == b'ba'"),           # different prefix
            ("token.startswith(b'ab')", "token.endswith(b'ab')"),   # prefix != suffix
            ("token[1:3] == b'ab'", "token.startswith(b'ab')"),     # offset slice, not a prefix
            ("token[:2] == b'a'", "token.startswith(b'a')"),        # len(LIT) != stop
        ],
    ),
]


def equivalent_pairs(engine: str = "orientation") -> list[tuple[str, str, str]]:
    """Every (a, b, group_name) that must canonicalize equal UNDER *engine*.

    Orientation is required on every group; egglog only on groups whose ``engines`` names
    it, so the shadow can lag without failing the contract for what it has not caught up on.
    """
    return [(a, b, g.name) for g in CATALOG if engine in g.engines for a, b in g.equivalent]


def distinct_pairs(engine: str = "orientation") -> list[tuple[str, str, str]]:
    """Every (a, b, group_name) that must canonicalize apart UNDER *engine*."""
    return [(a, b, g.name) for g in CATALOG if engine in g.engines for a, b in g.distinct]


def lagging_groups() -> list[str]:
    """Groups orientation satisfies that egglog has not caught up to — the agent's worklist."""
    return [g.name for g in CATALOG if "egglog" not in g.engines]
