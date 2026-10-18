"""``solve/fact_env``: eregion parity for container lengths, and the bridge to
``solve/values.Value`` ([internal ref], eregion facts port)."""

from __future__ import annotations

import random

from tharkun.solve.fact_env import FactEnv, VarFacts, from_value, join, refine, to_value
from tharkun.solve.values import IntFact, StrFact, Value


def test_container_length_ops_mirror_eregion():
    e = refine(FactEnv(), "s", "container_len_eq", 3)
    assert e.facts["s"].container_len == 3
    assert not refine(e, "s", "container_len_eq", 4).possible
    assert not refine(e, "s", "container_len_ne", 3).possible
    e2 = refine(FactEnv(), "s", "container_len_ne", 2)
    assert e2.facts["s"].container_len_ne == {2}
    assert not refine(e2, "s", "container_len_eq", 2).possible
    assert refine(FactEnv(), "s", "container_len_eq", -1).facts["s"].container_len is None
    # other refinements keep the length facts
    e3 = refine(refine(e, "s", "!=", "abc"), "s", "container_len_ne", 5)
    assert e3.facts["s"].container_len == 3 and e3.facts["s"].container_len_ne == {5}


def test_join_keeps_only_shared_length_facts():
    a = refine(refine(FactEnv(), "s", "container_len_eq", 3), "s", "container_len_ne", 7)
    b = refine(refine(FactEnv(), "s", "container_len_eq", 3), "s", "container_len_ne", 8)
    j = join(a, b).facts["s"]
    assert j.container_len == 3 and j.container_len_ne == set()
    c = refine(FactEnv(), "s", "container_len_eq", 4)
    assert join(a, c).facts["s"].container_len is None


def _satisfies(vf: VarFacts, x) -> bool:
    if vf.eq is not None and x not in vf.eq:
        return False
    if x in vf.neq:
        return False
    if isinstance(x, int) and not isinstance(x, bool):
        if (vf.lo is not None and x < vf.lo) or (vf.hi is not None and x > vf.hi):
            return False
    if isinstance(x, (str, bytes)):
        if vf.container_len is not None and len(x) != vf.container_len:
            return False
        if len(x) in vf.container_len_ne:
            return False
    return True


UNIVERSE = list(range(-12, 13)) + ["", "a", "ab", "abc", "xyz", True, False, None, 2.5, -0.5]


def _random_env(rng):
    env = FactEnv()
    for _ in range(rng.randint(1, 4)):
        op = rng.choice(["==", "!=", "<", "<=", ">", ">=", "in", "not in",
                         "container_len_eq", "container_len_ne"])
        if op in ("in", "not in"):
            val = set(rng.sample(UNIVERSE[:25] + ["a", "ab", None], 4))
        elif op.startswith("container"):
            val = rng.randint(0, 4)
        elif op in ("==", "!="):
            val = rng.choice(UNIVERSE)
        else:
            val = rng.randint(-10, 10)
        env = refine(env, "x", op, val)
        if not env.possible:
            return None
    return env.facts.get("x")


def test_to_value_contains_everything_the_facts_allow():
    rng = random.Random(1021)
    for _ in range(2000):
        vf = _random_env(rng)
        if vf is None:
            continue
        v = to_value(vf)
        for x in UNIVERSE:
            if _satisfies(vf, x):
                assert v.contains_value(x), (vf, x, v)


def test_from_value_facts_hold_for_every_member():
    rng = random.Random(7)
    samples = [
        Value.of_int(IntFact(1, 5)),
        Value.of_int(IntFact(-3, None)),
        Value.of_str(StrFact(frozenset({"a", "bc"}))),
        Value.const(None).join(Value.of_int(IntFact(0, 2))),
        Value.of_int(IntFact(0, 3)).meet(Value(neq=frozenset({1}))),
        Value.const(True).join(Value.const("x")),
    ]
    for v in samples:
        vf = from_value(v)
        for x in UNIVERSE:
            if v.contains_value(x):
                assert _satisfies(vf, x), (v, vf, x)
    # a round trip keeps the int bounds; the exact set widens, because x in {1, 2, 3} also
    # admits 1.0, 2.0, 3.0 and True, and a FloatFact is an interval
    vf = refine(FactEnv(), "x", "in", {1, 2, 3}).facts["x"]
    rt = from_value(to_value(vf))
    assert (rt.lo, rt.hi) == (1, 3)
    assert from_value(Value.of_int(IntFact(1, 3))).eq == {1, 2, 3}


def test_guard_facts_model_container_lengths():
    from tharkun.solve.claims import guard_facts
    c = guard_facts(None, 1, predicate="len(s) == 3 and s != \x27abc\x27")
    assert c.feasible and c.env.facts["s"].container_len == 3
    assert not guard_facts(None, 1, predicate="len(s) == 3 and len(s) != 3").feasible
    assert not guard_facts(None, 1, predicate="len(s) == 2 and 3 == len(s)").feasible
    # a length compared by order is not modelled: no claim, never infeasible
    assert guard_facts(None, 1, predicate="len(s) > 3 and len(s) < 2").feasible
