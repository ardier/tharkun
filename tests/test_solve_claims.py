"""``tharkun.solve.claims`` (#0611): sound value-set claims for a guard anchor,
and the fact carrier's refine / join arithmetic they are built on."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tharkun.solve import claims, loader
from tharkun.solve.claims import IntDomain, guard_domains, guard_facts
from tharkun.solve.fact_env import FactEnv, VarFacts, join, refine, refine_all

SOURCE = '''
import pathlib

pathlib.Path(__MARKER__).write_text("ran")


def f(x, y, s, mode):
    if x > 50 and x <= 100:          # L_conj
        return 1
    if x == 3 or x == 5:             # L_disj
        return 2
    if 10 < y < 20 and y != 15:      # L_chain
        return 3
    if not (x >= 7):                 # L_neg
        return 4
    if helper(x) and x < 4:          # L_call
        return 5
    if mode in ("a", "b") and s is None:   # L_membership
        return 6
    if x > 5 and x < 3:              # L_infeasible
        return 7
    return 0
'''


@pytest.fixture
def target(tmp_path: Path) -> tuple[str, dict[str, int], Path]:
    marker = tmp_path / "ran.marker"
    src = tmp_path / "guards_0611.py"
    src.write_text(textwrap.dedent(SOURCE).replace("__MARKER__", repr(str(marker))), encoding="utf-8")
    lines = {}
    for i, line in enumerate(src.read_text().splitlines(), 1):
        if "# L_" in line:
            lines[line.split("# ")[1].strip()] = i
    return str(src), lines, marker


def test_conjunction_refines_to_an_interval(target) -> None:
    path, lines, marker = target
    claim = guard_facts(path, lines["L_conj"])
    assert claim.feasible and claim.sound_superset
    assert claim.domains["x"] == IntDomain(lo=51, hi=100, values=None, excluded=frozenset())
    assert claim.domains["x"].kind == "interval"
    assert claim.unmodelled_atoms == 0
    assert claim.provenance.backend == "tharkun.solve" and claim.provenance.strategy == "static_intervals"
    assert claim.guard_source == "x > 50 and x <= 100"
    assert not marker.exists()  # the source was parsed, never run


def test_prior_facts_are_intersected(target) -> None:
    path, lines, _ = target
    prior = FactEnv(facts={"x": VarFacts(lo=0, hi=60), "z": VarFacts(eq={1})})
    claim = guard_facts(path, lines["L_conj"], facts=prior)
    assert claim.domains["x"] == IntDomain(lo=51, hi=60)
    assert claim.domains["z"] == IntDomain(values=frozenset({1}))  # untouched prior survives
    assert prior.facts["x"] == VarFacts(lo=0, hi=60)  # inputs are never mutated


def test_disjunction_joins_to_a_finite_set(target) -> None:
    path, lines, _ = target
    d = guard_domains(path, lines["L_disj"])["x"]
    assert d.values == frozenset({3, 5}) and d.lo == 3 and d.hi == 5 and d.kind == "finite_set"


def test_chained_comparison_and_exclusion(target) -> None:
    path, lines, _ = target
    d = guard_domains(path, lines["L_chain"])["y"]
    assert d == IntDomain(lo=11, hi=19, excluded=frozenset({15}))


def test_else_branch_negates(target) -> None:
    path, lines, _ = target
    assert guard_domains(path, lines["L_neg"])["x"] == IntDomain(hi=6)          # not (x >= 7)
    assert guard_domains(path, lines["L_neg"], desired_branch="else")["x"] == IntDomain(lo=7)
    assert guard_domains(path, lines["L_conj"], desired_branch="else")["x"].kind == "unknown"  # x <= 50 or x > 100


def test_unmodelled_atoms_never_narrow(target) -> None:
    path, lines, _ = target
    claim = guard_facts(path, lines["L_call"])
    assert claim.domains["x"] == IntDomain(hi=3) and claim.unmodelled_atoms == 1
    assert claim.sound_superset


def test_membership_and_is_none(target) -> None:
    path, lines, _ = target
    claim = guard_facts(path, lines["L_membership"])
    assert claim.env.facts["mode"].eq == {"a", "b"}
    assert claim.env.facts["s"].eq == {None}
    # #0657: a homogeneous str set IS a claim now -- it was declined before
    # only for not being ints, which lost a fact the guard states exactly.
    assert claim.domains["mode"].values == frozenset({"a", "b"})
    # ``s is None`` still yields no domain: None is not a scalar type with a
    # value set, so there is nothing sound to claim.
    assert claim.domains["s"].values is None


def test_infeasible_guard(target) -> None:
    path, lines, _ = target
    claim = guard_facts(path, lines["L_infeasible"])
    assert not claim.feasible and guard_domains(path, lines["L_infeasible"]) == {}


def test_no_guard_at_line_returns_the_prior(target) -> None:
    path, _lines, _ = target
    prior = FactEnv(facts={"x": VarFacts(lo=1, hi=2)})
    claim = guard_facts(path, 2, facts=prior)
    assert claim.env is prior and claim.provenance.strategy == "no_guard" and claim.domains["x"] == IntDomain(lo=1, hi=2)
    assert guard_facts("/no/such/file.py", 1).provenance.strategy == "no_guard"


def test_witness_is_separate_and_loads_no_target(target, monkeypatch: pytest.MonkeyPatch) -> None:
    path, lines, marker = target
    calls: list[str] = []
    monkeypatch.setattr(loader, "exec_source", lambda p: calls.append(p))
    claim = guard_facts(path, lines["L_conj"], witness=True)
    assert claim.witness is not None and 50 < claim.witness["x"] <= 100
    assert claim.domains["x"] == IntDomain(lo=51, hi=100)  # the witness never touches the domains
    assert claim.provenance.strategy.startswith("static_intervals+witness:")
    assert not marker.exists() and calls == []


def test_assert_and_assume_call_statements_are_guards(tmp_path: Path) -> None:
    src = tmp_path / "lowered_0612.py"
    src.write_text(textwrap.dedent('''
        def __bw_assume__(cond):
            return cond

        def f(x):
            __bw_assume__(x < 0)          # L_assume_then
            __bw_assume__(not x < 0 and x > 0)   # L_assume_else
            assert x <= 9                 # L_assert
            other(x > 100)                # L_other_call
            return x
    '''), encoding="utf-8")
    lines = {line.split("# ")[1].strip(): i for i, line in enumerate(src.read_text().splitlines(), 1) if "# L_" in line}
    calls = ("__bw_assume__",)
    assert guard_domains(str(src), lines["L_assume_then"], assume_calls=calls)["x"] == IntDomain(hi=-1)
    assert guard_domains(str(src), lines["L_assume_else"], assume_calls=calls)["x"] == IntDomain(lo=1)
    assert guard_domains(str(src), lines["L_assert"])["x"] == IntDomain(hi=9)
    assert guard_facts(str(src), lines["L_assume_then"]).provenance.strategy == "no_guard"  # not named: not a guard
    assert guard_facts(str(src), lines["L_other_call"], assume_calls=calls).provenance.strategy == "no_guard"
    # chained through the prior: then-guard, else-guard of a later branch, assert
    env = FactEnv()
    for key in ("L_assume_then", "L_assert"):
        env = guard_facts(str(src), lines[key], facts=env, assume_calls=calls).env
    assert env.facts["x"] == VarFacts(hi=-1)
    assert not guard_facts(str(src), lines["L_assume_else"], facts=env, assume_calls=calls).feasible


def test_tree_anchor_needs_no_file(tmp_path: Path) -> None:
    """#0622: the caller's own AST is the anchor; the path is display only."""
    import ast

    src = textwrap.dedent('''
        def __bw_assume__(cond):
            return cond

        def f(x):
            __bw_assume__(x > 50)      # G1
            __bw_assume__(x <= 100)    # G2
            return x
    ''')
    tree = ast.parse(src)
    lines = {line.split("# ")[1].strip(): i for i, line in enumerate(src.splitlines(), 1) if "# G" in line}
    calls = ("__bw_assume__",)
    assert guard_domains(None, lines["G1"], assume_calls=calls, tree=tree)["x"] == IntDomain(lo=51)
    env = guard_facts(None, lines["G1"], assume_calls=calls, tree=tree).env
    claim = guard_facts(None, lines["G2"], facts=env, assume_calls=calls, tree=tree)
    assert claim.domains["x"] == IntDomain(lo=51, hi=100) and claim.provenance.strategy == "static_intervals"
    # the tree wins over a path that would say something else
    other = tmp_path / "other.py"
    other.write_text("def g(x):\n    if x == 7:\n        return 1\n", encoding="utf-8")
    assert guard_domains(str(other), lines["G1"], assume_calls=calls, tree=tree)["x"] == IntDomain(lo=51)
    assert guard_domains(str(other), 2)["x"] == IntDomain(lo=7, hi=7, values=frozenset({7}))
    # neither a tree nor a path: no guard, never an exception
    assert guard_facts(None, 3).provenance.strategy == "no_guard"
    # a witness needs a path; with a tree only the claim is still complete
    claim = guard_facts(None, lines["G1"], assume_calls=calls, tree=tree, witness=True)
    assert claim.witness is None and claim.domains["x"] == IntDomain(lo=51)


def test_assignments_replay_a_path() -> None:
    """#0622: a guard on a variable that is later reassigned must not
    contradict the write -- replaying the path in order, writes kill."""
    import ast

    src = textwrap.dedent('''
        def __bw_assume__(cond):
            return cond

        def f(data):
            q = 3                          # A0
            __bw_assume__(q == 3)          # G1
            x = data[0]                    # A1
            __bw_assume__(x == 0)          # G2
            q = 2                          # A2
            __bw_assume__(q == 2)          # G3
            a, b = 1, 2                    # A3
            return q
    ''')
    tree = ast.parse(src)
    lines = {line.split("# ")[1].strip(): i for i, line in enumerate(src.splitlines(), 1) if "# " in line and line.strip().endswith(line.split("# ")[1].strip())}
    calls = ("__bw_assume__",)
    env = FactEnv()
    for key in ("A0", "G1", "A1", "G2", "A2", "G3", "A3"):
        claim = guard_facts(None, lines[key], facts=env, assume_calls=calls, tree=tree, assignments=True)
        assert claim.feasible, key
        assert claim.provenance.strategy == ("assignment" if key.startswith("A") else "static_intervals")
        env = claim.env
    assert env.facts["q"] == VarFacts(eq={2}, lo=2, hi=2)      # the write to q replaced the q == 3 fact
    assert env.facts["x"] == VarFacts(eq={0}, lo=0, hi=0)      # x = data[0] killed nothing it did not know; the guard pinned it
    assert env.facts["a"].eq == {1} and env.facts["b"].eq == {2}
    # without assignments=True the same replay contradicts itself
    env2 = FactEnv()
    for key in ("G1", "G3"):
        env2 = guard_facts(None, lines[key], facts=env2, assume_calls=calls, tree=tree).env
    assert not env2.possible
    # an assignment line is not a guard unless asked
    assert guard_facts(None, lines["A0"], tree=tree).provenance.strategy == "no_guard"


def test_predicates_need_no_source() -> None:
    """#0622: the caller hands over the guard (or the write) itself -- source
    text or an ``ast.expr`` -- so statements that share a line number (a
    lowering pass copying one location onto everything it injects) are still
    replayed one by one, in path order, with no file, tree or line lookup."""
    import ast

    env = FactEnv()
    path = (
        ("assignment", "q = 3"), ("predicate", "q == 3"), ("assignment", "x = data[0]"),
        ("predicate", "x == 0"), ("assignment", "q = 2"), ("predicate", "not (q != 2)"),
    )
    for kind, text in path:
        claim = guard_facts(None, None, facts=env, **{kind: text})
        assert claim.feasible, text
        assert claim.provenance.strategy == ("assignment" if kind == "assignment" else "static_intervals")
        env = claim.env
    assert env.facts["q"] == VarFacts(eq={2}, lo=2, hi=2)      # the write to q replaced the q == 3 fact
    assert env.facts["x"] == VarFacts(eq={0}, lo=0, hi=0)
    assert guard_facts(None, 25, predicate="x > 50").guard_source == "x > 50"   # the line is display only
    # polarity and a ready-made node work the same way
    assert guard_domains(None, None, predicate="x > 50", desired_branch="else")["x"] == IntDomain(hi=50)
    assert guard_domains(None, None, predicate=ast.parse("x > 50", mode="eval").body)["x"] == IntDomain(lo=51)
    # text that is not a predicate, a write that is not to plain names, two statements: no guard, never a raise
    assert guard_facts(None, None, predicate="x >").provenance.strategy == "no_guard"
    assert guard_facts(None, None, assignment="a[0] = 1").provenance.strategy == "no_guard"
    assert guard_facts(None, None, assignment="a = 1; b = 2").provenance.strategy == "no_guard"
    assert guard_facts(None, None, assignment="def f(): pass").provenance.strategy == "no_guard"
    # a witness needs a path and a line; the claim itself is still complete
    claim = guard_facts(None, None, predicate="x > 50", witness=True)
    assert claim.witness is None and claim.domains["x"] == IntDomain(lo=51)
    # an unmodelled atom narrows nothing and is counted
    claim = guard_facts(None, None, predicate="helper(x) and x < 9")
    assert claim.domains["x"] == IntDomain(hi=8) and claim.unmodelled_atoms == 1


# ------------------------------------------------------------ refine / join
def test_refine_is_pure_and_monotone() -> None:
    env = FactEnv()
    e1 = refine(env, "x", ">", 5)
    e2 = refine(e1, "x", "<=", 9)
    assert env.facts == {} and e1.facts["x"] == VarFacts(lo=6) and e2.facts["x"] == VarFacts(lo=6, hi=9)
    e3 = refine(e2, "x", "==", 7)
    assert e3.facts["x"] == VarFacts(eq={7}, lo=7, hi=7)
    assert not refine(e3, "x", "!=", 7).possible
    assert not refine(e2, "x", "<", 6).possible
    assert refine(e2, "x", ">", "str").facts["x"] == VarFacts(lo=6, hi=9)  # non-int ordering: no-op


def test_refine_membership_and_exclusions() -> None:
    env = refine(FactEnv(), "m", "in", {"a", "b", "c"})
    env = refine(env, "m", "not in", {"c"})
    assert env.facts["m"].eq == {"a", "b"} and env.facts["m"].neq == {"c"}
    ints = refine(FactEnv(), "k", "in", {3, 9, 5})
    assert (ints.facts["k"].lo, ints.facts["k"].hi) == (3, 9)
    assert refine(ints, "k", "in", {1}).possible is False
    assert refine_all(FactEnv(), [("x", ">=", 1), ("x", "<=", 3), ("x", "!=", 2)]).facts["x"] == VarFacts(neq={2}, lo=1, hi=3)


def test_join_is_the_union_of_possibilities() -> None:
    a = refine_all(FactEnv(), [("x", "==", 3), ("y", ">", 0)])
    b = refine_all(FactEnv(), [("x", "==", 5), ("y", "<", 10), ("z", "==", 1)])
    j = join(a, b)
    assert j.facts["x"] == VarFacts(eq={3, 5}, lo=3, hi=5)
    assert j.facts["y"] == VarFacts()          # one-sided bounds do not survive a join
    assert "z" not in j.facts                  # only vars known on both sides
    wide = join(refine(FactEnv(), "x", "in", set(range(6))), refine(FactEnv(), "x", "in", set(range(6, 12))))
    assert wide.facts["x"].eq is None          # > MAX_CONST_SET collapses to unknown
    assert join(FactEnv(possible=False), a).facts == a.facts


def test_solve_facade_exports_the_claims_api() -> None:
    import tharkun.solve as solve

    assert solve.guard_facts is guard_facts and solve.IntDomain is IntDomain and solve.refine is refine
    assert claims.BACKEND == "tharkun.solve"


# ------------------------------------------------------------------ #0657
# A guard on a string states a value set exactly as much as one on an int.
# Before #0657 ``IntDomain.of`` filtered every non-int away, so the claim came
# back "unknown" where the fact was known -- measured as 12 corpus requests
# across 3 toys (0512/0513 str, 0514 bytes, 0515 float).

def test_typed_eq_sets_are_carried():
    from tharkun.solve import IntDomain
    from tharkun.solve.fact_env import VarFacts

    for members, kind in (({"GET", "POST"}, "finite_set"), ({b"\x00"}, "point"),
                          ({2.5}, "point"), ({True}, "point")):
        d = IntDomain.of(VarFacts(eq=set(members)))
        assert d.values == frozenset(members), members
        assert d.kind == kind, (members, d.kind)


def test_typed_neq_sets_are_carried():
    from tharkun.solve import IntDomain
    from tharkun.solve.fact_env import VarFacts

    assert IntDomain.of(VarFacts(neq={"GET"})).excluded == frozenset({"GET"})


def test_mixed_and_nan_sets_decline():
    """Two shapes have no sound domain: a mixed set has no single type, and a
    NaN breaks the set equality every consumer of ``values`` relies on."""
    from tharkun.solve import IntDomain
    from tharkun.solve.fact_env import VarFacts

    assert IntDomain.of(VarFacts(eq={1, "a"})).values is None
    assert IntDomain.of(VarFacts(eq={float("nan")})).values is None
    assert IntDomain.of(VarFacts(eq={1.0, float("nan")})).values is None


def test_int_behaviour_is_unchanged():
    from tharkun.solve import IntDomain
    from tharkun.solve.fact_env import VarFacts

    d = IntDomain.of(VarFacts(eq={2, 5}, neq={5}, lo=0, hi=9))
    assert d.values == frozenset({2, 5}) and d.excluded == frozenset({5})
    assert d.lo == 0 and d.hi == 9 and d.kind == "point"


def test_exclusions_of_another_type_are_dropped_not_mixed():
    from tharkun.solve import IntDomain
    from tharkun.solve.fact_env import VarFacts

    d = IntDomain.of(VarFacts(eq={"GET"}, neq={3}))
    assert d.values == frozenset({"GET"}) and d.excluded == frozenset()


def test_a_guard_on_a_string_now_claims(tmp_path):
    """End to end through ``guard_facts``: the 0512/0513 shapes."""
    from tharkun.solve import guard_domains

    src = "def run(s):\n    if s == 'POST':\n        pass\n    return s\n"
    p = tmp_path / "m.py"; p.write_text(src, encoding="utf-8")
    assert guard_domains(str(p), 2)["s"].values == frozenset({"POST"})

    src2 = "def run(s):\n    if s in ('GET', 'POST'):\n        pass\n    return s\n"
    p2 = tmp_path / "m2.py"; p2.write_text(src2, encoding="utf-8")
    assert guard_domains(str(p2), 2)["s"].values == frozenset({"GET", "POST"})


def test_a_call_shaped_constant_is_folded(tmp_path):
    """Toy 0514's shape: the CONSTANT is ``bytes.fromhex('00')``, a Call, which
    the extractor used to decline -- losing the fact for a spelling."""
    from tharkun.solve import guard_domains

    src = "def run(b):\n    if b == bytes.fromhex('00'):\n        pass\n    return b\n"
    p = tmp_path / "m.py"; p.write_text(src, encoding="utf-8")
    assert guard_domains(str(p), 2)["b"].values == frozenset({b"\x00"})


# ------------------------------------------------------------------ #0666
def test_an_unmodelled_atom_reports_its_names_and_shape():
    """A count cannot answer "was it MY variable?".  A consumer that has to
    replay the guard through ``_atom`` to find out is a consumer
    re-implementing ``_atom``; it drifts the moment the vocabulary grows."""
    import ast

    def shape(src):
        return claims._unmodelled(ast.parse(src, mode="eval").body)

    assert shape("c3 == data[i]").shape == "subscript"
    assert shape("c3 == data[i]").names == frozenset({"c3", "data", "i"})
    assert shape("P3 % 2 == 1").shape == "binop-arith"
    assert shape("P3 % 2 == 1").names == frozenset({"P3"})
    assert shape("b & 240 == 48").shape == "binop-bitwise"
    assert shape("not x").shape == "truthiness"
    assert shape("x").shape == "truthiness"
    assert shape("not __call_succeeds__(int, s)").shape == "call"
    assert shape("x > y").shape == "name-vs-name"
    assert shape("obj.f == 1").shape == "attribute"
    # The source is kept so a diagnosis can quote the guard it lost.
    assert shape("P3 % 2 == 1").source == "P3 % 2 == 1"


def test_the_claim_carries_the_atoms_it_could_not_model(tmp_path):
    """Both halves on one guard: one atom models, one does not, and the claim
    says which -- so "an atom was dropped somewhere on the path" and "the atom
    about x was dropped" stop being the same observation."""
    src = (
        "def run(x, y, data, i):\n"
        "    if x == 3 and y == data[i]:\n"
        "        pass\n"
        "    return x\n"
    )
    p = tmp_path / "m.py"
    p.write_text(src, encoding="utf-8")
    claim = guard_facts(str(p), 2)

    assert claim.unmodelled_atoms == 1
    assert len(claim.unmodelled) == 1
    (atom,) = claim.unmodelled
    assert atom.shape == "subscript"
    assert "y" in atom.names and "x" not in atom.names
    # x modelled cleanly despite sharing the guard with an unmodelled atom.
    assert claim.domains["x"].values == frozenset({3})


def test_every_unmodelled_atom_is_reported_not_just_counted():
    """Two dropped atoms in one disjunction: the count says 2, and the records
    say which two and about whom."""
    import ast

    from tharkun.solve.fact_env import FactEnv

    expr = ast.parse("f(a, b) or g(c)", mode="eval").body
    _env, found = claims._refine_expr(FactEnv(), expr)
    assert [a.shape for a in found] == ["call", "call"]
    assert found[0].names == frozenset({"f", "a", "b"})
    assert found[1].names == frozenset({"g", "c"})
