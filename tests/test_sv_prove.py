"""tharkun.sv proofs ([internal ref]): ``solve/drawflow`` claims and the ``proof:stationary`` runs.

Soundness first: every MUTANT below has an execution that violates (or may
violate) the property, and none may come back TRUE.  The accepted programs are
the shapes of the SV-Benchmarks pyperformance tasks the proof exists for.
"""

from __future__ import annotations

import sys
import textwrap
import time

import pytest

from tharkun.solve.drawflow import plan_draws
from tharkun.sv import verify
from tharkun.sv.prove import prove_dynamic

DL = ("KeyError", "IndexError")
AR = ("ZeroDivisionError", "FloatingPointError")


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def _prove(path, classes=DL, seconds=120):
    t0 = time.monotonic()
    return prove_dynamic(path, classes, python=sys.executable, jobs=2,
                         remaining=lambda: seconds - (time.monotonic() - t0))


# ---------------------------------------------------------------- drawflow (static)

ACCEPT = {
    "param_loop": """
        import _sv_verifier
        def bench(loops):
            r = range(loops)
            for _ in r:
                x = [1, 2]
        if __name__ == "__main__":
            bench(abs(_sv_verifier.nondet_int()))
    """,
    "rebinding_target": """
        import _sv_verifier
        def bench(loops):
            for loops in range(loops):
                pass
        bench(abs(_sv_verifier.nondet_int()))
    """,
    "offset_and_quotient": """
        import _sv_verifier
        LEVELS = [1, 2, 3]
        def main(loops, level):
            for _ in range(loops):
                pass
            return LEVELS[level]
        level = abs(_sv_verifier.nondet_int()) % len(LEVELS)
        loops = 1 + abs(_sv_verifier.nondet_int())
        main(loops, level)
    """,
    # [internal ref], the sound part: a trip value reaches a call whose result is a function of its
    # CLASS only (it is always an exact int), so no result can vary with the trip count
    "isinstance_guard": """
        import _sv_verifier
        def bench(loops):
            assert isinstance(loops, int)
            for _ in range(loops):
                x = [1, 2]
        if __name__ == "__main__":
            bench(abs(_sv_verifier.nondet_int()))
    """,
    "type_guard_on_a_variable": """
        import _sv_verifier
        def bench(loops):
            if type(loops) is not int:
                raise TypeError("loops")
            for _ in range(loops):
                pass
        n = 1 + abs(_sv_verifier.nondet_int())
        assert isinstance(n, (int, float))
        bench(n)
    """,
    "shadowed_builtin": """
        import _sv_verifier
        class Node:
            def __init__(self, id):
                self.id = id
        def f(n):
            for _ in range(n):
                Node(3)
        f(abs(_sv_verifier.nondet_int()))
    """,
}

REFUSE = {
    "read_after_loop": ("""
        import _sv_verifier
        def f(n):
            for _ in range(n):
                pass
            return [0][n]
        f(abs(_sv_verifier.nondet_int()))
    """, "read by"),
    "target_read": ("""
        import _sv_verifier
        def f(n):
            for i in range(n):
                [0, 1][i]
        f(abs(_sv_verifier.nondet_int()))
    """, "loop variable i is read"),
    "no_abs": ("""
        import _sv_verifier
        for _ in range(_sv_verifier.nondet_int()):
            pass
    """, "not taken abs"),
    "range_object_escapes": ("""
        import _sv_verifier
        r = range(abs(_sv_verifier.nondet_int()))
        for _ in r:
            pass
        [0][len(r)]
    """, "trip range"),
    "rebound_param": ("""
        import _sv_verifier
        def f(n):
            n = n + 1
            for _ in range(n):
                pass
        f(abs(_sv_verifier.nondet_int()))
    """, "rebound"),
    # [internal ref]: transparent only for CLASS-only results. Every other builtin's result varies with
    # the value, and the induction stops tracking a value once the planner stops tracking it
    "builtin_value_flow": ("""
        import _sv_verifier
        def f(n):
            for _ in range(n):
                pass
            return divmod(n, 5)
        f(abs(_sv_verifier.nondet_int()))
    """, "unresolved callee"),
    "isinstance_user_class": ("""
        import _sv_verifier
        class M(type):
            def __instancecheck__(cls, inst):
                return inst == 13
        class K(metaclass=M):
            pass
        def f(n):
            for _ in range(n):
                pass
            return isinstance(n, K)
        f(abs(_sv_verifier.nondet_int()))
    """, "unresolved callee"),
    "isinstance_shadowed": ("""
        import _sv_verifier
        def isinstance(x, c):
            return x == 13
        def f(n):
            for _ in range(n):
                pass
            return isinstance(n, int)
        f(abs(_sv_verifier.nondet_int()))
    """, "read by"),
    "type_with_keywords": ("""
        import _sv_verifier
        def f(n):
            for _ in range(n):
                pass
            return type(object=n)
        f(abs(_sv_verifier.nondet_int()))
    """, "unresolved callee"),
    "isinstance_value_argument": ("""
        import _sv_verifier
        def f(n):
            for _ in range(n):
                pass
            return isinstance(5, n)
        f(abs(_sv_verifier.nondet_int()))
    """, "unresolved callee"),
    "method_callee": ("""
        import _sv_verifier
        class K:
            def run(self, n):
                for _ in range(n):
                    pass
        K().run(abs(_sv_verifier.nondet_int()))
    """, "unresolved callee"),
    "module_state_store": ("""
        import _sv_verifier, string
        def f(n):
            for _ in range(n):
                string.counter = 1
        f(abs(_sv_verifier.nondet_int()))
    """, "module string"),
    "private_module_attr": ("""
        import _sv_verifier, re
        def f(n):
            for _ in range(n):
                len(re._cache)
        f(abs(_sv_verifier.nondet_int()))
    """, "module re"),
    "unlisted_import": ("""
        import _sv_verifier, os
        for _ in range(abs(_sv_verifier.nondet_int())):
            pass
    """, "imports os"),
    "introspection": ("""
        import _sv_verifier
        def f(n):
            for _ in range(n):
                locals()
        f(abs(_sv_verifier.nondet_int()))
    """, "uses locals"),
    "id_builtin": ("""
        import _sv_verifier
        def f(n):
            for _ in range(n):
                id(n)
        f(abs(_sv_verifier.nondet_int()))
    """, "uses id"),
    "float_draw": ("""
        import _sv_verifier
        x = _sv_verifier.nondet_float()
    """, "unbounded draw kind"),
    "aliased_verifier": ("""
        import _sv_verifier
        nd = _sv_verifier.nondet_int
        for _ in range(abs(nd())):
            pass
    """, "used as a value"),
    "nested_rebinding": ("""
        import _sv_verifier
        def f(v):
            for _ in range(3):
                for v in range(v):
                    pass
        f(abs(_sv_verifier.nondet_int()))
    """, "nested loop"),
    "global_decl": ("""
        import _sv_verifier
        n = abs(_sv_verifier.nondet_int())
        def g():
            global n
            n = 5
        for _ in range(n):
            pass
    """, "global"),
    "finalizer": ("""
        import _sv_verifier
        class K:
            def __del__(self):
                pass
        for _ in range(abs(_sv_verifier.nondet_int())):
            K()
    """, "__del__"),
    "modulus_draws": ("""
        import _sv_verifier
        q = abs(_sv_verifier.nondet_int()) % (1 + abs(_sv_verifier.nondet_int()))
    """, "modulus draws"),
}


@pytest.mark.parametrize("name", sorted(ACCEPT))
def test_drawflow_accepts_trip_and_quotient_draws(name):
    plan = plan_draws(textwrap.dedent(ACCEPT[name]))
    assert plan.ok, plan.reason
    assert all(d.kind in ("trip", "quotient", "finite") for d in plan.draws)


@pytest.mark.parametrize("name", sorted(REFUSE))
def test_drawflow_refuses_what_it_cannot_see_through(name):
    src, why = REFUSE[name]
    plan = plan_draws(textwrap.dedent(src))
    assert not plan.ok and why in plan.reason, plan.reason


def test_a_class_only_guard_is_proved_true_end_to_end(tmp_path):
    p = _prog(tmp_path, "guard.py", ACCEPT["isinstance_guard"])
    v = _prove(p, classes=DL + ("AssertionError",))
    assert v is not None and v.verdict == "true" and v.component == "proof:stationary", v


def test_withdrawing_the_isinstance_contract_withdraws_the_class_only_exemption(monkeypatch):
    from tharkun.solve.contracts.registry import REGISTRY, load_all

    load_all()
    assert plan_draws(textwrap.dedent(ACCEPT["isinstance_guard"])).ok
    monkeypatch.delitem(REGISTRY, "isinstance")
    plan = plan_draws(textwrap.dedent(ACCEPT["isinstance_guard"]))
    assert not plan.ok and "unresolved callee" in plan.reason


def test_drawflow_names_offsets_and_loops():
    plan = plan_draws(textwrap.dedent(ACCEPT["offset_and_quotient"]))
    kinds = sorted((d.kind, d.offset) for d in plan.draws)
    assert kinds == [("quotient", 0), ("trip", 1)]
    (loop,) = plan.loops
    assert loop.offset == 1 and loop.dead == ("_",)


# ---------------------------------------------------------------- proof:stationary (runs)

def test_stationary_loop_is_proved_true(tmp_path):
    p = _prog(tmp_path, "st.py", """
        import _sv_verifier
        DATA = [3, 1, 2]
        def work(loops):
            for _ in range(loops):
                s = sorted(DATA)
                total = s[0] + s[2]
        if __name__ == "__main__":
            work(abs(_sv_verifier.nondet_int()))
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true", v
    assert v.component == "proof:stationary" and "line 5" in v.reason
    assert v.detail["stationary_instances"] >= 1


def test_stationary_probe_writes_uncertified_loop_trip_count_facts(tmp_path):
    """[internal ref] step 2 / W57's [internal ref] ask ([internal ref]): "the forward runs already see
    it" -- the stationary probe already observes each trip loop's actual trip count for
    the vector it just ran; passing a KnowledgeBase records that observation as an
    uncertified hint, one Fact per loop-head instance a probe run made, mapping the
    loop back to the trip count AND the draws that produced it. Never certified: an
    observed trip count on one vector says nothing about every other vector."""
    from tharkun.facts import Fact
    from tharkun.knowledge import KnowledgeBase

    p = _prog(tmp_path, "st_kb.py", """
        import _sv_verifier
        DATA = [3, 1, 2]
        def work(loops):
            for _ in range(loops):
                s = sorted(DATA)
                total = s[0] + s[2]
        if __name__ == "__main__":
            work(abs(_sv_verifier.nondet_int()))
    """)
    kb = KnowledgeBase()
    t0 = time.monotonic()
    v = prove_dynamic(p, DL, python=sys.executable, jobs=2,
                      remaining=lambda: 120 - (time.monotonic() - t0), knowledge=kb)
    assert v is not None and v.verdict == "true"
    facts = [f for f in kb if f.kind == "loop_trip_count"]
    assert facts, "expected at least one loop_trip_count fact from the probe's own runs"
    assert all(f.certified is False and f.source == "prove:stationary" for f in facts)
    assert all(f.subject.startswith("loop@") for f in facts)
    assert all("trip=" in f.value and "draws=" in f.value for f in facts)
    # at least one instance actually observed a trip count of 0 (a real, useful signal:
    # the loop at line 5 ran zero times for SOME probed vector, not just the "stationary"
    # head instances)
    assert any("trip=0 " in f.value for f in facts)


def test_loop_trip_count_facts_do_not_change_the_verdict(tmp_path):
    """Pure bookkeeping, same discipline as step 2's other writers: a real KnowledgeBase
    and no KnowledgeBase at all (the default) decide the identical proof."""
    from tharkun.knowledge import KnowledgeBase

    p = _prog(tmp_path, "st_inert.py", """
        import _sv_verifier
        DATA = [3, 1, 2]
        def work(loops):
            for _ in range(loops):
                s = sorted(DATA)
                total = s[0] + s[2]
        if __name__ == "__main__":
            work(abs(_sv_verifier.nondet_int()))
    """)
    t0 = time.monotonic()
    v1 = prove_dynamic(p, DL, python=sys.executable, jobs=2,
                       remaining=lambda: 120 - (time.monotonic() - t0))
    t0 = time.monotonic()
    v2 = prove_dynamic(p, DL, python=sys.executable, jobs=2,
                       remaining=lambda: 120 - (time.monotonic() - t0), knowledge=KnowledgeBase())
    assert v1.verdict == v2.verdict == "true"
    assert v1.component == v2.component
    assert v1.detail["stationary_instances"] == v2.detail["stationary_instances"]


def test_quotient_draw_is_enumerated(tmp_path):
    p = _prog(tmp_path, "q.py", """
        import _sv_verifier
        LEVELS = {1: "a", 2: "b", 3: "c"}
        keys = sorted(LEVELS)
        level = abs(_sv_verifier.nondet_int()) % len(keys)
        loops = 1 + abs(_sv_verifier.nondet_int())
        def main(loops, key):
            for _ in range(loops):
                out = LEVELS[key] * 2
        main(loops, keys[level])
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true", v
    # 3 residues x trip counts {1, 2} x 2 hash seeds
    assert v.runs == 12


MUTANTS = {
    # the state grows: iteration 3 differs from iteration 2
    "growing_list": """
        import _sv_verifier
        acc = []
        def f(n):
            for _ in range(n):
                acc.append(1)
                if len(acc) == 3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "class_attr_counter": """
        import _sv_verifier
        class K:
            count = 0
        def f(n):
            for _ in range(n):
                K.count += 1
                if K.count == 3:
                    {}["x"]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "closure_counter": """
        import _sv_verifier
        def mk():
            c = 0
            def inc():
                nonlocal c
                c += 1
                return c
            return inc
        inc = mk()
        def f(n):
            for _ in range(n):
                if inc() == 3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "default_arg_counter": """
        import _sv_verifier
        def tick(acc=[]):
            acc.append(0)
            return len(acc)
        def f(n):
            for _ in range(n):
                if tick() == 3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "function_attr_counter": """
        import _sv_verifier
        def tick():
            tick.n = getattr(tick, "n", 0) + 1
            return tick.n
        def f(n):
            for _ in range(n):
                if tick() == 3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "seeded_random_in_body": """
        import _sv_verifier, random
        random.seed(0)
        def f(n):
            for _ in range(n):
                if random.random() < 0.3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "stringio_grows": """
        import _sv_verifier, io
        buf = io.StringIO()
        def f(n):
            for _ in range(n):
                buf.write("x")
                if buf.tell() == 3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "array_grows": """
        import _sv_verifier
        from array import array
        a = array("i")
        def f(n):
            for _ in range(n):
                a.append(1)
                if len(a) == 3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "toggle_then_after_loop": """
        import _sv_verifier
        s = [0]
        ran = [0]
        def f(n):
            for _ in range(n):
                s[0] = 1 - s[0]
                ran[0] = 1
            if ran[0] and s[0] == 0:
                [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "generator_in_state": """
        import _sv_verifier
        def gen():
            i = 0
            while True:
                i += 1
                yield i
        g = gen()
        def f(n):
            for _ in range(n):
                if next(g) == 3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "iterator_in_state": """
        import _sv_verifier
        it = iter([1, 2, 3, 4])
        def f(n):
            for _ in range(n):
                if next(it) == 3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "draw_in_body": """
        import _sv_verifier
        def f(n):
            c = [0]
            for _ in range(n):
                if _sv_verifier.nondet_bool():
                    c[0] += 1
            if c[0] >= 3:
                [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "identity_keyed_set": """
        import _sv_verifier
        class P:
            pass
        seen = set()
        def f(n):
            for _ in range(n):
                seen.add(P())
                if len(seen) == 3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "dict_counter": """
        import _sv_verifier
        d = {"k": 0}
        def f(n):
            for _ in range(n):
                d["k"] += 1
                if d["k"] == 3:
                    {}["missing"]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "slot_counter": """
        import _sv_verifier
        class C:
            __slots__ = ("n",)
            def __init__(self):
                self.n = 0
        c = C()
        def f(n):
            for _ in range(n):
                c.n += 1
                if c.n == 3:
                    [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    # the loop state repeats, but the body iterates a transient set of objects: its order
    # depends on addresses, so iterations are not the same function of the state
    "transient_set_order": """
        import _sv_verifier
        class P:
            def __init__(self, k):
                self.k = k
        def f(n):
            for _ in range(n):
                s = {P(1), P(2)}
                for p in s:
                    if p.k == 2:
                        [][0]           # only when the address order puts P(2) first
                    break
        f(abs(_sv_verifier.nondet_int()))
    """,
    # W41's never-TRUE probe ([internal ref]): n also feeds divmod, whose result gates an IndexError that
    # is reached only at n == 13 (divmod(13, 5) == (2, 3)), beyond every probed trip count. A
    # contract-only "pure and total call is transparent" rule proved this TRUE.
    "divmod_gate": """
        import _sv_verifier
        def f(n):
            for _ in range(n):
                pass
            y, x = divmod(n, 5)
            if y == 2 and x == 3:
                [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    # the class-only exemption must not cover a call that merely LOOKS like the guard
    "isinstance_shadow_gate": """
        import _sv_verifier
        def isinstance(x, c):
            return x == 13
        def f(n):
            for _ in range(n):
                pass
            if isinstance(n, int):
                [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "isinstance_metaclass_gate": """
        import _sv_verifier
        class M(type):
            def __instancecheck__(cls, inst):
                return inst == 13
        class K(metaclass=M):
            pass
        def f(n):
            for _ in range(n):
                pass
            if isinstance(n, K):
                [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
    "caught_then_counted": """
        import _sv_verifier
        errs = [0]
        def f(n):
            for _ in range(n):
                try:
                    [][0]
                except IndexError:
                    errs[0] += 1
            if errs[0] >= 3:
                [][0]
        f(abs(_sv_verifier.nondet_int()))
    """,
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_never_proved(tmp_path, name):
    p = _prog(tmp_path, f"{name}.py", MUTANTS[name])
    v = _prove(p)
    assert v is None or v.verdict != "true", (name, v)


def test_trip_zero_violation_is_found_not_proved(tmp_path):
    p = _prog(tmp_path, "zero.py", """
        import _sv_verifier
        flag = [0]
        def f(n):
            for _ in range(n):
                flag[0] = 1
        f(abs(_sv_verifier.nondet_int()))
        if flag[0] == 0:
            [][0]
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "false" and v.exception == "IndexError"
    assert v.witness["vector"] == [{"kind": "int", "value": 0}]


def test_first_iteration_violation_is_found(tmp_path):
    p = _prog(tmp_path, "one.py", """
        import _sv_verifier
        def f(n):
            for _ in range(n):
                {}["x"]
        f(1 + abs(_sv_verifier.nondet_int()))
    """)
    v = _prove(p)
    assert v is not None and v.line() == "FALSE(KeyError)"


def test_verify_reports_the_proof_component(tmp_path):
    p = _prog(tmp_path, "v.py", """
        import _sv_verifier
        import re
        def work(loops):
            for _ in range(loops):
                m = re.match("a(b)", "ab")
                g = m.group(1)
        work(abs(_sv_verifier.nondet_int()))
    """)
    v = verify(p, DL, budget_seconds=60, jobs=2)
    assert v.line() == "TRUE" and v.component == "proof:stationary"
    off = verify(p, DL, budget_seconds=8, prove=False, jobs=2)
    assert off.verdict == "unknown"


@pytest.mark.parametrize("name", ["growing_list", "class_attr_counter", "stringio_grows"])
def test_verify_finds_the_mutants_violation(tmp_path, name):
    p = _prog(tmp_path, f"{name}.py", MUTANTS[name])
    v = verify(p, DL, budget_seconds=60, jobs=2)
    assert v.verdict == "false", v


# ---------------------------------------------------------------- determinism census (allowlist)

from pathlib import Path  # noqa: E402

from tharkun.solve.determinism import determinism_census  # noqa: E402

DET_OK = [
    "xs = [3, 1, 2]\nxs.sort()\nd = {'a': 1}\nfor k in d:\n    pass\n",
    "s = set()\ns.add(1)\nif 1 in s and len(s) == 1:\n    pass\nt = {2}\nok = s <= t\n",
    "for x in {1, 2, 3}:\n    pass\n",
    "def f(id):\n    return id(1)\nf(abs)\n",
    "def unused():\n    for x in set('ab'):\n        pass\n",
    "with open('out.txt', 'w') as fh:\n    fh.write('x')\n",
    "import math, random\nrandom.seed(3)\nx = math.sqrt(random.random())\n",
    "import random\nr = random.Random(7)\nx = r.randint(0, 9)\n",
    "class P:\n    def __repr__(self):\n        return 'P(%s)' % (self.x,)\n    def __init__(self):\n        self.x = 1\nprint(P())\n",
    "n = 3\ns = '%d items, %.2f each' % (n, 1.5)\nt = str(len(s)) + 'x'\n",
    "def f(k):\n    msg = 'bad %s' % k\n    raise ValueError(msg if k else 'x')\n",
    "class A:\n    def __init__(self):\n        self.v = 2\nclass B(A):\n    def __init__(self):\n        A.__init__(self)\nB()\n",
    "def g(x):\n    raise ValueError('got %s' % x)\n",
]

DET_NO = [
    ("s = {'a', 'b'}\nfor x in s:\n    print(x)\n", "order can matter"),
    ("s = set('abc')\nt = s.copy()\nprint(list(t))\n", "order can matter"),
    ("def g():\n    return {1, 2} | {3}\nfor x in g():\n    pass\n", "order can matter"),
    ("s = set()\nx = s.pop()\n", "order can matter"),
    ("s = {'a'}\nt = s\nx = t.pop()\n", "order can matter"),
    ("x = sorted(set('ab'), key=len)\n", "order can matter"),
    ("x = id(object())\n", "uses id"),
    ("x = hash('a')\n", "uses hash"),
    ("x = 'abc'.__hash__()\n", "reflective"),
    ("class K:\n    id = 1\n    def m(self):\n        return id(self)\nK().m()\n", "uses id"),
    # "os" itself is no longer an unresolvable import ([internal ref]) -- "xml" still is
    ("import xml\n", "imports xml"),
    ("import time\n", "imports time"),
    ("from datetime import datetime\n", "datetime"),
    ("import threading\n", "imports threading"),
    ("import sys\nx = sys.getrefcount(1)\n", "sys.getrefcount"),
    ("import sys\nx = sys.argv\n", "sys.argv"),
    ("x = open('in.txt').read()\n", "may read"),
    ("class A:\n    def __del__(self):\n        pass\n", "__del__"),
    ("eval('1')\n", "uses eval"),
    ("x = vars()\n", "uses vars"),
    ("x = [].__class__\n", "reflective"),
    ("x = getattr([], 'po' + 'p')\n", "computed name"),
    ("import random\nr = random.SystemRandom()\n", "SystemRandom"),
    ("import random\nr = random.Random()\n", "literal seed"),
    ("from random import Random\nr = Random().randrange(5)\n", "literal seed"),
    ("import random\nrandom.seed()\n", "without a seed"),
    ("o = object()\nx = int(repr(o)[-3], 16)\n", "address"),
    ("o = object()\nx = f'{o}'\n", "address"),
    ("def f(o):\n    return str(o)\nf(object())\n", "address"),
    ("class P:\n    pass\nfirst = sorted([P(), P()], key=repr)[0]\n", "address"),
    ("xs = list(map(str, [object()]))\n", "address"),
    ("x = '{0.real}'.format(3)\n", "attributes/items"),
    ("import io\nb = io.StringIO()\nprint(object(), file=b)\nx = b.getvalue()\n", "address"),
    ("try:\n    raise ValueError('%s' % object())\nexcept ValueError as e:\n    x = e.args[0]\n", "address"),
    ("s = {'b', 'a'}\nx = sorted(s)[0]\n", "order can matter"),
    ("import random\nr = random.Random(1)\nr.seed()\n", "without a seed"),
    ("import random\nr = random.Random(1)\nr.__init__()\n", "__init__"),
    ("import io\nf = io.open('x')\n", "may read"),
    ("except_ok = 1\ntry:\n    pass\nexcept ValueError as SIZE:\n    pass\nx = '%s' % SIZE\n", "address"),
]


@pytest.mark.parametrize("src", DET_OK)
def test_determinism_census_accepts(src):
    r = determinism_census(src)
    assert r.ok, r.reason


@pytest.mark.parametrize("src,why", DET_NO)
def test_determinism_census_refuses(src, why):
    r = determinism_census(src)
    assert not r.ok and why in r.reason, r.reason


NEVER_CERTIFIED = sorted((Path(__file__).parent / "fixtures" / "determinism" /
                          "never_certified").glob("*.py"))


@pytest.mark.parametrize("path", NEVER_CERTIFIED, ids=lambda p: p.stem)
def test_lane_f_nondeterminism_is_never_certified(path):
    """Lane F's probes ([internal ref]): each program's outcome depends on the hash seed, addresses,
    the clock, the environment or an unseeded generator; none may be certified."""
    r = determinism_census(path.read_text())
    assert not r.ok, (path.name, r.reason)


# ---------------------------------------------------------------- the certificate in verify

def test_deterministic_true_cites_its_certificate(tmp_path):
    p = _prog(tmp_path, "det.py", """
        xs = [1, 2, 3]
        print(xs[2])
    """)
    v = verify(p, DL, budget_seconds=60, prove=False)
    assert v.line() == "TRUE" and v.component == "deterministic"
    assert "resolves" in v.detail["certificate"]


def test_uncertified_determinism_is_not_true(tmp_path):
    # the order of a str set decides the lookup: the hash seed picks the execution
    p = _prog(tmp_path, "hs.py", """
        s = {"a", "b", "c", "d", "e", "f", "g", "h"}
        first = next(iter(s))
        d = {"a": 1}
        d[first]
    """)
    v = verify(p, DL, budget_seconds=60)
    assert v.verdict != "true", v


def test_finite_tree_needs_the_certificate(tmp_path):
    p = _prog(tmp_path, "ft.py", """
        import _sv_verifier
        s = {"x", "y", "z", "w"}
        if _sv_verifier.nondet_bool():
            order = list(s)
            {"x": 0}[order[0]]
    """)
    v = verify(p, DL, budget_seconds=60)
    assert v.verdict != "true", v


@pytest.mark.parametrize("path", NEVER_CERTIFIED, ids=lambda p: p.stem)
def test_lane_f_probes_are_never_true_end_to_end(path):
    """The whole verifier on each of lane F's programs: never TRUE ([internal ref])."""
    v = verify(str(path), DL, budget_seconds=60, run_timeout=30, jobs=2)
    assert v.verdict != "true", (path.name, v.component, v.reason)


# ---------------------------------------------------------------- proof:no-site (census)

from tharkun.solve.census import site_census  # noqa: E402

AS = ("AssertionError",)

CENSUS_SAFE = [
    (AR, "x = [1, 2]\nx.insert(0, x.pop(0))\ny = x[0] // 2 + x[1] % 3 + 2 ** 3 + 4 / 2.0\n"),
    (AR, "raise ValueError('no')\n"),
    (AR, "s = '%d items' % 3\n"),
    (AR, "try:\n    pass\nexcept ZeroDivisionError as e:\n    raise e\n"),
    (("IndexError",), "xs = [1, 2, 3]\nys = xs[1:]\nfor a, b in zip(xs, ys):\n    pass\n"),
    (AS, "assert True\nx = 1\n"),
    (AR, "import math\ny = math.sqrt(2) + math.log(3)\n"),
]

CENSUS_SITES = [
    (AR, "def f(a, b):\n    return a / b\nf(1, 0)\n", "division"),
    (AR, "x = 0\ny = 3 / 1e-400\n", "division"),
    (AR, "y = 2 ** -1\n", "power"),
    (AR, "q, r = divmod(7, 2)\n", "divmod"),
    (AR, "f = divmod\n", "used as a value"),
    (AR, "from math import log as lg\ny = lg(8, 2)\n", "math.log"),
    (AR, "import random\nrandom.seed(1)\nrandom.expovariate(0)\n", "expovariate"),
    (AR, "e = ZeroDivisionError('x')\nraise e\n", "raise e"),
    (AR, "class E(ZeroDivisionError):\n    pass\nraise E\n", "raise E"),
    (AR, "class E(ArithmeticError):\n    pass\nclass F(E, ZeroDivisionError):\n    pass\nraise F()\n",
     "raise F"),
    (AR, "ValueError = ZeroDivisionError\nraise ValueError\n", "raise ValueError"),
    (AR, "def g():\n    yield 1\nx = g()\nnext(x)\nx.throw(ValueError)\n", "throw"),
    (AR, "x = 3\nx //= 0\n", "division"),
    (DL, "d = {}\nd['k']\n", "subscript"),
    (DL, "d = {}\nd[1:2]\n", "subscript"),               # a slice is a dict key (3.12+)
    (DL, "d = {}\nd[1:2]\n", "subscript"),               # a slice is a dict key (3.12+)
    (DL, "d = {'a': 1}\nfmt = '%(b)s'\ny = fmt % d\n", "modulo"),
    (DL, "y = '%(b)s' % {'a': 1}\n", "modulo"),
    (DL, "y = '{0}'.format()\n", "format"),
    (DL, "import re\nre.sub('a', r'\\g<x>', 'a')\n", "re.sub"),
    (DL, "s = {1}\ns.remove(2)\n", "remove"),
    (DL, "import random\nrandom.choice([])\n", "random.choice"),
    (AS, "x = 1\nassert x == 2\n", "assert"),
]

CENSUS_REFUSE = [
    # "os" itself is no longer an unresolvable import ([internal ref]) -- "xml" still is
    (AR, "import xml\n", "imports xml"),
    (AR, "eval('1/0')\n", "uses eval"),
    (AR, "x = 1\ngetattr(x, 'real' if x else '__truediv__')\n", "computed name"),
    (AR, "import math\nm = math\n", "used as a value"),
    (AR, "class V:\n    pass\nv = V()\nv.__truediv__(0)\n", "reflective"),
    (AR, "f = object.__getattribute__(7, '__floordiv__')\n", "reflective"),
    (AR, "g = int.__dict__['__floordiv__']\n", "reflective"),
    (AR, "t = (0).__class__\n", "reflective"),
    (DL, "a = [1]\nf = a.__getattribute__('pop')\n", "reflective"),
    (AR, "x = getattr(7, '__floordiv__')\n", "reflective"),
    (AR, "x = 1\ny = x.frobnicate\n", "resolves to nothing"),
    (("TypeError",), "x = 1\n", "no site table"),
]


@pytest.mark.parametrize("classes,src", CENSUS_SAFE)
def test_census_finds_no_site(classes, src):
    r = site_census(src, classes)
    assert r.no_site, r.reason


@pytest.mark.parametrize("classes,src,what", CENSUS_SITES)
def test_census_counts_every_site(classes, src, what):
    r = site_census(src, classes)
    assert not r.no_site and what in r.reason, r.reason


@pytest.mark.parametrize("classes,src,why", CENSUS_REFUSE)
def test_census_refuses(classes, src, why):
    r = site_census(src, classes)
    assert not r.no_site and why in r.reason, r.reason




def test_no_site_proof_is_consulted_first(tmp_path):
    p = _prog(tmp_path, "ns.py", """
        import _sv_verifier
        def f(n):
            count = list(range(1, n + 1))
            count.insert(0, count.pop())
            return max(count)
        f(1 + abs(_sv_verifier.nondet_int()))
    """)
    from tharkun.sv.prove import prove_static
    v = prove_static(p, AR)
    assert v is not None and v.verdict == "true" and v.component == "proof:no-site"
    assert prove_static(p, DL) is None


def test_uncertified_determinism_falls_back_to_no_site(tmp_path):
    p = _prog(tmp_path, "hs2.py", """
        s = {"a", "b"}
        for x in s:
            print(x)
    """)
    v = verify(p, DL, budget_seconds=60)
    assert v.line() == "TRUE" and v.component == "proof:no-site"


NEVER_NO_SITE = sorted((Path(__file__).parent / "fixtures" / "census" / "never_no_site").glob("*.py"))


@pytest.mark.parametrize("path", NEVER_NO_SITE, ids=lambda p: p.stem)
def test_lane_f_reflection_is_never_no_site(path):
    """Lane F's probes ([internal ref]): a property exception reached through reflection
    (``__getattribute__``, ``__dict__``, ``__mro__``); the census must not claim no site."""
    src = path.read_text()
    for classes in (AR, DL):
        r = site_census(src, classes)
        assert not r.no_site, (path.name, classes, r.reason)
