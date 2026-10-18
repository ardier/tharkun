#!/usr/bin/env python3
"""A differential for user-class support in the symbolic executor ([internal ref]; ``sv/_symexec_objects.py``).

It generates small class-heavy programs -- classes with constructors, instance and class attributes, methods,
static and class methods, single inheritance with ``super()``, ``isinstance``; a main block that draws integers,
builds objects, aliases them, calls methods, stores through any reference, branches on the draws (the effects of
an ``if`` arm, a conditional expression or a short circuit are the point), loops -- and checks the symbolic
executor against NATIVE execution, one program at a time:

* every program ends in ``assert`` sites: some hold at a randomly drawn vector ``d0`` and may fail elsewhere,
  and the last states ``obs != <the value obs has natively at d0>``, which fails at ``d0`` at least;
* ``solve_targets`` (with the program's own ``call_target`` records, exactly as the scheduler hands them over)
  answers per site ``sat`` / ``unsat`` / ``unreached`` / ``unsupported``;
* a ``sat`` model MUST make exactly that assert the first one to fail when the program is run natively with the
  model's draws; an ``unsat`` / ``unreached`` site MUST never be the first failing assert natively on a box of
  draws (``d0`` included); ``unsupported`` is a refusal and is only counted;
* a program in which the executor SKIPPED a statement (a named refusal, ``_Runner.skipped``) is not compared at
  all: ``_Runner.run`` assumes a skipped statement does not raise, so what it says afterwards is a hint that the
  replay on the real program judges (the pre-existing design of ``sv/_symexec.py``); such programs are counted
  as ``refused_programs`` and the bar is about the programs the executor modelled completely.

A mismatch is a bug in the model (or in the generator); ``0`` over >= 2000 programs is the bar.  The second half
is the NEGATIVE CONTROL: the same check against deliberately broken copies of the tree (attribute stores
dropped, the heap of an ``if`` not merged, arms sharing one heap, class attributes ignoring stores, ``super()``
resolving to the wrong class) must catch every one of them, or the differential proves nothing.

    python scripts/class_differential.py --programs 2000 --jobs 16
    python scripts/class_differential.py --negative-control --programs 600 --jobs 16

DISABLED (owner, 2026-10-06: "we should disable differential fuzzing for now"): nothing in the suite or in a gate runs
this script; the numbers it produced when it last ran (3000 programs -> 2961 compared, 0 mismatches; 7 of 7 broken
copies caught) are in PR #149.  The unit tests ``tests/test_symexec_classes.py`` (each program run symbolically and
natively, and the same tests failing when a fix is undone: ``test_the_heap_tests_fail_when_the_fix_is_undone``) are the
bar for now.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import multiprocessing
import os
import random
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: a generated program runs under these caps (the campaign rule: no process above 15 GB; a generated run is
#: bounded far below it): address space of the process, and wall seconds per native run
MEMORY_BYTES = 2 << 30
NATIVE_SECONDS = 2


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()


def cap_process() -> None:
    """Cap the address space of this process (a pool worker, or the script's own process): never called from
    inside a pytest worker, which other tests share."""
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))


class _bounded:
    """``with _bounded():`` -- the native run raises :class:`_Timeout` after ``NATIVE_SECONDS``."""

    def __enter__(self):
        self.old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(NATIVE_SECONDS)

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old)

# ---------------------------------------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------------------------------------

OPS = ("+", "-", "*")


class Gen:
    def __init__(self, rng: random.Random):
        self.r = rng
        self.classes: list = []          # [(name, base or None)]
        self.src_classes = ""
        self.draws: list = []
        self.objs: dict = {}             # variable -> class name
        self.ints: list = []
        self.lines: list = []
        self.n = 0

    # -- classes -------------------------------------------------------------------------------------------

    def const(self, lo=-3, hi=5) -> int:
        return self.r.randint(lo, hi)

    def build_classes(self) -> None:
        r = self.r
        names = ["A", "B", "C"][: r.choice([1, 2, 3, 3])]
        out = []
        for i, name in enumerate(names):
            base = None if i == 0 else (names[i - 1] if r.random() < 0.8 else names[0])
            out.append((name, base))
        self.classes = out
        m, c1, c2 = r.randint(1, 3), self.const(), self.const()
        counts = r.random() < 0.5
        lines = [
            "class A:",
            f"    k = {self.const()}",
            "    count = 0",
            "    def __init__(self, a, b):",
            "        self.a = a",
            f"        self.b = b + {c1}",
            *(["        A.count = A.count + 1"] if counts else []),
            "    def get(self):",
            f"        return self.a + self.b * {m}",
            "    def bump(self, d):",
            "        self.a = self.a + d",
            "        return self.a",
            "    def cond(self, t):",
            "        if t > self.b:",
            "            self.a = self.a + 1",
            "            return 1",
            "        return 0",
            "    def twice(self):",
            "        return self.get() * 2",
            "    @staticmethod",
            "    def sm(v):",
            f"        return v * {r.randint(1, 3)}",
            "    @classmethod",
            "    def cm(cls):",
            f"        return cls.k + {c2}",
            "    def link(self, other):",
            "        other.a = other.a + self.b",
            "        return other.a",
            "    def note(self):",
            "        A.count = A.count + 1",
            "        return A.count",
            "",
        ]
        for name, base in out[1:]:
            lines += [f"class {name}({base}):"]
            body = []
            if r.random() < 0.5:
                body += ["    k = %d" % self.const()]
            has_init = r.random() < 0.6
            if has_init:
                body += ["    def __init__(self, a, b):", "        super().__init__(a, b)",
                         f"        self.c = a - b + {self.const()}"]
            if r.random() < 0.6:
                style = r.choice(["super", "super", "own"] + (["super_c"] if has_init else []))
                body += ["    def get(self):",
                         {"super": f"        return super().get() + {self.const()}",
                          "super_c": "        return super().get() + self.c",
                          "own": f"        return self.a - {self.const()}"}[style]]
            if r.random() < 0.5:
                body += ["    def bump(self, d):", "        super().bump(d)", f"        self.b = self.b + {r.randint(0, 2)}",
                         "        return self.b"]
            if r.random() < 0.3:
                body += ["    def cond(self, t):",
                         r.choice(["        return 1 if t < self.a else 0", "        return super().cond(t) + 1"])]
            if r.random() < 0.4:
                body += ["    def twice(self):",
                         r.choice([f"        return self.get() + {self.const()}",
                                   f"        return super().twice() + {self.const()}"])]
            if r.random() < 0.2:
                body += ["    @classmethod", "    def cm(cls):",
                         r.choice([f"        return cls.k * {r.randint(1, 3)}", f"        return super().cm() + cls.k"])]
            lines += body or ["    pass"]
            lines += [""]
        self.src_classes = "\n".join(lines)

    # -- expressions -----------------------------------------------------------------------------------------

    def iexpr(self, depth=0) -> str:
        r = self.r
        pool = []
        pool += [str(self.const())]
        pool += self.draws * 2
        pool += self.ints
        for o in self.objs:
            pool += [f"{o}.a", f"{o}.b"]
        e = r.choice(pool)
        if depth < 2 and r.random() < 0.45:
            return f"({e} {r.choice(OPS)} {self.iexpr(depth + 1)})"
        return e

    def cond(self) -> str:
        r = self.r
        k = r.random()
        if k < 0.4:
            return f"{r.choice(self.draws)} {r.choice(('>', '<', '=='))} {self.const()}"
        if k < 0.7:
            return f"{self.iexpr(1)} {r.choice(('>', '<', '>=', '=='))} {self.iexpr(1)}"
        o = r.choice(list(self.objs))
        return f"{o}.cond({self.iexpr(1)}) == 1"

    def call(self) -> str:
        r = self.r
        o = r.choice(list(self.objs))
        k = r.random()
        if k < 0.2:
            return f"{o}.get()"
        if k < 0.4:
            return f"{o}.bump({self.iexpr(1)})"
        if k < 0.5:
            return f"{o}.twice()"
        if k < 0.6:
            return f"{o}.cond({self.iexpr(1)})"
        if k < 0.7:
            return f"{o}.link({r.choice(list(self.objs))})"
        if k < 0.78:
            return f"{o}.sm({self.iexpr(1)})"
        if k < 0.86:
            return f"{r.choice([c for c, _b in self.classes])}.sm({self.iexpr(1)})"
        if k < 0.92:
            return f"{o}.cm()"
        if k < 0.96:
            return f"{r.choice([c for c, _b in self.classes])}.cm()"
        return f"{o}.note()"

    # -- statements ---------------------------------------------------------------------------------------------

    def new_int(self) -> str:
        self.n += 1
        return f"r{self.n}"

    def simple(self, indent: str) -> list:
        r = self.r
        k = r.random()
        o = r.choice(list(self.objs))
        top = indent == "    "                 # a name an arm defines would not exist on the other path
        if k < 0.30:
            v = self.new_int()
            line = f"{indent}{v} = {self.call()}"
            if top:
                self.ints.append(v)
            return [line]
        if k < 0.50:
            return [f"{indent}{o}.{r.choice('ab')} = {self.iexpr()}"]
        if k < 0.62:
            return [f"{indent}{o}.{r.choice('ab')} {r.choice(('+=', '-='))} {self.iexpr(1)}"]
        if k < 0.72:
            o2 = r.choice(list(self.objs))
            return [f"{indent}{o}.a, {o2}.b = {self.iexpr(1)}, {self.iexpr(1)}"]
        if k < 0.80:
            c = r.choice([c for c, _b in self.classes])
            return [f"{indent}{c}.k = {c}.k + {self.const()}"]
        if k < 0.88:
            return [f"{indent}{self.call()}"]
        v = self.new_int()
        if top:
            self.ints.append(v)
        cls = r.choice([c for c, _b in self.classes])
        return [f"{indent}{v} = 1 if isinstance({o}, {cls}) else 0"]

    def statement(self) -> list:
        r = self.r
        k = r.random()
        if k < 0.40:
            return self.simple("    ")
        if k < 0.50:
            name = f"o{len(self.objs) + 1}"
            cls = r.choice([c for c, _b in self.classes])
            line = f"    {name} = {cls}({self.iexpr(1)}, {self.iexpr(1)})"
            self.objs[name] = cls
            return [line]
        if k < 0.58:
            src = r.choice(list(self.objs))
            name = f"p{len(self.objs) + 1}"
            self.objs[name] = self.objs[src]
            return [f"    {name} = {src}"]
        if k < 0.66:
            o = r.choice(list(self.objs))
            cls = r.choice([c for c, _b in self.classes])
            self.objs[o] = cls
            return [f"    {o} = {cls}({self.iexpr(1)}, {self.iexpr(1)})"]
        if k < 0.88:
            out = [f"    if {self.cond()}:"]
            out += self.simple("        ")
            if r.random() < 0.5:
                out += self.simple("        ")
            if r.random() < 0.6:
                out += ["    else:"] + self.simple("        ")
            return out
        if k < 0.94:
            v = self.new_int()
            line = f"    {v} = {self.call()} if {self.cond()} else {self.call()}"
            self.ints.append(v)                  # defined after its own expression: no self reference
            return [line]
        if k < 0.97:
            v = self.new_int()
            self.ints.append(v)
            return [f"    {v} = 0", f"    {v} = {self.cond()} and {self.call()}"]
        o = r.choice(list(self.objs))
        i = f"i{self.n}"
        return [f"    for {i} in range({r.randint(1, 3)}):", f"        {o}.bump({i} + {r.choice(self.draws)})"]

    def build(self):
        r = self.r
        self.build_classes()
        ndraw = r.choice([1, 1, 2, 2, 3])
        self.draws = ["x", "y", "z"][:ndraw]
        main = [f"    {d} = _sv_verifier.nondet_int()" for d in self.draws]
        first = r.choice([c for c, _b in self.classes])
        main.append(f"    o1 = {first}({self.iexpr(1)}, {self.iexpr(1)})")
        self.objs["o1"] = first
        if r.random() < 0.7:
            second = r.choice([c for c, _b in self.classes])
            main.append(f"    o2 = {second}({self.iexpr(1)}, {self.iexpr(1)})")
            self.objs["o2"] = second
        for _ in range(r.randint(2, 9)):
            main += self.statement()
        observables = []
        for o in self.objs:
            observables += [f"{o}.a", f"{o}.b"]
        observables += self.ints
        observables += [f"{c}.k" for c, _b in self.classes]
        observables += ["A.count"]
        self.observables = observables
        return main


def make_program(seed: int):
    """``(source, d0 draws, draws names)`` for ``seed``: the asserts are appended by :func:`finish`."""
    rng = random.Random(seed)
    g = Gen(rng)
    main = g.build()
    d0 = [rng.randint(-3, 3) for _ in g.draws]
    head = "import _sv_verifier\n\n" + g.src_classes + '\nif __name__ == "__main__":\n'
    # evaluate observables at d0 natively
    base = head + "\n".join(main) + "\n"
    probe = base + "    __obs = [" + ", ".join(g.observables) + "]\n"
    values = run_native_values(probe, d0)
    if values is None:
        return None
    asserts = []
    for _ in range(rng.randint(0, 2)):
        a, b = rng.choice(g.observables), rng.choice(g.observables)
        pred = rng.choice([f"{a} + {b} != {rng.randint(-4, 8)}", f"{a} <= {b} + {rng.randint(0, 3)}",
                           f"not ({a} == {b} and {a} > {rng.randint(0, 3)})", f"{a} - {b} != {rng.randint(-3, 3)}"])
        got = run_native_values(base + f"    __obs = [bool({pred})]\n", d0)
        if got is None:
            continue
        asserts.append(f"    assert {pred}" if got[0] else f"    assert not ({pred})")
    k = rng.randrange(len(g.observables))
    asserts.append(f"    assert {g.observables[k]} != {values[k]}")
    return base + "\n".join(asserts) + "\n", d0, g.draws


# ---------------------------------------------------------------------------------------------------------
# native and symbolic runs
# ---------------------------------------------------------------------------------------------------------

def _stub(draws):
    queue = list(draws)
    stub = type(sys)("_sv_verifier")
    stub.nondet_int = lambda: queue.pop(0)
    return stub


def run_native_values(src: str, draws):
    """The value of ``__obs`` after the program ran with ``draws``, or ``None`` when it died."""
    old = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = _stub(draws)
    ns = {"__name__": "__main__"}
    try:
        with _bounded():
            exec(compile(src, "<native>", "exec"), ns)
    except (Exception, _Timeout):        # noqa: BLE001 -- a run that dies, runs out of memory or time is no value
        return None
    finally:
        if old is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old
    return ns.get("__obs")


def native_first_fail(code, draws):
    """The line of the first ``assert`` that fails natively, ``None`` when none does (or the run died)."""
    old = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = _stub(draws)
    try:
        with _bounded():
            exec(code, {"__name__": "__main__"})
    except AssertionError as e:
        return traceback.extract_tb(e.__traceback__)[-1].lineno
    except (Exception, _Timeout):        # noqa: BLE001
        return None
    finally:
        if old is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old
    return None


def symbolic(src: str, tmp: str):
    """``(outcomes in assert order, harness)`` from ``solve_targets`` with the program's call_target records."""
    from tharkun.solve.call_targets import call_targets
    from tharkun.solve.module_namespace import module_constants
    from tharkun.sv import _symexec
    from tharkun.sv.sites import build_harness

    class Spy(_symexec._Runner):                 # keeps the runner: what it skipped says which claims are hints
        last = None

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            Spy.last = self

    path = os.path.join(tmp, "task.py")
    with open(path, "w") as fh:
        fh.write(src)
    constants = module_constants(path)
    records = call_targets(path, constants)
    proven = [c["name"] for c in constants if c["proven"] and c["origin"] == "class"]
    h = build_harness(src, ("AssertionError",), None)
    mod = {"__name__": "sv_harness_diff"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    real = _symexec._Runner
    _symexec._Runner = Spy
    try:
        got = _symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites], h.params, h.kinds,
                                     {}, call_targets=records, proven_classes=proven)
    finally:
        _symexec._Runner = real
    sites = sorted(h.sites, key=lambda s: s["line"])
    return [got[s["line"]] for s in sites], h, list(Spy.last.skipped)


def box_for(n: int):
    return range(-12, 13) if n == 1 else range(-5, 6) if n == 2 else range(-3, 4)


def check_seed(seed: int) -> dict:
    made = make_program(seed)
    if made is None:
        return {"seed": seed, "skipped": "the program dies natively at d0"}
    src, d0, draws = made
    res = {"seed": seed, "mismatches": [], "sat": 0, "unsat": 0, "unsupported": 0, "sites": 0}
    with tempfile.TemporaryDirectory(prefix="class-diff-") as tmp:
        try:
            outcomes, h, skipped = symbolic(src, tmp)
        except Exception as e:           # noqa: BLE001 -- a crash of the executor is a bug, not a refusal
            res["mismatches"].append({"kind": "crash", "error": f"{type(e).__name__}: {e}"[:300]})
            res["source"] = src
            return res
        if skipped:
            # the executor skipped a statement (a refusal, named in ``skipped``): a skipped statement is assumed
            # not to raise, so what the run says afterwards is a HINT the replay judges -- not a claim to compare
            res["refused"] = True
            res["reasons"] = [s[:120] for s in skipped[:2]]
            return res
        asserts = sorted(n.lineno for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assert))
        if len(asserts) != len(outcomes):
            res["mismatches"].append({"kind": "sites", "asserts": len(asserts), "outcomes": len(outcomes)})
            res["source"] = src
            return res
        code = compile(src, "<native>", "exec")
        fired = {}
        for combo in itertools.chain([tuple(d0)], itertools.product(box_for(len(draws)), repeat=len(draws))):
            line = native_first_fail(code, list(combo))
            if line is not None:
                fired.setdefault(line, combo)
        for (outcome, payload), line in zip(outcomes, asserts):
            res["sites"] += 1
            if outcome == "sat":
                res["sat"] += 1
                got = native_first_fail(code, [payload[p] for p in h.params])
                if got != line:
                    res["mismatches"].append({"kind": "sat-not-replayed", "line": line, "model": payload,
                                              "native_first_fail": got})
            elif outcome in ("unsat", "unreached"):
                res["unsat"] += 1
                if line in fired:
                    res["mismatches"].append({"kind": "unsat-but-fires", "line": line, "outcome": outcome,
                                              "draws": list(fired[line])})
            else:
                res["unsupported"] += 1
                res.setdefault("reasons", []).append(str(payload)[:120])
    if res["mismatches"]:
        res["source"] = src
    return res


def _work(seed):
    try:
        return check_seed(seed)
    except Exception as e:               # noqa: BLE001
        return {"seed": seed, "mismatches": [{"kind": "harness-error", "error": f"{type(e).__name__}: {e}"[:300]}]}


def run(programs: int, start: int, jobs: int) -> dict:
    seeds = range(start, start + programs)
    if jobs > 1:
        with multiprocessing.Pool(jobs, initializer=cap_process) as pool:
            results = pool.map(_work, seeds, chunksize=8)
    else:
        results = [_work(s) for s in seeds]
    done = [r for r in results if "skipped" not in r and not r.get("refused")]
    refused = [r for r in results if r.get("refused")]
    bad = [r for r in done if r["mismatches"]]
    reasons: dict = {}
    for r in refused:
        for why in r.get("reasons", [])[:1]:
            reasons[why] = reasons.get(why, 0) + 1
    return {
        "programs": len(done), "refused_programs": len(refused),
        "dead_at_d0": len(results) - len(done) - len(refused), "mismatched_programs": len(bad),
        "mismatches": sum(len(r["mismatches"]) for r in bad),
        "sites": sum(r["sites"] for r in done), "sat": sum(r["sat"] for r in done),
        "unsat": sum(r["unsat"] for r in done), "unsupported": sum(r["unsupported"] for r in done),
        "first": [{"seed": r["seed"], "mismatches": r["mismatches"], "source": r.get("source")} for r in bad[:3]],
        "top_refusals": sorted(reasons.items(), key=lambda kv: -kv[1])[:5],
    }


# ---------------------------------------------------------------------------------------------------------
# the negative control
# ---------------------------------------------------------------------------------------------------------

#: (name, file under src/tharkun, text to find, replacement[, how many times it occurs]): each breaks one thing the
#: heap must get right; a replacement list breaks several places at once (the fact cross-check of ``super()`` would
#: turn the one-line mutant into a refusal, which is the defence working, not the differential)
MUTANTS = [
    ("attribute stores are dropped", "sv/_symexec_objects.py",
     "            self.heap.cells[base.oid] = {**self.heap.cells[base.oid], name: value}",
     "            pass"),
    ("the heap of a symbolic if is not merged (the then arm's wins)", "sv/_symexec.py",
     "            self.objects.heap = self.objects.merge(_bool(ga), ha, hb)",
     "            self.objects.heap = ha"),
    ("the arms of a fork share one heap (a store leaks into the other arm)", "sv/_symexec_objects.py",
     "        return Heap(dict(self.cells), self.stale)",
     "        return self"),
    ("class attribute stores are ignored", "sv/_symexec_objects.py",
     "            self.heap.cells[base] = {**self.heap.cells.get(base, {}), name: value}",
     "            pass"),
    ("super() skips a class of the chain (and the fact cross-check is off)", "sv/_symexec_objects.py",
     [("        hit = self._chain(owner.base, f.attr, self.heap)",
       "        hit = self._chain(owner.base.base or owner.base, f.attr, self.heap)", 1),
      ('        if ct["name"] != f"{m.owner.name}.{f.attr}":', "        if False:", 2)],
     None),
    ("an instance attribute does not shadow the class attribute", "sv/_symexec_objects.py",
     "            if name in cell:\n                return self._held((\"value\", cell[name]), what)",
     "            if False:\n                return self._held((\"value\", cell[name]), what)"),
    ("a short circuit applies its effects on every path", "sv/_symexec.py",
     "            self.objects.heap = self.objects.merge(_bool(guard), self.objects.heap, base_heap)",
     "            pass"),
]


def negative_control(programs: int, start: int, jobs: int) -> int:
    failures = 0
    for mutant in MUTANTS:
        name, rel = mutant[0], mutant[1]
        edits = mutant[2] if isinstance(mutant[2], list) else [(mutant[2], mutant[3], 1)]
        with tempfile.TemporaryDirectory(prefix="class-diff-broken-") as tmp:
            shutil.copytree(os.path.join(ROOT, "src", "tharkun"), os.path.join(tmp, "tharkun"),
                            ignore=shutil.ignore_patterns("__pycache__"))
            path = os.path.join(tmp, "tharkun", rel)
            text = open(path).read()
            broken = False
            for old, new, times in edits:
                if text.count(old) != times:
                    print(f"NEGATIVE CONTROL BROKEN: {name}: the text to replace occurs {text.count(old)} times "
                          f"in {rel}, expected {times}")
                    broken = True
                    break
                text = text.replace(old, new)
            if broken:
                failures += 1
                continue
            open(path, "w").write(text)
            env = dict(os.environ, PYTHONPATH=tmp, PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--programs", str(programs), "--start",
                                   str(start), "--jobs", str(jobs), "--json"], env=env, capture_output=True, text=True)
            try:
                summary = json.loads(proc.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                print(f"NEGATIVE CONTROL BROKEN: {name}: no summary ({proc.stderr[-300:]})")
                failures += 1
                continue
            caught = summary["mismatches"] > 0
            print(f"{'caught  ' if caught else 'MISSED  '} {name}: {summary['mismatched_programs']} of "
                  f"{summary['programs']} programs mismatch")
            failures += 0 if caught else 1
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="print the summary as one JSON line")
    ap.add_argument("--negative-control", action="store_true")
    args = ap.parse_args()
    try:
        import tharkun  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.join(ROOT, "src"))
    if args.negative_control:
        return 1 if negative_control(args.programs, args.start, args.jobs) else 0
    if args.jobs == 1:
        cap_process()
    summary = run(args.programs, args.start, args.jobs)
    if args.json:
        print(json.dumps(summary))
    else:
        print(json.dumps({k: v for k, v in summary.items() if k != "first"}, indent=1))
        for f in summary["first"]:
            print(f"--- seed {f['seed']}: {f['mismatches']}\n{f['source']}")
    return 1 if summary["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
