"""Draw domains: Tharkun's child model REFUSES a draw the grading harness refuses ([internal ref]).

The grading harness (``eregion-sv/src/eregion_sv/shim/_sv_verifier.py``, replayed by
``eregion_sv.replay``) accepts a primitive draw only inside the domain the helper's CODE produces ([internal ref]: the documented
domain is the larger one, a TRUE is proven over it, a FALSE is confirmed only inside the code's):

* ``nondet_float_range(a, b)``: a value ``random.uniform(a, b)`` can return, ``a + (b - a) * (k * 2**-53)`` for an integer
  ``0 <= k < 2**53`` in IEEE doubles (so ``1.0`` is out of the domain of ``(0.0, 1.0)``, ``1e-300`` too, and ``inf`` or
  ``nan`` is IN the domain of ``(-1e308, 1e308)``, where ``b - a`` overflows);
* ``nondet_float()``: ``inf``, ``-inf``, ``nan`` or ``-0.0`` (CPython 3.14: its second branch overflows to ``inf`` or ``nan``);
* ``nondet_str()``: every character in ``string.printable`` (so no non-ASCII, no DEL, no NUL);
* ``rindex`` / ``rfloat`` (unseeded ``random``): CLAMPED into range, and the clamped value is what is
  logged -- so a vector holding the raw value no longer equals the log (``replayed`` fails);
* a vector entry of the wrong kind falls back to the kind's default (same logging argument).

A run that makes an out-of-domain ``float_range`` / ``str`` draw is ``invalid`` there and is never a
confirmed violation.  If Tharkun's own child accepted such a draw it could confirm a FALSE the harness
refuses (a wrong FALSE).  These tests plant out-of-domain draws and require that Tharkun refuses them
exactly where the harness does: ``sv/_child.py::_draw`` / ``nondet_float_range`` and
``sv/engine.py::Outcome.violated``, every FALSE path (scheduler ``found``, the fork server, the
stationary probes, the product proof, ``engine.replay``) ending in one of those.

The last block is a DIFFERENTIAL against the real harness, skipped when ``eregion_sv`` is not around
(``EREGION_SV_SRC``, or ``repos/eregion-sv/src`` next to this checkout in the umbrella).
"""
from __future__ import annotations

import json
import os
import string
import sys
import textwrap
from pathlib import Path

import pytest

from tharkun.sv import engine
from tharkun.sv.engine import Outcome, replay, run_once, verify

IE = ("IndexError",)

#: strings string.printable rejects, each a way to plant an out-of-domain draw
NON_PRINTABLE = {
    "latin1-letter": "é",
    "DEL": "\x7f",
    "NUL": "\x00",
    "ESC": "\x1b[0m",
    "NBSP": " ",
    "emoji": "\U0001F600",
    "lone-surrogate": "\ud800",
    "ascii-then-nul": "a\x00b",
}
#: strings string.printable accepts, including every whitespace character it lists
PRINTABLE = {
    "empty": "",
    "letters": "abc",
    "space": " ",
    "all-printable": string.printable,
    "whitespace": "\t\n\r\x0b\x0c ",
    "punctuation": "~!@#$%^&*()",
}


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def _run(prog, vector):
    return run_once(sys.executable, prog, vector, timeout=60)


STR_TASK = """
    import _sv_verifier
    x = _sv_verifier.nondet_str()
    raise IndexError(len(x))
"""
RANGE_TASK = """
    import _sv_verifier
    x = _sv_verifier.nondet_float_range({a}, {b})
    raise IndexError(x)
"""


# ------------------------------------------------------------------ nondet_str

@pytest.mark.parametrize("name", sorted(NON_PRINTABLE))
def test_a_str_draw_outside_string_printable_is_refused_never_a_confirmed_violation(tmp_path, name):
    prog = _prog(tmp_path, "s.py", STR_TASK)
    vector = [{"kind": "str", "value": NON_PRINTABLE[name]}]

    out = _run(prog, vector)

    assert out.invalid_draw is True, out
    assert out.status == "raised" and out.exc == "ValueError"      # the call refuses, like the harness's shim
    assert out.violated(IE) is None
    assert out.draws == []                                         # refused BEFORE it is logged (harness: same)
    confirmed, _ = replay(prog, {"vector": vector}, IE)
    assert confirmed is False


@pytest.mark.parametrize("name", sorted(PRINTABLE))
def test_a_str_draw_inside_string_printable_is_a_confirmed_violation(tmp_path, name):
    prog = _prog(tmp_path, "s.py", STR_TASK)
    vector = [{"kind": "str", "value": PRINTABLE[name]}]

    out = _run(prog, vector)

    assert out.invalid_draw is False and out.violated(IE) == "IndexError", out
    assert engine._replayed(vector, out)
    assert replay(prog, {"vector": vector}, IE)[0] is True


def test_an_out_of_domain_str_draw_hides_nothing_behind_a_try_the_program_cannot_reach(tmp_path):
    """The refusal is out of band: a program that catches ValueError around the draw cannot turn an
    invalid run into a clean one or into a violation (``invalid_draw`` is set before the raise)."""
    prog = _prog(tmp_path, "catch.py", """
        import _sv_verifier
        try:
            x = _sv_verifier.nondet_str()
        except ValueError:
            x = "caught"
        raise IndexError(x)
    """)
    out = _run(prog, [{"kind": "str", "value": "é"}])
    assert out.invalid_draw is True and out.violated(IE) is None


def test_the_default_draws_of_an_exhausted_vector_are_in_domain(tmp_path):
    prog = _prog(tmp_path, "s.py", STR_TASK)
    out = _run(prog, [])
    assert out.invalid_draw is False and out.violated(IE) == "IndexError"


def test_verify_never_answers_false_for_a_violation_only_a_non_printable_string_reaches(tmp_path):
    """The end-to-end repro: the enumeration's own candidate list used to hold ``"\\u00e9"`` and the
    child accepted it, so this program was FALSE here and NOT CONFIRMED by the grading harness."""
    prog = _prog(tmp_path, "only_non_printable.py", """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        if s == "\\u00e9" or s == "\\x00" or len(s) == 1 and ord(s) > 127:
            raise IndexError(s)
    """)
    # the candidate that used to be accepted ("\u00e9") is among the first the enumeration tries, so a 12 s budget decides the
    # same question the 30 s one did (the verdict is "never FALSE", not "found it by the end of the budget")
    v = verify(prog, IE, budget_seconds=12, jobs=1, run_timeout=30)
    assert v.verdict != "false", v.line()


def test_verify_still_finds_a_violation_a_printable_string_reaches(tmp_path):
    prog = _prog(tmp_path, "printable.py", """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        if s == "a":
            raise IndexError(s)
    """)
    v = verify(prog, IE, budget_seconds=60, jobs=1, run_timeout=30)
    assert v.verdict == "false"
    assert replay(prog, v.witness, IE)[0] is True


def test_the_enumerations_own_candidates_are_all_admissible():
    """A candidate list the child would refuse is wasted work and, before this fix, wrong FALSEs."""
    c = engine.Candidates(engine.harvest_constants(__file__), deepen=False)
    for r in range(64):
        v = c.value("str", r)
        if v is None:
            break
        assert engine.admissible("str", v), repr(v)
    for v in engine.BODY["str"] + engine.HEAD["str"]:
        assert engine.admissible("str", v), repr(v)


def test_harvested_string_literals_that_the_harness_refuses_are_not_candidates(tmp_path):
    prog = _prog(tmp_path, "lits.py", """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        if s == "caf\\u00e9" or s == "ok" or s == "nul\\x00":
            raise IndexError(s)
    """)
    harvested = engine.harvest_constants(prog)["str"]
    assert "ok" in harvested
    assert all(engine.admissible("str", s) for s in harvested), harvested


def test_admissible_only_restricts_str():
    assert engine.admissible("int", 2 ** 80) and engine.admissible("float", float("nan")) and engine.admissible("float", 1.5)
    assert engine.admissible("bool", True) and engine.admissible("str", "") and engine.admissible("str", "a\tb")
    assert not engine.admissible("str", "é") and not engine.admissible("str", 5)


# ------------------------------------------------------------------ nondet_float_range

@pytest.mark.parametrize("raw", [
    float("nan"), float("inf"), float("-inf"), 2.0, -1e-300, 1.0000000000000002, -5e-324, 4.15e300,
    1.0,            # the bound itself: uniform(0.0, 1.0) never returns it ([internal ref], MIN's cmp_g07_float_range_neg)
    -0.0,           # 0.0 + 1.0 * 0.0 is +0.0
    1e-300, 5e-324, 0.1, 0.3,   # inside [0, 1] but not a multiple of 2**-53
])
def test_a_float_range_draw_outside_the_bounds_is_refused_never_a_confirmed_violation(tmp_path, raw):
    prog = _prog(tmp_path, "r.py", RANGE_TASK.format(a="0.0", b="1.0"))
    vector = [{"kind": "float", "value": raw.hex()}]

    out = _run(prog, vector)

    assert out.invalid_draw is True and out.violated(IE) is None, out
    assert replay(prog, {"vector": vector}, IE)[0] is False


@pytest.mark.parametrize("a, b, raw", [
    ("0.0", "1.0", 0.0), ("0.0", "1.0", 0.5), ("0.0", "1.0", 1 - 2.0 ** -53), ("0.0", "1.0", 2.0 ** -53),
    ("5", "10", 7.0), ("5", "10", 5.0),
    ("5", "10", 10.0),                                       # 5 + 5 * (1 - 2**-53) ROUNDS to 10.0: the bound is producible here
    ("10", "5", 7.0),                                        # swapped bounds are a range too
])
def test_a_float_range_draw_inside_the_bounds_is_a_confirmed_violation(tmp_path, a, b, raw):
    prog = _prog(tmp_path, "r.py", RANGE_TASK.format(a=a, b=b))
    vector = [{"kind": "float", "value": raw.hex()}]
    out = _run(prog, vector)
    assert out.invalid_draw is False and out.violated(IE) == "IndexError", out
    assert replay(prog, {"vector": vector}, IE)[0] is True


@pytest.mark.parametrize("a, b, k", [
    ("0.0", "1.0", 12345), ("-3.7", "123456.789", 2 ** 52), ("1e10", "1e10 + 1.0", 7), ("0.1", "0.7", 2 ** 53 - 1),
    ("5", "3", 2 ** 40), ("-1.5", "2.5", 0),
])
def test_a_value_the_helpers_code_returns_is_a_confirmed_violation(tmp_path, a, b, k):
    """The native control: ``random.uniform`` itself, with ``random()`` fixed to ``k * 2**-53``, returns a value the child and the
    harness both accept."""
    import random

    class Draw(random.Random):
        def random(self):
            return k * 2.0 ** -53

    v = Draw().uniform(eval(a), eval(b))
    prog = _prog(tmp_path, "r.py", RANGE_TASK.format(a=a, b=b))
    vector = [{"kind": "float", "value": v.hex()}]
    out = _run(prog, vector)
    assert out.invalid_draw is False and out.violated(IE) == "IndexError", (v, out)
    assert replay(prog, {"vector": vector}, IE)[0] is True


@pytest.mark.parametrize("raw, valid", [
    (float("inf"), True), (float("nan"), True),                           # the overflow of b - a: inf (k > 0) and nan (k == 0)
    (-float("inf"), False), (0.0, False), (1e308, False), (-1e308, False), (1.0, False),
])
def test_an_overflowing_span_produces_only_inf_and_nan(tmp_path, raw, valid):
    """``uniform(-1e308, 1e308)``: ``b - a`` is ``inf``; the old closed-interval rule accepted ``1e308`` and refused ``inf``."""
    import random

    class Draw(random.Random):
        def __init__(self, k):
            super().__init__(0)
            self.k = k

        def random(self):
            return self.k * 2.0 ** -53

    produced = {repr(Draw(k).uniform(-1e308, 1e308)) for k in (0, 1, 2 ** 52, 2 ** 53 - 1)}
    assert produced == {"nan", "inf"}                                      # the native control
    prog = _prog(tmp_path, "r.py", RANGE_TASK.format(a="-1e308", b="1e308"))
    out = _run(prog, [{"kind": "float", "value": raw.hex()}])
    assert (out.invalid_draw is False) == valid, (raw, out)


# ------------------------------------------------------------------ nondet_float

FLOAT_TASK = """
    import _sv_verifier
    x = _sv_verifier.nondet_float()
    raise IndexError(x)
"""


@pytest.mark.parametrize("raw", [float("inf"), -float("inf"), float("nan"), -0.0])
def test_a_nondet_float_draw_the_helper_produces_is_a_confirmed_violation(tmp_path, raw):
    prog = _prog(tmp_path, "f.py", FLOAT_TASK)
    vector = [{"kind": "float", "value": raw.hex()}]
    out = _run(prog, vector)
    assert out.invalid_draw is False and out.violated(IE) == "IndexError", out
    assert replay(prog, {"vector": vector}, IE)[0] is True


@pytest.mark.parametrize("raw", [0.0, 1.0, -1.0, 0.5, 1e300, -1e300, 5e-324, 1.7976931348623157e308])
def test_a_finite_nondet_float_draw_is_refused_never_a_confirmed_violation(tmp_path, raw):
    """TRUE is proven over every float (absint keeps ``FLOAT``); a FALSE rests only on what the helper's code produces on 3.14."""
    prog = _prog(tmp_path, "f.py", FLOAT_TASK)
    vector = [{"kind": "float", "value": raw.hex()}]
    out = _run(prog, vector)
    assert out.invalid_draw is True and out.violated(IE) is None, out
    assert replay(prog, {"vector": vector}, IE)[0] is False


def test_the_default_nondet_float_draw_is_in_domain(tmp_path):
    """An empty vector draws ``-0.0`` (a ``0.0`` would be refused): the first run of a program that draws a float is a valid run."""
    out = _run(_prog(tmp_path, "f.py", FLOAT_TASK), [])
    assert out.invalid_draw is False and out.violated(IE) == "IndexError"
    assert out.draws[0]["value"] == (-0.0).hex()


def test_the_helpers_own_code_produces_only_the_four_values_on_this_interpreter():
    """The native control of the domain: the sv-benchmarks ``nondet_float`` (MR !1752), run on this CPython."""
    import math as _m
    import random
    import sys as _s

    rnd = random.Random(1159)          # an instance: the module-level functions are patched by an in-process ``drive`` in the same worker

    def nondet_bool():
        return bool(rnd.randint(0, 1))

    def helper():
        if nondet_bool():
            return rnd.choice([float("inf"), float("-inf"), float("nan"), float("-0.")])
        return rnd.uniform(-_s.float_info.max, _s.float_info.max)

    draws = [helper() for _ in range(20000)]
    assert not any(_m.isfinite(d) and (d != 0.0 or _m.copysign(1.0, d) > 0) for d in draws)
    seen = {"inf" if d == float("inf") else "-inf" if d == float("-inf") else "nan" if d != d else "-0.0" for d in draws}
    assert seen == {"inf", "-inf", "nan", "-0.0"}
    from tharkun.sv import _child

    assert all(_child.float_producible(d) for d in draws)


def test_the_default_draw_outside_a_float_range_is_invalid_not_clamped(tmp_path):
    """An empty vector draws the float default 0.0: outside [5, 10], so invalid -- never clamped to 5."""
    out = _run(_prog(tmp_path, "r.py", RANGE_TASK.format(a="5", b="10")), [])
    assert out.invalid_draw is True and out.violated(IE) is None


# ------------------------------------------------------------------ rindex / rfloat / kind mismatch

def test_a_raw_out_of_range_rindex_is_clamped_in_the_log_so_it_never_replays_as_written(tmp_path):
    """The harness clamps an rindex draw and logs the clamped value; the validator then sees a log that
    differs from the witness and refuses it.  Tharkun's own self-check (``engine._replayed``) is the same
    comparison, so a vector holding the raw value is not a confirmed witness here either."""
    prog = _prog(tmp_path, "c.py", """
        import random
        x = random.choice([1, 2, 3])
        raise IndexError(x)
    """)
    vector = [{"kind": "rindex", "value": 9, "n": 3}]
    out = _run(prog, vector)
    assert out.violated(IE) == "IndexError"
    assert out.draws[0]["value"] == 2                 # clamped into [0, n - 1]
    assert engine._replayed(vector, out) is False     # ... so the raw vector is not what ran


def test_a_raw_out_of_range_rfloat_is_replaced_by_zero_in_the_log(tmp_path):
    prog = _prog(tmp_path, "rf.py", """
        import random
        x = random.random()
        raise IndexError(x)
    """)
    vector = [{"kind": "rfloat", "value": (1.5).hex()}]
    out = _run(prog, vector)
    assert out.draws[0]["value"] == (0.0).hex()
    assert engine._replayed(vector, out) is False


def test_a_vector_entry_of_the_wrong_kind_draws_the_default_and_never_replays_as_written(tmp_path):
    prog = _prog(tmp_path, "k.py", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        raise IndexError(x)
    """)
    vector = [{"kind": "float", "value": (3.0).hex()}]
    out = _run(prog, vector)
    assert out.draws == [{"kind": "int", "value": 0, "site": out.draws[0]["site"]}]
    assert engine._replayed(vector, out) is False


# ------------------------------------------------------------------ every path that can write a confirmed FALSE

def test_the_fork_server_refuses_the_same_out_of_domain_draws(tmp_path):
    """``explore`` runs vectors in a fork server (``forkserver.py`` -> ``_forkserver.py`` -> ``_child.main``);
    ``found()`` then replays a forked find fresh, but the forked outcome must already agree."""
    if sys.platform != "linux":
        pytest.skip("the fork server is linux-only")
    from tharkun.sv.forkserver import ForkServer

    prog = _prog(tmp_path, "s.py", STR_TASK)
    server = ForkServer(sys.executable)
    try:
        bad = server.run(sys.executable, prog, [{"kind": "str", "value": "\u00e9"}], timeout=60)
        good = server.run(sys.executable, prog, [{"kind": "str", "value": "ok"}], timeout=60)
    finally:
        server.close()
    assert bad.invalid_draw is True and bad.violated(IE) is None, bad
    assert good.invalid_draw is False and good.violated(IE) == "IndexError", good


def test_the_in_process_drive_refuses_the_same_out_of_domain_draws(tmp_path):
    from tharkun.sv.nondet_harness import drive

    prog = _prog(tmp_path, "s.py", STR_TASK)
    assert drive(prog, [{"kind": "str", "value": "\u00e9"}]).invalid_draw is True
    assert drive(prog, [{"kind": "str", "value": "ok"}]).invalid_draw is False
    rng = _prog(tmp_path, "r.py", RANGE_TASK.format(a="0.0", b="1.0"))
    assert drive(rng, [{"kind": "float", "value": (2.0).hex()}]).invalid_draw is True
    assert drive(rng, [{"kind": "float", "value": (0.5).hex()}]).invalid_draw is False


def _sv_modules():
    root = Path(engine.__file__).resolve().parent
    return {p.name: p for p in sorted(root.glob("*.py"))}


def test_only_the_child_defines_the_draw_model_every_other_model_is_a_proposal_generator():
    """Every confirmed FALSE is a run of ``_child.py`` (``run_once``, the fork server, the stationary probe,
    ``drive``, the product proof's replay).  This pins WHO defines a ``_sv_verifier`` stand-in: ``_child.py``
    is the judge; ``_hybrid_worker.py`` and ``_concolic_worker.py`` are proposal generators whose output is
    ALWAYS replayed through ``_child.py`` (they log in-domain values on purpose, ``_clamp_float_range`` /
    ``_printable_str``); ``avengers_seed.py`` only stubs the module so a class can be built.  A new module that
    defines its own ``nondet_str`` / ``nondet_float_range`` would be a second model that could confirm
    something the harness refuses: add it here only after reading why it must exist."""
    import ast

    definers = set()
    for name, path in _sv_modules().items():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("nondet_str", "nondet_float_range"):
                definers.add(name)
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr in ("nondet_str", "nondet_float_range"):
                        definers.add(name)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "def nondet_float_range" in node.value:
                definers.add(name)
    assert definers == {"_child.py", "_hybrid_worker.py", "_concolic_worker.py", "avengers_seed.py"}, definers


def test_the_probe_and_the_fork_server_take_their_draw_model_from_the_child():
    mods = _sv_modules()
    assert "import _child as C" in mods["_probe.py"].read_text()
    assert "import _child" in mods["_forkserver.py"].read_text()
    assert "from tharkun.sv import _child" in mods["nondet_harness.py"].read_text()


def test_a_violation_is_written_only_by_found_after_a_replay():
    """The ``violate`` writer (``ObligationTable.violate``) has exactly one caller: the scheduler's ``found``,
    and ``found`` is fed only verdicts whose witness is the draws a ``_child`` run consumed."""
    import ast

    callers = []

    class Visit(ast.NodeVisitor):
        def __init__(self, module):
            self.module, self.stack = module, []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "violate":
                callers.append((self.module, self.stack[-1] if self.stack else "<module>"))
            self.generic_visit(node)

    for name, path in _sv_modules().items():
        Visit(name).visit(ast.parse(path.read_text()))
    assert callers == [("scheduler.py", "found")], callers


# ------------------------------------------------------------------ the differential against the real harness

def _harness_src():
    here = Path(__file__).resolve()
    candidates = [os.environ.get("EREGION_SV_SRC"), str(here.parents[2] / "eregion-sv" / "src"),
                  str(here.parents[1] / ".." / "eregion-sv" / "src")]
    for c in candidates:
        if c and (Path(c) / "eregion_sv" / "replay.py").is_file():
            return c
    return None


HARNESS = _harness_src()


def _fl(x):
    return float(x).hex()


I = lambda v: {"kind": "int", "value": v}
Fl = lambda v: {"kind": "float", "value": _fl(v)}
S = lambda v: {"kind": "str", "value": v}
B = lambda v: {"kind": "bool", "value": v}
RI = lambda v, n=3: {"kind": "rindex", "value": v, "n": n}
RF = lambda v: {"kind": "rfloat", "value": _fl(v)}

TASKS = {
    "argv": "import sys\nsys.argv.append('mutation')\nraise IndexError\n",
    "fr01": "x = _sv_verifier.nondet_float_range(0.0, 1.0)\nraise IndexError(x)\n",
    "fr510": "x = _sv_verifier.nondet_float_range(5, 10)\nraise IndexError(x)\n",
    "fr105": "x = _sv_verifier.nondet_float_range(10, 5)\nraise IndexError(x)\n",
    "frinf": "x = _sv_verifier.nondet_float_range(float('-inf'), float('inf'))\nraise IndexError(x)\n",
    "frov": "x = _sv_verifier.nondet_float_range(-1e308, 1e308)\nraise IndexError(x)\n",
    "frneg": "x = _sv_verifier.nondet_float_range(-3.7, 123456.789)\nraise IndexError(x)\n",
    "frzero": "x = _sv_verifier.nondet_float_range(2.5, 2.5)\nraise IndexError(x)\n",
    "frdesc": "x = _sv_verifier.nondet_float_range(1.0, 0.0)\nraise IndexError(x)\n",
    "float": "x = _sv_verifier.nondet_float()\nraise IndexError(x)\n",
    "int": "x = _sv_verifier.nondet_int()\nraise IndexError(x)\n",
    "bool": "x = _sv_verifier.nondet_bool()\nraise IndexError(x)\n",
    "str": "x = _sv_verifier.nondet_str()\nraise IndexError(x)\n",
    "two": "a = _sv_verifier.nondet_int()\nb = _sv_verifier.nondet_float()\nraise IndexError(a, b)\n",
    "choice": "x = random.choice([1, 2, 3])\nraise IndexError(x)\n",
    "rand": "x = random.random()\nraise IndexError(x)\n",
    "uniform": "x = random.uniform(2, 4)\nraise IndexError(x)\n",
    "list": "x = _sv_verifier.nondet_list(_sv_verifier.nondet_int)\nraise IndexError(x)\n",
}

DIFF_CASES = (
    [("argv", [{"kind": k, "value": v}]) for k in ("argv", "argv_full")
     for v in ([], [""], ["arbitrary-zero"], ["雪", "\0", "\ud800"], "bad", ["ok", 1], None)]
    +     [("fr01", [v]) for v in (Fl(0.0), Fl(1.0), Fl(-0.0), Fl(0.5), Fl(1.0000000000000002), Fl(-1e-300), Fl(2.0),
                              Fl(float("nan")), Fl(float("inf")), Fl(float("-inf")))] + [("fr01", [])]
    + [("fr01", [Fl(v)]) for v in (1 - 2.0 ** -53, 2.0 ** -53, 1e-300, 5e-324, 0.1, 0.3, 0.25, 0.75)]
    + [("fr510", []), ("fr510", [Fl(7.0)]), ("fr510", [Fl(4.999)]), ("fr510", [Fl(10.0)]),
       ("fr105", [Fl(7.0)]), ("fr105", [Fl(11.0)]), ("frinf", [Fl(float("nan"))]), ("frinf", [Fl(1e308)])]
    + [("frov", [Fl(v)]) for v in (float("inf"), float("nan"), -float("inf"), 0.0, 1e308, -1e308, -0.0)] + [("frov", [])]
    + [("frneg", [Fl(v)]) for v in (-3.7, 0.0, 123456.789, 61726.5445, 1.0)] + [("frneg", [])]
    + [("frzero", [Fl(v)]) for v in (2.5, 2.4999999999999996, 0.0)] + [("frdesc", [Fl(v)]) for v in (1.0, 0.0, 0.5, 1e-17, 2.0 ** -53)]
    + [("float", [Fl(float("nan"))]), ("float", [Fl(float("inf"))]), ("float", [Fl(-0.0)]), ("float", [Fl(-float("inf"))]),
       ("float", [Fl(0.0)]), ("float", [Fl(1.5)]), ("float", [Fl(1e308)]), ("float", [])]
    + [("int", [I(v)]) for v in (0, -1, 2 ** 70, True, 1.5, "7", None)] + [("int", []), ("int", [Fl(1.0)])]
    + [("bool", [B(v)]) for v in (True, False, 2, "", "x", 0.0)] + [("bool", [I(1)])]
    + [("str", [S(v)]) for v in list(NON_PRINTABLE.values()) + list(PRINTABLE.values())] + [("str", [])]
    + [("str", [S(5)]), ("str", [S(["x"])])]
    + [("two", [I(1), Fl(2.0)]), ("two", [I(1)]), ("two", [Fl(2.0), I(1)]), ("two", [I(1), Fl(2.0), I(9)])]
    + [("choice", [RI(v)]) for v in (0, 2, 5, -1)] + [("choice", [RI(1, 9)]), ("choice", [I(1)]), ("choice", [])]
    + [("rand", [RF(v)]) for v in (0.5, 1.0, -0.1, float("nan"), 0.9999999999999999)] + [("rand", [Fl(0.5)])]
    + [("uniform", [RF(0.5)]), ("uniform", [RF(2.0)])]
    + [("list", [B(True), I(1), B(True), I(2), B(False)]), ("list", [B(False)]), ("list", [B(True), I(1)])]
)


@pytest.mark.skipif(HARNESS is None, reason="needs the grading harness (eregion_sv): EREGION_SV_SRC or the umbrella's repos/eregion-sv")
@pytest.mark.parametrize("task, vector", DIFF_CASES,
                         ids=[f"{t}-{json.dumps(v)[:40]}" for t, v in DIFF_CASES])
def test_tharkun_confirms_exactly_what_the_grading_harness_confirms(tmp_path, task, vector):
    """For every draw kind and edge value: same confirmation, same invalid-draw verdict, same draws
    consumed.  (A case the harness confirms and Tharkun does not would be harmless; the reverse is a wrong
    FALSE -- this test asserts equality so neither drifts.)"""
    sys.path.insert(0, HARNESS)
    try:
        from eregion_sv import replay as H
    finally:
        sys.path.remove(HARNESS)
    prog = _prog(tmp_path, f"{task}.py", "import _sv_verifier\nimport random\n" + TASKS[task])

    h = H.run(sys.executable, prog, source={"vector": vector}, timeout=60)
    t = run_once(sys.executable, prog, vector, timeout=60)

    h_replayed = len(h.get("draws", [])) == len(vector) and all(
        isinstance(w, dict) and all(w.get(k) == d[k] for k in ("kind", "value"))
        for w, d in zip(vector, h.get("draws", [])))
    h_confirmed = H.violates(h, IE) is not None and h_replayed
    t_confirmed = t.violated(IE) is not None and engine._replayed(vector, t)
    h_invalid = bool(h.get("invalid_draw")) or h.get("status") == "invalid-draw"
    t_invalid = bool(t.invalid_draw) or t.status == "invalid-draw"

    assert t_confirmed == h_confirmed, (h, t)
    assert t_invalid == h_invalid, (h, t)
    assert [(d["kind"], d["value"]) for d in t.draws] == [(d["kind"], d["value"]) for d in h.get("draws", [])]
