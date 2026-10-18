"""The ``random`` models of the absint pass: call shapes ([internal ref]) and ``uniform`` ([internal ref]).

[internal ref] (a wrong TRUE): ``random.randint(a=0, b=5)``, ``uniform(a=0, b=1)``, ``choice(seq=[1])``, ``seed(a=1)`` and ``seed(1, 2)`` are valid
calls (the module functions are bound methods of ``random.Random``; each parameter may be given by name, and ``seed`` takes
``version`` too), but ``Analyzer._arity`` counted a keyword, or the second positional argument of ``seed``, as an arity error.  The model
recorded an unproven ``TypeError`` and ended the path, so every statement after the call went unchecked: ``x = random.randint(a=0, b=5);
c = 1 // 0`` was TRUE for ``ZeroDivisionError`` and natively raises it.  The same pattern was in ``_sv_verifier.nondet_float_range(a, b)``.

[internal ref] (a wrong TRUE): ``random.uniform(a, b)`` was modelled as the interval ``[a, b]``, but CPython computes
``a + (b - a) * random()``, which is ``inf`` for ``uniform(-1e308, 1e308)`` (``b - a`` overflows), ``nan`` for an infinite bound, and an
``OverflowError`` for an int too large for a float.  ``assert y <= 1e308`` after ``y = random.uniform(-1e308, 1e308)`` was TRUE and
fails natively on every draw.

Every program here is small and safe to run natively.  No generated programs: each case is one named program, run natively and by
absint.  The controls: a test of a fix fails when the fix is undone (``scripts`` of the PR body list the runs).
"""

from __future__ import annotations

import io
import math
import random
import sys
import textwrap
import types
import warnings

import pytest

from tharkun.solve.absint import prove

HEAD = "import math\nimport random\nimport _sv_verifier as sv\n"
SENTINEL = "reached = True\nc = 1 // 0\n"


def _stub_verifier() -> types.ModuleType:
    """The grading shim's ``nondet_float_range(a, b)`` (``sv/_child.py``), with a fixed draw."""
    m = types.ModuleType("_sv_verifier")
    m.nondet_float_range = lambda a, b: a
    m.nondet_int = lambda: 7
    m.nondet_bool = lambda: True
    return m


def native(src: str):
    """``(sentinel reached, the exception class names a native run raises before it)``."""
    ns: dict = {"__name__": "__main__"}
    warnings.simplefilter("ignore")
    old_out, old_state, old_mod = sys.stdout, random.getstate(), sys.modules.get("_sv_verifier")
    sys.stdout = io.StringIO()
    sys.modules["_sv_verifier"] = _stub_verifier()
    try:
        exec(compile(src, "<program>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        reached = bool(ns.get("reached"))
        return reached, None if reached else {c.__name__ for c in type(e).__mro__}
    finally:
        sys.stdout = old_out
        random.setstate(old_state)
        if old_mod is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old_mod
    return False, None


def verdict(src: str, cls: str):
    return prove(src, [cls], filename="<program>", time_budget=20.0)


def never_true(consumer: str, cls: str | None = None) -> None:
    """``consumer`` then the sentinel: native reaches the sentinel (or raises ``cls`` in ``consumer``) and absint must NOT say TRUE
    for it."""
    src = HEAD + textwrap.dedent(consumer) + "\n" + SENTINEL
    reached, raised = native(src)
    if cls is None:
        assert reached, ("the consumer must succeed natively", consumer, raised)
        cls = "ZeroDivisionError"
    else:
        assert not reached and cls in raised, ("native must raise " + cls, consumer, reached, raised)
    assert verdict(src, cls).verdict != "TRUE", src


def refused(src: str, cls: str, why: str) -> None:
    r = verdict(HEAD + src, cls)
    assert r.verdict == "UNKNOWN" and r.reason.startswith("unsupported:") and why in r.reason, (src, r.verdict, r.reason)


# ---------------------------------------------------------------------------------------------------------------------------
# [internal ref]: a call binds its arguments as CPython binds a Python function's
# ---------------------------------------------------------------------------------------------------------------------------

VALID_SHAPES = [
    "x = random.randint(a=0, b=5)", "x = random.randint(0, b=5)", "x = random.randint(b=5, a=0)",
    "x = random.uniform(a=0, b=1)", "x = random.uniform(0, b=1)", "x = random.uniform(b=1.0, a=0.0)",
    "x = random.choice(seq=[1, 2])", "x = random.choice(seq=(1, 2))", "x = random.choice(seq='ab')",
    "random.seed(a=1)", "random.seed(1, 2)", "random.seed(1, version=2)", "random.seed(version=2)", "random.seed(a=None, version=1)",
    "random.seed()", "random.seed('x', 3)", "random.seed(1.5, 'v')",
    "x = random.random()", "st = random.getstate()",
    "x = sv.nondet_float_range(a=0.0, b=1.0)", "x = sv.nondet_float_range(0.0, b=1.0)", "x = sv.nondet_float_range(b=1.0, a=0.0)",
]


@pytest.mark.parametrize("call", VALID_SHAPES)
def test_a_valid_call_shape_continues_the_path(call):
    never_true(call)


INVALID_SHAPES = [
    "random.randint(1)", "random.randint(1, 2, 3)", "random.randint(1, c=2)", "random.randint(1, b=2, a=1)", "random.randint()",
    "random.uniform(1)", "random.uniform(1, 2, 3)", "random.uniform(a=1, c=2)", "random.uniform(1, a=1)",
    "random.choice()", "random.choice([1], 2)", "random.choice(x=[1])", "random.choice([1], seq=[1])",
    "random.seed(1, 2, 3)", "random.seed(b=1)", "random.seed(1, a=1)",
    "random.random(1)", "random.random(a=1)", "random.getstate(1)", "random.getstate(a=1)",
    "sv.nondet_float_range(1.0)", "sv.nondet_float_range(1.0, 2.0, 3.0)", "sv.nondet_float_range(a=1.0, c=2.0)",
]


@pytest.mark.parametrize("call", INVALID_SHAPES)
def test_an_invalid_call_shape_is_a_certain_recorded_type_error(call):
    """Native raises ``TypeError`` before anything else; the verdict for ``TypeError`` must not be TRUE (the check is recorded and
    unproven), and the path ends there, so the sentinel after it is TRUE."""
    src = HEAD + call + "\n" + SENTINEL
    reached, raised = native(src)
    assert not reached and "TypeError" in raised, (call, reached, raised)
    assert verdict(src, "TypeError").verdict != "TRUE", call
    assert verdict(src, "ZeroDivisionError").verdict == "TRUE", call


def test_keywords_bind_by_name_not_by_position():
    """``randint(b=4, a=3)`` is ``randint(3, 4)``: bound by position it would be ``randint(4, 3)``, a certain ``ValueError`` and the end
    of the path, which proves everything after it."""
    assert {random.randint(b=4, a=3) for _ in range(200)} == {3, 4}  # the native control: 3 or 4, so x - 3 may be 0
    for call in ("random.randint(b=4, a=3)", "random.randint(a=3, b=4)", "random.randint(3, b=4)"):
        assert verdict(HEAD + f"x = {call}\nc = 10 // (x - 3)\n", "ZeroDivisionError").verdict == "UNKNOWN", call
    for call in ("random.randint(b=5, a=4)", "random.randint(a=4, b=5)"):
        assert verdict(HEAD + f"x = {call}\nc = 10 // (x - 3)\n", "ZeroDivisionError").verdict == "TRUE", call
    assert {random.randint(b=5, a=4) for _ in range(200)} == {4, 5}


def test_random_randint_with_a_greater_than_b_ends_the_path():
    """``randint(5, 3)`` raises ``ValueError`` on every run (not a property class): what follows is not reached."""
    assert verdict(HEAD + "x = random.randint(5, 3)\nassert False\n", "AssertionError").verdict == "TRUE"
    assert verdict(HEAD + "x = random.randint(a=5, b=3)\nassert False\n", "AssertionError").verdict == "TRUE"


@pytest.mark.parametrize("call", [
    "random.randint(a=[1], b=3)", "random.randint(0, 'x')", "random.randint(1.5, 3)", "random.randint(0, 2.5)", "random.randint(None, 3)",
])
def test_randint_of_a_non_int_is_a_type_error(call):
    never_true(call, "TypeError")


def test_random_with_user_objects_is_a_named_refusal():
    """A user object's methods (``__index__`` is refused at the class, but ``__add__``, ``__sub__``, ``__eq__`` run in these calls) are
    not analysed here: refused by name, not a ``TypeError`` that ends the path."""
    cls = "class B:\n    def __add__(self, o):\n        return 7\n    def __sub__(self, o):\n        return 1.0\n    def __rsub__(self, o):\n        return 1.0\n"
    refused(cls + "x = random.randint(0, B())\n" + SENTINEL, "ZeroDivisionError", "random.randint of a user object")
    refused(cls + "x = random.uniform(0.5, B())\n" + SENTINEL, "ZeroDivisionError", "random.uniform of a user object")
    refused(cls + "x = random.choice(B())\n" + SENTINEL, "ZeroDivisionError", "random.choice of a user object")
    refused(cls + "random.seed(1, B())\n" + SENTINEL, "ZeroDivisionError", "random.seed of a user object")


def test_the_value_of_a_keyword_call_is_the_value_of_the_positional_one():
    for pos, kw in (("random.randint(3, 5)", "random.randint(a=3, b=5)"), ("random.choice([7, 8])", "random.choice(seq=[7, 8])")):
        for body in ("x = {}\nc = 10 // (x - 2)\n", "x = {}\nc = 10 // (x - 7)\n"):
            a = verdict(HEAD + body.format(pos), "ZeroDivisionError").verdict
            b = verdict(HEAD + body.format(kw), "ZeroDivisionError").verdict
            assert a == b, (pos, kw, body, a, b)


# ---------------------------------------------------------------------------------------------------------------------------
# [internal ref]: uniform(a, b) is a + (b - a) * random(), not the interval [a, b]
# ---------------------------------------------------------------------------------------------------------------------------


def test_uniform_overflows_to_infinity_natively():
    """The native control of the fix: every draw of ``uniform(-1e308, 1e308)`` is ``inf`` (``b - a`` is ``inf``, and ``inf * r`` is ``inf``
    for ``r > 0``, ``nan`` for ``r == 0``)."""
    assert all(math.isinf(random.uniform(-1e308, 1e308)) for _ in range(50))
    assert math.isnan(random.uniform(math.inf, 1.0))
    assert math.isnan(random.uniform(0.0, math.nan))
    with pytest.raises(OverflowError):
        random.uniform(1, 10**400)


@pytest.mark.parametrize("body", [
    "y = random.uniform(-1e308, 1e308)\nassert y <= 1e308",                       # the probe g15_uniform_ok_hidden: y is inf
    "y = random.uniform(-1e308, 1e308)\nassert y < 1e308",
    "y = random.uniform(-1e308, 1e308)\nassert not math.isinf(y)",
    "y = random.uniform(a=-1e308, b=1e308)\nassert y <= 1e308",
    "y = random.uniform(1e308, -1e308)\nassert y >= -1e308",
    "y = random.uniform(-1.7e308, 1.7e308)\nassert y <= 1.7e308",
    "y = random.uniform(math.inf, 1.0)\nassert not math.isnan(y)",                # inf - ... : nan
    "y = random.uniform(0.0, math.inf)\nassert not math.isinf(y) and y == y",    # inf * 0.0 is nan, inf * r is inf
    "y = random.uniform(0.0, math.nan)\nassert y == y",
])
def test_uniform_is_not_the_interval_of_its_bounds(body):
    """Native raises ``AssertionError``; absint must not say TRUE for it."""
    never_true(body, "AssertionError")


def test_the_hidden_branch_probe_g15():
    """``g15_uniform_ok_hidden.py`` (MAIN, [internal ref]), with the hidden condition replaced by a branch absint cannot decide."""
    src = ("import _sv_verifier as sv, random\nb = sv.nondet_int()\nif b == 12345678901234567890123:\n"
           "    y = random.uniform(-1e308, 1e308)\n    assert y <= 1e308\n")
    assert verdict(src, "AssertionError").verdict != "TRUE"


def _lit(x: float) -> str:
    return "math.nan" if math.isnan(x) else "math.inf" if x == math.inf else "-math.inf" if x == -math.inf else repr(x)


class _Draw(random.Random):
    """A generator whose ``random()`` is a chosen draw: ``uniform`` is CPython's own, so its answer is a value a program can see."""

    def __init__(self, r: float) -> None:
        super().__init__(0)
        self.r = r

    def random(self) -> float:
        return self.r


@pytest.mark.parametrize("a, b", [
    (-1e308, 1e308), (0.0, 1.0), (1.0, 2.0), (2.0, 1.0), (-5, 5), (0, 1), (1e308, 1.7e308), (-1e-300, 1e-300), (5e-324, 1.0),
    (math.inf, 1.0), (0.0, math.inf), (-math.inf, math.inf), (0.0, math.nan), (-0.0, 0.0),
])
def test_every_value_the_formula_can_take_is_reachable_for_absint(a, b):
    """The draws ``random()`` can return include ``0.0``, ``0.5`` and ``1 - 2**-53``; for each, CPython's ``uniform`` (the native
    control: a ``Random`` whose ``random()`` is that draw) gives a value the program may see, so ``assert y != <that value>`` must not
    be TRUE (a ``nan`` is tested with ``y == y``, an infinity with the largest finite float)."""
    for r in (0.0, 0.5, 1.0 - 2.0 ** -53):
        v = _Draw(r).uniform(a, b)
        assert v == a + (b - a) * r or (math.isnan(v) and math.isnan(a + (b - a) * r))
        big = repr(sys.float_info.max)  # an infinity is compared against the largest finite float: `math.inf` is an unknown float here
        cond = ("y == y" if math.isnan(v) else f"y <= {big}" if v == math.inf else f"y >= -{big}" if v == -math.inf
                else f"y != {_lit(v)}")
        src = HEAD + f"y = random.uniform({_lit(a)}, {_lit(b)})\nassert {cond}\n"
        assert verdict(src, "AssertionError").verdict != "TRUE", (a, b, r, v, src)


def test_uniform_without_overflow_stays_precise():
    """The formula, not a refusal: the hull of an ordinary range is still exact enough to prove a bound."""
    assert verdict(HEAD + "y = random.uniform(0, 1)\nassert 0 <= y <= 1\n", "AssertionError").verdict == "TRUE"
    assert verdict(HEAD + "y = random.uniform(2, 1)\nassert 1 <= y <= 2\n", "AssertionError").verdict == "TRUE"
    assert verdict(HEAD + "y = random.uniform(1, 2)\nassert y < 1\n", "AssertionError").verdict != "TRUE"
    assert verdict(HEAD + "y = random.uniform(1.0, 2.0)\nassert 1.0 <= y <= 2.0\n", "AssertionError").verdict == "TRUE"


@pytest.mark.parametrize("call", ["random.uniform('a', 1)", "random.uniform(1, 'a')", "random.uniform(None, 1)", "random.uniform([1], 2)",
                                  "random.uniform(1, (2,))"])
def test_uniform_of_a_non_number_is_a_type_error(call):
    never_true(call, "TypeError")


# ---------------------------------------------------------------------------------------------------------------------------
# the rest of the random float family, and the draws no model covers: refused by name (never a value)
# ---------------------------------------------------------------------------------------------------------------------------

UNMODELLED = {
    "triangular": "(0.0, 1.0)", "gauss": "(0.0, 1.0)", "normalvariate": "(0.0, 1.0)", "lognormvariate": "(0.0, 1.0)",
    "expovariate": "(1.0)", "betavariate": "(1.0, 1.0)", "gammavariate": "(1.0, 1.0)", "vonmisesvariate": "(0.0, 1.0)",
    "paretovariate": "(1.0)", "weibullvariate": "(1.0, 1.0)", "binomialvariate": "(1, 0.5)", "randbytes": "(2)",
    "choices": "([1, 2], k=2)",
}


@pytest.mark.parametrize("name", sorted(UNMODELLED))
def test_the_unmodelled_draws_are_refused_by_name(name):
    """Each is a valid call natively (the sentinel is reached), and absint refuses it: a verdict of UNKNOWN naming the call."""
    never_true(f"x = random.{name}{UNMODELLED[name]}")
    refused(f"x = random.{name}{UNMODELLED[name]}\n" + SENTINEL, "ZeroDivisionError", f"random.{name}")


@pytest.mark.parametrize("body, cls", [
    ("x = random.expovariate(0.0)", "ZeroDivisionError"),                        # -log(1 - r) / lambd
    ("x = random.paretovariate(0.0)", "ZeroDivisionError"),                      # u ** (-1.0 / alpha)
    ("x = random.weibullvariate(1.0, 0.0)", "ZeroDivisionError"),                # alpha * (-log(u)) ** (1.0 / beta)
    ("y = random.triangular(-1e308, 1e308)\nassert -1e308 <= y <= 1e308", "AssertionError"),  # the overflow of uniform: +-inf
    ("x = random.lognormvariate(1000.0, 1.0)", "OverflowError"),
])
def test_the_float_family_has_hazards_a_model_would_have_to_carry(body, cls):
    """Why the family is refused rather than modelled by the hull of its parameters: each of these raises, or leaves the interval of its
    arguments, natively (a ``ZeroDivisionError`` for a zero scale, ``inf`` for a wide triangle).  ``OverflowError`` is not a property
    class, so that one only has to run and be refused."""
    src = HEAD + body + "\n" + SENTINEL
    reached, raised = native(src)
    if cls == "OverflowError":
        assert not reached and "OverflowError" in raised
    else:
        assert not reached and cls in raised, (body, reached, raised)
        assert verdict(src, cls).verdict != "TRUE", body
    assert verdict(src, "ZeroDivisionError").verdict != "TRUE" or cls == "OverflowError"


def test_the_names_of_the_random_module_are_all_accounted_for():
    """Every public function of ``random`` is either modelled (``MODULES['random']``), contract-driven (``contracts/rules_random.py``: a
    sound havoc, never a value) or in ``UNMODELLED`` above (refused); a function a later Python adds is a name to decide on."""
    from tharkun.solve.absint.interp import MODULES
    from tharkun.solve.contracts import registry

    modelled = set(MODULES["random"]["funcs"])
    contracted = {c.qualname.split(".", 1)[1] for c in registry.for_module("random")}
    functions = {n for n in random.__all__ if callable(getattr(random, n)) and not isinstance(getattr(random, n), type)}
    classes = {n for n in random.__all__ if isinstance(getattr(random, n), type)}
    state = {"setstate"}
    assert functions - modelled - contracted - set(UNMODELLED) - state == set(), "decide on the new function"
    assert classes == {"Random", "SystemRandom"}, classes  # an instance of either: refused (a call of an unknown value)


@pytest.mark.parametrize("prefix", [
    "", "random.seed()\n", "random.seed(0)\n", "random.random()\n",
    "random.getrandbits(2000)\n", "from random import getstate as peek\n",
])
def test_getstate_proves_only_seed_independent_shape_and_index(prefix):
    call = "peek()" if " as peek" in prefix else "random.getstate()"
    src = HEAD + prefix + f"st = {call}\nassert len(st) == 3\nassert len(st[1]) == 625\nassert 1 <= st[1][-1] <= 624\n"
    assert native(src) == (False, None)
    assert verdict(src, "AssertionError").verdict == "TRUE"


@pytest.mark.parametrize("condition", ["st[1][1] != 123456789", "st[1][-1] == 624", "st[2] is None"])
def test_getstate_keeps_state_words_exact_index_and_cache_unknown(condition):
    src = HEAD + "st = random.getstate()\nassert " + condition + "\n"
    assert verdict(src, "AssertionError").verdict == "UNKNOWN"


@pytest.mark.parametrize("restore", ["random.setstate", "restore"])
def test_getstate_never_proves_a_setstate_installed_zero_index(restore):
    src = HEAD + "from random import setstate as restore\nst = random.getstate()\n"
    src += restore + "((st[0], st[1][:-1] + (0,), st[2]))\nassert random.getstate()[1][-1] != 0\n"
    reached, raised = native(src)
    assert not reached and "AssertionError" in raised
    assert verdict(src, "AssertionError").verdict == "UNKNOWN"


def test_getstate_refuses_module_monkeypatches_and_callback_mutations():
    patched = HEAD + "def fake():\n    return (3, (0,), None)\nrandom.getstate = fake\nassert random.getstate()[1][-1] != 0\n"
    assert verdict(patched, "AssertionError").verdict == "UNKNOWN"
    callback = HEAD + """class Seq:
    def __len__(self):
        return 2
    def __getitem__(self, index):
        return index
    def __setitem__(self, index, value):
        st = random.getstate()
        random.setstate((st[0], st[1][:-1] + (0,), st[2]))
random.shuffle(Seq())
assert random.getstate()[1][-1] != 0
"""
    reached, raised = native(callback)
    assert not reached and "AssertionError" in raised
    assert verdict(callback, "AssertionError").verdict == "UNKNOWN"


def test_getstate_repeated_allocation_after_draws_keeps_the_index_invariant():
    src = HEAD + """for _ in range(4):
    random.random()
    st = random.getstate()
    assert len(st[1]) == 625
    assert 1 <= st[1][-1] <= 624
"""
    assert native(src) == (False, None)
    assert verdict(src, "AssertionError").verdict == "TRUE"
