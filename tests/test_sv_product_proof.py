"""tharkun.sv.product_proof: the exact whole-program Z3 proof for loop-free "product"
programs ([internal ref]'s equivalent-mutant proving; [internal ref], owner ruling 2026-09-29: a
GENERAL proof over the whole inter-procedural call stack before any mutant-specific
rewrite).

Soundness first, same discipline as ``test_sv_prove.py``: every MUTANT below has an
execution that genuinely violates the property, and this module must find it (FALSE,
replay-confirmed) rather than guess TRUE; every EQUIVALENT pair must come back TRUE;
anything outside this module's modelled scope (a loop, an unlifted draw, an operator
:mod:`tharkun.sv._symexec` cannot model exactly) must come back ``None`` (UNKNOWN),
never a guess either way.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from tharkun.sv import product_proof
from tharkun.sv import verify

AR = ("AssertionError",)


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def _prove(path):
    return product_proof.prove(open(path).read(), AR, program=path, python=sys.executable)


# ---------------------------------------------------------------------------
# the toy's own mutants ([internal ref]'s mutation-toy demo)
# ---------------------------------------------------------------------------

_TOY_F = """
        import _sv_verifier


        def f(x, y):
            d = x - y
            a = abs(d)
            if a > 10:
                return a // 2
            return 0

"""


def test_m1_sub_to_add_is_killable(tmp_path):
    # abs() clamps the SIGN, not the MAGNITUDE: x + y has a different magnitude
    # from x - y whenever y != 0.
    p = _prog(tmp_path, "m1.py", _TOY_F + """
        def f_mut(x, y):
            d = x + y
            a = abs(d)
            if a > 10:
                return a // 2
            return 0


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "false"
    assert v.exception == "AssertionError"
    assert v.component == "symbolic:product"


def test_m2_swap_operands_is_equivalent(tmp_path):
    # y - x == -(x - y) always; abs() erases the sign entirely.
    p = _prog(tmp_path, "m2_swap.py", _TOY_F + """
        def f_mut(x, y):
            d = y - x
            a = abs(d)
            if a > 10:
                return a // 2
            return 0


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true"
    assert v.component == "symbolic:product"


def test_m2_cosmic_ray_sub_to_mul_is_killable(tmp_path):
    # the REAL Cosmic Ray mutant for M2 (Sub -> Mul, not the earlier operand-swap
    # shape): x * y is a different magnitude from x - y in general, e.g. x=20,y=1.
    p = _prog(tmp_path, "m2_mul.py", _TOY_F + """
        def f_mut(x, y):
            d = x * y
            a = abs(d)
            if a > 10:
                return a // 2
            return 0


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "false"
    assert v.exception == "AssertionError"


def test_m3_gt_to_ge_is_killable(tmp_path):
    # differs exactly at a == 10: f returns 0, f_mut returns a // 2 == 5.
    p = _prog(tmp_path, "m3.py", _TOY_F + """
        def f_mut(x, y):
            d = x - y
            a = abs(d)
            if a >= 10:
                return a // 2
            return 0


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "false"
    assert v.exception == "AssertionError"


def test_m4_floordiv_to_rshift_is_equivalent(tmp_path):
    # a // 2 == a >> 1 for EVERY Python int, clamp or no clamp (see the
    # never-needs-the-clamp positive control below) -- proven here under the
    # clamp, matching the toy's actual mutant.
    p = _prog(tmp_path, "m4.py", _TOY_F + """
        def f_mut(x, y):
            d = x - y
            a = abs(d)
            if a > 10:
                return a >> 1
            return 0


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true"
    assert v.component == "symbolic:product"


# ---------------------------------------------------------------------------
# controls beyond the toy: never a wrong TRUE, never a missed kill
# ---------------------------------------------------------------------------

def test_floordiv_by_2_and_rshift_by_1_are_equivalent_for_any_sign(tmp_path):
    # d // 2**k == d >> k is a Python identity for EVERY integer, negative or not
    # (both round toward -infinity) -- no abs()/non-negativity clamp is needed for
    # THIS pair, unlike what a narrower, clamp-gated rewrite might assume.
    p = _prog(tmp_path, "shift_identity.py", """
        import _sv_verifier


        def f(x, y):
            d = x - y          # d may be negative: no abs() clamp anywhere
            return d // 2


        def f_mut(x, y):
            d = x - y
            return d >> 1


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true"


def test_shift_by_wrong_amount_is_killable_despite_looking_like_m4(tmp_path):
    # a // 4 is NOT a >> 1 (that would be a // 2) -- a mutant that merely LOOKS
    # like M4's shape (abs clamp, floordiv vs shift both present) must still be
    # killed, not waved through by surface resemblance.
    p = _prog(tmp_path, "wrong_shift.py", """
        import _sv_verifier


        def f(x, y):
            d = x - y
            a = abs(d)
            if a > 10:
                return a // 4
            return 0


        def f_mut(x, y):
            d = x - y
            a = abs(d)
            if a > 10:
                return a >> 1
            return 0


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "false"


def test_operand_swap_without_abs_is_killable(tmp_path):
    # without abs(), the sign is observable: x - y != y - x whenever x != y.
    p = _prog(tmp_path, "swap_no_abs.py", """
        import _sv_verifier


        def f(x, y):
            return x - y


        def f_mut(x, y):
            return y - x


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "false"


def test_floor_division_vs_truncation_toward_zero_is_killable(tmp_path):
    # d // 2 (floor, toward -infinity) vs -((-d) // 2) (truncation toward zero)
    # differ whenever d is odd and negative.
    p = _prog(tmp_path, "trunc.py", """
        import _sv_verifier


        def f(x, y):
            d = x - y
            return d // 2


        def f_mut(x, y):
            d = x - y
            return -((-d) // 2)


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "false"


# ---------------------------------------------------------------------------
# scope refusals: never a guess, always None (UNKNOWN) outside the modelled shape
# ---------------------------------------------------------------------------

def test_a_float_draw_refuses_rather_than_guess(tmp_path):
    # MG review, PR #114: float is deliberately OUT of scope -- _symexec models
    # it as a z3 Real (exact rational arithmetic), the wrong domain for IEEE-754
    # double semantics, so this must refuse rather than risk an unsound TRUE.
    p = _prog(tmp_path, "float_draw.py", """
        import _sv_verifier


        def f(x):
            return x


        def f_mut(x):
            return x


        x = _sv_verifier.nondet_float()
        assert f(x) == f_mut(x)
    """)
    assert _prove(p) is None


def test_true_division_on_int_draws_refuses_rather_than_guess(tmp_path):
    # MG review, PR #114: excluding FLOAT DRAWS is not enough by itself -- a
    # program could still produce a float internally from int draws (x / 2,
    # math.sqrt(x), float(x)). Verify the exactness gate ALSO catches this: none
    # of these are in _symexec's modelled operator set for a symbolic operand
    # (ast.Div has no symbolic case at all, unlike FloorDiv/Mod/Pow), so each
    # raises Unsupported and run_exact refuses, exactly like any other unmodelled
    # operator -- there is no separate path that could silently produce a
    # z3 Real from int-only inputs.
    p = _prog(tmp_path, "true_div.py", """
        import _sv_verifier


        def f(x):
            return x / 2


        def f_mut(x):
            return x / 2


        x = _sv_verifier.nondet_int()
        assert f(x) == f_mut(x)
    """)
    assert _prove(p) is None


def test_math_sqrt_on_int_draws_refuses_rather_than_guess(tmp_path):
    p = _prog(tmp_path, "sqrt.py", """
        import _sv_verifier
        import math


        def f(x):
            return math.sqrt(x * x)


        def f_mut(x):
            return math.sqrt(x * x)


        x = _sv_verifier.nondet_int()
        assert f(x) == f_mut(x)
    """)
    assert _prove(p) is None


def test_float_builtin_on_int_draws_refuses_rather_than_guess(tmp_path):
    p = _prog(tmp_path, "float_builtin.py", """
        import _sv_verifier


        def f(x):
            return float(x)


        def f_mut(x):
            return float(x)


        x = _sv_verifier.nondet_int()
        assert f(x) == f_mut(x)
    """)
    assert _prove(p) is None


# ---------------------------------------------------------------------------
# MG review, PR #114: declarations are NOT always inert -- four never-TRUE
# counterexamples where a flat "strip imports/defs, they have no runtime
# effect" assumption silently produced a wrong TRUE
# ---------------------------------------------------------------------------

def test_a_redefined_function_is_never_true(tmp_path):
    # def f(): return 1 / a = f() / def f(): return 2 -- a is 1 (the FIRST f,
    # the one in effect when a = f() actually ran), so a != 1 is FALSE and the
    # assert ALWAYS fails, for every x, regardless of the second def. A flat
    # {name: def} dict (tharkun.sv._symexec._Runner.defs) always resolves `f`
    # to the LAST definition in the tree, silently using "return 2" instead.
    p = _prog(tmp_path, "redef.py", """
        import _sv_verifier


        x = _sv_verifier.nondet_int()


        def f():
            return 1


        a = f()


        def f():
            return 2


        assert a != 1
    """)
    assert _prove(p) is None


def test_a_default_argument_evaluated_at_def_time_is_never_true(tmp_path):
    # def g(): assert False / def f(a=g()): return a -- Python evaluates a
    # default argument expression EAGERLY, at the def statement itself, so
    # g() -- and its assert False -- runs unconditionally, before the
    # program's own final assert is ever reached. Stripping the def as having
    # "no runtime effect" drops that eager call entirely.
    p = _prog(tmp_path, "default.py", """
        import _sv_verifier


        x = _sv_verifier.nondet_int()


        def g():
            assert False


        def f(a=g()):
            return a


        assert x != x + 1
    """)
    assert _prove(p) is None


def test_a_decorator_evaluated_at_def_time_is_never_true(tmp_path):
    # @g def f(): ... is sugar for f = g(f), called at the def statement
    # itself -- same eager-evaluation hazard as the default-argument case.
    p = _prog(tmp_path, "deco.py", """
        import _sv_verifier


        x = _sv_verifier.nondet_int()


        def g(fn):
            assert False


        @g
        def f():
            return 1


        assert x != x + 1
    """)
    assert _prove(p) is None


def test_an_import_shadowing_abs_is_never_true(tmp_path):
    # from operator import neg as abs rebinds the name `abs` to negation, so
    # abs(x) really computes -x; for x = 1, -1 >= 0 is False. This
    # interpreter's abs() handling is a hardcoded name match
    # (tharkun.sv._symexec.ev_Call), blind to what the program actually bound
    # that name to -- an import other than _sv_verifier must refuse outright.
    p = _prog(tmp_path, "absimport.py", """
        import _sv_verifier
        from operator import neg as abs


        x = _sv_verifier.nondet_int()
        assert abs(x) >= 0
    """)
    assert _prove(p) is None


def test_a_loop_refuses_rather_than_guess(tmp_path):
    p = _prog(tmp_path, "loop.py", """
        import _sv_verifier


        def f(n):
            total = 0
            for i in range(n):
                total += i
            return total


        n = _sv_verifier.nondet_int()
        assert f(n) == f(n)
    """)
    assert _prove(p) is None


def test_a_draw_in_a_return_expression_is_a_fresh_draw_per_execution(tmp_path):
    # ``f() == f()`` draws TWICE: it fails whenever the two draws differ.  A proof that bound one term per
    # draw STATEMENT would call this TRUE; a model has no single vector here, so the answer is None, not TRUE.
    p = _prog(tmp_path, "buried_draw.py", """
        import _sv_verifier


        def f():
            return _sv_verifier.nondet_int()


        assert f() == f()
    """)
    assert _prove(p) is None
    assert verify(p, AR, python=sys.executable, budget_seconds=60, run_timeout=30, canary_seconds=3,
                  early_stop_seconds=3, witness_dir=tmp_path / "proofs").verdict == "false"


def test_no_property_site_refuses(tmp_path):
    p = _prog(tmp_path, "no_site.py", """
        import _sv_verifier

        x = _sv_verifier.nondet_int()
        y = x + 1
    """)
    assert _prove(p) is None


def test_unsupported_call_refuses_rather_than_guess(tmp_path):
    p = _prog(tmp_path, "unsupported_call.py", """
        import _sv_verifier
        import math


        def f(x, y):
            return math.gcd(x, y)


        def f_mut(x, y):
            return math.gcd(x, y)


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    assert _prove(p) is None


# ---------------------------------------------------------------------------
# PR #114 review round 2 (MG): exactness by construction, whole-module allowlist
# ---------------------------------------------------------------------------

def _never_true(path):
    v = _prove(path)
    assert v is None or v.verdict != "true", v


def test_boolor_returns_the_operand_not_its_truth_false(tmp_path):
    # ``x or 7`` is x when x is truthy: x == 3 makes a == 3. (Was a wrong TRUE.)
    p = _prog(tmp_path, "or_false.py", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        a = x or 7
        assert a != 3
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "false"


def test_boolor_returns_the_operand_not_its_truth_true(tmp_path):
    # x truthy -> a = x != 0; x == 0 -> a = 7: a != 0 always.
    p = _prog(tmp_path, "or_true.py", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        a = x or 7
        assert a != 0
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true"


def test_booland_returns_the_operand_not_its_truth(tmp_path):
    # x == 0 -> a = 0; x != 0 -> a = 5.
    p = _prog(tmp_path, "and_true.py", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        a = x and 5
        assert a == 0 or a == 5
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true"
    q = _prog(tmp_path, "and_false.py", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        a = x and 5
        assert a != 5
    """)
    v = _prove(q)
    assert v is not None and v.verdict == "false"


def test_chained_comparison_refuses(tmp_path):
    # a chain short-circuits natively; the interpreter evaluates every link, so a
    # divide inside a never-reached link would prune paths it must not.
    p = _prog(tmp_path, "chain.py", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert x < 0 < 1 // y or True
        assert not (x >= 0 and y == 0)
    """)
    _never_true(p)


def test_a_non_assertion_property_refuses(tmp_path):
    p = _prog(tmp_path, "zd.py", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        if x == 7:
            raise ZeroDivisionError
        y = 10 // (x - 3)
    """)
    assert product_proof.prove(open(p).read(), ("ZeroDivisionError",), program=p,
                               python=sys.executable) is None


def test_builtins_attribute_store_is_never_true_at_cli_level(tmp_path):
    # a22: ``__builtins__.abs = lambda v: -1`` then ``assert abs(x) >= 0`` fails for
    # EVERY x. Once a wrong TRUE from component=symbolic:product.
    p = _prog(tmp_path, "a22.py", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        __builtins__.abs = lambda v: -1
        assert abs(x) >= 0
    """)
    _never_true(p)
    v = verify(p, AR, python=sys.executable, budget_seconds=60, run_timeout=30,
               canary_seconds=3, early_stop_seconds=3, witness_dir=tmp_path / "proofs")
    assert v.verdict != "true", v


@pytest.mark.parametrize("name,body", [
    ("nested_import", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        if True:
            from operator import neg as abs
        assert abs(x) >= 0 or x == 0
    """),
    ("nested_def", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        def g(v):
            return 1
        if x == 3:
            def g(v):
                return 2
        assert g(x) == 1
    """),
    ("user_sv_site", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        def __sv_site():
            assert False
        if x == 3:
            __sv_site()
    """),
    ("attribute_store_on_module", """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        _sv_verifier.nondet_int = 3
        assert x != 3
    """),
])
def test_review2_shapes_are_never_true(tmp_path, name, body):
    p = _prog(tmp_path, name + ".py", body)
    _never_true(p)


def test_exact_subset_refuses_each_construct_outside_the_allowlist():
    import ast
    from tharkun.sv.exact_subset import audit

    def refused(src):
        tree = ast.parse(textwrap.dedent(src))
        return audit(tree, set()) is not None

    ok = "x = 1\ndef f(a):\n    return a + 1\nassert f(x) == 2\n"
    assert not refused(ok)
    for fine in [
        "x = 1\ny = 0 < x < 2\n",                                # a chain over total operands
        "x = 1\ny = -1 < x * 2 + 1 <= 3 != x - 1\n",
        "def f(a: int, b: bool) -> int:\n    return a\n",       # annotations int / bool / object
        "def f(a: object) -> None:\n    pass\nf(1)\n",           # a procedure, called as a statement
        "def f(a):\n    if a:\n        return\n    assert a\nf(1)\n",   # bare return and falling off
        "def f(a: int):\n    return int(a)\n",
        "xs = [0] * 3\nxs[1] = 2\ny = xs[1] + len(xs)\n",       # a list: indexed, stored into, measured ([internal ref] PR 3)
    ]:
        assert not refused(fine), fine
    for bad in [
        "x = 1\nwhile x:\n    x = 0\n",                       # loop
        "x = 1\ny = x.real\n",                                 # attribute
        "x = 1\ny = {x: x}\n",                                 # dict container
        "x = 1\ny = [[x]]\n",                                  # a list of lists
        "x = 1\ny = [x]\nz = y\n",                            # a list used as a value (an alias)
        "x = 1.5\n",                                            # float constant
        "x = 'a'\n",                                            # str constant
        "x = 1\ny = x / 2\n",                                  # true division
        "x = 1\ny = x ** 2\n",                                 # pow
        "x = 1\ny = x & 2\n",                                  # bit and
        "x = 1\ny = 0 < x // 2 < 2\n",                        # chain over an operand that can raise
        "x = 1\ny = 0 < x < 2 < abs(x)\n",                     # chain over a call
        "x = 1\ny = x is 1\n",                                 # identity comparison
        "x = 1\ny = (z := x)\n",                               # walrus
        "x = 1\ny = lambda: x\n",                              # lambda
        "x = 1\ntry:\n    pass\nexcept Exception:\n    pass\n",
        "x = 1\nclass C: pass\n",
        "abs = 1\n",                                            # rebinds a builtin
        "__sv_site = 1\n",                                      # dunder name
        "x = 1\ny = __builtins__\n",
        "x = 1\ny = sum(x)\n",                                 # un-audited builtin
        "x = 1\ny = len(x)\n",                                 # len of something that is not a list
        "x = 1\ny = abs(x, x)\n",                              # wrong arity
        "def f():\n    pass\nx = f()\n",                      # a procedure's None used as a value
        "def f():\n    pass\nx = 1 + f()\n",
        "def f(a):\n    if a:\n        return 1\n    return\n",   # a value on some paths only
        "def f(a):\n    if a:\n        return 1\n",                # a function that falls off its end
        "def f(a: float):\n    return a\n",                     # an annotation outside int / bool / object
        "def f(a) -> str:\n    return a\n",
        "def f(a: 'int'):\n    return a\n",                     # a string annotation
        "import _sv_verifier as int\n",                         # the alias would shadow the builtin ``int``
        "def f(a: int, b):\n    return int(a, b)\n",            # int() takes one argument
        "def f(a):\n    return _sv_verifier.nondet_int(a)\n",   # not a marked draw
        "def f(a=1):\n    return a\n",                         # default
        "@g\ndef f():\n    return 1\n",                        # decorator
        "def f():\n    def g():\n        return 1\n    return 1\n",   # nested def
        "def f():\n    import os\n    return 1\n",            # nested import
        "def f():\n    global x\n    return 1\n",
        "x = 1\nif x:\n    import os\n",                      # import under if
        "def f():\n    return 1\ndef f():\n    return 2\n",  # def twice
        "x = 1\nprint(x)\n",                                   # unmodelled call
        "x = 1\nassert x, 'msg'\n",                            # assert message
        "x = 1\ndel x\n",
        "def f(a):\n    return 1\nf = 3\n",                   # name is def and var
        "x = 1\ndef f(a):\n    x = a\n    return x\n",         # assigned local shadows a module name
        "def g():\n    return 1\ndef f(g):\n    return g\n",   # parameter takes a def's name
        "x = 1\nraise x\n",                                    # raise of a non-class
    ]:
        assert refused(bad), bad


def test_run_exact_is_not_exact_when_any_statement_is_skipped():
    # the root cause of round 2: ``_Runner.run`` dropped a statement it could not model
    # unless it stored a name / drew / called a def; run_exact still said exact=True.
    import ast
    from tharkun.sv._symexec import run_exact

    for src in ["x.a = 1\n", "import os\n", "print(1)\n", "__sv_site()\n"]:
        tree = ast.parse(src)
        r = run_exact(tree, {}, {}, list(tree.body), [])
        assert r["exact"] is False, (src, r)
    tree = ast.parse("y = 1\n")
    assert run_exact(tree, {}, {}, list(tree.body), [])["exact"] is True


# ---------------------------------------------------------------------------
# MG's attack corpora (PR #114 reviews 1 and 2): none may EVER come back TRUE
# ---------------------------------------------------------------------------

import os

_ATTACKS = os.path.join(os.path.dirname(__file__), "fixtures", "product_proof_attacks")
#: every attack whose program genuinely violates the assertion natively (MG's harness:
#: native ground truth over x in -40..40 and large values). The rest of each corpus
#: (a18, a24, ...) are correct-TRUE or no-violation controls and are not listed.
_VIOLATING = [
    "p/t01_redef.py",
    "p/t02_default.py",
    "p/t03_deco.py",
    "p/t04_absimport.py",
    "p/t05_absdef.py",
    "p/t06_abslambda.py",
    "p/t07_bvor.py",
    "p/t08_bvand.py",
    "p/t09_xor.py",
    "p/t10_invert.py",
    "p/t11_floordiv.py",
    "p/t12_negmod.py",
    "p/t13_tryexc.py",
    "p/t14_boolmix.py",
    "p/t15_rshift.py",
    "p/t16_global.py",
    "p/t17_recur.py",
    "p/t18_twice.py",
    "p/t19_implicitzd.py",
    "p/t20_pow.py",
    "p/t21_bigpow.py",
    "p/t22_divtrue.py",
    "p/t23_chaincmp.py",
    "p/t24_intbool.py",
    "p/t25_hash.py",
    "p/t26_walrus.py",
    "p/t27_ternary_str.py",
    "p/t28_whilefalse.py",
    "p/t29_exitearly.py",
    "p/t30_finally.py",
    "p2/a01_svokbinop.py",
    "p2/a02_svindex.py",
    "p2/a03_svnext.py",
    "p2/a04_svsite.py",
    "p2/a05_try_nameerr.py",
    "p2/a06_usebeforedef.py",
    "p2/a07_nested_ifdef.py",
    "p2/a08_nested_import.py",
    "p2/a09_globalrebind.py",
    "p2/a10_boolor.py",
    "p2/a11_booland.py",
    "p2/a12_isinst_bool.py",
    "p2/a13_boolarith.py",
    "p2/a14_annassign_shadow.py",
    "p2/a15_walrus_shadow.py",
    "p2/a16_forshadow.py",
    "p2/a17_nestedassign.py",
    "p2/a19_minmax3.py",
    "p2/a20_mod_neg.py",
    "p2/a21_star_import.py",
    "p2/a22_builtins_attr.py",
    "p2/a23_fn_local_shadow.py",
    "p2/a25_del.py",
    "p2/a26_nested_class_in_fn.py",
    "p2/a27_svverifier_attr.py",
    "p2/a28_from_import_alias.py",
    "p2/a29_ifexp_shadow.py",
    "p2/a30_divbool.py",
    "p2/a31_global_read_late.py",
    "p2/a32_svdraw_name.py",
    "p2/a33_svfoo_inline.py",
    "p2/a34_sys_exit.py",
]


@pytest.mark.parametrize("rel", _VIOLATING)
def test_mg_attack_corpus_is_never_true(rel):
    path = os.path.join(_ATTACKS, rel)
    _never_true(path)


# ---------------------------------------------------------------------------
# wired into the real pipeline (tharkun.sv.verify, scheduler.py stage 1b)
# ---------------------------------------------------------------------------

def test_verify_end_to_end_true_via_symbolic_product(tmp_path):
    p = _prog(tmp_path, "m4_e2e.py", _TOY_F + """
        def f_mut(x, y):
            d = x - y
            a = abs(d)
            if a > 10:
                return a >> 1
            return 0


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = verify(p, AR, python=sys.executable, budget_seconds=60, run_timeout=30,
              canary_seconds=3, early_stop_seconds=3, witness_dir=tmp_path / "proofs")
    assert v.verdict == "true"
    assert v.component == "symbolic:product"


def test_verify_end_to_end_false_via_symbolic_product(tmp_path):
    p = _prog(tmp_path, "m1_e2e.py", _TOY_F + """
        def f_mut(x, y):
            d = x + y
            a = abs(d)
            if a > 10:
                return a // 2
            return 0


        x = _sv_verifier.nondet_int()
        y = _sv_verifier.nondet_int()
        assert f(x, y) == f_mut(x, y)
    """)
    v = verify(p, AR, python=sys.executable, budget_seconds=60, run_timeout=30,
              canary_seconds=3, early_stop_seconds=3, witness_dir=tmp_path / "proofs")
    assert v.verdict == "false"
    assert v.component == "symbolic:product"
    assert v.exception == "AssertionError"


def test_a_huge_shift_amount_refuses_instead_of_computing_2_to_the_huge(tmp_path):
    # symbolic x << HUGE and a concrete 1 << HUGE (via a name): both must be UNKNOWN,
    # promptly, not a MemoryError caught after a memory spike
    head = "import _sv_verifier\nx = _sv_verifier.nondet_int()\n"
    for name, body in [("sym", "a = x << 1000000000000\n"),
                       ("conc", "k = 1000000000000\na = 1 << k\n"),
                       ("rsh", "a = x >> 1000000000000\n")]:
        p = _prog(tmp_path, f"shift_{name}.py", head + body + "assert a != 3\n")
        assert _prove(p) is None, name
    ok = _prog(tmp_path, "shift_ok.py", head + "a = x << 4\nassert a != 3\n")
    v = _prove(ok)
    assert v is not None and v.verdict == "true"


# ---------------------------------------------------------------------------
# [internal ref]: draws INSIDE the functions the program calls (``def main(): x = nondet_int(); ...``)
# ---------------------------------------------------------------------------

_MAIN = """
        import _sv_verifier

        def main():
            x = _sv_verifier.nondet_int()
            if not (1 <= x <= 100):
                return
            assert x * x {rel} x

        main()
"""


def test_an_early_return_guard_then_assert_in_main_is_proved(tmp_path):
    v = _prove(_prog(tmp_path, "main_true.py", _MAIN.format(rel=">=")))
    assert v is not None and v.verdict == "true" and v.component == "symbolic:product"


def test_the_same_program_with_one_value_that_breaks_it_is_never_true(tmp_path):
    # x == 1 breaks ``x * x > x``; a draw inside main has no single vector, so this stage is silent (None)
    # and the search stages own the FALSE -- but it is never TRUE
    p = _prog(tmp_path, "main_false.py", _MAIN.format(rel=">"))
    assert _prove(p) is None
    v = verify(p, AR, python=sys.executable, budget_seconds=60, run_timeout=30, canary_seconds=3,
               early_stop_seconds=3, witness_dir=tmp_path / "proofs")
    assert v.verdict == "false"


def test_the_main_guard_is_true_for_the_program_the_engine_runs(tmp_path):
    p = _prog(tmp_path, "guard.py", _MAIN.replace("\n        main()\n", '\n        if __name__ == "__main__":\n'
                                                  '            main()\n').format(rel=">="))
    v = _prove(p)
    assert v is not None and v.verdict == "true"


def test_a_main_guard_with_an_else_proves_the_branch_that_runs(tmp_path):
    p = _prog(tmp_path, "guard_else.py", """
        import _sv_verifier

        x = _sv_verifier.nondet_int()
        if __name__ == "__main__":
            assert x == x
        else:
            assert False
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true"


def test_a_name_comparison_other_than_the_main_guard_is_not_assumed(tmp_path):
    p = _prog(tmp_path, "not_main.py", """
        import _sv_verifier

        x = _sv_verifier.nondet_int()
        if __name__ == "other":
            assert x == x
    """)
    assert _prove(p) is None


def test_chained_comparison_over_total_operands_is_exact(tmp_path):
    ok = _prog(tmp_path, "chain_ok.py", """
        import _sv_verifier

        def main():
            w = _sv_verifier.nondet_int()
            if not (0 <= w <= 6):
                return
            r = w < 5
            assert r == (w < 5)
            assert 0 <= w * 2 - w < 7

        main()
    """)
    v = _prove(ok)
    assert v is not None and v.verdict == "true"
    bad = _prog(tmp_path, "chain_bad.py", """
        import _sv_verifier

        def main():
            w = _sv_verifier.nondet_int()
            if not (0 <= w <= 6):
                return
            assert 0 <= w <= 5

        main()
    """)
    assert _prove(bad) is None                    # w == 6 breaks it: never TRUE
    assert not _native_holds(bad)


def _native_holds(path):
    """Whether the program, run natively with every draw in a small window, never raises AssertionError."""
    import types
    src = open(path).read()
    for draw in range(-3, 10):
        stub = types.ModuleType("_sv_verifier")
        stub.nondet_int = lambda d=draw: d
        old = sys.modules.get("_sv_verifier")
        sys.modules["_sv_verifier"] = stub
        try:
            exec(compile(src, path, "exec"), {"__name__": "__main__"})
        except AssertionError:
            return False
        finally:
            if old is None:
                sys.modules.pop("_sv_verifier", None)
            else:
                sys.modules["_sv_verifier"] = old
    return True


def test_a_procedure_returns_early_or_falls_off_its_end(tmp_path):
    p = _prog(tmp_path, "procedure.py", """
        import _sv_verifier

        def check(v: int) -> None:
            if v > 10:
                return
            assert v <= 10

        def main() -> None:
            x: int = _sv_verifier.nondet_int()
            check(x)
            check(x + 1)

        main()
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true"


def test_the_value_of_a_procedure_is_never_used(tmp_path):
    # f() returns None; ``1 + f()`` is a TypeError natively, and ``x = f()`` makes a None a later operand:
    # a def with no return value is only ever called as a statement
    for name, body in [("add", "y = 1 + f()"), ("assign", "y = f()"), ("arg", "y = abs(f())")]:
        p = _prog(tmp_path, f"proc_{name}.py", f"""
            import _sv_verifier

            def f():
                pass

            x = _sv_verifier.nondet_int()
            {body}
            assert x == x
        """)
        assert _prove(p) is None, name


def test_a_def_that_returns_a_value_on_some_paths_only_is_refused(tmp_path):
    p = _prog(tmp_path, "some_paths.py", """
        import _sv_verifier

        def f(a):
            if a > 0:
                return 1

        x = _sv_verifier.nondet_int()
        assert f(x) == 1 or x <= 0
    """)
    assert _prove(p) is None


def test_a_draw_in_a_called_function_is_fresh_at_every_call(tmp_path):
    # g() draws once per CALL: ``a != b`` is possible, so ``assert a == b`` is violated.  This is the NEGATIVE
    # CONTROL of the fresh draw: binding one term per draw statement (what _fresh_draw replaces) proves it TRUE.
    src = """
        import _sv_verifier

        def g():
            v = _sv_verifier.nondet_int()
            return v

        def main():
            a = g()
            b = g()
            assert a == b

        main()
    """
    p = _prog(tmp_path, "twice.py", src)
    assert _prove(p) is None
    from tharkun.sv import _symexec

    memo = {}
    orig = _symexec._Runner._fresh_draw

    def one_term_per_statement(self, kind):
        return memo.setdefault(kind, _symexec._var("shared", kind))

    _symexec._Runner._fresh_draw = one_term_per_statement
    try:
        wrong = _prove(p)
    finally:
        _symexec._Runner._fresh_draw = orig
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"
    assert _prove(p) is None


def test_a_draw_in_a_branch_of_a_called_function(tmp_path):
    p = _prog(tmp_path, "branch_draw.py", """
        import _sv_verifier

        def pick(c: bool) -> int:
            if c:
                return _sv_verifier.nondet_int()
            v = _sv_verifier.nondet_int()
            return v * 0

        def main():
            c = _sv_verifier.nondet_bool()
            r = pick(c)
            if not c:
                assert r == 0

        main()
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true"


def test_excluded_draw_kinds_and_draw_arguments_stay_refused_inside_a_function(tmp_path):
    for name, call in [("float", "_sv_verifier.nondet_float()"), ("str", "_sv_verifier.nondet_str()"),
                       ("range", "_sv_verifier.nondet_float_range(0.0, 1.0)")]:
        p = _prog(tmp_path, f"kind_{name}.py", f"""
            import _sv_verifier

            def main():
                x = {call}
                assert x == x

            main()
        """)
        assert _prove(p) is None, name


def test_a_draw_at_module_level_inside_an_if_is_still_refused(tmp_path):
    p = _prog(tmp_path, "if_draw.py", """
        import _sv_verifier

        c = _sv_verifier.nondet_int()
        if c > 0:
            y = _sv_verifier.nondet_int()
            assert y == y
    """)
    assert _prove(p) is None


def test_unbounded_recursion_through_a_drawing_function_is_unsupported(tmp_path):
    p = _prog(tmp_path, "recur.py", """
        import _sv_verifier

        def f(n):
            x = _sv_verifier.nondet_int()
            if x > n:
                return 0
            return f(n + 1)

        assert f(0) == 0
    """)
    assert _prove(p) is None


def test_int_and_object_annotations_are_inert(tmp_path):
    p = _prog(tmp_path, "annot.py", """
        import _sv_verifier

        BOUND = 1 << 20

        def widen(_cfg: object, n: int) -> int:
            return int(n * 2)

        def main() -> None:
            cfg: object = 0
            n: int = _sv_verifier.nondet_int()
            if not (0 <= n <= BOUND):
                return
            m = widen(cfg, n)
            assert m == n + n

        main()
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true"


def test_an_annotation_that_runs_code_is_refused(tmp_path):
    for name, ann in [("call", "f()"), ("attr", "int.real"), ("name", "Foo")]:
        p = _prog(tmp_path, f"ann_{name}.py", f"""
            import _sv_verifier

            def f():
                return int

            def g(a: {ann}):
                return a

            x = _sv_verifier.nondet_int()
            assert g(x) == x
        """)
        assert _prove(p) is None, name


def test_verify_end_to_end_true_for_the_main_shape(tmp_path):
    p = _prog(tmp_path, "e2e_main.py", _MAIN.format(rel=">="))
    v = verify(p, AR, python=sys.executable, budget_seconds=60, run_timeout=30,
               canary_seconds=3, early_stop_seconds=3, witness_dir=tmp_path / "proofs")
    assert v.verdict == "true" and v.component == "symbolic:product"


def test_rebinding_dunder_name_is_never_assumed_main(tmp_path):
    # the guard is only read as true when nothing in the program can change ``__name__``
    p = _prog(tmp_path, "rebind_name.py", """
        import _sv_verifier

        x = _sv_verifier.nondet_int()
        __name__ = "other"
        if __name__ == "__main__":
            assert False
    """)
    assert _prove(p) is None


def test_a_program_variable_named_like_a_fresh_draw_is_a_different_term(tmp_path):
    # the fresh draws of a def are named by the interpreter; a top-level draw is named by the PROGRAM.  ``draw1`` and
    # ``draw2`` are what a naive counter would call the first two fresh terms: if they were the same terms, the
    # draws would be forced equal and this violating program (a != b) would be proved
    p = _prog(tmp_path, "named_like.py", """
        import _sv_verifier

        draw1 = _sv_verifier.nondet_int()
        draw2 = _sv_verifier.nondet_int()

        def g():
            return _sv_verifier.nondet_int()

        def main():
            assert g() + g() == draw1 + draw2

        main()
    """)
    assert _prove(p) is None
    q = _prog(tmp_path, "named_like2.py", """
        import _sv_verifier

        draw1 = _sv_verifier.nondet_int()

        def g():
            return _sv_verifier.nondet_int()

        def main():
            assert g() == draw1

        main()
    """)
    assert _prove(q) is None                     # a fresh draw differs from draw1 whenever it likes


def test_the_naming_control_fails_the_way_a_collision_would(tmp_path):
    # NEGATIVE CONTROL: name the fresh terms ``draw<n>`` (a name a program can use) and the same program is wrongly TRUE
    from tharkun.sv import _symexec

    p = _prog(tmp_path, "collide.py", """
        import _sv_verifier

        draw1 = _sv_verifier.nondet_int()

        def g():
            return _sv_verifier.nondet_int()

        def main():
            assert g() == draw1

        main()
    """)
    orig = _symexec._Runner._fresh_draw

    def collides(self, kind):
        self.draw_count += 1
        return _symexec._var(f"draw{self.draw_count}", kind)

    _symexec._Runner._fresh_draw = collides
    try:
        wrong = _prove(p)
    finally:
        _symexec._Runner._fresh_draw = orig
    assert wrong is not None and wrong.verdict == "true"
    assert _prove(p) is None


def test_the_proof_assumes_the_child_runs_the_program_as_main(tmp_path):
    """``_normalize`` reads ``if __name__ == "__main__":`` as true and drops its ``else``: right only if the program RUNS as
    ``__main__``.  The engine's child (``sv/_child.py``, ``runpy.run_path(program, run_name="__main__")``) does, for the search,
    the replay and the canary alike: the dead ``else`` below would fail an assertion, the proof says TRUE, and the same
    program run by the child reaches the guard's body (``y`` is bound) and never the ``else``."""
    from tharkun.sv import engine

    p = _prog(tmp_path, "guard_else.py", """
        import _sv_verifier

        x = _sv_verifier.nondet_int()
        if __name__ == "__main__":
            y = x + 1
        else:
            assert x != x
        assert y == x + 1
    """)
    v = product_proof.prove(open(p).read(), ("AssertionError",), program=p, python=sys.executable)
    assert v is not None and v.verdict == "true" and v.component == "symbolic:product"
    confirmed, out = engine.replay(p, {"vector": [{"kind": "int", "value": 3}]}, ("AssertionError",), python=sys.executable)
    assert not confirmed and out.status == "clean" and out.violated(("AssertionError",)) is None
    # a guard that is NOT the main guard is not rewritten (``__name__`` is not an audited name): the proof refuses it
    q = _prog(tmp_path, "guard_other.py", """
        import _sv_verifier

        x = _sv_verifier.nondet_int()
        if __name__ != "__main__":
            assert x != x
    """)
    assert product_proof.prove(open(q).read(), ("AssertionError",), program=q, python=sys.executable) is None


def test_the_product_proof_z3_timeout_never_exceeds_what_remains():
    z = product_proof.z3_timeout_ms
    assert z(9.5) == 9500                        # a 10 s triage run: not the 10 s floor
    assert z(3.0) == 3000
    assert z(30.0) == 10000                      # the floor, when it fits
    assert z(100.0) == 15000                     # 15% of the remaining budget
    assert z(1000.0) == 60000                    # the cap
