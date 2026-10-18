"""[internal ref]: the exact subset of the product proof grows, one construct at a time (tuples first), and every construct is
pinned three ways:

* a NATIVE differential on a finite domain checked EXHAUSTIVELY (every draw vector of a small domain, not a sample): for each
  hand-written program the proof says TRUE exactly when no vector makes the real Python program raise ``AssertionError``;
* the REFUSALS: every shape the audit does not model exactly comes back ``None``, by name in ``audit``;
* a NEGATIVE CONTROL: the one mutation of the interpreter that would make the construct unsound is applied and the program
  that exposes it must come back wrongly TRUE (so the differential above would catch the bug), then the fix is restored.

``exhaustive_native`` is the ground truth: it runs the UNMODIFIED program with every vector of the domains.  Nothing here is
random.
"""
from __future__ import annotations

import ast
import itertools
import sys
import textwrap
import types

import pytest

pytest.importorskip("z3")

from tharkun.sv import _symexec, product_proof  # noqa: E402
from tharkun.sv.exact_subset import audit  # noqa: E402

AR = ("AssertionError",)


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def _prove(path):
    return product_proof.prove(open(path).read(), AR, program=path, python=sys.executable)


def audit_reason(src):
    """Why the audit refuses ``src`` (the draws are marked the way ``product_proof._lift`` marks them), or ``None``."""
    tree = ast.parse(src)
    product_proof._normalize(tree)
    aliases, direct = product_proof._nondet_aliases(tree), product_proof._direct_imports(tree)
    lifted = {id(st) for st in tree.body if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call)
              and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
              and product_proof._draw_kind(st.value, aliases, direct)}
    fresh = {id(n) for st in tree.body if isinstance(st, ast.FunctionDef) for n in ast.walk(st)
             if isinstance(n, ast.Call) and product_proof._draw_kind(n, aliases, direct)}
    return audit(tree, lifted, frozenset(fresh))


def exhaustive_native(path, *domains):
    """``True`` when the program, run natively with EVERY vector of ``domains`` (one iterable per draw, in the order the draws
    are consumed; a ``bool`` domain feeds ``nondet_bool``), never ends in ``AssertionError``.  Any other exception is the
    program's own business (a ``ZeroDivisionError`` or ``IndexError`` ends that run, it is not the property)."""
    src = open(path).read()
    code = compile(src, path, "exec")
    for vector in itertools.product(*domains):
        feed = iter(vector)
        stub = types.ModuleType("_sv_verifier")
        stub.nondet_int = lambda feed=feed: next(feed)
        stub.nondet_bool = lambda feed=feed: bool(next(feed))
        old = sys.modules.get("_sv_verifier")
        sys.modules["_sv_verifier"] = stub
        try:
            exec(code, {"__name__": "__main__"})
        except AssertionError:
            return False
        except Exception:  # noqa: BLE001 -- the program's own exception: not the property
            pass
        finally:
            if old is None:
                sys.modules.pop("_sv_verifier", None)
            else:
                sys.modules["_sv_verifier"] = old
    return True


def agrees_with_native(tmp_path, name, body, *domains):
    """The proof's answer against the exhaustive native one: TRUE iff native never asserts, and never TRUE otherwise."""
    p = _prog(tmp_path, name, body)
    native_ok = exhaustive_native(p, *domains)
    v = _prove(p)
    if native_ok:
        return v is not None and v.verdict == "true"
    assert v is None or v.verdict != "true", f"{name}: a wrong TRUE (native raises AssertionError)"
    return True


# ---------------------------------------------------------------------------
# tuples ([internal ref] PR 1): a tuple function, unpacked into names
# ---------------------------------------------------------------------------

_VOTE = """
    import _sv_verifier

    def record(no_votes, b_request, b_respond):
        if no_votes:
            {early}
        changed = False
        written = 0
        if b_request:
            written += 1
            changed = True
        if b_respond:
            written += 1
            {respond}
        return {ret}

    def main():
        no_votes = _sv_verifier.nondet_bool()
        b_request = _sv_verifier.nondet_bool()
        b_respond = _sv_verifier.nondet_bool()
        changed, written = record(no_votes, b_request, b_respond)
        assert {claim}

    main()
"""

_BOOLS = ([False, True],) * 3

#: (name, early return, respond arm, return, claim): the correct program and the mutants that each break one thing
_VARIANTS = [
    ("correct", "return False, 0", "changed = True", "changed, written", "changed == (written >= 1)"),
    ("early_arm_wrong", "return True, 0", "changed = True", "changed, written", "changed == (written >= 1)"),
    ("respond_forgets_changed", "return False, 0", "pass", "changed, written", "changed == (written >= 1)"),
    ("elements_swapped", "return False, 0", "changed = True", "written, changed", "changed == (written >= 1)"),
    ("claim_off_by_one", "return False, 0", "changed = True", "changed, written", "changed == (written >= 2)"),
    ("early_arm_nonzero_written", "return False, 3", "changed = True", "changed, written", "changed == (written >= 1)"),
    ("claim_on_written_only", "return False, 0", "changed = True", "changed, written", "written <= 2"),
    ("early_arm_makes_claim_false", "return True, 0", "changed = True", "changed, written", "written <= 2"),
]


@pytest.mark.parametrize("name,early,respond,ret,claim", _VARIANTS, ids=[v[0] for v in _VARIANTS])
def test_a_tuple_function_agrees_with_the_native_program_on_the_whole_domain(tmp_path, name, early, respond, ret, claim):
    body = _VOTE.format(early=early, respond=respond, ret=ret, claim=claim)
    assert agrees_with_native(tmp_path, name + ".py", body, *_BOOLS)


def test_the_correct_tuple_program_is_proved_true_and_a_mutant_never_is(tmp_path):
    ok = _prove(_prog(tmp_path, "ok.py", _VOTE.format(early="return False, 0", respond="changed = True",
                                                      ret="changed, written", claim="changed == (written >= 1)")))
    assert ok is not None and ok.verdict == "true" and ok.component == "symbolic:product"
    bad = _prog(tmp_path, "bad.py", _VOTE.format(early="return False, 0", respond="pass",
                                                  ret="changed, written", claim="changed == (written >= 1)"))
    v = _prove(bad)
    assert v is None or v.verdict != "true"


def test_a_tuple_of_ints_over_a_bounded_draw(tmp_path):
    # int elements and a path-dependent second arm; the domain is the precondition's own interval, so the native run is exhaustive
    body = """
        import _sv_verifier

        def split(total, parts):
            if parts <= 0:
                return 0, total
            each = total // parts
            return each, total - each * parts

        def main():
            total = _sv_verifier.nondet_int()
            parts = _sv_verifier.nondet_int()
            if not (0 <= total <= 20 and -2 <= parts <= 6):
                return
            each, rest = split(total, parts)
            assert each * parts + rest == total
            assert 0 <= rest
            assert parts <= 0 or rest < parts

        main()
    """
    assert agrees_with_native(tmp_path, "split.py", body, range(-1, 22), range(-3, 8))
    v = _prove(_prog(tmp_path, "split2.py", body))
    assert v is not None and v.verdict == "true"
    broken = body.replace("return each, total - each * parts", "return each, total - each * parts + 1")
    assert agrees_with_native(tmp_path, "split3.py", broken, range(-1, 22), range(-3, 8))


def test_a_tuple_dropped_by_a_bare_call_statement_is_fine(tmp_path):
    body = """
        import _sv_verifier

        def pair(a):
            return a, a + 1

        def main():
            x = _sv_verifier.nondet_int()
            if not (0 <= x <= 5):
                return
            pair(x)
            assert x >= 0

        main()
    """
    v = _prove(_prog(tmp_path, "drop.py", body))
    assert v is not None and v.verdict == "true"


# ---------------------------------------------------------------------------
# the audit refuses every other use of a tuple, by name
# ---------------------------------------------------------------------------

_HEAD = "import _sv_verifier\n\ndef f(a):\n    return a, a + 1\n\n"


@pytest.mark.parametrize("name,src,why", [
    ("stored", _HEAD + "x = _sv_verifier.nondet_int()\nt = f(x)\nassert x == x\n", "used as a value"),
    ("indexed", _HEAD + "x = _sv_verifier.nondet_int()\ny = f(x)[0]\nassert x == x\n", "subscript of something that is not a list"),
    ("compared", _HEAD + "x = _sv_verifier.nondet_int()\nassert f(x) == f(x)\n", "used as a value"),
    ("operand", _HEAD + "x = _sv_verifier.nondet_int()\ny = f(x) + f(x)\nassert x == x\n", "used as a value"),
    ("passed", _HEAD + "def g(t):\n    return t\n\nx = _sv_verifier.nondet_int()\ny = g(f(x))\nassert x == x\n", "used as a value"),
    ("arity_mismatch", _HEAD + "x = _sv_verifier.nondet_int()\na, b, c = f(x)\nassert x == x\n", "tuple function of that arity"),
    ("unpack_a_non_tuple_function",
     "import _sv_verifier\n\ndef g(a):\n    return a\n\nx = _sv_verifier.nondet_int()\na, b = g(x)\nassert x == x\n",
     "tuple function of that arity"),
    ("swap", "import _sv_verifier\n\nx = _sv_verifier.nondet_int()\na = x + 1\nb = x\na, b = b, a\nassert x == x\n",
     "tuple function of that arity"),
    ("duplicate_names", _HEAD + "x = _sv_verifier.nondet_int()\na, a = f(x)\nassert x == x\n", "distinct plain names"),
    ("starred_target", _HEAD + "x = _sv_verifier.nondet_int()\na, *b = f(x)\nassert x == x\n", "distinct plain names"),
    ("nested_target", _HEAD + "x = _sv_verifier.nondet_int()\n(a, b), c = f(x)\nassert x == x\n", "distinct plain names"),
    ("mixed_returns", "import _sv_verifier\n\ndef g(a):\n    if a:\n        return 1, 2\n    return 3\n", "tuple on some paths only"),
    ("different_arities", "import _sv_verifier\n\ndef g(a):\n    if a:\n        return 1, 2\n    return 1, 2, 3\n", "different arity"),
    ("starred_element", "import _sv_verifier\n\ndef g(a):\n    return a, *a\n", "starred"),
    ("tuple_in_a_draw_position", "import _sv_verifier\n\na, b = _sv_verifier.nondet_int(), _sv_verifier.nondet_int()\nassert a == a\n",
     "tuple function of that arity"),
    ("tuple_literal_outside_return", "import _sv_verifier\n\nx = _sv_verifier.nondet_int()\nt = (x, x)\nassert x == x\n",
     "expression kind Tuple"),
])
def test_every_other_use_of_a_tuple_is_refused_by_name(name, src, why):
    reason = audit_reason(src)
    assert reason is not None and why in reason, (name, reason)


def test_a_returned_tuple_is_never_a_value_the_proof_computes_with(tmp_path):
    p = _prog(tmp_path, "bad_use.py", _HEAD + """
        x = _sv_verifier.nondet_int()
        assert f(x) != f(x + 1)
    """)
    assert _prove(p) is None


# ---------------------------------------------------------------------------
# negative control: a merge that keeps one arm's tuple is a wrong TRUE
# ---------------------------------------------------------------------------

def test_the_elementwise_merge_of_two_returned_tuples_is_what_keeps_the_proof_sound(tmp_path):
    p = _prog(tmp_path, "merge.py", """
        import _sv_verifier

        def pick(c):
            if c:
                return 1, 2
            return 3, 4

        def main():
            c = _sv_verifier.nondet_bool()
            a, b = pick(c)
            assert a == 1

        main()
    """)
    # sound: the second arm (c False) gives a == 3, so the assertion can fail: never TRUE
    assert exhaustive_native(p, [False, True]) is False
    v = _prove(p)
    assert v is None or v.verdict != "true"

    real = _symexec._ite

    def keeps_the_first_arm(c, a, b):
        if isinstance(a, tuple) and isinstance(b, tuple):
            return a                    # the bug: the merged tuple is the first arm's
        return real(c, a, b)

    _symexec._ite = keeps_the_first_arm
    try:
        wrong = _prove(p)
    finally:
        _symexec._ite = real
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"
    again = _prove(p)
    assert again is None or again.verdict != "true"


# ---------------------------------------------------------------------------
# for i in range(<constant>) ([internal ref] PR 2): unrolled exactly (sv/exact_unroll.py)
# ---------------------------------------------------------------------------

from tharkun.sv import exact_unroll  # noqa: E402

_SCHED = """
    import _sv_verifier

    K = 3

    def main():
        budget = _sv_verifier.nondet_int()
        if not (0 <= budget <= 3):
            return
        total = budget
        spent = 0
        for _i in range(K):
            e = _sv_verifier.nondet_int()
            if not (0 <= e <= 3):
                return
            if {guard}:
                budget = budget - e
                spent = spent + {spend}
            assert {claim}

    main()
"""

_SCHED_DOMAINS = (range(-1, 5), range(-1, 5), range(-1, 5), range(-1, 5))

#: (name, guard, spend, claim): the correct accounting and the mutants that each break one thing
_SCHED_VARIANTS = [
    ("correct", "e <= budget", "e", "budget >= 0 and budget + spent == total"),
    ("no_guard", "True", "e", "budget >= 0 and budget + spent == total"),
    ("spend_plus_one", "e <= budget", "e + 1", "budget >= 0 and budget + spent == total"),
    ("claim_strict", "e <= budget", "e", "budget > 0"),
    ("guard_lt", "e < budget", "e", "budget >= 0 and budget + spent == total"),
    ("claim_total_only", "e <= budget", "e", "spent <= total"),
]


@pytest.mark.parametrize("name,guard,spend,claim", _SCHED_VARIANTS, ids=[v[0] for v in _SCHED_VARIANTS])
def test_a_constant_range_loop_agrees_with_the_native_program_on_the_whole_domain(tmp_path, name, guard, spend, claim):
    body = _SCHED.format(guard=guard, spend=spend, claim=claim)
    assert agrees_with_native(tmp_path, name + ".py", body, *_SCHED_DOMAINS)


def test_the_correct_loop_is_proved_true(tmp_path):
    v = _prove(_prog(tmp_path, "sched.py", _SCHED.format(guard="e <= budget", spend="e",
                                                         claim="budget >= 0 and budget + spent == total")))
    assert v is not None and v.verdict == "true" and v.component == "symbolic:product"


@pytest.mark.parametrize("name,loop,claim", [
    ("sum_and_last_value", "for i in range(4):\n    total = total + i", "total == 6 and i == 3"),
    ("target_rebound_in_the_body", "for i in range(3):\n    i = i + 10\n    total = total + i", "total == 33 and i == 12"),
    ("start_stop", "for i in range(2, 5):\n    total = total + i", "total == 9 and i == 4"),
    ("step", "for i in range(2, 9, 3):\n    total = total + i", "total == 15 and i == 8"),
    ("negative_step", "for i in range(5, 0, -2):\n    total = total + i", "total == 9 and i == 1"),
    ("empty_range_leaves_the_target", "i = 7\nfor i in range(3, 3):\n    total = total + 1", "total == 0 and i == 7"),
    ("an_expression_of_constants", "for i in range(1 << 2):\n    total = total + 1", "total == 4"),
    ("nested", "for i in range(3):\n    for j in range(2):\n        total = total + i * 10 + j", "total == 63 and i == 2 and j == 1"),
    ("a_wrong_claim", "for i in range(4):\n    total = total + i", "total == 7"),
    ("a_wrong_last_value", "for i in range(4):\n    total = total + i", "i == 4"),
])
def test_loop_results_match_native_python(tmp_path, name, loop, claim):
    body = ("import _sv_verifier\n\ndef main():\n    x = _sv_verifier.nondet_int()\n    if not (0 <= x <= 1):\n        return\n"
            "    total = 0\n" + textwrap.indent(loop, "    ") + f"\n    assert {claim}\n\nmain()\n")
    assert agrees_with_native(tmp_path, name + ".py", body, range(-1, 3))


def test_an_early_return_inside_the_loop_ends_the_function_not_the_loop(tmp_path):
    body = """
        import _sv_verifier

        def scan(limit):
            seen = 0
            for i in range(5):
                if i == limit:
                    return seen
                seen = seen + 1
            return seen + 100

        def main():
            limit = _sv_verifier.nondet_int()
            if not (-2 <= limit <= 7):
                return
            r = scan(limit)
            assert (0 <= limit <= 4 and r == limit) or ((limit < 0 or limit > 4) and r == 105)

        main()
    """
    assert agrees_with_native(tmp_path, "early.py", body, range(-3, 9))
    v = _prove(_prog(tmp_path, "early2.py", body))
    assert v is not None and v.verdict == "true"
    wrong = body.replace("r == 105", "r == 104")
    assert agrees_with_native(tmp_path, "early3.py", wrong, range(-3, 9))


def test_each_iteration_draws_a_fresh_value(tmp_path):
    body = """
        import _sv_verifier

        def main():
            first = 0
            for i in range(2):
                x = _sv_verifier.nondet_int()
                if i == 0:
                    first = x
                else:
                    assert x == first

        main()
    """
    p = _prog(tmp_path, "fresh.py", body)
    assert exhaustive_native(p, range(0, 3), range(0, 3)) is False        # the two draws differ
    v = _prove(p)
    assert v is None or v.verdict != "true"


# ---- refusals: every loop that is not exactly unrollable stays a `for` and is refused by name -------------------------------

_LOOP = "import _sv_verifier\n\nK = 3\n\n"


@pytest.mark.parametrize("name,src", [
    ("bound_is_a_draw", _LOOP + "def main():\n    n = _sv_verifier.nondet_int()\n    for i in range(n):\n        pass\n\nmain()\n"),
    ("bound_is_a_parameter_passed_two_constants", _LOOP + "def f(n):\n    for i in range(n):\n        pass\n\nf(2)\nf(3)\n"),
    ("bound_is_a_parameter_passed_a_draw", _LOOP + "def f(n):\n    for i in range(n):\n        pass\n\nx = _sv_verifier.nondet_int()\nf(x)\n"),
    ("bound_is_len", _LOOP + "def f(n):\n    for i in range(len(n)):\n        pass\n\nf(2)\n"),
    ("bound_is_a_float", _LOOP + "for i in range(2.0):\n    pass\n"),
    ("bound_is_a_bool", _LOOP + "for i in range(True):\n    pass\n"),
    ("keyword_argument", _LOOP + "for i in range(K, step=1):\n    pass\n"),
    ("zero_step", _LOOP + "for i in range(0, 3, 0):\n    pass\n"),
    ("too_many_trips", _LOOP + "for i in range(65):\n    pass\n"),
    ("tuple_target", _LOOP + "for i, j in range(3):\n    pass\n"),
    ("iterating_a_name", _LOOP + "for i in K:\n    pass\n"),
    ("for_else", _LOOP + "for i in range(3):\n    pass\nelse:\n    pass\n"),
    ("break", _LOOP + "def f():\n    for i in range(3):\n        if i == 1:\n            break\n\nf()\n"),
    ("continue", _LOOP + "def f():\n    for i in range(3):\n        if i == 1:\n            continue\n\nf()\n"),
    ("break_in_a_nested_loop_keeps_the_outer_one",
     _LOOP + "def f():\n    for i in range(3):\n        for j in range(3):\n            if j == 1:\n                break\n\nf()\n"),
    ("constant_rebound", "import _sv_verifier\n\nK = 3\nK = 4\nfor i in range(K):\n    pass\n"),
    ("constant_augmented", "import _sv_verifier\n\nK = 3\nK += 1\nfor i in range(K):\n    pass\n"),
    ("constant_shadowed_by_a_parameter", _LOOP + "def f(K):\n    for i in range(K):\n        pass\n\nf(2)\n"),
    ("constant_shadowed_by_a_local", _LOOP + "def f():\n    K = 2\n    for i in range(K):\n        pass\n\nf()\n"),
    ("constant_declared_global", _LOOP + "def f():\n    global K\n    for i in range(K):\n        pass\n\nf()\n"),
    ("constant_from_a_call", "import _sv_verifier\n\nK = abs(3)\nfor i in range(K):\n    pass\n"),
    ("constant_division_by_zero", "import _sv_verifier\n\nK = 3 // 0\nfor i in range(K):\n    pass\n"),
])
def test_a_for_loop_that_is_not_exactly_unrollable_is_refused(name, src):
    tree = ast.parse(src)
    product_proof._normalize(tree)
    assert any(isinstance(n, ast.For) for n in ast.walk(tree)), f"{name}: the loop was unrolled"
    reason = audit_reason(src)
    assert reason is not None, name
    if name not in ("constant_shadowed_by_a_local", "constant_declared_global"):     # another refusal may name the program first
        assert "for loop that is not range(<constant>)" in reason, (name, reason)


def test_a_while_loop_is_still_refused_by_name():
    reason = audit_reason("import _sv_verifier\n\ndef f():\n    n = 3\n    while n:\n        n = n - 1\n\nf()\n")
    assert reason is not None and "while loop" in reason


def test_the_trip_limit_is_the_documented_one(tmp_path):
    assert exact_unroll.MAX_TRIPS == 64
    ok = _prog(tmp_path, "sixty_four.py", "import _sv_verifier\n\nx = _sv_verifier.nondet_int()\nt = 0\nfor i in range(64):\n    t = t + 1\nassert t == 64\n")
    v = _prove(ok)
    assert v is not None and v.verdict == "true"


# ---- negative controls: the mutation of the unrolling that would be unsound ------------------------------------------------

def test_dropping_the_last_iteration_would_be_a_wrong_true(tmp_path):
    p = _prog(tmp_path, "last.py", """
        import _sv_verifier

        def main():
            x = _sv_verifier.nondet_int()
            if not (0 <= x <= 1):
                return
            for i in range(3):
                assert i != 2

        main()
    """)
    assert exhaustive_native(p, range(0, 2)) is False        # the last iteration (i == 2) fails the assertion
    v = _prove(p)
    assert v is None or v.verdict != "true"
    real = exact_unroll._trips
    exact_unroll._trips = lambda start, stop, step: real(start, stop, step)[:-1]       # the bug
    try:
        wrong = _prove(p)
    finally:
        exact_unroll._trips = real
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"
    again = _prove(p)
    assert again is None or again.verdict != "true"


def test_unrolling_in_the_wrong_order_would_be_a_wrong_true(tmp_path):
    p = _prog(tmp_path, "order.py", """
        import _sv_verifier

        def main():
            x = _sv_verifier.nondet_int()
            if not (0 <= x <= 1):
                return
            acc = 0
            for i in range(3):
                acc = acc * 2 + i
            assert acc != 4

        main()
    """)
    assert exhaustive_native(p, range(0, 2)) is False        # in order: ((0*2+0)*2+1)*2+2 == 4
    v = _prove(p)
    assert v is None or v.verdict != "true"
    real = exact_unroll._trips
    exact_unroll._trips = lambda start, stop, step: real(start, stop, step)[::-1]      # the bug: 2, 1, 0 gives 10
    try:
        wrong = _prove(p)
    finally:
        exact_unroll._trips = real
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"


def test_a_loop_proof_is_the_end_to_end_answer_too(tmp_path):
    from tharkun.sv import verify

    p = _prog(tmp_path, "e2e.py", _SCHED.format(guard="e <= budget", spend="e", claim="budget >= 0 and budget + spent == total"))
    v = verify(p, AR, python=sys.executable, budget_seconds=60, run_timeout=30, canary_seconds=3, early_stop_seconds=3,
               witness_dir=tmp_path / "proofs")
    assert v.verdict == "true" and v.component == "symbolic:product"
    bad = _prog(tmp_path, "e2e_bad.py", _SCHED.format(guard="True", spend="e", claim="budget >= 0 and budget + spent == total"))
    v = verify(bad, AR, python=sys.executable, budget_seconds=60, run_timeout=30, canary_seconds=3, early_stop_seconds=3,
               witness_dir=tmp_path / "proofs2")
    assert v.verdict == "false"


# ---------------------------------------------------------------------------
# lists ([internal ref] PR 3): alias-free lists of ints / bools, indexed, measured with len, stored into, built by [x] * K / a literal /
# [e for i in range(K)]
# ---------------------------------------------------------------------------
# What makes them exact: a list of this subset has a fixed length known when it is built; every cell is an int or a bool; the
# only things done with a list NAME are ``xs[i]``, ``xs[i] = v``, ``xs[i] += v``, ``len(xs)`` and handing it to a def that only
# reads it, so no two names can ever hold the same list (aliasing is what makes a copy-on-write model wrong); and an index
# outside ``-len <= i < len`` ends the run (IndexError is not the property), a negative index counts from the end.

import z3  # noqa: E402

_POOL = """
    import _sv_verifier

    N = 4

    def main():
        i = _sv_verifier.nondet_int()
        j = _sv_verifier.nondet_int()
        used = [0] * N
        used[i] = 1
        used[j] += 1
        total = used[0] + used[1] + used[2] + used[3]
        assert {claim}

    main()
"""

_POOL_DOMAINS = (range(-5, 6), range(-5, 6))      # N == 4: -4..3 are cells, -5 / 4 / 5 end the run

#: (name, claim): the claims that hold on every run that does not end in IndexError, and the ones some vector breaks
_POOL_CLAIMS = [
    ("two_marks", "total == 2"),
    ("marked_cell_is_at_least_one", "used[i] >= 1 and used[j] >= 1"),
    ("marked_cell_is_exactly_one", "used[i] == 1"),                         # i == j makes it 2
    ("both_cells_are_one", "used[i] + used[j] == 2"),                      # i == j makes it 4
    ("three_marks", "total == 3"),
    ("length_is_n", "len(used) == N and total == 2"),
    ("last_by_negative_index", "used[-1] == used[N - 1] and used[-4] == used[0]"),
    ("same_cell_or_one", "used[i] == 1 or i == j or i + N == j or j + N == i"),
    ("negative_index_aliases_the_cell", "i != j or used[i] == 2"),
    ("cell_after_the_end_is_never_reached", "i < N and j < N and i >= -N and j >= -N"),
]


@pytest.mark.parametrize("name,claim", _POOL_CLAIMS, ids=[c[0] for c in _POOL_CLAIMS])
def test_list_cells_agree_with_the_native_program_on_the_whole_domain(tmp_path, name, claim):
    assert agrees_with_native(tmp_path, name + ".py", _POOL.format(claim=claim), *_POOL_DOMAINS)


def test_a_list_program_is_proved_true_by_the_product_proof(tmp_path):
    v = _prove(_prog(tmp_path, "pool.py", _POOL.format(claim="total == 2")))
    assert v is not None and v.verdict == "true" and v.component == "symbolic:product"


_PARAMS = """
    import _sv_verifier

    def weight(xs, w):
        total = 0
        for k in range(4):
            total = total + xs[k] * w
        return total

    def pick(xs: list[int], i: int) -> int:
        return xs[i]

    def count_set(flags) -> int:
        n = 0
        for k in range(3):
            if flags[k]:
                n = n + 1
        return n

    def size(xs) -> int:
        return len(xs)

    def main():
        a = _sv_verifier.nondet_int()
        b = _sv_verifier.nondet_int()
        c = _sv_verifier.nondet_int()
        if not (0 <= a <= 2 and 0 <= b <= 2):
            return
        cells = [a, b, a + b, 7]
        bits = [a > 0, b > 0, a == b]
        assert {claim}

    main()
"""

_PARAM_CLAIMS = [
    ("weighted_sum", "weight(cells, 2) == 4 * a + 4 * b + 14"),
    ("wrong_weighted_sum", "weight(cells, 2) == 4 * a + 4 * b + 13"),
    ("annotated_pick", "pick(cells, 2) == a + b and pick(cells, -1) == 7"),
    ("pick_out_of_range_ends_the_run", "pick(cells, c) <= 7"),                # c outside -4..3 ends it; a + b <= 4 < 7
    ("wrong_pick", "pick(cells, c) < 7"),
    ("count_flags", "count_set(bits) >= 1"),                                  # a == b makes bits[2] true, else one of a, b is > 0
    ("wrong_count_flags", "count_set(bits) >= 2"),                            # a == 0, b == 1: only bits[1]
    ("len_of_a_param", "size(bits) == 3 and size(cells) == 4 and count_set(cells) <= 3"),
    ("wrong_len_of_a_param", "size(bits) == 4"),
]
_PARAM_DOMAINS = (range(-1, 4), range(-1, 4), range(-5, 6))


@pytest.mark.parametrize("name,claim", _PARAM_CLAIMS, ids=[c[0] for c in _PARAM_CLAIMS])
def test_lists_passed_to_a_def_agree_with_native_python(tmp_path, name, claim):
    assert agrees_with_native(tmp_path, name + ".py", _PARAMS.format(claim=claim), *_PARAM_DOMAINS)


_BRANCHES = """
    import _sv_verifier

    def main():
        c = _sv_verifier.nondet_bool()
        i = _sv_verifier.nondet_int()
        flags = [False] * 3
        if c:
            flags[0] = True
        else:
            flags[1] = True
        flags[i] = True
        n = 0
        for k in range(3):
            if flags[k]:
                n = n + 1
        assert {claim}

    main()
"""


@pytest.mark.parametrize("claim", ["n >= 1", "n <= 2", "n == 2", "n == 1", "(n == 2) == (i != 0 and c or i != 1 and not c)",
                                   "flags[0] or flags[1] or flags[2]", "len(flags) == 3"])
def test_a_list_stored_on_one_branch_agrees_with_native_python(tmp_path, claim):
    body = _BRANCHES.format(claim=claim)
    assert agrees_with_native(tmp_path, "branches.py", body, (0, 1), range(-4, 5))


_TABLE = """
    import _sv_verifier

    TABLE = [3, 1, 4, 1, 5]

    def lookup(i):
        return TABLE[i]

    i = _sv_verifier.nondet_int()
    {use}
"""


@pytest.mark.parametrize("name,use", [
    ("no_nine", "assert lookup(i) != 9"),
    ("a_four_somewhere", "assert lookup(i) != 4"),
    ("from_the_end", "assert TABLE[i] != 5 or i == 4 or i == -1"),
    ("a_one_at_two_places", "assert lookup(i) != 1 or i == 1 or i == 3 or i == -4 or i == -2"),
    ("length", "assert len(TABLE) == 5 and lookup(i) >= 1"),
])
def test_a_module_level_table_read_in_a_def_agrees_with_native_python(tmp_path, name, use):
    assert agrees_with_native(tmp_path, name + ".py", _TABLE.format(use=use), range(-7, 8))


_COMP = """
    import _sv_verifier

    def main():
        xs = [_sv_verifier.nondet_int() for _ in range(3)]
        sq = [i * i + {shift} for i in range(5)]
        if xs[0] < 0 or xs[0] > 2 or xs[1] < 0 or xs[1] > 2 or xs[2] < 0 or xs[2] > 2:
            return
        assert {claim}

    main()
"""


@pytest.mark.parametrize("name,shift,claim", [
    ("squares", 0, "sq[3] == 9 and sq[4] == 16 and len(sq) == 5"),
    ("squares_wrong", 0, "sq[4] == 15"),
    ("shifted", 1, "sq[0] == 1 and sq[4] == 17"),
    ("the_draws_are_independent", 0, "xs[0] == xs[1]"),
    ("the_draws_are_independent_in_a_sum", 0, "xs[0] + xs[1] + xs[2] == 3 * xs[0]"),
    ("each_draw_is_in_its_own_range", 0, "xs[0] + xs[1] + xs[2] <= 6 and sq[2] == 4"),
])
def test_a_comprehension_over_a_constant_range_agrees_with_native_python(tmp_path, name, shift, claim):
    assert agrees_with_native(tmp_path, name + ".py", _COMP.format(shift=shift, claim=claim), range(0, 3), range(0, 3), range(0, 3))


def test_an_augmented_store_evaluates_its_index_once(tmp_path):
    """``xs[pick()] += 1`` reads and writes the SAME cell: the index draws once.  Evaluated twice it would read one fresh
    draw's cell and write another's, and the proof of the (true) conservation claim below would be lost."""
    body = """
        import _sv_verifier

        def pick():
            return _sv_verifier.nondet_int()

        def main():
            xs = [0, 10, 20]
            xs[pick()] += 1
            assert xs[0] + xs[1] + xs[2] == 31

        main()
    """
    assert exhaustive_native(_prog(tmp_path, "aug.py", body), range(-4, 5)) is True
    v = _prove(_prog(tmp_path, "aug2.py", body))
    assert v is not None and v.verdict == "true"
    wrong = body.replace("== 31", "== 30")
    assert agrees_with_native(tmp_path, "aug3.py", wrong, range(-4, 5))


def test_a_list_is_a_value_per_call_not_shared_between_calls(tmp_path):
    body = """
        import _sv_verifier

        def bump(n):
            cells = [0] * 3
            cells[n] += 1
            cells[n] += 1
            return cells[0] + cells[1] + cells[2]

        def main():
            n = _sv_verifier.nondet_int()
            m = _sv_verifier.nondet_int()
            assert bump(n) + bump(m) == {total}

        main()
    """
    for total, expected in ((4, True), (3, False)):
        p = _prog(tmp_path, f"calls{total}.py", body.format(total=total))
        assert exhaustive_native(p, range(-4, 4), range(-4, 4)) is expected
        v = _prove(p)
        assert (v is not None and v.verdict == "true") if expected else (v is None or v.verdict != "true")


# ---- refusals: what a list may NOT do stays out of the proof, by name ------------------------------------------------------

_LISTS = "import _sv_verifier\n\n"


def _in_main(body, pre=""):
    return _LISTS + pre + "def main():\n    x = _sv_verifier.nondet_int()\n" + textwrap.indent(textwrap.dedent(body), "    ") + "\nmain()\n"


@pytest.mark.parametrize("name,src,fragment", [
    ("alias_by_assignment", _in_main("xs = [0] * 3\nys = xs\nys[0] = 1\nassert xs[0] == 0"), "used as a value"),
    ("returning_a_list", _in_main("assert g()[0] == 0", "def g():\n    xs = [0] * 3\n    return xs\n\n"), "used as a value"),
    ("list_as_a_scalar_argument", _in_main("xs = [0] * 3\nassert f(xs) == 1", "def f(x):\n    return x + 1\n\n"), "used as a value"),
    ("list_plus_list", _in_main("xs = [0]\nys = [1]\nzs = xs + ys\nassert x == x"), "used as a value"),
    ("list_equality", _in_main("xs = [0]\nys = [0]\nassert xs == ys"), "used as a value"),
    ("list_truth", _in_main("xs = [0]\nassert xs"), "used as a value"),
    ("membership", _in_main("xs = [0, 1]\nassert x in xs"), "comparison shape"),
    ("slice", _in_main("xs = [0, 1]\nys = xs[0:1]\nassert x == x"), "slice or tuple subscript"),
    ("tuple_subscript", _in_main("xs = [0, 1]\nassert xs[0, 1] == 0"), "slice or tuple subscript"),
    ("nested_list", _in_main("xs = [[0], [1]]\nassert xs[0][0] == 0"), "expression kind List"),
    ("append", _in_main("xs = [0]\nxs.append(1)\nassert len(xs) == 2"), "call shape"),
    ("size_is_a_draw", _in_main("xs = [0] * x\nassert len(xs) == 0"), "list repetition whose count is not a constant"),
    ("size_is_too_big", _in_main("xs = [0] * 257\nassert len(xs) == 257"), "list repetition whose count is not a constant"),
    ("negative_size_constant", _in_main("xs = [0] * -1\nassert len(xs) == 0"), "list repetition whose count is not a constant"),
    ("size_from_a_rebound_constant", "import _sv_verifier\n\nK = 3\nK = 4\ndef main():\n    xs = [0] * K\n    assert len(xs) == 4\n\nmain()\n",
     "list repetition whose count is not a constant"),
    ("store_into_a_parameter", _in_main("xs = [0] * 2\nf(xs)\nassert xs[0] == 0", "def f(ys):\n    ys[0] = 1\n\n"),
     "not a list local to this scope"),
    ("store_into_a_module_list", "import _sv_verifier\n\nT = [0, 1]\n\ndef f():\n    T[0] = 5\n\nf()\nassert T[0] == 0\n",
     "not a list local to this scope"),
    ("store_into_a_scalar", _in_main("n = 3\nn[0] = 1"), "not a list local to this scope"),
    ("list_rebound_by_a_scalar", _in_main("xs = [0] * 2\nxs = 5\nassert x == x"), "holds a list and is assigned something else"),
    ("list_parameter_rebound", _in_main("assert f([1]) == 1", "def f(ys):\n    t = ys[0]\n    ys = [t]\n    return ys[0]\n\n"),
     "list parameter ys is rebound"),
    ("iteration_over_a_list", _in_main("xs = [0] * 2\nfor v in xs:\n    pass\nassert x == x"), "for loop that is not range"),
    ("augmented_list", _in_main("xs = [0]\nxs += [1]\nassert x == x"), "augmented assignment to a list"),
    ("delete", _in_main("xs = [0, 1]\ndel xs[0]\nassert x == x"), "statement kind Delete"),
    ("len_of_a_scalar", _in_main("assert len(x) == 0"), "len of something that is not a list"),
    ("len_of_a_call", _in_main("assert len(g()) == 0", "def g():\n    return 1\n\n"), "len of something that is not a list"),
    ("subscript_of_a_scalar", _in_main("assert x[0] == 0"), "subscript of something that is not a list"),
    ("annotation_without_a_list_argument", _in_main("assert f(3) == 3", "def f(xs: list[int]) -> int:\n    return xs[0]\n\n"),
     "must be a list"),
    ("annotation_of_another_element_type", _in_main("assert f([3]) == 3", "def f(xs: list[str]) -> int:\n    return 1\n\n"),
     "annotation other than"),
    ("list_annotation_on_a_scalar_parameter", _in_main("assert f(3) == 3", "def f(xs: list[int]) -> int:\n    return 1\n\n"),
     "must be a list"),
    ("unpacking_into_a_list_name", _in_main("xs = [0]\nxs, y = 1, 2\nassert x == x"), "tuple assignment into a name that holds a list"),
    ("filtered_comprehension", _in_main("xs = [i for i in range(3) if i]\nassert x == x"), "expression kind ListComp"),
    ("comprehension_over_a_draw", _in_main("xs = [i for i in range(x)]\nassert x == x"), "expression kind ListComp"),
    ("two_generators", _in_main("xs = [i + j for i in range(2) for j in range(2)]\nassert x == x"), "expression kind ListComp"),
    ("comprehension_with_a_lambda", _in_main("xs = [(lambda: i)() for i in range(2)]\nassert x == x"), "expression kind ListComp"),
    ("chain_over_a_list_cell", _in_main("xs = [0, 1]\nassert 0 <= xs[1] <= 2"), "comparison chain"),
    ("string_cells", _in_main("xs = ['a', 'b']\nassert x == x"), ""),
    ("list_of_lists_by_repetition", _in_main("xs = [[0] * 2] * 2\nassert x == x"), "expression kind List"),
])
def test_a_list_use_outside_the_exact_subset_is_refused_by_name(name, src, fragment):
    reason = audit_reason(src)
    assert reason is not None, f"{name}: audited clean"
    assert fragment in reason, (name, reason)


def test_the_list_size_limit_is_the_documented_one(tmp_path):
    p = _prog(tmp_path, "big.py", _in_main("xs = [0] * 256\nxs[255] = 1\nassert xs[-1] == 1 and len(xs) == 256"))
    assert audit_reason(open(p).read()) is None
    v = _prove(p)
    assert v is not None and v.verdict == "true"


# ---- negative controls: the mutation that would make lists unsound ----------------------------------------------------------

def test_a_store_that_ignores_negative_indexes_would_be_a_wrong_true(tmp_path):
    p = _prog(tmp_path, "neg_store.py", """
        import _sv_verifier

        def main():
            i = _sv_verifier.nondet_int()
            xs = [0] * 4
            xs[i] = 1
            assert not (xs[3] == 1 and i < 0)

        main()
    """)
    assert exhaustive_native(p, range(-5, 6)) is False        # i == -1 writes the last cell
    v = _prove(p)
    assert v is None or v.verdict != "true"

    def forgets_the_end(self, t, v):                           # the bug: only the cell numbered k, no k - m
        k = _symexec._num(self.ev(t.slice))
        c = list(self.env[t.value.id])
        self._holds(z3.And(k >= -len(c), k < len(c)))
        self.env[t.value.id] = [_symexec._ite(k == i, v, x) for i, x in enumerate(c)]

    real = _symexec._Exec._model_list___setitem__
    _symexec._Exec._model_list___setitem__ = forgets_the_end
    try:
        wrong = _prove(p)
    finally:
        _symexec._Exec._model_list___setitem__ = real
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"
    again = _prove(p)
    assert again is None or again.verdict != "true"


def test_a_store_through_a_parameter_would_be_a_wrong_true_which_is_why_the_audit_refuses_it(tmp_path):
    from tharkun.sv import exact_subset

    p = _prog(tmp_path, "alias.py", """
        import _sv_verifier

        def fill(xs):
            xs[0] = 1

        def main():
            x = _sv_verifier.nondet_int()
            a = [0] * 2
            fill(a)
            assert a[0] == 0

        main()
    """)
    assert exhaustive_native(p, range(0, 2)) is False         # fill wrote through the caller's list
    v = _prove(p)
    assert v is None, "refused by the audit"
    assert "not a list local to this scope" in audit_reason(open(p).read())

    real = exact_subset._Audit._subscript_target
    real_mutable = _symexec._Runner._mutable
    exact_subset._Audit._subscript_target = lambda self, t, scope: self._expr(t.slice, scope)      # the bug: a store is a store
    try:
        protected = _prove(p)
        assert protected is None or protected.verdict != "true"   # the shared alias guard also refuses
        _symexec._Runner._mutable = lambda self, name, kinds: self.env[name]
        wrong = _prove(p)
    finally:
        exact_subset._Audit._subscript_target = real
        _symexec._Runner._mutable = real_mutable
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"
    assert _prove(p) is None


def test_expanding_a_comprehension_with_the_wrong_trips_would_be_a_wrong_true(tmp_path):
    p = _prog(tmp_path, "squares.py", """
        import _sv_verifier

        def main():
            x = _sv_verifier.nondet_int()
            sq = [i * i for i in range(5)]
            assert sq[3] != 9

        main()
    """)
    assert exhaustive_native(p, range(0, 2)) is False
    v = _prove(p)
    assert v is None or v.verdict != "true"
    real = exact_unroll._trips
    exact_unroll._trips = lambda start, stop, step: [0] * len(real(start, stop, step))      # the bug: every element is e(0)
    try:
        wrong = _prove(p)
    finally:
        exact_unroll._trips = real
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"


def test_a_list_proof_is_the_end_to_end_answer_too(tmp_path):
    from tharkun.sv import verify

    p = _prog(tmp_path, "e2e_list.py", _POOL.format(claim="total == 2"))
    v = verify(p, AR, python=sys.executable, budget_seconds=60, run_timeout=30, canary_seconds=3, early_stop_seconds=3,
               witness_dir=tmp_path / "proofs")
    assert v.verdict == "true" and v.component == "symbolic:product"
    bad = _prog(tmp_path, "e2e_list_bad.py", _POOL.format(claim="used[i] == 1"))
    v = verify(bad, AR, python=sys.executable, budget_seconds=60, run_timeout=30, canary_seconds=3, early_stop_seconds=3,
               witness_dir=tmp_path / "proofs2")
    assert v.verdict == "false"


# ---- the remaining admitted shapes, each against native Python on its whole domain (LFT2 soundness pass) --------------------

_SHAPES = """
    import _sv_verifier

    N = 3

    def first(zs):
        return zs[0]

    def through(ys):
        return first(ys) + len(ys)

    def main():
        x = _sv_verifier.nondet_int()
        b = _sv_verifier.nondet_bool()
        left = N * [x]
        sized = [0] * (N - 1)
        mixed = [0, 1]
        mixed[b] = True
        drawn = [x, _sv_verifier.nondet_int(), 7]
        assert {claim}

    main()
"""

_SHAPE_CLAIMS = [
    ("count_on_the_left", "left[0] + left[1] + left[2] == 3 * x and len(left) == N"),
    ("size_from_a_constant_expression", "len(sized) == 2 and sized[-1] == 0"),
    ("wrong_size", "len(sized) == 3"),
    ("a_bool_index_is_zero_or_one", "mixed[1] == 1 or not b"),
    ("a_true_cell_counts_as_one", "mixed[b] == 1 and mixed[b]"),
    ("wrong_mixed_sum", "mixed[0] + mixed[1] == 2"),
    ("a_true_index_is_one", "mixed[True] == mixed[1]"),
    ("a_second_draw_in_a_literal_is_its_own", "drawn[1] == drawn[0]"),
    ("a_literal_cell_is_the_draw", "drawn[0] == x and drawn[2] == 7"),
    ("a_list_through_two_defs", "through(drawn) == x + 3 and first(left) == x"),
    ("wrong_list_through_two_defs", "through(drawn) == x + 2"),
]


@pytest.mark.parametrize("name,claim", _SHAPE_CLAIMS, ids=[c[0] for c in _SHAPE_CLAIMS])
def test_the_other_list_shapes_agree_with_native_python(tmp_path, name, claim):
    assert agrees_with_native(tmp_path, name + ".py", _SHAPES.format(claim=claim), range(-2, 3), (0, 1), range(-2, 3))


_MODULE_STORE = """
    import _sv_verifier

    T = [0] * 2

    def peek():
        return T[0] + T[1]

    i = _sv_verifier.nondet_int()
    T[0] = i
    before = peek()
    T[1] = 7
    assert {claim}
"""


@pytest.mark.parametrize("claim", ["before == i and peek() == i + 7", "before == i and peek() == i", "peek() == 7"])
def test_a_def_reads_a_module_list_as_it_is_at_the_call(tmp_path, claim):
    """``_symexec.py::_inline`` pushes the module environment itself (not a copy) as the callee's global frame, so a
    module-level store before a call is what the callee reads; a snapshot would make ``peek() == 0`` a wrong TRUE."""
    assert agrees_with_native(tmp_path, "module_store.py", _MODULE_STORE.format(claim=claim), range(-2, 3))


_EMPTY = """
    import _sv_verifier

    def main():
        i = _sv_verifier.nondet_int()
        xs = [0] * 0
        ys = []
        assert {claim}

    main()
"""


@pytest.mark.parametrize("claim", ["len(xs) == 0 and len(ys) == 0", "xs[i] == 5", "ys[0] == 1 and len(xs) == 1", "len(xs) == 1"])
def test_an_empty_list_agrees_with_native_python(tmp_path, claim):
    """Every index into an empty list ends the run (IndexError), so a claim behind one is vacuously TRUE natively too."""
    assert agrees_with_native(tmp_path, "empty.py", _EMPTY.format(claim=claim), range(-2, 3))


def test_a_list_left_with_two_lengths_by_two_arms_is_not_a_proof(tmp_path):
    """``_symexec.py::_ite`` refuses to merge lists of different lengths and ``_merge_if`` poisons the name: the run is
    inexact, so neither the true nor the false claim is answered by the product proof (a precision limit, not a hole)."""
    body = """
        import _sv_verifier

        def main():
            c = _sv_verifier.nondet_bool()
            if c:
                xs = [0] * 2
            else:
                xs = [0] * 3
            assert {claim}

        main()
    """
    for claim, native in (("len(xs) >= 2", True), ("len(xs) == 2", False)):
        p = _prog(tmp_path, f"two_lengths_{native}.py", body.format(claim=claim))
        assert exhaustive_native(p, (0, 1)) is native
        v = _prove(p)
        assert v is None or v.verdict != "true"


@pytest.mark.parametrize("name,src,fragment", [
    ("double_subscript", _in_main("xs = [0, 1]\nassert xs[0][0] == 0"), "subscript of something that is not a list"),
    ("a_list_as_a_cell", _in_main("xs = [0, 1]\nxs[0] = [1]\nassert x == x"), "expression kind List"),
    ("a_list_in_a_returned_tuple", _in_main("a, b = g()\nassert a == 0", "def g():\n    xs = [0]\n    return xs, 1\n\n"),
     "used as a value"),
    ("a_list_as_the_count", _in_main("xs = [0]\nys = [1] * xs\nassert x == x"), "list repetition whose count is not a constant"),
    ("a_float_index", _in_main("xs = [0, 1]\nassert xs[1.0] == 1"), "constant of type float"),
    ("store_with_a_slice", _in_main("xs = [0, 1]\nxs[0:1] = [5]\nassert x == x"), "slice or tuple subscript"),
    ("augmented_store_into_a_parameter", _in_main("xs = [0]\nf(xs)\nassert xs[0] == 0", "def f(ys):\n    ys[0] += 1\n\n"),
     "not a list local to this scope"),
    ("a_list_in_a_truth_test", _in_main("xs = [0]\nif xs:\n    pass\nassert x == x"), "used as a value"),
    ("a_list_in_a_conditional_expression", _in_main("xs = [0]\nys = [1]\nzs = xs if x else ys\nassert x == x"), "used as a value"),
    ("len_with_two_arguments", _in_main("xs = [0]\nassert len(xs, xs) == 1"), "len of something that is not a list"),
    ("a_scalar_parameter_named_like_a_module_list_is_indexed",
     "import _sv_verifier\n\nT = [0, 1]\n\ndef f(T):\n    return T[0]\n\nx = _sv_verifier.nondet_int()\nassert f(x) == 0\n",
     "must be a list"),
])
def test_more_list_shapes_outside_the_exact_subset_are_refused_by_name(name, src, fragment):
    reason = audit_reason(src)
    assert reason is not None, f"{name}: audited clean"
    assert fragment in reason, (name, reason)


_SHADOW = """
    import _sv_verifier

    T = [4, 5]

    def plus(T):
        return T + 1

    def cell(T, j):
        return T[j]

    x = _sv_verifier.nondet_int()
    assert {claim}
"""


@pytest.mark.parametrize("claim", ["plus(x) == x + 1 and cell(T, 1) == 5 and T[0] == 4", "plus(x) == T[0]"])
def test_a_parameter_may_shadow_a_module_list(tmp_path, claim):
    """A parameter named like a module-level list is that parameter (the call frame holds exactly the parameters): a
    scalar in ``plus``, a list in ``cell``; the module's ``T`` is read where no parameter shadows it."""
    assert agrees_with_native(tmp_path, "shadow.py", _SHADOW.format(claim=claim), range(-2, 3))


# ---- negative controls for the read side and the expansion's length ----------------------------------------------------------

def test_a_read_that_ignores_negative_indexes_would_be_a_wrong_true(tmp_path):
    """``xs[-4]`` is the first cell natively; a read that only matched ``k == i`` would hand every negative index the
    LAST cell (the fold's default), so a claim the first cell breaks under a negative index would be wrongly proved."""
    p = _prog(tmp_path, "neg_read.py", """
        import _sv_verifier

        xs = [1, 0, 0, 0]
        i = _sv_verifier.nondet_int()
        assert xs[i] != 1 or i >= 0
    """)
    assert exhaustive_native(p, range(-5, 6)) is False       # i == -4 reads the 1
    v = _prove(p)
    assert v is None or v.verdict != "true"

    real = _symexec._Runner.ev_Subscript

    def forgets_the_end(self, n):                              # the bug: cell i only when k == i, never k == i - m
        if isinstance(n.value, ast.Name) and isinstance(self.env.get(n.value.id), list) and isinstance(n.slice, ast.Name):
            c, k = self.env[n.value.id], _symexec._num(self.ev(n.slice))
            self._holds(z3.And(k >= -len(c), k < len(c)))
            out = c[-1]
            for i in range(len(c) - 2, -1, -1):
                out = _symexec._ite(k == i, c[i], out)
            return out
        return real(self, n)

    _symexec._Runner.ev_Subscript = forgets_the_end
    try:
        wrong = _prove(p)
    finally:
        _symexec._Runner.ev_Subscript = real
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"
    again = _prove(p)
    assert again is None or again.verdict != "true"


def test_expanding_a_comprehension_with_too_few_trips_would_be_a_wrong_true(tmp_path):
    """The expansion's LENGTH is the trip count: one element short, ``len`` lies and a claim on it is wrongly proved."""
    p = _prog(tmp_path, "short.py", """
        import _sv_verifier

        def main():
            x = _sv_verifier.nondet_int()
            sq = [i + x for i in range(5)]
            assert len(sq) == 4

        main()
    """)
    assert exhaustive_native(p, range(0, 2)) is False
    v = _prove(p)
    assert v is None or v.verdict != "true"
    real = exact_unroll._trips
    exact_unroll._trips = lambda start, stop, step: real(start, stop, step)[:-1]      # the bug: the last trip is dropped
    try:
        wrong = _prove(p)
    finally:
        exact_unroll._trips = real
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"
    assert exact_unroll._trips(0, 5, 1) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# counter while ([internal ref] PR 4): unrolled exactly before the audit (sv/exact_unroll.py::_fuse_counter_loops), with the
# parameters every call passes the same constant folded like module constants (exact_unroll.py::_param_constants)
# ---------------------------------------------------------------------------

_NONE = -(1 << 20)

_NOT_NONE = """
    import _sv_verifier

    NONE = -(1 << 20)

    N = 4

    def all_not_none(xs: list[int], n: int) -> bool:
        i = 0
        while i < n:
            if {absent}:
                return {early}
            i = i + 1
        return True

    def main():
        xs = [0, 0, 0, 0]
        i = 0
        while i < N:
            v = _sv_verifier.nondet_int()
            if not (v == NONE or (0 <= v and v <= 1)):
                return
            xs[i] = v
            i = i + 1
        result = all_not_none(xs, N)
        any_none = False
        k = 0
        while k < N:
            if xs[k] == NONE:
                any_none = True
            k = k + 1
        assert {claim}

    main()
"""

_NOT_NONE_DOMAINS = ((_NONE, 0, 1, 2),) * 4

#: (name, absent test, early return, claim): boto3's all_not_none and the mutants that each break one thing
_NOT_NONE_VARIANTS = [
    ("correct", "xs[i] == NONE", "False", "result == (not any_none)"),
    ("truthiness_rejects_zero", "not xs[i]", "False", "result == (not any_none)"),
    ("early_arm_wrong", "xs[i] == NONE", "True", "result == (not any_none)"),
    ("claim_one_direction", "xs[i] == NONE", "False", "result or any_none"),
    ("claim_wrong", "xs[i] == NONE", "False", "result"),
    ("counter_left_at_n", "xs[i] == NONE", "False", "result == (not any_none) and i == N and k == N"),
]


@pytest.mark.parametrize("name,absent,early,claim", _NOT_NONE_VARIANTS, ids=[v[0] for v in _NOT_NONE_VARIANTS])
def test_counter_loops_agree_with_the_native_program_on_the_whole_domain(tmp_path, name, absent, early, claim):
    body = _NOT_NONE.format(absent=absent, early=early, claim=claim)
    assert agrees_with_native(tmp_path, name + ".py", body, *_NOT_NONE_DOMAINS)


def test_the_correct_counter_program_is_proved_true(tmp_path):
    v = _prove(_prog(tmp_path, "not_none.py", _NOT_NONE.format(absent="xs[i] == NONE", early="False",
                                                              claim="result == (not any_none)")))
    assert v is not None and v.verdict == "true" and v.component == "symbolic:product"


@pytest.mark.parametrize("name,loop,claim", [
    ("lt", "i = 0\nwhile i < 4:\n    total = total + i\n    i += 1", "total == 6 and i == 4"),
    ("lte", "i = 0\nwhile i <= 4:\n    total = total + i\n    i = i + 1", "total == 10 and i == 5"),
    ("step_two", "i = 1\nwhile i < 8:\n    total = total + i\n    i += 2", "total == 16 and i == 9"),
    ("down_gt", "i = 5\nwhile i > 0:\n    total = total + i\n    i -= 2", "total == 9 and i == -1"),
    ("down_gte", "i = 5\nwhile i >= 1:\n    total = total + i\n    i = i - 2", "total == 9 and i == -1"),
    ("empty_leaves_the_init", "i = 5\nwhile i < 3:\n    total = total + 1\n    i += 1", "total == 0 and i == 5"),
    ("init_from_a_constant_expression", "i = K - 3\nwhile i < K:\n    total = total + 1\n    i += 1", "total == 3 and i == K"),
    ("bound_from_a_constant_expression", "i = 0\nwhile i < K + 1:\n    total = total + 1\n    i += 1", "total == 4"),
    ("nested", "i = 0\nwhile i < 3:\n    j = 0\n    while j < 2:\n        total = total + i * 10 + j\n        j += 1\n    i += 1",
     "total == 63 and i == 3 and j == 2"),
    ("counter_read_in_the_body", "i = 0\nwhile i < 3:\n    if i == x:\n        total = total + 10\n    i += 1", "total == 10"),
    ("a_wrong_claim", "i = 0\nwhile i < 4:\n    total = total + i\n    i += 1", "total == 7"),
    ("a_wrong_final_value", "i = 0\nwhile i < 4:\n    total = total + i\n    i += 1", "i == 3"),
    ("a_wrong_nested_total", "i = 0\nwhile i < 3:\n    j = 0\n    while j < 2:\n        total = total + 1\n        j += 1\n    i += 1",
     "total == 5"),
])
def test_counter_loop_results_match_native_python(tmp_path, name, loop, claim):
    body = ("import _sv_verifier\n\nK = 3\n\ndef main():\n    x = _sv_verifier.nondet_int()\n    if not (0 <= x <= 1):\n"
            "        return\n    total = 0\n" + textwrap.indent(loop, "    ") + f"\n    assert {claim}\n\nmain()\n")
    assert agrees_with_native(tmp_path, name + ".py", body, range(-1, 3))


def test_an_early_return_inside_a_counter_loop_ends_the_function_not_the_loop(tmp_path):
    body = """
        import _sv_verifier

        def scan(limit):
            seen = 0
            i = 0
            while i < 5:
                if i == limit:
                    return seen
                seen = seen + 1
                i += 1
            return seen + 100

        def main():
            limit = _sv_verifier.nondet_int()
            if not (-2 <= limit <= 7):
                return
            r = scan(limit)
            assert (0 <= limit <= 4 and r == limit) or ((limit < 0 or limit > 4) and r == 105)

        main()
    """
    assert agrees_with_native(tmp_path, "early.py", body, range(-3, 9))
    v = _prove(_prog(tmp_path, "early2.py", body))
    assert v is not None and v.verdict == "true"
    wrong = body.replace("r == 105", "r == 104")
    assert agrees_with_native(tmp_path, "early3.py", wrong, range(-3, 9))


def test_each_trip_of_a_counter_loop_draws_a_fresh_value(tmp_path):
    body = """
        import _sv_verifier

        def main():
            first = 0
            i = 0
            while i < 2:
                x = _sv_verifier.nondet_int()
                if i == 0:
                    first = x
                else:
                    assert x == first
                i += 1

        main()
    """
    p = _prog(tmp_path, "fresh.py", body)
    assert exhaustive_native(p, range(0, 3), range(0, 3)) is False        # the two draws differ
    v = _prove(p)
    assert v is None or v.verdict != "true"


_PARAM_CONST = """
    import _sv_verifier

    N = 3

    def count_up(n):
        total = 0
        i = 0
        while i < n:
            total = total + i
            i += 1
        return total

    def main():
        x = _sv_verifier.nondet_int()
        if not (0 <= x <= 1):
            return
        a = count_up(N)
        b = count_up({second})
        assert {claim}

    main()
"""


@pytest.mark.parametrize("name,second,claim", [
    ("the_same_constant_at_every_call", "N", "a == 3 and b == 3"),
    ("the_same_value_as_a_literal", "3", "a == b"),
    ("a_wrong_claim", "N", "a == 4"),
])
def test_a_parameter_every_call_passes_the_same_constant_is_a_constant(tmp_path, name, second, claim):
    assert agrees_with_native(tmp_path, name + ".py", _PARAM_CONST.format(second=second, claim=claim), range(-1, 3))


def test_the_parameter_constant_is_the_one_every_call_passes():
    tree = ast.parse(textwrap.dedent(_PARAM_CONST.format(second="N", claim="True")))
    assert exact_unroll.module_constants(tree) == {"N": 3, "n": 3}
    tree = ast.parse(textwrap.dedent(_PARAM_CONST.format(second="2", claim="True")))
    assert exact_unroll.module_constants(tree) == {"N": 3}                 # two values: not a constant
    chain = textwrap.dedent("""
        K = 4

        def inner(m):
            return m

        def outer(n):
            return inner(n)

        outer(K)
    """)
    assert exact_unroll.module_constants(ast.parse(chain)) == {"K": 4, "n": 4, "m": 4}   # through a constant parameter


# ---- refusals: every while that is not a counter loop stays a `while` and is refused by name ------------------------------

_WHILE = "while loop that is not a counter loop"


@pytest.mark.parametrize("name,src", [
    ("bound_is_a_draw", _in_main("i = 0\nwhile i < x:\n    i += 1")),
    ("bound_is_a_parameter_passed_two_constants", _in_main("assert f(2) == 2 and f(3) == 3", "def f(n):\n    i = 0\n    while i < n:\n        i += 1\n    return i\n\n")),
    ("bound_is_a_parameter_passed_a_draw", _in_main("assert f(x) >= 0", "def f(n):\n    i = 0\n    while i < n:\n        i += 1\n    return i\n\n")),
    ("bound_is_a_parameter_of_a_def_never_called", _in_main("assert x == x", "def f(n):\n    i = 0\n    while i < n:\n        i += 1\n    return i\n\n")),
    ("bound_is_a_parameter_passed_by_keyword", _in_main("assert f(n=2) == 2", "def f(n):\n    i = 0\n    while i < n:\n        i += 1\n    return i\n\n")),
    ("counter_stored_in_the_body", _in_main("i = 0\nwhile i < 3:\n    i = i + 1\n    i += 1")),
    ("counter_rebound_by_an_inner_for", _in_main("i = 0\nwhile i < 3:\n    for i in range(2):\n        pass\n    i += 1")),
    ("step_under_an_if", _in_main("i = 0\nwhile i < 3:\n    if x:\n        i += 1")),
    ("step_not_last", _in_main("i = 0\nt = 0\nwhile i < 3:\n    i += 1\n    t = t + 1")),
    ("step_zero", _in_main("i = 0\nwhile i < 3:\n    i += 0")),
    ("step_against_the_test", _in_main("i = 0\nwhile i < 3:\n    i -= 1")),
    ("step_is_a_draw", _in_main("i = 0\nwhile i < 3:\n    i += x")),
    ("step_multiplies", _in_main("i = 1\nwhile i < 8:\n    i = i * 2")),
    ("break_in_the_body", _in_main("i = 0\nwhile i < 3:\n    if x:\n        break\n    i += 1")),
    ("continue_in_the_body", _in_main("i = 0\nwhile i < 3:\n    if x:\n        continue\n    i += 1")),
    ("else_clause", _in_main("i = 0\nwhile i < 3:\n    i += 1\nelse:\n    pass")),
    ("init_not_just_before", _in_main("i = 0\nt = 1\nwhile i < 3:\n    i += 1")),
    ("init_is_the_target_of_an_unrolled_for", _in_main("for i in range(3):\n    pass\nwhile i < 5:\n    i += 1")),
    ("init_is_a_draw", _in_main("i = x\nwhile i < 3:\n    i += 1")),
    ("no_init", _in_main("while x < 3:\n    x += 1")),
    ("not_equal_test", _in_main("i = 0\nwhile i != 3:\n    i += 1")),
    ("bound_on_the_left", _in_main("i = 0\nwhile 3 > i:\n    i += 1")),
    ("chained_test", _in_main("i = 0\nwhile 0 <= i < 3:\n    i += 1")),
    ("more_than_64_trips", _in_main("i = 0\nwhile i < 65:\n    i += 1")),
    ("truth_test", _in_main("i = 3\nwhile i:\n    i -= 1")),
])
def test_a_while_that_is_not_a_counter_loop_is_refused_by_name(name, src):
    reason = audit_reason(src)
    assert reason is not None, f"{name}: audited clean"
    assert _WHILE in reason, (name, reason)


def test_the_counter_loop_limit_is_the_documented_one(tmp_path):
    p = _prog(tmp_path, "sixty_four.py", _in_main("t = 0\ni = 0\nwhile i < 64:\n    t = t + 1\n    i += 1\nassert t == 64 and i == 64"))
    assert audit_reason(open(p).read()) is None
    v = _prove(p)
    assert v is not None and v.verdict == "true"


# ---- negative controls: the mutations of the unroller that would make a counter loop unsound ----------------------------------

def test_reading_lte_as_lt_would_be_a_wrong_true(tmp_path):
    p = _prog(tmp_path, "lte.py", _in_main("t = 0\ni = 0\nwhile i <= 2:\n    t = t + 1\n    i += 1\nassert t == 2"))
    assert exhaustive_native(p, range(0, 2)) is False                     # three trips
    v = _prove(p)
    assert v is None or v.verdict != "true"
    real = exact_unroll._TEST_OPS[ast.LtE]
    exact_unroll._TEST_OPS[ast.LtE] = 0                                     # the bug: <= unrolled like <
    try:
        wrong = _prove(p)
    finally:
        exact_unroll._TEST_OPS[ast.LtE] = real
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"
    assert _prove(p) is None or _prove(p).verdict != "true"


def test_unrolling_a_loop_whose_body_moves_the_counter_would_be_a_wrong_true(tmp_path):
    p = _prog(tmp_path, "moved.py", _in_main("t = 0\ni = 0\nwhile i < 3:\n    t = t + 1\n    i = 10\n    i += 1\nassert t == 3"))
    assert exhaustive_native(p, range(0, 2)) is False                     # one trip: i jumps to 11
    assert _WHILE in audit_reason(open(p).read())
    real = exact_unroll._body_leaves_the_counter
    exact_unroll._body_leaves_the_counter = lambda stmts, name: True      # the bug: the body is not checked
    try:
        wrong = _prove(p)
    finally:
        exact_unroll._body_leaves_the_counter = real
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"
    assert _prove(p) is None


def test_folding_a_parameter_to_the_first_call_s_constant_would_be_a_wrong_true(tmp_path):
    p = _prog(tmp_path, "first_call.py", _PARAM_CONST.format(second="2", claim="b == 3"))
    assert exhaustive_native(p, range(0, 2)) is False                     # count_up(2) is 1
    assert _WHILE in audit_reason(open(p).read())

    def takes_the_first(tree, consts, count):                              # the bug: the first call decides
        defs = {st.name: st for st in tree.body if isinstance(st, ast.FunctionDef)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defs:
                for j, a in enumerate(defs[n.func.id].args.args):
                    v = exact_unroll.fold_int(n.args[j], consts) if j < len(n.args) else None
                    if v is not None and a.arg not in consts:
                        consts[a.arg] = v

    real = exact_unroll._param_constants
    exact_unroll._param_constants = takes_the_first
    try:
        wrong = _prove(p)
    finally:
        exact_unroll._param_constants = real
    assert wrong is not None and wrong.verdict == "true", "the control must fail the way the bug would"
    assert _prove(p) is None


def test_a_counter_loop_proof_is_the_end_to_end_answer_too(tmp_path):
    from tharkun.sv import verify

    p = _prog(tmp_path, "e2e_while.py", _NOT_NONE.format(absent="xs[i] == NONE", early="False", claim="result == (not any_none)"))
    v = verify(p, AR, python=sys.executable, budget_seconds=60, run_timeout=30, canary_seconds=3, early_stop_seconds=3,
               witness_dir=tmp_path / "proofs")
    assert v.verdict == "true" and v.component == "symbolic:product"
    bad = _prog(tmp_path, "e2e_while_bad.py", _NOT_NONE.format(absent="not xs[i]", early="False", claim="result == (not any_none)"))
    v = verify(bad, AR, python=sys.executable, budget_seconds=60, run_timeout=30, canary_seconds=3, early_stop_seconds=3,
               witness_dir=tmp_path / "proofs2")
    assert v.verdict == "false"
