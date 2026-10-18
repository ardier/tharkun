"""The source checker (``sv/validate.py``) learns two spellings real tasks use:
``abs()`` on the three scalar sorts and ``import _sv_verifier`` with attribute draws
([internal ref] step 3: the facts stage can only certify what the checker can model, and the
SV-Benchmarks tasks write ``_sv_verifier.nondet_int()`` and ``abs(...)``).

Every positive case has a violating twin the checker must NOT confirm: a checker
that accepts a falsifiable invariant would make a wrong TRUE out of a producer's nomination.
CPython 3.14 only (the checker's own requirement).
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from tharkun.solve.census import obligation_census
from tharkun.sv.validate import check_candidates

pytestmark = pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="the source checker needs CPython 3.14")

ZD = "ZeroDivisionError"


def proves(src: str, exc: str = ZD) -> bool:
    """Does the checker confirm the census's own invariants of ``src`` (what ``finish`` asks)?"""
    src = textwrap.dedent(src).lstrip("\n")
    census = obligation_census(src, [exc], "t.py")
    spec = f"CHECK( init(main()), ! uncaught({exc}))\n"
    invariants = [o.invariant for o in census.obligations if not o.gap]
    assert invariants and not any(o.gap for o in census.obligations), "the program must have plain obligations"
    return check_candidates(src, "t.py", spec, census, invariants, seconds=20.0).confirmed


def test_abs_of_an_int_is_the_absolute_value_over_unbounded_ints():
    assert proves("""
        from _sv_verifier import nondet_int
        x = nondet_int()
        d = 1 + abs(x)
        r = 10 // d
    """)
    assert not proves("""
        from _sv_verifier import nondet_int
        x = nondet_int()
        d = abs(x)
        r = 10 // d
    """)


def test_abs_of_a_float_clears_the_sign_bit_and_keeps_nan():
    assert proves("""
        from _sv_verifier import nondet_float
        x = nondet_float()
        d = abs(x) + 1.0
        r = 1.0 / d
    """)
    # abs(-0.0) is 0.0 and abs(0.0) is 0.0: the division can still fault
    assert not proves("""
        from _sv_verifier import nondet_float
        x = nondet_float()
        d = abs(x)
        r = 1.0 / d
    """)


def test_abs_of_a_bool_is_the_int_zero_or_one():
    assert proves("""
        from _sv_verifier import nondet_bool
        b = nondet_bool()
        d = 1 + abs(b)
        r = 10 // d
    """)
    assert not proves("""
        from _sv_verifier import nondet_bool
        b = nondet_bool()
        d = abs(b)
        r = 10 // d
    """)


@pytest.mark.parametrize("rebinding", [
    "abs = 0\n",                                       # an assignment
    "def abs(v):\n    return 0\n",                       # a function with a parameter
    "from _sv_verifier import nondet_int as abs\n",      # an import under the name
])
def test_a_program_that_rebinds_abs_is_never_confirmed_with_the_builtins_meaning(rebinding):
    src = ("from _sv_verifier import nondet_int\n" + rebinding +
           "x = nondet_int()\nd = 1 + abs(x)\nr = 10 // d\n")
    census = obligation_census(src, [ZD], "t.py")
    verdict = check_candidates(src, "t.py", f"CHECK( init(main()), ! uncaught({ZD}))\n", census,
                               [o.invariant for o in census.obligations if not o.gap], seconds=20.0)
    assert not verdict.confirmed


def test_an_unmodeled_operand_of_abs_is_refused_not_guessed():
    src = textwrap.dedent("""
        from _sv_verifier import nondet_int
        x = nondet_int()
        d = 1 + abs("s")
        r = 10 // d
    """).lstrip("\n")
    census = obligation_census(src, [ZD], "t.py")
    assert not check_candidates(src, "t.py", f"CHECK( init(main()), ! uncaught({ZD}))\n", census,
                                [o.invariant for o in census.obligations if not o.gap], seconds=20.0).confirmed


def test_the_tracks_own_spelling_of_a_draw_is_modeled():
    assert proves("""
        import _sv_verifier

        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            d = 2 * x + 1
            r = 10 // d
    """)
    assert not proves("""
        import _sv_verifier

        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            d = 2 * x + 2
            r = 10 // d
    """)


def test_an_aliased_module_works_and_a_misused_one_is_refused():
    aliased = """
        import _sv_verifier as sv

        x = sv.nondet_int()
        d = 2 * x + 1
        r = 10 // d
    """
    assert proves(aliased)
    for misuse in ("y = sv()\n",                  # a module is not callable
                   "y = sv.nondet_str()\n",       # only the scalar draws are modeled
                   "y = sv.nondet_int(3)\n",      # a draw takes no arguments
                   "sv = 3\n",                    # the module name is not rebindable
                   "y = sv\n"):                   # nor usable as a value
        src = textwrap.dedent(aliased).lstrip("\n") + misuse
        census = obligation_census(src, [ZD], "t.py")
        verdict = check_candidates(src, "t.py", f"CHECK( init(main()), ! uncaught({ZD}))\n", census,
                                   [o.invariant for o in census.obligations if not o.gap], seconds=20.0)
        assert not verdict.confirmed, misuse


def test_only_the_verifier_module_may_be_imported_by_name():
    src = textwrap.dedent("""
        import os
        from _sv_verifier import nondet_int
        x = nondet_int()
        d = 2 * x + 1
        r = 10 // d
    """).lstrip("\n")
    census = obligation_census(src, [ZD], "t.py")
    invariants = [o.invariant for o in census.obligations if not o.gap]
    assert invariants
    assert not check_candidates(src, "t.py", f"CHECK( init(main()), ! uncaught({ZD}))\n", census,
                                invariants, seconds=20.0).confirmed
