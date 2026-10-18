"""[internal ref] / #1102c: ``isinstance()`` and str-method/str-predicate dispatch in
``tharkun.solve.absint`` (docs/ABSINT.md).

``isinstance(x, str/int/float/bool)`` and a small, fixed set of zero-arg, TOTAL
str methods (``_TOTAL_STR_METHODS``: never raise for ANY str receiver in real
Python) were entirely unmodelled before [internal ref]. #1102c extends the same shape
with the str PREDICATES (``_TOTAL_STR_PREDICATES``: also zero-arg and TOTAL,
routed to ``BOOL`` instead of ``ANYSTR``) and adds ``casefold`` (the one
zero-arg, total, string-returning method [internal ref]'s set was missing). Mirrors
``test_absint.py``'s own PROVABLE/NEVER_TRUE/UNSUPPORTED convention.
"""
from __future__ import annotations

import textwrap

import pytest

from tharkun.solve.absint import prove

DL = ("KeyError", "IndexError")
DT = ("TypeError", "AttributeError")


def _src(body: str) -> str:
    return textwrap.dedent(body)


# ------------------------------------------------------------------ isinstance()

def test_isinstance_str_true_lets_a_safe_branch_through():
    r = prove(_src("""
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        if isinstance(s, str):
            print(s.strip())
        else:
            [][0]
    """), DL)
    assert r.verdict == "TRUE", r.explain()


def test_isinstance_int_false_for_a_definite_str_is_provable():
    r = prove(_src("""
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        if isinstance(s, int):
            [][0]
    """), DL)
    assert r.verdict == "TRUE", r.explain()


def test_isinstance_bool_counts_as_int_membership():
    # bool is a subclass of int in real Python -- isinstance(a_bool, int) is True.
    r = prove(_src("""
        import _sv_verifier
        b = _sv_verifier.nondet_bool()
        if isinstance(b, int):
            print(1)
        else:
            [][0]
    """), DL)
    assert r.verdict == "TRUE", r.explain()


# ------------------------------------------------------------------ str methods / predicates

def test_total_str_method_result_is_provably_a_string():
    r = prove(_src("""
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        y = s.strip().lower().casefold()
        print(len(y) >= 0)
    """), DT)
    assert r.verdict == "TRUE", r.explain()


def test_total_str_predicate_result_is_provably_a_bool():
    r = prove(_src("""
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        ok = s.isascii()
        print(1 if ok else 0)
    """), DT)
    assert r.verdict == "TRUE", r.explain()


def test_isinstance_guard_then_predicate_and_total_method_is_provable():
    """The realistic shape this dispatch exists for (string_utils-style validation):
    an isinstance guard, then a predicate and a total method on the narrowed value."""
    r = prove(_src("""
        import _sv_verifier
        x = _sv_verifier.nondet_str()
        if isinstance(x, str):
            ok = x.isidentifier()
            y = x.casefold()
            print(len(y), ok)
    """), DT)
    assert r.verdict == "TRUE", r.explain()


# --- NEVER_TRUE: a predicate's BOOL result must not collapse to one branch ------------

NEVER_TRUE = {
    "predicate_result_keeps_both_branches_reachable": (DL, """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        xs = [1, 2]
        if s.isdigit():
            print(xs[5])
        else:
            print(xs[0])
    """),
    "isinstance_true_branch_still_has_its_own_violation": (DL, """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        xs = [1, 2]
        if isinstance(s, str):
            print(xs[5])
    """),
}


@pytest.mark.parametrize("name", sorted(NEVER_TRUE))
def test_programs_with_a_reachable_violation_are_never_true(name):
    classes, body = NEVER_TRUE[name]
    r = prove(_src(body), classes)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "unproven check" in r.reason, r.reason


# --- UNSUPPORTED: every shape this dispatch does NOT cover still aborts to UNKNOWN ----

UNSUPPORTED = {
    "isinstance_tuple_of_types": """
        import _sv_verifier
        x = _sv_verifier.nondet_str()
        if isinstance(x, (str, int)):
            print(1)
        [][0]
    """,
    "isinstance_ambiguous_kind": """
        import _sv_verifier
        n = _sv_verifier.nondet_int()
        x = "a" if n else 1
        if isinstance(x, str):
            print(1)
        [][0]
    """,
    "isinstance_aliased_type": """
        T = str
        import _sv_verifier
        x = _sv_verifier.nondet_str()
        if isinstance(x, T):
            print(1)
        [][0]
    """,
    "isinstance_of_a_list": """
        x = []
        if isinstance(x, list):
            print(1)
        [][0]
    """,
    "str_method_with_a_positional_arg_is_unsupported": """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        s.strip(" ")
        [][0]
    """,
    "str_predicate_with_a_positional_arg_is_unsupported": """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        s.isdigit(1)
        [][0]
    """,
    "str_method_with_a_keyword_arg_is_unsupported": """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        s.strip(chars=" ")
        [][0]
    """,
    "unmodelled_str_method_still_refuses": """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        s.replace("a", "b")
        [][0]
    """,
    "unmodelled_str_method_split_still_refuses": """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        s.split()
        [][0]
    """,
}


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_unmodelled_shapes_are_unknown(name):
    r = prove(_src(UNSUPPORTED[name]), DL)
    assert r.verdict == "UNKNOWN" and r.reason
