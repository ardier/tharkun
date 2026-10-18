"""[internal ref]: ``io.StringIO`` in ``tharkun.solve.absint`` (docs/ABSINT.md).

A SOUND, COARSE model, matching ``re.Pattern``'s precision level: a ``StringIOObj``
heap kind carries one field, ``content`` (an abstract string value), coarsened to
"some string" (``ANYSTR``) the instant anything is written to it -- no exact-content
or position tracking (``seek``/``tell`` return an unconstrained int, never connected
back to ``content``). A fresh, unwritten ``io.StringIO()`` is the one case precise
enough to stay exact: the literal empty string. ``write``/``getvalue``/``read``/
``seek``/``tell`` are modelled, plus ``print(..., file=a_stringio)`` as a write;
every other method (``readline``, ``truncate``, ``close``, ...) and every other
``file=`` destination (``sys.stdout``, a user object) still refuse. Motivated by
hexiom's absint blocker ([internal ref]): ``bm_hexiom_det.py``/
``bm_hexiom_nondet.py`` collect their solver's output via
``io.StringIO()``/``print(..., file=stream)``/``.getvalue()``. Mirrors
``test_absint.py``'s PROVABLE/NEVER_TRUE/UNSUPPORTED convention.
"""
from __future__ import annotations

import textwrap

import pytest

from tharkun.solve.absint import prove

DL = ("KeyError", "IndexError")
DT = ("TypeError", "AttributeError")


def _src(body: str) -> str:
    return textwrap.dedent(body)


# ------------------------------------------------------------------ PROVABLE

def test_write_then_getvalue_is_provably_a_string():
    r = prove(_src("""
        import io
        buf = io.StringIO()
        buf.write("hello")
        y = buf.getvalue()
        print(len(y))
    """), DT)
    assert r.verdict == "TRUE", r.explain()


def test_print_file_is_a_write():
    r = prove(_src("""
        import io
        buf = io.StringIO()
        print("hi", file=buf)
        y = buf.getvalue()
        print(len(y))
    """), DT)
    assert r.verdict == "TRUE", r.explain()


def test_fresh_buffer_is_precisely_empty():
    """The one case this coarse model keeps exact: a fresh, unwritten StringIO()."""
    r = prove(_src("""
        import io
        buf = io.StringIO()
        y = buf.getvalue()
        if y == "":
            print(1)
        else:
            [][0]
    """), DL)
    assert r.verdict == "TRUE", r.explain()


def test_seek_and_tell_round_trip():
    r = prove(_src("""
        import io
        buf = io.StringIO("abc")
        buf.seek(0)
        pos = buf.tell()
        print(pos >= 0)
    """), DT)
    assert r.verdict == "TRUE", r.explain()


def test_hexiom_shaped_solve_file_pattern_is_provable():
    """The actual shape bm_hexiom_det.py/bm_hexiom_nondet.py use: a helper function
    prints into a stream it's handed, the caller reads it back afterward."""
    r = prove(_src("""
        import io
        def solve_file(stream):
            print("line1", file=stream)
            print("line2", file=stream)

        stream = io.StringIO()
        solve_file(stream)
        output = stream.getvalue()
        print(len(output))
    """), DT)
    assert r.verdict == "TRUE", r.explain()


def test_read_after_write_is_provably_a_string():
    r = prove(_src("""
        import io
        buf = io.StringIO()
        buf.write("x")
        y = buf.read()
        print(len(y))
    """), DT)
    assert r.verdict == "TRUE", r.explain()


# --- NEVER_TRUE: content coarsens to "some string" -- must not collapse to one -----
# branch or silently pass an unproven TypeError check

NEVER_TRUE = {
    "write_result_not_assumed_equal_to_literal": (DL, """
        import io
        buf = io.StringIO()
        buf.write("x")
        xs = [1, 2]
        if buf.getvalue() == "x":
            print(xs[5])
        else:
            print(xs[0])
    """),
    "print_file_result_not_assumed_equal_to_literal": (DL, """
        import io
        buf = io.StringIO()
        print("x", end="", file=buf)
        xs = [1, 2]
        if buf.getvalue() == "x":
            print(xs[5])
        else:
            print(xs[0])
    """),
    "seek_then_read_not_assumed_to_be_the_original_content": (DL, """
        import io
        buf = io.StringIO("abc")
        buf.write("z")
        buf.seek(0)
        xs = [1, 2]
        if buf.read() == "abc":
            print(xs[5])
        else:
            print(xs[0])
    """),
}


@pytest.mark.parametrize("name", sorted(NEVER_TRUE))
def test_programs_with_a_reachable_violation_are_never_true(name):
    classes, body = NEVER_TRUE[name]
    r = prove(_src(body), classes)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "unproven check" in r.reason, r.reason


def test_non_str_initial_value_is_never_true_for_typeerror():
    """io.StringIO(5) raises TypeError in real Python -- a construction whose
    argument type this pass cannot confirm must never let a TypeError proof pass."""
    r = prove(_src("""
        import io
        buf = io.StringIO(5)
        print(buf.getvalue())
    """), ("TypeError",))
    assert r.verdict == "UNKNOWN", r.explain()
    assert "unproven check" in r.reason, r.reason


# --- UNSUPPORTED: every shape this model does NOT cover still aborts to UNKNOWN ----

UNSUPPORTED = {
    "print_file_sys_stdout_is_unmodelled": """
        import sys
        print("hi", file=sys.stdout)
        [][0]
    """,
    "print_file_ambiguous_receiver_is_unmodelled": """
        import io
        import _sv_verifier
        n = _sv_verifier.nondet_bool()
        buf = io.StringIO() if n else 5
        print("hi", file=buf)
        [][0]
    """,
    "truncate_is_unmodelled": """
        import io
        buf = io.StringIO()
        buf.write("x")
        buf.truncate()
        [][0]
    """,
    "close_is_unmodelled": """
        import io
        buf = io.StringIO()
        buf.close()
        [][0]
    """,
    "write_with_extra_positional_arg_is_unmodelled": """
        import io
        buf = io.StringIO()
        buf.write("x", 1)
        [][0]
    """,
    "write_with_a_non_str_arg_is_unmodelled": """
        import io
        buf = io.StringIO()
        buf.write(5)
        [][0]
    """,
    "stringio_with_both_positional_and_keyword_initial_value_is_unmodelled": """
        import io
        buf = io.StringIO("a", initial_value="b")
        [][0]
    """,
    "stringio_with_an_unknown_keyword_is_unmodelled": """
        import io
        buf = io.StringIO(errors="strict")
        [][0]
    """,
}


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_unmodelled_shapes_are_unknown(name):
    r = prove(_src(UNSUPPORTED[name]), DL)
    assert r.verdict == "UNKNOWN" and r.reason
