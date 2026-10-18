"""``//`` and ``%`` over a symbolic, nonzero divisor in ``tharkun.sv._symexec``: the z3 encoding must equal Python's
floor semantics for every sign of both operands, and must stay cheap for z3 (egraph PoC 2026-10-19).

The encoding is ``If(b > 0, a / b, (-a) / (-b))`` for ``//`` -- z3 only ever divides by a positive term -- and
``a - b * (a // b)`` for ``%``, over the same quotient term.  The earlier one, ``(a - r) / b`` with
``r = If(b > 0, a % b, If(a % b == 0, 0, a % b + b))``, is kept below as the reference it must agree with.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

z3 = pytest.importorskip("z3")

from tharkun.sv import product_proof
from tharkun.sv._symexec import _py_floordiv_sym, _py_mod_sym

AR = ("AssertionError",)
RANGE = range(-13, 14)


def _old_mod(a, b):
    return z3.If(b > 0, a % b, z3.If(a % b == 0, 0, a % b + b))


def _old_floordiv(a, b):
    return (a - _old_mod(a, b)) / b


def _eval(term, a, b, va, vb):
    return z3.simplify(z3.substitute(term, (a, z3.IntVal(va)), (b, z3.IntVal(vb)))).as_long()


def test_floordiv_and_mod_equal_python_for_every_sign():
    a, b = z3.Ints("a b")
    fd, md = _py_floordiv_sym(a, b), _py_mod_sym(a, b)
    old_fd, old_md = _old_floordiv(a, b), _old_mod(a, b)
    for va in RANGE:
        for vb in RANGE:
            if vb == 0:
                continue
            assert _eval(fd, a, b, va, vb) == va // vb == _eval(old_fd, a, b, va, vb), (va, vb)
            assert _eval(md, a, b, va, vb) == va % vb == _eval(old_md, a, b, va, vb), (va, vb)


def test_floordiv_and_mod_equal_python_beyond_machine_words():
    a, b = z3.Ints("a b")
    fd, md = _py_floordiv_sym(a, b), _py_mod_sym(a, b)
    big = 3 ** 90
    for va in (big, -big, big + 1, -big - 1, 0):
        for vb in (7, -7, big, -big, big - 1, -(big + 1)):
            assert _eval(fd, a, b, va, vb) == va // vb, (va, vb)
            assert _eval(md, a, b, va, vb) == va % vb, (va, vb)


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def _prove(path):
    return product_proof.prove(open(path).read(), AR, program=path, python=sys.executable)


_CEILDIV = """
        import _sv_verifier

        SMALL_BOUND = 1 << 10

        def cdiv(a: int, b: int) -> int:
            return -(a // -b)

        def main():
            a = _sv_verifier.nondet_int()
            b = _sv_verifier.nondet_int()
            if not (0 <= a <= SMALL_BOUND):
                return
            if not (1 <= b <= SMALL_BOUND):
                return
            q = cdiv(a, b)
            %s

        main()
"""


def test_ceildiv_postcondition_is_proved(tmp_path):
    # the MIN task ta_ceildiv_nonlinear_true: z3 needed ~8-12 s for it before the encoding change
    p = _prog(tmp_path, "ceildiv_true.py", _CEILDIV % "assert q * b >= a\n            assert (q - 1) * b < a")
    v = _prove(p)
    assert v is not None and v.verdict == "true" and v.component == "symbolic:product"


def test_ceildiv_wrong_postcondition_is_killed_with_a_replayed_witness(tmp_path):
    # q * b > a fails exactly when b divides a (a = 0, b = 1 is the smallest).  Top-level draws: a draw inside a def has no
    # single draw vector, so product_proof only ever PROVES that shape; here a model must be found AND replayed.
    p = _prog(tmp_path, "ceildiv_false.py", """
        import _sv_verifier

        a = _sv_verifier.nondet_int()
        b = _sv_verifier.nondet_int()
        if 0 <= a <= 1024 and 1 <= b <= 1024:
            q = -(a // -b)
            assert q * b > a
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "false" and v.exception == "AssertionError"


def test_mod_of_negative_divisor_keeps_the_divisors_sign(tmp_path):
    # Python: a % b has the sign of b (and |a % b| < |b|); z3's own % is never negative
    p = _prog(tmp_path, "mod_sign_true.py", """
        import _sv_verifier

        a = _sv_verifier.nondet_int()
        b = _sv_verifier.nondet_int()
        if b < 0:
            r = a % b
            assert r <= 0 and r > b
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true"
    q = _prog(tmp_path, "mod_sign_false.py", """
        import _sv_verifier

        a = _sv_verifier.nondet_int()
        b = _sv_verifier.nondet_int()
        if b < 0:
            r = a % b
            assert r < 0
    """)
    v = _prove(q)           # a = 0: 0 % b == 0
    assert v is not None and v.verdict == "false"


def test_divmod_identity_is_proved_for_every_sign(tmp_path):
    # the claim a separate positive-divisor form for % made z3 give up on (10 s timeout): a // b and a % b must share the quotient
    p = _prog(tmp_path, "divmod_identity.py", """
        import _sv_verifier

        a = _sv_verifier.nondet_int()
        b = _sv_verifier.nondet_int()
        if b != 0:
            assert (a // b) * b + a % b == a
            assert a % b == 0 or (a % b < 0) == (b < 0)
    """)
    v = _prove(p)
    assert v is not None and v.verdict == "true" and v.component == "symbolic:product"
