import _sv_verifier


class A:
    pass


class B(A):
    pass


class C(A):
    pass


flag = _sv_verifier.nondet_bool()
o = B() if flag else C()
_sv_verifier.check_type(o, B)
