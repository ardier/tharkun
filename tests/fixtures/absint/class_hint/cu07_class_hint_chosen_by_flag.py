import _sv_verifier


class A:
    pass


class B:
    pass


flag = _sv_verifier.nondet_bool()
o = A()
k = A if flag else B
_sv_verifier.check_type(o, k)
