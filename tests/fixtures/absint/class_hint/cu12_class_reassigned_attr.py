import _sv_verifier


class A:
    pass


class B:
    pass


a = A()
a.__class__ = B
_sv_verifier.check_type(a, A)
