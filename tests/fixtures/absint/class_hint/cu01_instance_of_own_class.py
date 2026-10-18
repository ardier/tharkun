import _sv_verifier


class A:
    pass


a = A()
_sv_verifier.check_type(a, A)
