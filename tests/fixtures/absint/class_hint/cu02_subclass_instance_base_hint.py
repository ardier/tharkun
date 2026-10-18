import _sv_verifier


class A:
    pass


class B(A):
    pass


b = B()
_sv_verifier.check_type(b, A)
