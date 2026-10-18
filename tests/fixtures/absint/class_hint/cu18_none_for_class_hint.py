import _sv_verifier


class A:
    pass


x = None
_sv_verifier.check_type(x, A)
