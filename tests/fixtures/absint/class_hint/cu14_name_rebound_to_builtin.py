import _sv_verifier


class A:
    pass


A = int
_sv_verifier.check_type(3, A)
_sv_verifier.check_type("s", A)
