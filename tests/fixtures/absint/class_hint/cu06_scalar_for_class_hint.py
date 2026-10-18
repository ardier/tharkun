import _sv_verifier


class A:
    pass


_sv_verifier.check_type(1, A)
