import _sv_verifier


class A:
    pass


xs = [A(), A()]
_sv_verifier.check_type(xs, list)
_sv_verifier.check_type(xs[0], A)
