import _sv_verifier


class A:
    def __init__(self):
        self.v = 1


class B:
    def __init__(self):
        self.v = 1


_sv_verifier.check_type(A(), B)
