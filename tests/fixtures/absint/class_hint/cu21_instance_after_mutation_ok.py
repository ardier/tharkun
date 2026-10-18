import _sv_verifier


class A:
    def __init__(self):
        self.v = 1


class B(A):
    def __init__(self):
        A.__init__(self)
        self.w = 2


objs = [B(), A()]
_sv_verifier.check_type(objs[0], A)
_sv_verifier.check_type(objs[1], A)
