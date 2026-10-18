import _sv_verifier


class B:
    pass


class A:
    def __init__(self):
        self.__class__ = B


a = A()
_sv_verifier.check_type(a, A)
