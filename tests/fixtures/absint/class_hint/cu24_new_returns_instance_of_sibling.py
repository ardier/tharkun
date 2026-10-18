import _sv_verifier


class B:
    pass


class A:
    def __new__(cls):
        return B()


a = A()
_sv_verifier.check_type(a, B)
_sv_verifier.check_type(a, A)
