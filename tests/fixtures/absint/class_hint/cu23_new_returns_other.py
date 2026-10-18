import _sv_verifier


class A:
    def __new__(cls):
        return 5


a = A()
_sv_verifier.check_type(a, A)
