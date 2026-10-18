import _sv_verifier


class M(type):
    def __instancecheck__(cls, inst):
        return False


class A(metaclass=M):
    pass


class B(A):
    pass


b = B()
_sv_verifier.check_type(b, A)
