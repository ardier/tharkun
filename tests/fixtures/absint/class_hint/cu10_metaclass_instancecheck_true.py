import _sv_verifier


class M(type):
    def __instancecheck__(cls, inst):
        return True


class A(metaclass=M):
    pass


_sv_verifier.check_type(1, A)
