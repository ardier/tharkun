import _sv_verifier


class A:
    def __getattr__(self, name):
        raise KeyError(name)


class B(A):
    pass


b = B()
_sv_verifier.check_type(b, A)
_sv_verifier.check_type(b, B)
