import _sv_verifier


class A:
    pass


class V:
    def __repr__(self):
        raise KeyError("repr")


_sv_verifier.check_type(V(), A)
