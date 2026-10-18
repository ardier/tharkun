import _sv_verifier


class A:
    pass


def f():
    return A()


_sv_verifier.check_type(f, A)
