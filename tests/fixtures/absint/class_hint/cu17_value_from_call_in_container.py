import _sv_verifier


class A:
    pass


class B:
    pass


def make(flag):
    return A() if flag else B()


box = [make(_sv_verifier.nondet_bool())]
_sv_verifier.check_type(box[0], A)
