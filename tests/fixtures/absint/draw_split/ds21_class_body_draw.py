import _sv_verifier


class C:
    flag = _sv_verifier.nondet_bool()
    x = 1 if flag else "s"
    y = 0


assert isinstance(C.x, int)
