import _sv_verifier


def f():
    flag = _sv_verifier.nondet_bool()
    if flag:
        return "s"
    x = 1
    return x


r = f()
_sv_verifier.check_type(r, int | str)
