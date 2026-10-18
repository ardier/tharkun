import _sv_verifier


def f():
    flag = _sv_verifier.nondet_bool()
    v = "s" if flag else 1
    x = 0
    if flag:
        _sv_verifier.check_type(v, str)
    return x


try:
    f()
except KeyError:
    pass
