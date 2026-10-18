import _sv_verifier


def f(n):
    flag = _sv_verifier.nondet_bool()
    v = "s" if flag else 1
    x = 0
    if n > 0:
        f(n - 1)
    if flag:
        _sv_verifier.check_type(v, int)
    return x


f(2)
