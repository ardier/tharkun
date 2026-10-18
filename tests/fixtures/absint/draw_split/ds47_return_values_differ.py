import _sv_verifier


def f():
    flag = _sv_verifier.nondet_bool()
    if flag:
        return 1
    return 2


r = f()
assert r == 1
