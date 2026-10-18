import _sv_verifier


def pick():
    flag = _sv_verifier.nondet_bool()
    r = "s" if flag else 1
    if flag:
        _sv_verifier.check_type(r, str)
    else:
        _sv_verifier.check_type(r, int)
    return flag


a = pick()
b = pick()
assert a == b
