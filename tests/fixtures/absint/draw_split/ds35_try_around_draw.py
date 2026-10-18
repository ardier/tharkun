import _sv_verifier

try:
    flag = _sv_verifier.nondet_bool()
    v = "s" if flag else 1
    if flag:
        _sv_verifier.check_type(v, int)
except TypeError:
    pass
x = 0
