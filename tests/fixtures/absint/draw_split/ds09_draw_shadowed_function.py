import _sv_verifier
from _sv_verifier import nondet_bool


def nondet_bool():
    return _sv_verifier.nondet_bool()


flag = nondet_bool()
v = "s" if flag else 1
x = 0
if flag:
    _sv_verifier.check_type(v, int)
