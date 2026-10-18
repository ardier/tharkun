import _sv_verifier
from _sv_verifier import nondet_bool


def nondet_bool():
    return True


flag = nondet_bool()
v = "s" if flag else 1
x = 0
_sv_verifier.check_type(v, str)
assert flag
