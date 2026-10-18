import _sv_verifier
x = _sv_verifier.nondet_int()
def abs(v):
    return -v
assert abs(x) >= 0
