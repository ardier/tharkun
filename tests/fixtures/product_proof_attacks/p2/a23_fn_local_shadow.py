import _sv_verifier
x = _sv_verifier.nondet_int()
def g(v):
    abs = len
    return abs([v, v])
assert g(x) != 2
