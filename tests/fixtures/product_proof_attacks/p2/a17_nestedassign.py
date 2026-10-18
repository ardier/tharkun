import _sv_verifier
x = _sv_verifier.nondet_int()
def g(v):
    return 1
def h(v):
    return 2
if x == 3:
    g = h
assert g(x) == 1
