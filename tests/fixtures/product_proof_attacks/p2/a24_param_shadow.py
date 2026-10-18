import _sv_verifier
x = _sv_verifier.nondet_int()
def k(v):
    return 2
def g(k):
    return k
assert g(x) == x
