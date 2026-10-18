import _sv_verifier
x = _sv_verifier.nondet_int()
def g(v):
    return 1
def h(v):
    return 2
r = (g if x != 3 else h)(x)
assert r == 1
