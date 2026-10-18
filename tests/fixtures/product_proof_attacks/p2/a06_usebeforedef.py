import _sv_verifier
x = _sv_verifier.nondet_int()
try:
    a = g(x)
except NameError:
    a = 0
def g(v):
    return v
assert a == x
