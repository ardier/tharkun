import _sv_verifier
try:
    z = x
except NameError:
    z = 5
x = _sv_verifier.nondet_int()
assert z == x
