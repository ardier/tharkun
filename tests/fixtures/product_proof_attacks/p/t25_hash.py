import _sv_verifier
x = _sv_verifier.nondet_int()
assert hash(x) != -2 or x == -2
