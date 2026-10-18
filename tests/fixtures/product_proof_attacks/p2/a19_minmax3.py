import _sv_verifier
x = _sv_verifier.nondet_int()
assert max(x, 3, 1) != 5
