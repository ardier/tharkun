import _sv_verifier
x = _sv_verifier.nondet_int()
assert (x ^ (1 << 70)) != x - (1 << 70) or x < (1 << 70)
