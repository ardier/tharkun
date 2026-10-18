import _sv_verifier
x = _sv_verifier.nondet_int()
assert 2 ** x != 8 or x < 0
