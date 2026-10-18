import _sv_verifier
x = _sv_verifier.nondet_int()
a = x and 7
assert a != 0 or x != 0
