import _sv_verifier
x = _sv_verifier.nondet_int()
a = x or 7
assert a != 3
