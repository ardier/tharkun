import _sv_verifier
x = _sv_verifier.nondet_int()
assert x % -3 <= 0
assert x % 3 != 2
