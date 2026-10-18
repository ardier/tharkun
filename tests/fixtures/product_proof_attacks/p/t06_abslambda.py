import _sv_verifier
x = _sv_verifier.nondet_int()
abs = lambda v: -v
assert abs(x) >= 0
