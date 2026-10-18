import _sv_verifier
x = _sv_verifier.nondet_int()
b = _sv_verifier.nondet_bool()
assert b + b != 2
