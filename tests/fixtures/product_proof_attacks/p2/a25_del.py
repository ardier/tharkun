import _sv_verifier
x = _sv_verifier.nondet_int()
y = x
del y
y2 = 1
assert x != 3
