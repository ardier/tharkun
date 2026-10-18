import _sv_verifier

xs = [1, 2]
y = xs[_sv_verifier.nondet_bool()]
z = 0
assert y == 1
