import _sv_verifier
x = _sv_verifier.nondet_int()
if True:
    from operator import neg as abs
assert abs(x) >= 0 or x == 0
