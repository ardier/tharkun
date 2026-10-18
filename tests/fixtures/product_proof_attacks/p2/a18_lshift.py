import _sv_verifier
x = _sv_verifier.nondet_int()
y = _sv_verifier.nondet_int()
if 0 <= y < 3:
    assert (x << y) != 12
