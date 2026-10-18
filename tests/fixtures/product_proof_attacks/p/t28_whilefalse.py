import _sv_verifier
x = _sv_verifier.nondet_int()
while x == 5:
    assert False
