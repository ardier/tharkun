import _sv_verifier
x = _sv_verifier.nondet_int()
try:
    y = 1 // x
except ZeroDivisionError:
    assert False
