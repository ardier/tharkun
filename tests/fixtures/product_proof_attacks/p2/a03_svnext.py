import _sv_verifier
x = _sv_verifier.nondet_int()
def __sv_next(a, b):
    assert a != 3
    return 0
y = __sv_next(x, x)
