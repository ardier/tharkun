import _sv_verifier
x = _sv_verifier.nondet_int()
def f(v):
    assert v != 3
    return v
f(x)
f(x + 1)
