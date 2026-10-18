import _sv_verifier
x = _sv_verifier.nondet_int()
def f(v):
    try:
        return 1
    finally:
        assert v != 4
f(x)
