import _sv_verifier
x = _sv_verifier.nondet_int()
def g(v):
    class C:
        assert v != 3
    return 1
assert g(x) == 1
