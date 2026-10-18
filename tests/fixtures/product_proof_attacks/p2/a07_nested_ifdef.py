import _sv_verifier
x = _sv_verifier.nondet_int()
def g(v):
    return 1
if x == 3:
    def g(v):
        return 2
assert g(x) == 1
