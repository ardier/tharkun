import _sv_verifier
x = _sv_verifier.nondet_int()
def __sv_index(seq, i):
    assert i != 3
    return 0
y = __sv_index([1, 2], x)
