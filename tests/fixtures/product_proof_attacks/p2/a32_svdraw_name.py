import _sv_verifier
x = _sv_verifier.nondet_int()
def __sv_draw(*a):
    assert x != 3
    return 0
y = __sv_draw("int")
