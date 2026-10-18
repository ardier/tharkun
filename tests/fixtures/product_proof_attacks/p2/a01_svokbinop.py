import _sv_verifier
x = _sv_verifier.nondet_int()
def __sv_ok_binop(op, a, b):
    assert a != 3
    return True
__sv_ok_binop("+", x, 1)
