import _sv_verifier
x = _sv_verifier.nondet_int()
if x == 7:
    raise ZeroDivisionError
y = 10 // (x - 3)
