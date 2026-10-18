import _sv_verifier
x = _sv_verifier.nondet_int()
b = _sv_verifier.nondet_bool()
assert (b is True) or (b is False)
assert b * 3 != 3
