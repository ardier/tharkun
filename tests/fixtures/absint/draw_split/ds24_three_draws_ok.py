import _sv_verifier

a = _sv_verifier.nondet_bool()
b = _sv_verifier.nondet_bool()
c = _sv_verifier.nondet_bool()
assert (a and b and c) == (a and (b and c))
v = "s" if a else (1 if b else 1.5)
if a:
    _sv_verifier.check_type(v, str)
elif b:
    _sv_verifier.check_type(v, int)
else:
    _sv_verifier.check_type(v, float)
