import _sv_verifier

a = _sv_verifier.nondet_bool()
b = _sv_verifier.nondet_bool()
v = "s" if a else 1
if b:
    _sv_verifier.check_type(v, str)
else:
    _sv_verifier.check_type(v, int)
