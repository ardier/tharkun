import _sv_verifier
x = _sv_verifier.nondet_int()
_sv_verifier.nondet_int = lambda: 3
assert x != 3
