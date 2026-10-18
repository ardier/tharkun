import _sv_verifier
x = _sv_verifier.nondet_int()
__builtins__.abs = lambda v: -1
assert abs(x) >= 0
