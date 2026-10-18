import _sv_verifier

c = _sv_verifier.nondet_bool()
v = (_sv_verifier.nondet_bool() if c else True)
x = 0
assert v
