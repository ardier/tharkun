import _sv_verifier

if _sv_verifier.nondet_bool() == _sv_verifier.nondet_bool():
    x = 1
else:
    x = 2
y = 0
assert x == 1
