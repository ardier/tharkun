import _sv_verifier

if _sv_verifier.nondet_bool():
    x = 1
else:
    x = "s"
y = 0
assert isinstance(x, int)
