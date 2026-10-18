import _sv_verifier

if _sv_verifier.nondet_bool():
    x = 1
else:
    x = "s"
y = x
assert isinstance(y, int) or isinstance(y, str)
