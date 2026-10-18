import _sv_verifier
x = _sv_verifier.nondet_int()
s = "a" if x > 0 else "b"
assert s != "a"
