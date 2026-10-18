import _sv_verifier

flag = _sv_verifier.nondet_bool()
if flag:
    raise ValueError("v")
y = 1
assert y == 1
