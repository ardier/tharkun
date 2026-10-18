import _sv_verifier

flag = _sv_verifier.nondet_bool()
if flag:
    raise KeyError("k")
y = 1
