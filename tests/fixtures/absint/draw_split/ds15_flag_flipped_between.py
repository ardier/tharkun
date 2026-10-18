import _sv_verifier

flag = _sv_verifier.nondet_bool()
v = "s" if flag else 1
flag = not flag
if flag:
    _sv_verifier.check_type(v, str)
