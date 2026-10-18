import _sv_verifier
from _sv_verifier import nondet_bool as nb

flag = nb()
v = "s" if flag else 1
if flag:
    _sv_verifier.check_type(v, int)
else:
    _sv_verifier.check_type(v, int)
