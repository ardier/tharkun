import _sv_verifier


class C:
    pass


c = C()
c.v = 0
flag = _sv_verifier.nondet_bool()
if flag:
    c.v = 1
d = c
assert c.v == (1 if flag else 0)
