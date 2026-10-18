import _sv_verifier


class W:
    def __bool__(self):
        raise KeyError("w")


flag = _sv_verifier.nondet_bool()
v = W() if flag else 1
x = 0
if v:
    pass
