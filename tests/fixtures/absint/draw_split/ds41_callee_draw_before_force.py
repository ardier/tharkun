import _sv_verifier


def callee():
    k = _sv_verifier.nondet_bool()
    n = 0
    for i in range(2):
        n += 1
    r = "s" if k else n
    return r


x = (callee(), _sv_verifier.nondet_bool())
y = callee()
_sv_verifier.check_type(y, int)
