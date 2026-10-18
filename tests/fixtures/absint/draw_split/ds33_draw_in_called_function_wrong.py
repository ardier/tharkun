import _sv_verifier


def pick(flag):
    if flag:
        return "s"
    return 1


flag = _sv_verifier.nondet_bool()
r = pick(flag)
if flag:
    _sv_verifier.check_type(r, int)
else:
    _sv_verifier.check_type(r, int)
