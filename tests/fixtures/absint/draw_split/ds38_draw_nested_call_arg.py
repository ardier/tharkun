import _sv_verifier


def pick(flag):
    return "s" if flag else 1


r = pick(_sv_verifier.nondet_bool())
y = 0
_sv_verifier.check_type(r, str)
