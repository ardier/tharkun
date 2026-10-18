import _sv_verifier


def g():
    b = _sv_verifier.nondet_bool()
    return b


x = g()
y = g()
assert x == y
