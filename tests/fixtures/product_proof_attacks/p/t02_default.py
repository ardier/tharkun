import _sv_verifier
x = _sv_verifier.nondet_int()
def g():
    assert False
def f(a=g()):
    return a
assert x == x or True
assert x != x + 1
