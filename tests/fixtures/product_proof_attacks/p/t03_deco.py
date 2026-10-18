import _sv_verifier
x = _sv_verifier.nondet_int()
def g(fn):
    assert False
@g
def f():
    return 1
assert x != x + 1
