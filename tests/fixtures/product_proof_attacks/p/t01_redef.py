import _sv_verifier
x = _sv_verifier.nondet_int()
def f():
    return 1
a = f()
def f():
    return 2
assert a != 1
