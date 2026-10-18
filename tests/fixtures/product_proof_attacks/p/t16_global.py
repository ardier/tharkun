import _sv_verifier
x = _sv_verifier.nondet_int()
g = 0
def f():
    global g
    g += 1
    return g
f()
assert f() != 2
