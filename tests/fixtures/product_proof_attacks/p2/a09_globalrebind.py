import _sv_verifier
x = _sv_verifier.nondet_int()
def g(v):
    return 1
def h():
    global g
    g = len
h()
assert g([1, 2]) == 1 or x != 3
