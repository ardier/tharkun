import _sv_verifier
x = _sv_verifier.nondet_int()
def g():
    return k
try:
    a = g()
except NameError:
    a = 9
k = 1
assert a == 1 or x != 3
