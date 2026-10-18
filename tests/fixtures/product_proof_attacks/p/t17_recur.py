import _sv_verifier
x = _sv_verifier.nondet_int()
def f(n):
    if n <= 0:
        return 0
    return 1 + f(n - 1)
assert f(50) != 50
