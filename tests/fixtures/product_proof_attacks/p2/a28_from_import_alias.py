from _sv_verifier import nondet_int as abs2
import _sv_verifier
x = abs2()
y = _sv_verifier.nondet_int()
assert x != y + 1 or x != 3
