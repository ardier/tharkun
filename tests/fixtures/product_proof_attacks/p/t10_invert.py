import _sv_verifier
x = _sv_verifier.nondet_int()
assert ~x == -x - 1
assert ~x != 5
