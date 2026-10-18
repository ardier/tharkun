import _sv_verifier
x = _sv_verifier.nondet_int()
b = _sv_verifier.nondet_bool()
assert x // (b + 1) != x or b or True
assert -x // 2 * 2 != -x - 1 or x % 2 == 0 or x != 3
