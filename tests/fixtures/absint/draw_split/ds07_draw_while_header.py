import _sv_verifier

n = 0
while _sv_verifier.nondet_bool():
    n += 1
assert n < 3
