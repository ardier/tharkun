import _sv_verifier

t = (_sv_verifier.nondet_bool(), _sv_verifier.nondet_bool())
x = 1
assert t[0] == t[1]
