import _sv_verifier

flag = _sv_verifier.nondet_bool()
v = 0
for i in range(3):
    if flag:
        v += 1
assert v == 3
