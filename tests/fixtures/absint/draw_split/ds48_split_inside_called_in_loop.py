import _sv_verifier


def f():
    flag = _sv_verifier.nondet_bool()
    if flag:
        return 1
    return 2


total = 0
for i in range(3):
    total += f()
assert total != 4
