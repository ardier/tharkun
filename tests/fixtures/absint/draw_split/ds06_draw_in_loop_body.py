import _sv_verifier

xs = []
for i in range(2):
    b = _sv_verifier.nondet_bool()
    xs.append(b)
    n = len(xs)
assert xs[0] == xs[1]
