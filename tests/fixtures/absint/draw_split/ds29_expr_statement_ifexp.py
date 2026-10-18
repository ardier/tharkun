import _sv_verifier

xs = []
xs.append(1) if _sv_verifier.nondet_bool() else None
y = 0
assert len(xs) == 1
