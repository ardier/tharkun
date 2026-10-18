import _sv_verifier
def nd():
    return _sv_verifier.nondet_int()
def nb():
    return _sv_verifier.nondet_bool()
class P:
    __slots__ = ('v', 'w')
    def __init__(self, v):
        self.v = v
        if v > 2:
            self.w = v
    def get(self):
        return self.v + 1
def f(q):
    return q - 1 if q > 0 else q + 1
a = nd()
b = nd()
c = 1
fl = 1.0
p = P(b)
xs = [0, 1]
t = (0, 1)
if b != nd():
    c = (a - (c % nd()))
    if nd() <= -2:
        xs = list(range(min((len(xs) // c), 5)))
        xs.pop()
a = b
c = abs(abs((a * 3)))
b = p.get()
a = f(b)
if xs[c] >= len(xs):
    xs.append(len(xs))
    xs[len(t)] = (ord("bc") - min(b, len(xs)))
    p = P(nd())
else:
    for j in range(min(len(xs), 6)):
        xs = [0] * min((c * 1), 5)
        p = None
        c = xs[(len(xs) // (len(xs) // a))]
    a = f(c)
if ord("") > c:
    xs = [1, 2, 3] * min(len(xs), 5)
    xs = list(range(min(len(xs), 5)))
