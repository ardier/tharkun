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
for i in range(min(min((b // c), len(xs)), 6)):
    g = 0
    while g < 5 and b <= tuple(range(min(len(xs), 4)))[(len(xs) + a)]:
        g += 1
        b = ((abs(-3) * 2) // len(xs))
        c = c
    xs[len(xs)] = 0
    fl = fl * 2.0
if len(xs) > (c - ord("Z")):
    b = abs(nd())
    b = a
    if -1 == (tuple(xs)[c] * -1):
        a = 1
        a = b
    else:
        c = ((len(t) * 2) * 2)
else:
    a = xs[abs(a)]
xs.append(((nd() + c) * 1))
b = p.v
g = 0
while g < 5 and ord("Z") != c:
    g += 1
    for j in range(min((-2 + (c - nd())), 6)):
        b = abs(nd())
        c = abs(a)
        a = len(xs)
t = tuple(range(min((min(b, len(xs)) // min(len(xs), b)), 4)))
