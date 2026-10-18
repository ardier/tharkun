# PROPS: arithmetic
# WITNESS: [{"kind":"int","value":1657}]
import _sv_verifier
n = _sv_verifier.nondet_int()
h = 0
for i in range(8):
    h = (h * 31 + n) % 10007
d = 1 if h != 4242 else 0
f = object.__getattribute__(7, "__floordiv__")
print(f(d))
