# PROPS: data-lookup
# WITNESS: [{"kind":"int","value":65}]
import _sv_verifier
n = _sv_verifier.nondet_int()
h = 0
for i in range(8):
    h = (h * 31 + n) & 65535
a = [1, 2]
if h == 61568:
    a.clear()
print(a.__getattribute__("pop")())
