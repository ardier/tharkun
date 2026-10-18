# PROPS: arithmetic
# WITNESS: [{"kind":"int","value":0}]
import _sv_verifier
n = _sv_verifier.nondet_int()
t = (0).__class__.__mro__[0].__dict__["__rtruediv__"]
print(t(n, 1))
