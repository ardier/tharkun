# PROPS: arithmetic
# WITNESS: [{"kind":"int","value":0}]
import _sv_verifier
n = _sv_verifier.nondet_int()
f = object.__getattribute__(7, "__floordiv__")
print(f(n))
