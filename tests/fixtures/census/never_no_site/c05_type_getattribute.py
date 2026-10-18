# PROPS: arithmetic
# WITNESS: [{"kind":"int","value":0}]
import _sv_verifier
n = _sv_verifier.nondet_int()
m = type.__getattribute__(int, "__mod__")
print(m(5, n))
