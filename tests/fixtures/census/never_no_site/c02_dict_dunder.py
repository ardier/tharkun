# PROPS: arithmetic
# WITNESS: [{"kind":"int","value":0}]
import _sv_verifier
n = _sv_verifier.nondet_int()
print(int.__dict__["__floordiv__"](7, n))
