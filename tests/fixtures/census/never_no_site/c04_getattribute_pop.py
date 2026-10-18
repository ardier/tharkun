# PROPS: data-lookup
# WITNESS: [{"kind":"int","value":0}]
import _sv_verifier
n = _sv_verifier.nondet_int()
a = [1] if n != 0 else []
print(a.__getattribute__("pop")())
