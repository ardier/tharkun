# PROPS: arithmetic
# WITNESS: [{"kind":"float","value":"0x0.0p+0"}]
import _sv_verifier
import random
x = _sv_verifier.nondet_float()
random.seed(1)
g = random.Random(2).__getattribute__("expovariate")
print(g(x))
