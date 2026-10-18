# PROPS: data-lookup
# WITNESS: [{"kind":"int","value":0}]
import _sv_verifier
n = _sv_verifier.nondet_int()
fmt = "{0}" if n != 0 else "{1}"
print(fmt.__getattribute__("format")(1))
