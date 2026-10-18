# PROPS: data-lookup
# WITNESS: [{"kind":"str","value":""}]
import _sv_verifier
k = _sv_verifier.nondet_str()
d = {"x": 1}
print(object.__getattribute__(d, "__getitem__")(k))
