import _sv_verifier

a1 = _sv_verifier.nondet_bool()
a2 = _sv_verifier.nondet_bool()
a3 = _sv_verifier.nondet_bool()
a4 = _sv_verifier.nondet_bool()
a5 = _sv_verifier.nondet_bool()
a6 = _sv_verifier.nondet_bool()
a7 = _sv_verifier.nondet_bool()
a8 = _sv_verifier.nondet_bool()
assert not (a1 and a2 and a3 and a4 and a5 and a6 and a7 and a8)
