import _sv_verifier
x = _sv_verifier.nondet_int()
import sys
if x != 5:
    sys.exit(0)
assert False
