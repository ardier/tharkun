import _sv_verifier
x = _sv_verifier.nondet_int()
def __sv_site():
    assert False
if x == 3:
    __sv_site()
