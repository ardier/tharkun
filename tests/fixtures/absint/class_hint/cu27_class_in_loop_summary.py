import _sv_verifier


class A:
    pass


objs = []
for i in range(3):
    objs.append(A())
_sv_verifier.check_type(objs[0], A)
_sv_verifier.check_type(objs[2], A)
