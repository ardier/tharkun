import _sv_verifier


class A:
    def f(self):
        return 1


class B(A):
    pass


class C(A):
    def f(self):
        return "c"


flag = _sv_verifier.nondet_bool()
o = B() if flag else C()
_sv_verifier.check_type(o, A)
r = o.f()
if flag:
    _sv_verifier.check_type(r, int)
else:
    _sv_verifier.check_type(r, str)
