import _sv_verifier

flag = _sv_verifier.nondet_bool()


def f():
    return "s" if flag else 1


a = f()
flag = _sv_verifier.nondet_bool()
b = f()
assert isinstance(a, str) == isinstance(b, str)
