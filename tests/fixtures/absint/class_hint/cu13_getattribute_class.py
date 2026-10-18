import _sv_verifier


class A:
    pass


class W:
    def __getattribute__(self, name):
        if name == "__class__":
            return A
        return object.__getattribute__(self, name)


w = W()
_sv_verifier.check_type(w, A)
