import _sv_verifier


class P:
    pass


class Q:
    @property
    def __class__(self):
        return P


q = Q()
_sv_verifier.check_type(q, P)
_sv_verifier.check_type(q, Q)
