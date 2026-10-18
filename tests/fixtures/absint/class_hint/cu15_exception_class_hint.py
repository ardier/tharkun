import _sv_verifier


class MyErr(Exception):
    pass


e = MyErr("x")
_sv_verifier.check_type(e, MyErr)
