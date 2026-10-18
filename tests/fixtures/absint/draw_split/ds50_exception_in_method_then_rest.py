import _sv_verifier


class A:
    def run(self):
        flag = _sv_verifier.nondet_bool()
        data = [1] if flag else []
        z = 0
        return data[0]


A().run()
