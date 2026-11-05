"""Batched child of ``proof:finite_domain``: run one program for a list of integer draw vectors in ONE process (stdlib only).

Request on stdin (JSON): ``{"source", "filename", "vectors": [[int, ...], ...], "alarm": seconds}``.  Reply on stdout: ``{"n": count,
"suspect": [[index, status, exception name, draws used, exception MRO names, is an ``Exception``], ...]}`` for every vector that did NOT
finish cleanly (an exception of any kind, ``SystemExit`` included, a timeout, or a number of draws other than the vector's length).  A clean
vector is not listed.

The parent (``sv/finite_domain.py::sweep_batch``) trusts this child for ONE claim only, "this vector ran to its end without raising", and
re-runs every suspect in the engine's own child, which alone decides a violation.  The claim holds only for programs the parent admitted
(``batch_admissible``: imports of ``_sv_verifier``, ``math`` and ``string`` only, ``nondet_int`` the only verifier function, no reflection, no file or
clock, determinism certified): each vector gets a fresh module namespace and a fresh stub module, so nothing but ``math`` (pure) survives
from one run to the next.
"""
import builtins
import contextlib
import io
import json
import signal
import sys
import types


class _Timeout(BaseException):
    pass


def main():
    req = json.load(sys.stdin)
    code = compile(req["source"], req["filename"], "exec")
    alarm = float(req.get("alarm", 5))
    out, sink = [], io.StringIO()

    def on_alarm(*_):
        raise _Timeout()

    signal.signal(signal.SIGALRM, on_alarm)
    for index, vector in enumerate(req["vectors"]):
        it, used = iter(vector), [0]

        def nondet_int(it=it, used=used):
            used[0] += 1
            try:
                return next(it)
            except StopIteration:
                raise RuntimeError("a draw past the vector") from None

        stub = types.ModuleType("_sv_verifier")
        stub.nondet_int = nondet_int
        sys.modules["_sv_verifier"] = stub
        namespace = {"__name__": "__main__", "__file__": req["filename"], "__builtins__": builtins}
        sink.seek(0)
        sink.truncate()
        status = exception = ""
        mro, plain = [], False
        signal.setitimer(signal.ITIMER_REAL, alarm)
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                exec(code, namespace)
            status = "clean"
        except BaseException as error:        # SystemExit and the timeout included: every non-clean vector goes back to the engine's child
            status, exception = "raised", type(error).__name__
            mro, plain = [c.__name__ for c in type(error).__mro__], isinstance(error, Exception)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if status != "clean" or used[0] != len(vector):
            out.append([index, status, exception, used[0], mro, plain])
    sys.stdout.write(json.dumps({"n": len(req["vectors"]), "suspect": out}))


if __name__ == "__main__":
    main()
