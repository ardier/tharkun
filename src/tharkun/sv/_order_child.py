"""Batch child of the order-only small model (``sv/order_only.py``): run one program on many draw vectors.

One interpreter, one ``fork`` per vector.  The forked grandchild installs a minimal ``_sv_verifier`` that replays the
vector, executes the program as a fresh ``__main__`` module (so the program's own globals and imports are never shared
between vectors), and reports how it ended: clean, ``SystemExit``, an uncaught exception (the builtin classes in its
MRO), a resource error, or a death.  Nothing here is modelled: no random/clock patches, no import hooks -- the order-only
rule admits only programs whose runs are a function of the draws, and the scheduler checks that with the determinism
certificate before it believes a TRUE.  A vector the program did not consume exactly (a draw of another kind, a draw
past the end, draws left over) is an ANOMALY: the vector was not the program's input, so nothing is concluded.

Imports nothing from tharkun (``tests/test_layering.py``).  Request: ``{"program", "vectors": [[kinds, values], ...],
"classes", "out", "memory", "run_cpu"}``; result: ``{"ran", "violation": {"index", "exc", "mro"} | null, "anomaly"}``.
Stops at the first violation or anomaly.
"""
import builtins
import json
import os
import resource
import signal
import sys
import tokenize
import types

_BUILTIN_CLASSES = {name: obj for name, obj in vars(builtins).items() if isinstance(obj, type)}


def _mro(exc):
    return [c.__name__ for c in type(exc).__mro__ if _BUILTIN_CLASSES.get(c.__name__) is c]


def _verifier(kinds, values, log):
    m = types.ModuleType("_sv_verifier")
    pos = [0]

    def take(kind):
        i = pos[0]
        pos[0] += 1
        if i >= len(kinds) or kinds[i] != kind:
            log.append("draw %d is not a %s draw of the vector" % (i, "bool" if kind == "b" else "int"))
            return False if kind == "b" else 0
        return bool(values[i]) if kind == "b" else int(values[i])

    def nondet_bool():
        return take("b")

    def nondet_int():
        return take("i")

    def nondet_list(elem):
        out = []
        while nondet_bool():
            out.append(elem())
        return out

    def missing(name):
        log.append("unmodelled _sv_verifier attribute %s" % name)
        raise AttributeError(name)

    m.nondet_bool, m.nondet_int, m.nondet_list, m.__getattr__ = nondet_bool, nondet_int, nondet_list, missing
    return m, pos


def _one(program, code, kinds, values, run_cpu, memory):
    """In the grandchild: run the program on one vector; the verdict dict."""
    log = []
    mod, pos = _verifier(kinds, values, log)
    sys.modules["_sv_verifier"] = mod
    main = types.ModuleType("__main__")
    main.__file__ = program
    sys.modules["__main__"] = main
    sys.path.insert(0, os.path.dirname(program))
    sys.argv = [program]
    resource.setrlimit(resource.RLIMIT_CPU, (run_cpu, run_cpu + 1))
    if memory:
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    status, mro, exc = "clean", [], None
    try:
        exec(code, main.__dict__)
    except SystemExit:
        status = "exit"
    except (RecursionError, MemoryError) as e:
        status, exc = "resource", type(e).__name__
    except Exception as e:
        status, mro, exc = "raised", _mro(e), type(e).__name__ + ": " + str(e)[:160]
    except BaseException as e:
        status, exc = "other", type(e).__name__
    if pos[0] != len(kinds):
        log.append("the program consumed %d of %d draws (it ended %s %s)" % (pos[0], len(kinds), status, exc))
    return {"status": status, "mro": mro, "exc": exc, "log": log}


def _forked(program, code, kinds, values, run_cpu, memory):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        rc = 0
        try:
            os.close(r)
            null = os.open(os.devnull, os.O_RDWR)
            for fd in (0, 1, 2):
                os.dup2(null, fd)
            res = _one(program, code, kinds, values, run_cpu, memory)
            os.write(w, json.dumps(res).encode())
        except BaseException:
            rc = 1
        finally:
            os._exit(rc)
    os.close(w)
    chunks = []
    while True:
        b = os.read(r, 65536)
        if not b:
            break
        chunks.append(b)
    os.close(r)
    _, wait = os.waitpid(pid, 0)
    try:
        return json.loads(b"".join(chunks))
    except ValueError:
        return {"status": "died", "mro": [], "exc": "signal %d" % (wait & 0x7F), "log": []}


def main(req_path):
    with open(req_path) as fh:
        req = json.load(fh)
    program, classes = req["program"], set(req["classes"])
    with tokenize.open(program) as fh:       # honours a PEP 263 encoding cookie, like sv/engine.py::read_source
        source = fh.read()
    code = compile(source, program, "exec")   # once, before any fork: a syntax error is every vector's outcome
    result = {"ran": 0, "violation": None, "anomaly": None}
    for i, (kinds, values) in enumerate(req["vectors"]):
        res = _forked(program, code, kinds, values, req.get("run_cpu", 5), req.get("memory"))
        result["ran"] = i + 1
        if res["log"]:
            result["anomaly"] = "vector %d: %s" % (i, "; ".join(res["log"]))
            break
        if res["status"] in ("resource", "other", "died"):
            result["anomaly"] = "vector %d did not complete (%s %s)" % (i, res["status"], res["exc"])
            break
        if res["status"] == "raised" and any(c in classes for c in res["mro"]):
            result["violation"] = {"index": i, "exc": res["exc"], "mro": res["mro"]}
            break
    with open(req["out"], "w") as fh:
        json.dump(result, fh)


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    main(sys.argv[1])
