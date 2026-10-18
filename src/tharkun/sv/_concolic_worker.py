"""Helper process of ``tharkun.sv.concolic``: drive the harness to each violation site.

Runs as ``python _concolic_worker.py REQUEST.json`` in its own process (the target's
code executes here, under the parent's address-space cap).  A stub ``_sv_verifier``
answers every draw the harness could not lift with the child's default and logs it,
so the logged sequence is the child's draw vector.
"""

import importlib.util
import json
import os
import signal
import sys
import time
import types

# the draw domains of the float helpers live in ``_child.py`` (the judge of every confirmed FALSE); the insert lives only for the import
_SV_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SV_DIR)
try:
    import _child
finally:
    sys.path.remove(_SV_DIR)

#: ``float``: the default ``_child.py`` gives ``nondet_float()`` ([internal ref]: a value of its domain, ``0.0`` is refused)
_DEFAULTS = {"bool": False, "int": 0, "float": -0.0, "str": "", "argv": [], "argv_full": []}


def _stub():
    m = types.ModuleType("_sv_verifier")
    m.LOG = []

    def mk(kind):
        def f(*a, **k):
            v = _DEFAULTS[kind]
            m.LOG.append((kind, v))
            return v
        return f

    def float_range(a, b):
        # [internal ref]-followup: a default logged verbatim can fall outside the call's domain (e.g. nondet_float_range(5, 10)) --
        # snap it into the call's own domain BEFORE logging, same discipline as the hybrid/Atheris worker fix, so a
        # candidate vector this worker's unlifted stub produces is always something _child.py's domain check can
        # confirm, never something it must refuse.  [internal ref]: the domain is what random.uniform(a, b) can return.
        v = _child.snap_float_range(0.0, a, b)
        m.LOG.append(("float", v))
        return v

    for name, kind in (("nondet_bool", "bool"), ("nondet_int", "int"), ("nondet_float", "float"),
                       ("nondet_str", "str")):
        setattr(m, name, mk(kind))
    m.nondet_float_range = float_range
    m.nondet_list = lambda elem: []
    m.nondet_dict = lambda k, v: {}
    return m


def _enc(kind, v):
    return float(v).hex() if kind == "float" else list(v) if kind in ("argv", "argv_full") else v


def _vector(harness, inputs):
    """The draws one run of the harness consumes, in execution order."""
    stub = sys.modules["_sv_verifier"]
    spec = importlib.util.spec_from_file_location("sv_harness_replay", harness)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    del stub.LOG[:]
    try:
        mod.__sv_harness(**inputs)
    except BaseException:  # noqa: BLE001 -- the run is meant to raise
        pass
    return [{"kind": k, "value": _enc(k, v)} for k, v in stub.LOG]


class _SiteTimeout(BaseException):
    """A site's (or the stage's) time is spent.  A BaseException ON PURPOSE: the alarm fires inside the program under test
    (``_load``, the symbolic run, ``driver.drive_to_line``'s concrete runs), and a broad ``except Exception`` there -- the
    program's own, or the driver's, which reads an exception of the subject as "a valid trace up to the raise" -- would swallow
    a one-shot alarm; the worker would then run on past its deadline until the parent kills it at the budget + 5 s
    (measured on bm_go_nondet at the 9 s triage budget: 11.4 s).  A subject that catches ``BaseException`` in a loop still holds
    the worker until that kill: an in-process alarm cannot do more."""


def _alarm(_signum, _frame):
    raise _SiteTimeout()


def _default(kind):
    return [] if kind.startswith("list:") else _DEFAULTS[kind]


def _order(sites):
    """The frontier: the sites nearest the draws first -- sites in the main block before
    sites in callees, fewer enclosing guards first, then source order (lazy: each site is
    solved only when its turn comes and the budget still allows)."""
    return sorted(sites, key=lambda s: (s["function"] != "__sv_harness", len(s.get("guards") or []),
                                        s["line"]))


def _load(harness, name):
    spec = importlib.util.spec_from_file_location(name, harness)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stream_lens(harness, seed, seconds):
    """How many draws each stream parameter serves in one concrete run of the harness
    with the default seed -- the length the symbolic executor models it with."""
    streams = {p: [] for p, v in seed.items() if isinstance(v, list)}
    if not streams:
        return {}
    mod = _load(harness, "sv_harness_lens")
    inputs = dict(seed, **streams)
    signal.alarm(max(1, seconds))
    try:
        mod.__sv_harness(**inputs)
    except BaseException:  # noqa: BLE001
        pass
    finally:
        signal.alarm(0)
    pos = getattr(mod, "__SV_POS", {})
    return {p: pos.get(id(lst), 0) for p, lst in streams.items()}


def _symbolic(req, site, lens, seconds):
    """``(inputs, note)``: the draws the symbolic executor solves for, or ``None``."""
    try:
        from tharkun.sv import _symexec
    except ImportError as e:            # z3 not installed
        return None, f"symexec unavailable: {e}"
    signal.alarm(max(1, seconds))
    try:
        mod = _load(req["harness"], "sv_harness_sym")
        with open(req["harness"]) as fh:
            text = fh.read()
        got = _symexec.solve_site(text, vars(mod), "__sv_harness", site["line"], req["params"],
                                  req["kinds"], lens, timeout_ms=max(1000, seconds * 1000 - 500))
        return got, "sat" if got is not None else "unsat or unknown"
    except _SiteTimeout:
        return None, "symexec: site budget spent"
    except _symexec.Unsupported as e:
        return None, f"symexec unsupported: {e}"
    except Exception as e:  # noqa: BLE001
        return None, f"symexec error {type(e).__name__}: {e}"[:200]
    finally:
        signal.alarm(0)


def _symbolic_all(req, sites, lens, seconds):
    """``({line: (outcome, inputs_or_reason)}, note)`` from ONE symbolic run of the whole
    harness toward every site (:func:`tharkun.sv._symexec.solve_targets`)."""
    try:
        from tharkun.sv import _symexec
    except ImportError as e:            # z3 not installed
        return {}, f"symexec unavailable: {e}"
    signal.alarm(max(1, seconds))
    try:
        mod = _load(req["harness"], "sv_harness_all")
        with open(req["harness"]) as fh:
            text = fh.read()
        per_site = max(500, int(seconds * 1000 / max(1, len(sites))))
        got = _symexec.solve_targets(text, vars(mod), "__sv_harness", [s["line"] for s in sites],
                                     req["params"], req["kinds"], lens, timeout_ms=per_site,
                                     call_targets=req.get("call_targets"),
                                     proven_classes=req.get("proven_classes", ()))
        return got, "ok"
    except _SiteTimeout:
        return {}, "symexec: budget spent"
    except Exception as e:  # noqa: BLE001
        return {}, f"symexec error {type(e).__name__}: {e}"[:200]
    finally:
        signal.alarm(0)


def _vector_capped(harness, inputs, seconds, confine):
    signal.alarm(max(1, seconds))
    try:
        with confine():
            return _vector(harness, inputs)
    except _SiteTimeout:
        return []
    finally:
        signal.alarm(0)


def main(req_path, *, persistent=False):
    with open(req_path) as fh:
        req = json.load(fh)
    files = tuple(dict.fromkeys([*req.get("files", [req["harness"]]), req["harness"]]))
    scratch = req.get("scratch", os.path.dirname(os.path.abspath(req_path)))
    try:
        return _run_request(req_path, req)
    finally:
        if persistent:
            from tharkun.solve.environment import install_files, refuse_threads

            flags = set()
            # Result I/O is complete; retain the model through interpreter cleanup,
            # including exceptional exits after target objects were created.
            install_files(files, scratch, flags)
            refuse_threads(flags)


def _run_request(req_path, req):
    from tharkun.solve.environment import confined

    files = tuple(dict.fromkeys([*req.get("files", [req["harness"]]), req["harness"]]))
    flags = set()

    def confine():
        return confined(files, scratch=req.get("scratch", os.path.dirname(os.path.abspath(req_path))), flags=flags, threads=False)

    from tharkun.solve.extern import native_import_reason
    from tharkun.sv.sites import harness_namespace_refusal
    source_binding = req.get("source_binding")
    if (not isinstance(source_binding, (list, tuple)) or len(source_binding) != 3
            or not all(isinstance(value, str) for value in source_binding)
            or not os.path.isabs(source_binding[1])):
        refusal = "missing or malformed source binding"
    else:
        source_binding = tuple(source_binding)
        generated, original_path, original_source = source_binding
        with open(req["harness"], "rb") as stream:
            matches = stream.read() == generated.encode("utf-8")
        refusal = ((harness_namespace_refusal(original_source)
                    or native_import_reason(original_source, original_path, reversible=True)) if matches
                   else "generated source changed after admission")
    if refusal:
        with open(req["out"], "w") as stream:
            json.dump({"reached": [], "missed": [], "solver_calls": 0,
                       "flags": ["import_state_unmodelled"], "refusal": refusal}, stream)
        return

    sys.path.insert(0, req["program_dir"])
    # Normally a fresh, disposable subprocess (python _concolic_worker.py req.json)
    # that exits right after writing the result, so a leaked
    # sys.modules["_sv_verifier"] never mattered -- restoring it anyway makes
    # main() safe in-process too (see _child.py::main's same fix; a leftover raw
    # stub here is exactly what makes avengers_seed.py's
    # _sv_verifier_stub_on_path crash for whatever test runs next in the same
    # process).
    old_sv = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = _stub()
    try:
        reached, missed = [], []
        kinds = dict(zip(req["params"], req["kinds"]))
        signal.signal(signal.SIGALRM, _alarm)
        sites = _order(req["sites"])
        calls = 0
        # the engine runs this worker twice: first only the cheap symbolic stage (before any
        # execution of the program), later the per-site drive on the sites left open
        symbolic, drive = req.get("symbolic", True), req.get("drive", True)
        with confine():
            lens = _stream_lens(req["harness"], req["seed"], 30)
        # stage 1: one symbolic run of the whole harness toward every site at once, with at
        # most half of the time; what it cannot decide goes to the per-site stages below
        left = req["deadline"] - time.time()
        if symbolic and sites:
            with confine():
                sym, sym_note = _symbolic_all(req, sites, lens, max(1, int(left / 2) if drive else int(left)))
            calls += 1
        else:
            sym, sym_note = {}, "symbolic stage done before"
        open_sites = []
        for site in sites:
            outcome, got = sym.get(site["line"], ("open", sym_note))
            if outcome == "sat":
                reached.append({"line": site["line"], "exc": site["exc"], "rounds": 1,
                                "entry": "__sv_harness", "techniques": ["symexec"],
                                "vector": _vector_capped(req["harness"], got, min(10, max(1, int(req["deadline"] - time.time()))), confine)})
            elif outcome in ("unsat", "unreached"):
                # no modelled run fires it: a guard-flipping drive would search in vain
                missed.append({"line": site["line"], "outcome": outcome, "status": f"symexec: {got}"})
            else:
                open_sites.append((site, f"symexec {outcome}: {got}"))
        if not drive:
            missed.extend({"line": site["line"], "outcome": "open", "status": note} for site, note in open_sites)
            open_sites = []
        from tharkun.driver import drive_to_line
        for i, (site, first_note) in enumerate(open_sites):
            left = req["deadline"] - time.time()
            if left <= 1:
                missed.append({"line": site["line"], "outcome": "open", "status": "no time left"})
                continue
            # each site gets its share of what is left, so one hard site cannot starve the rest
            share = max(1, int(left / (len(open_sites) - i)))
            attempts = [("__sv_harness", dict(req["seed"]), None)]
            feed = req["feeds"].get(site["function"])
            if feed:
                # the site sits in a function the main block calls with draws: solve that
                # function's own guards, then map its parameters back to the draws
                attempts.append((site["function"],
                                 {p: _default(kinds[d]) for p, d in feed.items()}, feed))
            status = [first_note]
            if site["function"] == "__sv_harness":
                # the backward slice to this one site: what the whole-run model could not reach
                with confine():
                    got, note = _symbolic(req, site, lens, share)
                calls += 1
                if got is not None:
                    reached.append({"line": site["line"], "exc": site["exc"], "rounds": 1,
                                    "entry": "__sv_harness", "techniques": ["symexec"],
                                    "vector": _vector_capped(req["harness"], got, share, confine)})
                    continue
                status.append(note)
            for fn, seed, feed_map in attempts:
                signal.alarm(share)
                try:
                    r = drive_to_line(req["harness"], fn, site["line"], seed, max_rounds=16,
                                      confine=confine, _source_binding=source_binding)
                except _SiteTimeout:
                    status.append(f"{fn}: site budget ({share}s) spent")
                    continue
                except Exception as e:  # noqa: BLE001
                    with confine():
                        status.append(f"{fn}: error {type(e).__name__}: {e}"[:200])
                    continue
                finally:
                    signal.alarm(0)
                calls += r.rounds
                if r.status != "reached":
                    status.append(f"{fn}: {r.status}")
                    continue
                inputs = r.inputs
                if feed_map is not None:
                    inputs = dict(req["seed"], **{d: r.inputs[p] for p, d in feed_map.items()})
                reached.append({"line": site["line"], "exc": site["exc"], "rounds": r.rounds,
                                "entry": fn,
                                "techniques": [s.technique for s in r.history if s.technique],
                                "vector": _vector_capped(req["harness"], inputs, share, confine)})
                break
            else:
                missed.append({"line": site["line"], "outcome": "open", "status": "; ".join(status)})
    finally:
        if old_sv is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old_sv
    with confine():
        document = json.dumps({"reached": reached, "missed": missed, "solver_calls": calls}, default=repr)
    if flags:
        # Candidates still require fresh original replay. Environment-dependent
        # observations must not close a site or certify an exhausted search.
        payload = json.loads(document)
        payload["flags"] = sorted(flags)
        document = json.dumps(payload)
    with open(req["out"], "w") as fh:
        fh.write(document)


if __name__ == "__main__":
    main(sys.argv[1], persistent=True)
