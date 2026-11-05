"""Helper process of ``tharkun.sv.concolic``: drive the harness to each violation site.

Runs as ``python _concolic_worker.py REQUEST.json`` in its own process (the target's
code executes here, under the parent's address-space cap).  A stub ``_sv_verifier``
answers every draw the harness could not lift with the child's default and logs it,
so the logged sequence is the child's draw vector.

``"mode": "reach"`` ([internal ref], ``concolic.py::reach``) is the same process, the same admission (source
binding, reversible import admission, confinement) and the same driver: one
:func:`tharkun.driver.drive_to_line` per hint target, from the request's entry function, with no draw
harness and no stub.  It reports the inputs of each reached target and the facts the drive's knowledge
base gained (:func:`_reach`).
"""

import contextlib
import importlib.util
import io
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

#: the REAL result writer: a sealed worker ([internal ref]) has ``open`` under the file model, and its own record must never be
#: redirected to a scratch shadow (``_child.py::_RESULT_FILE_IO``, same reason)
_RESULT_FILE_IO = io.FileIO


def _write_out(path, obj):
    with _RESULT_FILE_IO(path, "w") as fh:
        fh.write(json.dumps(obj, default=repr).encode())


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


def _symbolic_all(req, sites, lens, seconds, world=None, world_results=None):
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
                                     proven_classes=req.get("proven_classes", ()),
                                     module_sources=req.get("module_sources"),
                                     **({"world": world, "world_results": world_results} if world is not None else {}))
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


def _operand(value):
    """``{"kind", "value"}`` for a value a draw can hold and the fuzzer can use (short printable text, a 64-bit int)."""
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int and -2**63 <= value < 2**63:
        return {"kind": "int", "value": value}
    if type(value) is str and 0 < len(value) <= 64 and value.isprintable():
        return {"kind": "str", "value": value}
    return None


def _frontier_inputs(req, site, seconds, confine, operands=None):
    """Solve the live blocking guard with the rest of a fuzzed prefix fixed; replay decides.  The constants of that
    guard, the live values it compares and the solved values are appended to ``operands`` (hints for the fuzzer)."""
    import ast
    from tharkun.driver import _run_traced, enclosing_guards
    from tharkun.sv import _symexec

    signal.alarm(max(1, seconds))
    try:
        with confine():
            tree = ast.parse(open(req["harness"]).read())
            frames = enclosing_guards(tree, site["line"])
            observed = _run_traced(req["harness"], "__sv_harness", req["seed"],
                                   {f.branch_line for f in frames}, capture_real=True)
            if not observed.ok:
                return None
            for frame in frames:
                side = frame.then_span if frame.desired_branch == "then" else frame.else_span
                if (frame.branch_line not in observed.executed or side is None
                        or any(side[0] <= line <= side[1] for line in observed.executed)):
                    continue
                snapshots = observed.real.get(frame.branch_line) or []
                if not snapshots:
                    return None
                live = {k: v for k, v in snapshots[-1].items() if type(v) in (str, int, bool, float)}
                node = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and n.lineno == frame.branch_line)
                test = ast.unparse(node.test)
                if frame.desired_branch == "else":
                    test = f"not ({test})"
                names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                if operands is not None:
                    operands.extend(o for o in (_operand(v) for v in
                                                [n.value for n in ast.walk(node.test) if isinstance(n, ast.Constant)]
                                                + [live[k] for k in names if k in live]) if o)
                for param, kind in zip(req["params"], req["kinds"]):
                    if (param not in names or param not in live or live[param] != req["seed"][param]
                            or kind not in ("str", "int", "bool", "float")):
                        continue
                    text = f"def __frontier({param}):\n    if {test}:\n        if True:\n            __sv_site(0)\n"
                    try:
                        got = _symexec.solve_site(text, live, "__frontier", 4, [param], [kind], {},
                                                 timeout_ms=max(1, seconds * 1000 // max(1, len(names))))
                    except _symexec.Unsupported:
                        continue
                    if got is not None:
                        if operands is not None:
                            operands.extend(o for o in map(_operand, got.values()) if o)
                        return dict(req["seed"], **got)
                return None
    except (_SiteTimeout, _symexec.Unsupported):
        return None
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
    from tharkun.solve.environment import confined, preload_infrastructure

    # Solver initialization probes host library paths, not the target's file world.
    preload_infrastructure()
    try:
        from tharkun.sv import _symexec  # noqa: F401
    except ImportError:
        pass  # The symbolic helpers report the optional solver's absence.

    # a sealed worker's driver reads its own target files (there is no per-block confine to fall back on, [internal ref]), so the
    # reach targets are declared with the harness
    targets = [t["path"] for t in req.get("targets") or () if isinstance(t, dict) and isinstance(t.get("path"), str)]
    files = tuple(dict.fromkeys([*req.get("files", [req["harness"]]), req["harness"], *targets]))
    scratch = req.get("scratch", os.path.dirname(os.path.abspath(req_path)))
    flags = set()
    world = None
    if req.get("world"):
        from tharkun.solve.world import WorldStore
        world = WorldStore.from_snapshot(req["world"])

    def select_world(snapshot):
        # A sealed worker's installed file model retains these same instances.
        restored = WorldStore.from_snapshot(snapshot)
        for path, obj in restored.objects.items():
            current = world.objects[path]
            current.concrete, current.facts, current.mode, current.origins = (
                obj.concrete, obj.facts, obj.mode, obj.origins)

    def confine():
        return confined(files, scratch=scratch, flags=flags, threads=False,
                        **({"world": world, "box": True} if world is not None else {}))

    from tharkun.solve.extern import native_import_reason
    from tharkun.sv.sites import harness_namespace_refusal
    source_binding = req.get("source_binding")
    sealed = False
    if (not isinstance(source_binding, (list, tuple)) or len(source_binding) != 3
            or not all(isinstance(value, str) for value in source_binding)
            or not os.path.isabs(source_binding[1])):
        refusal = "missing or malformed source binding"
    else:
        source_binding = tuple(source_binding)
        generated, original_path, original_source = source_binding
        with open(req["harness"], "rb") as stream:
            matches = stream.read() == generated.encode("utf-8")
        refusal = (harness_namespace_refusal(original_source) if matches
                   else "generated source changed after admission")
        reversible = (native_import_reason(original_source, original_path, reversible=True)
                      if matches and not refusal else None)
        if reversible and not native_import_reason(original_source, original_path):
            # [internal ref] C4: ONLY the reversible tier refuses (a deferred finalizer/callback import or type access).  This
            # process is disposable, so seal it once -- the file and spawn model installed for its whole life and thread
            # creation refused -- instead of refusing the search.  Nothing here proves: every proposal is replayed in a
            # fresh child (``concolic.py::reach``/``solve``), so the run may only ever add a validated FALSE.
            from tharkun.solve.environment import install_files, refuse_threads
            install_files(files, scratch, flags, **({"world": world, "box": True} if world is not None else {}))
            refuse_threads(flags)
            sealed = True

            def confine():
                return contextlib.nullcontext()
        else:
            refusal = refusal or reversible
    if refusal:
        _write_out(req["out"], {"reached": [], "missed": [], "solver_calls": 0,
                                "flags": ["import_state_unmodelled"], "refusal": refusal})
        return

    sys.path.insert(0, req["program_dir"])
    if req.get("mode") == "reach":
        result = _reach(req, confine, source_binding, sealed, world)
        with confine():
            document = dict(result, flags=sorted(flags)) if flags else result
        _write_out(req["out"], document)
        return
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
        reached, missed, seeds, operands = [], [], [], []
        from tharkun.knowledge import KnowledgeBase
        store = KnowledgeBase()
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
        world_results = {}
        if symbolic and sites:
            with confine():
                sym, sym_note = _symbolic_all(req, sites, lens, max(1, int(left / 2) if drive else int(left)),
                                             world, world_results)
            calls += 1
        else:
            sym, sym_note = {}, "symbolic stage done before"
        open_sites = []
        for site in sites:
            outcome, got = sym.get(site["line"], ("open", sym_note))
            if outcome == "sat":
                if world is not None:
                    # Each target starts from the request, not another target's refinements.
                    candidate = world_results[site["line"]]
                    select_world(candidate)
                reached.append({"line": site["line"], "exc": site["exc"], "rounds": 1,
                                "entry": "__sv_harness", "techniques": ["symexec"],
                                **({"world": candidate} if world is not None else {}),
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
            if world is not None:
                select_world(req["world"])
            left = req["deadline"] - time.time()
            if left <= 1:
                missed.append({"line": site["line"], "outcome": "open", "status": "no time left"})
                continue
            # each site gets its share of what is left, so one hard site cannot starve the rest
            share = max(1, int(left / (len(open_sites) - i)))
            attempts = [("__sv_harness", dict(req["seed"]), None)]
            if req.get("seeded"):
                proposal = _frontier_inputs(req, site, min(share, 3), confine, operands)
                if proposal is not None:
                    attempts.insert(0, ("__sv_harness", proposal, None))
            feed = req["feeds"].get(site["function"])
            if feed:
                # the site sits in a function the main block calls with draws: solve that
                # function's own guards, then map its parameters back to the draws
                attempts.append((site["function"],
                                 {p: _default(kinds[d]) for p, d in feed.items()}, feed))
            status = [first_note]
            if world is None and site["function"] == "__sv_harness" and not req.get("seeded"):
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
                                      confine=confine, _source_binding=source_binding, store=store,
                                      **({"world": world} if world is not None else {}))
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
                    if fn == "__sv_harness" and r.inputs != req["seed"] and req["deadline"] - time.time() > 1:
                        seeds.append(_vector_capped(req["harness"], r.inputs,
                                                    min(3, int(req["deadline"] - time.time())), confine))
                    status.append(f"{fn}: {r.status}")
                    continue
                inputs = r.inputs
                if feed_map is not None:
                    inputs = dict(req["seed"], **{d: r.inputs[p] for p, d in feed_map.items()})
                reached.append({"line": site["line"], "exc": site["exc"], "rounds": r.rounds,
                                "entry": fn,
                                "techniques": (["symexec:frontier"] if req.get("seeded") and seed != req["seed"] else [])
                                              + [s.technique for s in r.history if s.technique],
                                **({"world": world.snapshot()} if world is not None else {}),
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
        document = json.dumps({"reached": reached, "missed": missed, "solver_calls": calls,
                               "seeds": seeds,
                               "operands": operands + [o for o in (_operand(f.value) for f in store
                                                                   if f.kind in ("operand", "candidate")) if o]},
                              default=repr)
    if flags:
        # Candidates still require fresh original replay. Environment-dependent
        # observations must not close a site or certify an exhausted search.
        payload = json.loads(document)
        payload["flags"] = sorted(flags)
        document = json.dumps(payload)
    with open(req["out"], "w") as fh:
        fh.write(document)


def _reach(req, confine, source_binding, sealed=False, world=None):
    """[internal ref]: one :func:`tharkun.driver.drive_to_line` per ``req["targets"]`` entry (``{"path", "line"}``, a line in
    the code the entry calls), from ``req["entry"]`` of the generated drive harness with ``req["seed"]``, each with its
    share of the deadline.  The drive's portfolio is unchanged (cheap flippers, the preimage tier, minleaf, Avengers,
    Hypothesis); its knowledge base is one store for every target, and what it gained comes back as plain facts.
    A reached target reports its inputs as ``repr`` text (the parent reads them back with ``ast.literal_eval``: a
    value that is not a literal is no proposal).  Nothing here is a row: the parent replays every proposal."""
    from tharkun.driver import drive_to_line
    from tharkun.knowledge import KnowledgeBase

    import ast

    store = KnowledgeBase()
    if world is not None:
        from tharkun.facts import Fact
        world.knowledge = store
        store.add(Fact(kind="world_fact", subject="files", value=world, source="world", certified=False))
    seed = {k: ast.literal_eval(v) for k, v in req["seed"].items()}       # repr text (``concolic.py::reach``)
    bounds = {k: tuple(v) for k, v in (req.get("param_bounds") or {}).items()}
    signal.signal(signal.SIGALRM, _alarm)
    targets = list(req["targets"])
    reached, missed, rounds = [], [], 0
    for i, target in enumerate(targets):
        where = {"path": target["path"], "line": target["line"]}
        left = req["deadline"] - time.time()
        if left <= 1:
            missed.append(dict(where, status="no time left"))
            continue
        share = max(1, int(left / (len(targets) - i)))
        signal.alarm(share)
        try:
            r = drive_to_line(req["harness"], req["entry"], target["line"], dict(seed), max_rounds=16,
                              target_path=target["path"], param_bounds=bounds or None, store=store,
                              **({"world": world} if world is not None else {}),
                              # a sealed worker admitted the source already (``confine=None`` skips the driver's scan)
                              confine=None if sealed else confine, _source_binding=source_binding)
        except _SiteTimeout:
            missed.append(dict(where, status=f"drive budget ({share}s) spent"))
            continue
        except Exception as e:  # noqa: BLE001 -- a crashed drive proposes nothing
            with confine():
                missed.append(dict(where, status=f"error {type(e).__name__}: {e}"[:300]))
            continue
        finally:
            signal.alarm(0)
        rounds += r.rounds
        with confine():
            if r.status == "reached":
                reached.append(dict(where, rounds=r.rounds, inputs={k: repr(v) for k, v in r.inputs.items()},
                                    **({"world": world.snapshot()} if world is not None else {}),
                                    techniques=[st.technique for st in r.history if st.technique]))
            else:
                missed.append(dict(where, status=f"{r.status}: {r.diagnostics}"[:300]))
    with confine():
        facts = [{"kind": e.kind, "key": e.key, "value": e.value,
                  "producers": sorted({d.producer for d in e.derivations})} for e in store.evidence()]
    return {"reached": reached, "missed": missed, "solver_calls": rounds, "facts": facts}


if __name__ == "__main__":
    main(sys.argv[1], persistent=True)
