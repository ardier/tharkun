"""Probe child of ``tharkun.sv``'s stationarity proof ([internal ref]): one run that also OBSERVES.

Executed as a script in a fresh interpreter, like ``_child.py``, and like it
imports nothing from tharkun; it reuses ``_child``'s model of ``_sv_verifier``,
its random interposition and its clock/stdin flags (imported as a sibling
module), so a probe run consumes draws exactly as a ``_child`` run does.

The request adds to ``_child``'s (``program``, ``vector``, ``out``):

* ``loops``: ``[{"line", "col", "dead": [names], "m": m}]`` -- the trip loops of
  ``tharkun.solve.drawflow``.  Each such ``for`` statement's iterable is wrapped
  in :func:`_probe_loop`, a generator that yields the same items (an AST rewrite
  that keeps every line and column).  At the loop's heads ``m + 1`` and
  ``m + 2`` -- the states after ``m`` and ``m + 1`` iterations -- of an instance
  whose trip count is ``m + 1`` it takes a canonical SNAPSHOT of everything the
  rest of the run can read (:func:`snapshot`).
* ``quotients``: ``[[line, col]]`` -- draw sites that ``drawflow`` proved reach the
  program only as ``abs(d) % M``; the probe records each draw's modulus ``M``.

The output adds ``instances`` (one record per trip-loop instance: its trip count,
the heads it reached, whether the two snapshots are equal, or why a snapshot was
refused), ``moduli`` (draw index -> modulus) and a ``col`` in every draw's site.

What a snapshot covers: the locals of every frame from the loop's frame out to
the program's module frame (the loop's dead target names excluded), everything
reachable from them, the real ``random`` generator state, and the model's draw
count and flags.  An object of a kind it cannot canonicalize -- an iterator, a
generator, a module outside the whitelist, an instance of a class that is not
the program's own, a set or dict keyed by identity-hashed objects -- REFUSES the
snapshot rather than guessing; a refused snapshot is never "equal".
"""

import ast
import builtins
import hashlib
import importlib.machinery
import json
import os
import sys
import tokenize
import traceback
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _child as C  # noqa: E402  -- the sibling model; never a tharkun import


def _read_source(path: str) -> str:
    """Read the way CPython reads it: PEP 263 encoding-cookie detection,
    defaulting to UTF-8, no newline normalization ([internal ref]). Duplicated from
    ``tharkun.sv.engine.read_source`` rather than imported -- this module
    "imports nothing from tharkun" by design (see the module docstring)."""
    with open(path, "rb") as raw:
        encoding, _ = tokenize.detect_encoding(raw.readline)
        raw.seek(0)
        return raw.read().decode(encoding)

#: whitelisted modules may be referenced from the state; drawflow.MODULE_WHITELIST
#: (the program's imports) plus the modules those import and the model's own.
_MODULES_OK = {"_sv_verifier", "math", "random", "re", "io", "array", "string", "builtins",
               "_random", "_io", "_sre", "re._compiler", "re._parser", "re._constants",
               "_child", "__main__"}
_NODE_LIMIT = 5_000_000


class Refused(Exception):
    pass


class _OutputFile:
    """Successful writes with real argument/encoding checks, no readable contents."""
    def __init__(self, binary, encoding):
        self.binary, self.encoding, self.closed = binary, encoding, False

    def write(self, value):
        if self.closed:
            raise ValueError("write to closed output")
        if self.binary:
            if type(value) not in (bytes, bytearray):
                raise Refused("unmodelled binary output buffer")
        else:
            if type(value) is not str:
                raise Refused("unmodelled text output value")
            str.encode(value, self.encoding)
        return len(value)

    def flush(self):
        if self.closed:
            raise ValueError("flush of closed output")

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def _output_open(path, mode='r', *, encoding=None, newline=None):
    if type(path) not in (str, bytes):
        raise Refused("unmodelled output path protocol")
    return _OutputFile('b' in mode, encoding or 'utf-8')


class _P:
    loops = {}
    quotients = set()
    instances = []
    moduli = {}
    program = ""
    module_code = None


# --- draws: the column of the call, and the modulus of a quotient draw --------

def _site(depth):
    try:
        f = sys._getframe(depth)
        col = -1
        try:
            pos = list(f.f_code.co_positions())[f.f_lasti // 2]
            col = pos[2] if pos[2] is not None else -1
        except (IndexError, ValueError, TypeError):
            pass
        return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}:{col}"
    except ValueError:
        return "?"


class _QAbs(int):
    """``abs(d)`` of a quotient draw: records the modulus it is reduced by."""

    def __mod__(self, m):
        _P.moduli[self.index] = (type(m).__name__, m if isinstance(m, int) else repr(m))
        return int(self) % m


class _QInt(int):

    def __abs__(self):
        a = _QAbs(abs(int(self)))
        a.index = self.index
        return a


def _nondet_int():
    v = C._draw("int")
    rec = C._State.log[-1]
    try:
        _, line, col = rec["site"].rsplit(":", 2)
        key = (int(line), int(col))
    except ValueError:
        key = None
    if key in _P.quotients:
        q = _QInt(v)
        q.index = len(C._State.log) - 1
        return q
    return v


# --- the canonical snapshot ----------------------------------------------------

_ATOMS = (type(None), bool, int, float, complex, str, bytes, type(Ellipsis), type(NotImplemented))
_VALUE_KEYS = (type(None), bool, int, float, complex, str, bytes, range)


def _value_key(k) -> bool:
    """A dict/set key whose hash and equality depend on its value, not its address."""
    t = type(k)
    if t in _VALUE_KEYS:
        return True
    if t in (tuple, frozenset):
        return all(_value_key(x) for x in k)
    return False


def _user_class(t) -> bool:
    return isinstance(t, type) and t.__module__ == "__main__" and type(t) is type


def snapshot(frame, dead) -> str:
    """The sha256 of a canonical, address-free rendering of the run's reachable state."""
    import random as _random
    h = hashlib.sha256()
    memo = {}
    keep = []
    count = 0
    stack = []

    def tok(s):
        h.update(s.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")

    def push_seq(items):
        for x in reversed(items):
            stack.append(x)

    def T(s):
        return ("tok", s)

    def O(x):
        return ("obj", x)

    # roots: the frame chain out to the program's module frame
    roots = []
    f = frame
    first = True
    while f is not None:
        items = list(f.f_locals.items())
        if first:
            items = [(k, v) for k, v in items if k not in dead]
            first = False
        roots.append(T(f"frame {f.f_code.co_qualname} {f.f_lasti}"))
        for k, v in items:
            roots.append(T(f"local {k}"))
            roots.append(O(v))
        if f.f_code is _P.module_code:
            break
        f = f.f_back
    else:
        raise Refused("the module frame is not on the stack")
    roots.append(T("random"))
    roots.append(O(C.real_random_getstate()))
    roots.append(T(f"model draws={len(C._State.log)} seeded={C._State.seeded} "
                   f"flags={sorted(C._State.flags)}"))
    push_seq(roots)

    while stack:
        act, x = stack.pop()
        if act == "tok":
            tok(x)
            continue
        count += 1
        if count > _NODE_LIMIT:
            raise Refused("state too large")
        t = type(x)
        if t in _ATOMS:
            if t is int:
                tok("i" + hex(x))
            elif t is float:
                tok("f" + x.hex())
            elif t is complex:
                tok("c" + x.real.hex() + "," + x.imag.hex())
            else:
                tok(t.__name__ + ":" + repr(x) if t is not str else "s" + x)
            continue
        k = id(x)
        if k in memo:
            tok(f"@{memo[k]}")
            continue
        memo[k] = len(memo)
        keep.append(x)
        if t is tuple:
            tok(f"tuple {len(x)}")
            push_seq([O(e) for e in x])
        elif t is list:
            tok(f"list {len(x)}")
            push_seq([O(e) for e in x])
        elif t is dict:
            for key in x:
                if not _value_key(key):
                    raise Refused(f"dict key of type {type(key).__name__}")
            tok(f"dict {len(x)}")
            push_seq([y for kv in x.items() for y in (O(kv[0]), O(kv[1]))])
        elif t in (set, frozenset):
            for key in x:
                if not _value_key(key):
                    raise Refused(f"set element of type {type(key).__name__}")
            tok(f"{t.__name__} {len(x)}")
            push_seq([O(e) for e in x])
        elif t is _OutputFile:
            tok(f"output binary={x.binary} encoding={x.encoding} closed={x.closed}")
        elif t is bytearray:
            tok("bytearray " + bytes(x).hex())
        elif t is range or t is slice:
            tok(f"{t.__name__}")
            push_seq([O(x.start), O(x.stop), O(x.step)])
        elif t is types.FunctionType:
            code = x.__code__
            tok(f"fn {code.co_filename} {code.co_qualname} {code.co_firstlineno}")
            cells = []
            for c in x.__closure__ or ():
                try:
                    cells.append(O(c.cell_contents))
                except ValueError:
                    cells.append(T("empty-cell"))
            push_seq([O(x.__defaults__), O(x.__kwdefaults__), T("closure")] + cells
                     + [T("fdict"), O(x.__dict__)])
        elif t is types.MethodType:
            tok("method")
            push_seq([O(x.__func__), O(x.__self__)])
        elif t is types.BuiltinFunctionType:
            owner = x.__self__
            if owner is None or isinstance(owner, types.ModuleType):
                name = getattr(owner, "__name__", "builtins")
                if owner is not None and name not in _MODULES_OK:
                    raise Refused(f"builtin of module {name}")
                tok(f"bf {name}.{x.__name__}")
            else:
                tok(f"bm {x.__name__}")
                push_seq([O(owner)])
        elif t is types.ModuleType:
            if x.__name__ not in _MODULES_OK:
                raise Refused(f"module {x.__name__}")
            tok(f"module {x.__name__}")
        elif t is importlib.machinery.SourceFileLoader:
            # the program's own ``__loader__`` (``C.native_main_module``): a name and a path, which no loop changes
            tok(f"loader {x.name} {x.path}")
        elif t is types.CellType:
            try:
                push_seq([T("cell"), O(x.cell_contents)])
            except ValueError:
                tok("empty-cell")
        elif t is staticmethod or t is classmethod:
            tok(t.__name__)
            push_seq([O(x.__func__)])
        elif t is property:
            tok("property")
            push_seq([O(x.fget), O(x.fset), O(x.fdel)])
        elif t in (types.MemberDescriptorType, types.GetSetDescriptorType,
                   types.WrapperDescriptorType, types.MethodDescriptorType):
            tok(f"descr {getattr(x, '__objclass__', type).__qualname__}.{x.__name__}")
        elif t is type:
            if _user_class(x):
                tok(f"class {x.__qualname__}")
                seq = [T("bases"), O(x.__bases__)]
                for name, v in x.__dict__.items():
                    if name in ("__dict__", "__weakref__", "__doc__", "__module__",
                                "__qualname__", "__firstlineno__", "__static_attributes__"):
                        continue
                    seq += [T("attr " + name), O(v)]
                push_seq(seq)
            elif x.__module__ in _MODULES_OK:
                tok(f"type {x.__module__}.{x.__qualname__}")
            else:
                raise Refused(f"class {x.__module__}.{x.__qualname__}")
        elif _user_class(t):
            for base in t.__mro__[:-1]:
                if not _user_class(base):
                    raise Refused(f"instance of a {base.__qualname__} subclass")
            seq = [T("inst"), O(t)]
            try:
                d = object.__getattribute__(x, "__dict__")
            except AttributeError:
                d = None
            if d is not None:
                seq += [T("idict"), O(d)]
            for cls in t.__mro__[:-1]:
                for name, descr in cls.__dict__.items():
                    if type(descr) is types.MemberDescriptorType:
                        try:
                            seq += [T("slot " + name), O(descr.__get__(x, cls))]
                        except AttributeError:
                            seq += [T("slot-unset " + name)]
            push_seq(seq)
        elif _is_re_pattern(x):
            tok(f"re {x.flags}")
            push_seq([O(x.pattern)])
        elif _is_re_match(x):
            tok(f"rematch {x.regs} {x.pos} {x.endpos}")
            push_seq([O(x.re), O(x.string)])
        elif _is_array(x):
            tok(f"array {x.typecode}")
            push_seq([O(x.tolist())])
        elif _is_stringio(x):
            if x.closed:
                tok("stringio closed")
            else:
                tok(f"stringio {x.tell()}")
                push_seq([O(x.getvalue())])
        else:
            raise Refused(f"object of type {t.__module__}.{t.__qualname__}")
    return h.hexdigest()


def _is_re_pattern(x):
    m = sys.modules.get("re")
    return m is not None and type(x) is m.Pattern


def _is_re_match(x):
    m = sys.modules.get("re")
    return m is not None and type(x) is m.Match


def _is_array(x):
    m = sys.modules.get("array")
    return m is not None and type(x) is m.array


def _is_stringio(x):
    m = sys.modules.get("io")
    return m is not None and type(x) is m.StringIO


# --- the loop wrapper ------------------------------------------------------------

def _probe_loop(iterable, lid):
    spec = _P.loops[lid]
    m = spec["m"]
    trip = len(iterable) if type(iterable) is range else None
    inst = {"loop": [spec["line"], spec["col"]], "trip": trip, "heads": 0}
    _P.instances.append(inst)
    if trip is None:
        inst["refused"] = f"the loop iterates a {type(iterable).__name__}, not a range"
    it = iter(iterable)
    want = (m + 1, m + 2) if trip == m + 1 else ()
    snaps = {}
    k = 0
    while True:
        k += 1
        inst["heads"] = k
        if k in want and "refused" not in inst:
            try:
                snaps[k] = snapshot(sys._getframe(1), set(spec["dead"]))
            except Refused as e:
                inst["refused"] = str(e)
            except Exception as e:  # noqa: BLE001 -- a snapshot fault is a refusal, never a pass
                inst["refused"] = f"snapshot fault {type(e).__name__}: {e}"
            if len(snaps) == 2:
                inst["equal"] = snaps[m + 1] == snaps[m + 2]
        try:
            x = next(it)
        except StopIteration:
            return
        yield x


class _Wrap(ast.NodeTransformer):
    def __init__(self, keys):
        self.keys = keys
        self.found = set()

    def visit_For(self, node):
        self.generic_visit(node)
        lid = self.keys.get((node.lineno, node.col_offset))
        if lid is not None:
            self.found.add(lid)
            call = ast.Call(func=ast.Name("__tharkun_probe_loop__", ast.Load()),
                            args=[node.iter, ast.Constant(lid)], keywords=[])
            node.iter = ast.copy_location(call, node.iter)
            ast.fix_missing_locations(node)
        return node


def main(req_path, *, persistent=False):
    with open(req_path) as fh:
        req = json.load(fh)
    program = os.path.abspath(req["program"])
    _P.program = program
    _P.loops = {i: spec for i, spec in enumerate(req.get("loops", []))}
    _P.quotients = {tuple(q) for q in req.get("quotients", [])}
    C._State.vector = req.get("vector", [])
    C._site = _site
    sv = C._make_sv_verifier()
    sv.nondet_int = _nondet_int
    # Normally a fresh, disposable subprocess that exits right after writing the
    # result, so a leaked sys.modules["_sv_verifier"] never mattered -- restoring
    # it anyway makes main() safe in-process too (see _child.py::main's same fix;
    # avengers_seed.py's _sv_verifier_stub_on_path crashes on exactly this kind
    # of leftover from a caller that didn't clean up).
    old_sv = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = sv
    C._patch_random()
    C._patch_clock()
    C.watch_threads()
    C.native_sys_path(program)
    res = {"status": "clean"}
    try:
        try:
            source = _read_source(program)
            # No stand-in for sys.stdin here either ([internal ref]); same static pre-scan
            # as _child.py, reused via the sibling import.
            if C.references_stdin(source):
                C._State.flags.add("stdin")
            if C.native_import_reason(source, program):
                C._State.flags.add("import_state_unmodelled")
            if not persistent and C.native_import_reason(source, program, reversible=True):
                C._State.flags.update(("import_state_unmodelled", "stack_unmodelled"))
                res = {"status": "probe-error", "msg": "reversible target execution refused"}
                raise SystemExit
            if C.references_stack(source):
                C._State.flags.add("stack_unmodelled")
            tree = ast.parse(source, program)
            wrap = _Wrap({(s["line"], s["col"]): i for i, s in _P.loops.items()})
            tree = wrap.visit(tree)
            if wrap.found != set(_P.loops):
                res = {"status": "probe-error", "msg": "a trip loop was not found"}
                raise SystemExit
            code = compile(tree, program, "exec", dont_inherit=True)
        except SystemExit:
            code = None
        except BaseException as e:  # noqa: BLE001
            res = {"status": "probe-error", "msg": f"{type(e).__name__}: {e}"}
            code = None
        if code is not None:
            from contextlib import nullcontext

            if persistent:
                C.install_files(req.get("files", [program]), req.get("scratch", os.path.dirname(os.path.abspath(req_path))),
                                C._State.flags)
                C.refuse_threads(C._State.flags)

            context = nullcontext() if persistent else C.confined(
                req.get("files", [program]), scratch=req.get("scratch"), flags=C._State.flags, threads=False)
            with context:
                _P.module_code = code
                builtins.__tharkun_probe_loop__ = _probe_loop
                mod = C.native_main_module(program)
                if req.get("output_files"):
                    # Preserve module globals: dir() must not observe an injected open.
                    target_builtins = types.ModuleType("builtins")
                    target_builtins.__dict__.update(vars(builtins), open=_output_open)
                    mod.__dict__["__builtins__"] = target_builtins
                saved = sys.modules.get("__main__")
                sys.modules["__main__"] = mod
                try:
                    sys.argv = C.initial_argv(program, source)
                    if "invalid_draw" in C._State.flags:
                        res = {"status": "invalid-draw"}
                    else:
                        C.run_as_main(code, mod)
                except Refused as e:
                    res = {"status": "probe-error", "msg": str(e)}
                except SystemExit as e:
                    res = {"status": "exit", "code": repr(e.code)}
                except BaseException as e:  # noqa: BLE001
                    frames = traceback.extract_tb(e.__traceback__)
                    user = [fr for fr in frames if os.path.abspath(fr.filename) == program]
                    fr = (user or frames or [None])[-1]
                    res = {"status": "raised", "exc": type(e).__name__,
                           "mro": C.exception_mro(e), "msg": str(e)[:300],
                           "site": ({"file": os.path.basename(fr.filename), "line": fr.lineno,
                                     "function": fr.name} if fr is not None else None)}
                finally:
                    if saved is not None:
                        sys.modules["__main__"] = saved
    finally:
        if old_sv is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old_sv
    res["draws"] = C._State.log
    res["flags"] = sorted(C._State.flags)
    res["instances"] = _P.instances
    res["moduli"] = {str(k): v for k, v in _P.moduli.items()}
    with open(req["out"], "w") as fh:
        json.dump(res, fh)


if __name__ == "__main__":
    main(sys.argv[1], persistent=True)
