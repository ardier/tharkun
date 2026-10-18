"""Decide an SV-Benchmarks ``uncaught(E)`` property for one Python program.

Proof-supporting runs and FALSE replays use fresh interpreters (``_child.py``).
Exploratory draw vectors may use an exec-started fork server; every
``_sv_verifier.nondet_*`` call and every unseeded ``random`` call is a DRAW the
engine controls.  The verdict rules are the competition's:

* **FALSE(E)** -- some run ended with an uncaught exception that is an instance
  of a property class ``E``, and the run drew nothing outside a documented
  domain (``Outcome.invalid_draw``, [internal ref] -- a raw ``nondet_float_range``
  draw outside ``[a, b]``, a ``nondet_str`` draw with a character outside
  ``string.printable``; the grading harness's own replay refuses such a
  draw, so a run that made one is never a witness).  The run's draw vector is
  the witness; replaying it reproduces the run exactly.
* **TRUE** -- only when the explored runs cover EVERY execution:
  the program consumed no draw and touched no uncontrolled nondeterminism, and
  two runs under different hash seeds agree (a deterministic program has one
  execution); or every draw has a finite domain and the search enumerated the
  whole draw tree without a timeout.
* **UNKNOWN** -- anything else.  A bounded search that found nothing is not a
  proof.

The search is generational and small-first: a run whose forced prefix has
length ``L`` opens, for every later draw position ``j``, a chain of alternative
values for that draw; the chain is walked lazily (running one value pushes the
next), so the domains of the unbounded kinds never end ([internal ref]).  Vectors run
cheapest first, where a value's cost is its rank in the kind's candidate
sequence: the small values first (``0, 1, -1, 2, ...`` for ints), then the
program's own constants (literals and their neighbours, [internal ref]), then a
growing tail of magnitudes.  Up to ``jobs`` vectors run side by side ([internal ref]);
their outcomes are read in cost order, so the verdict does not depend on which
child finishes first.  The budget is CPU seconds (the verifier plus its
children), the quantity BenchExec limits.
"""

from __future__ import annotations
import threading

import ast
import heapq
import itertools
import json
import math
import os
import queue
import shutil
import string
import subprocess
import sys
import tempfile
import time
import tokenize
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from tharkun.solve.environment import TaskEnvironment, TaskMetadataError, argv_candidates, proof_blocker
from tharkun.solve.extern import native_import_reason

CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_child.py")


def _python314(python: str) -> bool:
    """Fresh runtime identity; resolving our executable needs no child process."""
    executable = shutil.which(python)
    if executable is None:
        return False
    try:
        if os.path.samefile(executable, sys.executable):
            return sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 14)
        result = subprocess.run(
            [executable, "-I", "-c", "import sys; print(sys.implementation.name, *sys.version_info[:2])"],
            capture_output=True, text=True, timeout=2, check=True)
    except (OSError, UnicodeError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False
    return result.stdout.strip() == "cpython 3 14"


def admit_task(program: str, *, task: str | None = None, python: str | None = None,
               language_version: str | None = None) -> tuple[TaskEnvironment, str | None]:
    """Bind task metadata and runtime before analysis. Bare library calls stay legacy."""
    path = task or program
    try:
        environment = TaskEnvironment.from_task(path)
    except TaskMetadataError as error:
        return TaskEnvironment(error.files), str(error)
    except (OSError, ValueError) as error:
        return TaskEnvironment(()), f"unsupported task metadata: {error}"
    yaml_task = path.endswith((".yml", ".yaml"))
    if yaml_task and (not environment.files or os.path.abspath(program) != environment.files[0]):
        return environment, "unsupported task metadata: requested program differs from primary input_files"
    if task is None and language_version is None and not yaml_task:
        return environment, None
    version = environment.language_version if yaml_task or language_version is None else language_version
    if yaml_task and language_version is not None and language_version != version:
        return environment, "unsupported language version: task declaration conflicts with requested version"
    if type(version) is not str or version != "3.14":
        return environment, f"unsupported language version: {version!r}"
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 14):
        return environment, "unsupported runtime: verifier requires CPython 3.14"
    if not _python314(python or sys.executable):
        return environment, "unsupported runtime: target requires CPython 3.14"
    return environment, None


def read_source(path: str) -> str:
    """Read a Python source file the way CPython does: detect its PEP 263
    encoding cookie (``tokenize.detect_encoding``, defaulting to UTF-8, BOM
    aware) rather than assuming UTF-8. [internal ref] review: a strict UTF-8 read here
    crashed on a Latin-1 source with a coding cookie, turning a real FALSE into
    ERROR. Newlines are NOT normalized (``bytes.decode`` never touches them):
    the witness/tamper-detection hash and the "did the source change" check
    below need the exact original bytes, decoded, not universal-newline text.
    """
    with open(path, "rb") as raw:
        encoding, _ = tokenize.detect_encoding(raw.readline)
        raw.seek(0)
        return raw.read().decode(encoding)


def read_source_bytes(path: str) -> bytes:
    """The exact on-disk bytes of *path* ([internal ref], folding in [internal ref]'s note):
    ``input_file_hashes`` must bind the file as it actually is, not a UTF-8
    re-encoding of :func:`read_source`'s decoded text -- those differ whenever
    the source's own PEP 263 encoding isn't UTF-8."""
    with open(path, "rb") as raw:
        return raw.read()


def verifier_api_reason(source: str) -> str | None:
    """Refuse unsupported verifier APIs from source, before proof or search."""
    try:
        nodes = list(ast.walk(ast.parse(source)))
    except (SyntaxError, ValueError):
        return None
    aliases = {a.asname or a.name for node in nodes if isinstance(node, ast.Import)
               for a in node.names if a.name == "_sv_verifier"}
    imported = [a.name for node in nodes if isinstance(node, ast.ImportFrom)
                and node.module == "_sv_verifier" and not node.level for a in node.names]
    if not aliases and not imported:
        return None
    from tharkun.sv._child import _make_sv_verifier

    supported = set(dir(_make_sv_verifier()))
    for name in imported:
        if name == "*":
            return "unsupported _sv_verifier star import"
        if name not in supported:
            return f"unsupported _sv_verifier attribute: {name}"
    getters = {"getattr"} | {a.asname or a.name for node in nodes if isinstance(node, ast.ImportFrom)
                             and node.module == "builtins" and not node.level
                             for a in node.names if a.name == "getattr"}
    builtins_names = {a.asname or a.name for node in nodes if isinstance(node, ast.Import)
                      for a in node.names if a.name == "builtins"}
    # As in sites.py's import scan, aliases are conservative across source scopes.
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                continue
            value = node.value
            if not isinstance(value, ast.Name):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for names in (aliases, getters):
                if value.id in names:
                    added = {target.id for target in targets if isinstance(target, ast.Name)} - names
                    names.update(added)
                    changed |= bool(added)
    for node in nodes:
        name = None
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name) and node.value.id in aliases):
            name = node.attr
        elif (isinstance(node, ast.Call)
              and ((isinstance(node.func, ast.Name) and node.func.id in getters)
                   or (isinstance(node.func, ast.Attribute) and node.func.attr == "getattr"
                       and isinstance(node.func.value, ast.Name) and node.func.value.id in builtins_names))
              and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in aliases):
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or type(node.args[1].value) is not str:
                return "unsupported _sv_verifier attribute: dynamic getattr"
            name = node.args[1].value
        if name is not None and name not in supported:
            return f"unsupported _sv_verifier attribute: {name}"
    return None


def shadowed_property_names(source: str, classes) -> tuple[str, ...]:
    """The exception class names of the property that the program binds itself (owner 2026-10-18).

    ``uncaught(E)`` names a class.  Whether an uncaught exception of a program-defined class that is merely
    *named* ``E`` (``class IndexError(Exception)``, ``IndexError = ...``, an import alias, a parameter, an
    ``except ... as IndexError`` target, a change of ``builtins``) violates the property is not settled for
    SV-COMP 2027.  FALSE needs an instance of the built-in class (class identity), which violates the property
    under either reading, so FALSE is unaffected; a TRUE on such a program would rest on the class-identity
    reading alone, so it is withheld (UNKNOWN; ``scheduler.run`` and :func:`verify`).  Conservative: every
    binding of the name in any scope counts, and so does the name as a string constant subscript store,
    ``setattr``/``delattr`` argument or attribute store, or key/keyword of a namespace ``update``/``setdefault``/
    ``__setitem__``.  A class may also be *named* so dynamically: a ``__name__``/``__qualname__`` store,
    ``type(name, bases, namespace)`` or ``new_class(name, ...)`` with that name counts, and with a name that is
    not a string constant every class counts.  Unparsable source fails closed: every class counts.
    """
    names = set(classes)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return tuple(sorted(names))

    def text(node):
        """The str a constant or a ``+`` of constants spells, else None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = text(node.left), text(node.right)
            return None if left is None or right is None else left + right
        return None

    def named(node):
        """The class name ``node`` spells: that name, or every class when it is not a constant."""
        value = text(node)
        return list(names) if value is None else [value]

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is not None and any(isinstance(t, ast.Attribute) and t.attr in ("__name__", "__qualname__")
                                              for t in targets):
                found.update(name for name in named(node.value) if name in names)
        if isinstance(node, ast.Call):
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if (fname == "setattr" and len(node.args) > 2 and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in ("__name__", "__qualname__")):
                found.update(name for name in named(node.args[2]) if name in names)
            elif (fname == "type" and isinstance(func, ast.Name) and len(node.args) == 3) or fname == "new_class":
                if node.args:
                    found.update(name for name in named(node.args[0]) if name in names)
            elif fname in ("update", "setdefault", "__setitem__") and isinstance(func, ast.Attribute):
                keys = [kw.arg for kw in node.keywords if kw.arg]
                if node.args and isinstance(node.args[0], ast.Dict):
                    keys += [k.value for k in node.args[0].keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                elif node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    keys.append(node.args[0].value)
                found.update(key for key in keys if key in names)
        bound: list = []
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bound = [node.name]
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound = [node.id]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = [alias.asname or alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound = list(node.names)
        elif isinstance(node, ast.arg):
            bound = [node.arg]
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound = [node.name]
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound = [node.name]
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound = [node.rest]
        elif type(node).__name__ in ("TypeVar", "ParamSpec", "TypeVarTuple"):
            bound = [node.name]
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound = [node.attr]
        elif (isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del))
              and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            bound = [node.slice.value]
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in ("setattr", "delattr") and len(node.args) > 1
              and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            bound = [node.args[1].value]
        found.update(name for name in bound if name in names)
    return tuple(sorted(found))


def local_import_files(program: str, limit: int = 64) -> tuple[str, ...] | None:
    """The local modules ``program`` imports by absolute import from its own directory, recursively (``x`` ->
    ``x.py`` or ``x/__init__.py``, ``x.y`` -> also ``x/y.py`` or ``x/y/__init__.py``; ``from x import y`` -> also
    the submodule ``x/y``).  None when the graph is unreadable or larger than ``limit``: fail closed."""
    root = os.path.dirname(os.path.abspath(program))

    def candidates(dotted):
        parts = dotted.split(".")
        for i in range(1, len(parts) + 1):
            base = os.path.join(root, *parts[:i])
            for path in (base + ".py", os.path.join(base, "__init__.py")):
                if os.path.isfile(path):
                    yield os.path.abspath(path)

    seen, pending, found = {os.path.abspath(program)}, [os.path.abspath(program)], []
    while pending:
        path = pending.pop()
        try:
            tree = ast.parse(read_source(path), path)
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            return None
        for node in ast.walk(tree):
            dotted = []
            if isinstance(node, ast.Import):
                dotted = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                dotted = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*"]
            for name in dotted:
                for file in candidates(name):
                    if file not in seen:
                        seen.add(file)
                        found.append(file)
                        pending.append(file)
                        if len(found) > limit:
                            return None
    return tuple(found)


def withhold_true(v: "Verdict", shadowed) -> None:
    """Turn a TRUE that a shadowed property name makes indefinite into UNKNOWN, removing the correctness
    witness files this run wrote for it (``proof.finish`` records them in ``detail``)."""
    v.detail.update(withheld_component=v.component, shadowed_property_names=list(shadowed))
    v.verdict, v.reason, v.component = "unknown", shadow_reason(shadowed), "none"
    removed = []
    for key in ("witness_path", "obligations_path", "store_witness_path"):
        path = v.detail.pop(key, None)
        if path and os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(path)
            except OSError:
                v.detail["withheld_witness_left"] = v.detail.get("withheld_witness_left", []) + [path]
    if removed:
        v.detail["withheld_witness_removed"] = removed
        try:
            os.rmdir(os.path.dirname(removed[0]))   # the run's own directory, only if now empty
        except OSError:
            pass


def shadow_reason(names) -> str:
    """The UNKNOWN reason for a withheld TRUE (:func:`shadowed_property_names`); no verdict words."""
    return ("the program binds the property exception name " + ", ".join(names)
            + " itself; whether a class merely named so counts is unresolved")


_INF = float("inf")
#: the cheapest candidates of each kind, in rank order (the head of every sequence)
HEAD: dict[str, list] = {
    "bool": [False, True],
    "int": [0, 1, -1, 2, -2],
    "float": [0.0, 1.0, -1.0, 0.5, -0.0],
    "str": ["", "a", "0", " "],
    "argv": list(itertools.islice(argv_candidates(), 5)),
    "argv_full": list(itertools.islice(argv_candidates(), 5)),
    "rfloat": [0.0, 0.5, 0.25, 0.75, 0.999999, 1e-300],
}
#: after the head and the program's constants: fixed interesting values
BODY: dict[str, list] = {
    "int": [3, 4, 5, -3, 7, 8, 10, 16, 32, 64, 100, 255, 256, 1000, -1000, 65536,
            2**31 - 1, -2**31, 2**63 - 1],
    "float": [2.0, 1e-300, 1e300, -1e300, _INF, -_INF, math.nan, 5e-324],
    "str": ["ab", "\n", "%", "{}", "a" * 100],
    "argv": list(itertools.islice(argv_candidates(), 5, 21)),
    "argv_full": list(itertools.islice(argv_candidates(), 5, 21)),
}
FINITE_KINDS = ("bool", "rindex")
UNBOUNDED_KINDS = ("int", "float", "str", "argv", "argv_full")
_RINDEX_CAP = 32
_HARVEST_CAP = 64
_RINDEX_VALUES_CAP = 512


#: the characters a ``nondet_str`` draw may hold (the grading harness's rule, ``_child._PRINTABLE``)
_PRINTABLE = frozenset(string.printable)


def admissible(kind: str, value) -> bool:
    """Is ``value`` a draw the grading harness's model accepts for ``kind`` ([internal ref])?  Argv lists keep strict list/string types. Only ``str`` has
    a domain beyond its type here: every character in ``string.printable``.  (The ``float`` domains depend on
    the helper: ``nondet_float`` takes only ``inf``, ``-inf``, ``nan`` and ``-0.0``, ``nondet_float_range(a, b)`` only
    what ``random.uniform(a, b)`` can return -- [internal ref]; a candidate generator does not know which call a position is,
    and the child refuses a draw outside either at the call.)  A candidate the child would refuse is never
    worth a run, and one it accepted used to be confirmed here and refused by the harness."""
    if kind in ("argv", "argv_full"):
        return type(value) is list and all(type(s) is str for s in value)
    return kind != "str" or (isinstance(value, str) and all(c in _PRINTABLE for c in value))


#: the draw kinds a producer's vector may hold (the ``sv-py-witness/1`` vocabulary minus those whose value
#: the child clamps or models specially: ``rindex``, ``rfloat``, ``argv``, ``argv_full``)
VECTOR_KINDS = ("bool", "int", "float", "str")


def vector_admissible(vector, first) -> bool:
    """The out-of-band pre-check of a PROPOSED draw vector (docs/SV.md "Draw vectors from a producer"): would the
    child refuse a draw in it?  A refusal here costs no run and can never decide anything -- the child still
    refuses such a draw itself (``invalid_draw`` is set before anything can raise or swallow) and a vector is
    only ever a candidate, replayed natively.

    * every entry is a dict of a kind in :data:`VECTOR_KINDS` whose value has the kind's type, and a ``str``
      stays inside ``string.printable`` (:func:`admissible`);
    * a ``float`` entry at a position where the FIRST run called ``nondet_float_range(a, b)`` is a value
      ``random.uniform(a, b)`` can return (``_child.float_range_producible``: not just "in ``[a, b]``", the helper's
      code is the domain, [internal ref]).  The call at a position is fixed by the draws BEFORE it (the program is
      deterministic given its draws), so the first run's bounds there hold for a vector that repeats the first run's
      draws up to that position, whatever value it puts there; the check stops at the first position where the
      vector differs from the first run."""
    if not isinstance(vector, list):
        return False
    for entry in vector:
        if not isinstance(entry, dict) or entry.get("kind") not in VECTOR_KINDS or "value" not in entry:
            return False
        kind, value = entry["kind"], entry["value"]
        if kind == "bool" and type(value) is not bool or kind == "int" and type(value) is not int:
            return False
        if kind == "float":
            try:
                float.fromhex(value) if isinstance(value, str) else float(value)
            except (TypeError, ValueError, OverflowError):
                return False
        if not admissible(kind, value):
            return False
    ranges = {pos: (float.fromhex(a), float.fromhex(b)) for pos, a, b in getattr(first, "ranges", None) or []}
    observed = getattr(first, "draws", None) or []
    for i, entry in enumerate(vector):
        if i >= len(observed):
            break
        if i in ranges and entry["kind"] == "float":
            # the call at position i is fixed by the draws BEFORE it, which the vector repeats up to here
            from tharkun.sv import _child

            a, b = ranges[i]
            if not _child.float_range_producible(a, b, _decode(entry)):
                return False
        if _key(entry["kind"], _decode(entry)) != _key(observed[i]["kind"], _decode(observed[i])):
            break
    return True


def _key(kind: str, value) -> tuple:
    if kind in ("float", "rfloat"):
        return (kind, float(value).hex())
    return (kind, tuple(value) if kind in ("argv", "argv_full") else value)


def _decode(entry: dict):
    v = entry["value"]
    if entry["kind"] in ("float", "rfloat"):
        return float.fromhex(v) if isinstance(v, str) else float(v)
    return v


def _encode(kind: str, value) -> object:
    return float(value).hex() if kind in ("float", "rfloat") else value


def _tail(kind: str):
    """The unbounded continuation of a kind's candidates ([internal ref]): growing magnitudes."""
    if kind == "int":
        for e in itertools.count(3):
            yield 10**e
            yield -(10**e)
            yield 2**(e + 5) + 1
            yield -(2**(e + 5))
    elif kind == "float":
        # stop before 2.0**e overflows (e = 1024): OverflowError used to crash the
        # verifier on a long float search; inf and nan are already in BODY
        for e in range(1, 1023):
            yield 2.0**e * 1.5
            yield -(2.0**e * 1.5)
            yield 2.0**-e
            yield float(10**min(e, 308))
    elif kind in ("argv", "argv_full"):
        yield from itertools.islice(argv_candidates(), 21, None)
    elif kind == "str":
        # strings up to 2**20 characters: past that a draw only exhausts memory
        for n in range(2, 21):
            yield "a" * (2**n)
            yield str(n)
            yield chr(0x20 + n % 0x5f) * n


def _num_literal(node) -> float | None:
    """A bare or unary-minus numeric literal (``5``, ``5.0``, ``-5``), as a ``float``;
    ``None`` for anything else (a variable, a computed bound, ...)."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node, sign = node.operand, -1.0
    else:
        sign = 1.0
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return sign * float(node.value)
    return None


def harvest_constants(program: str) -> dict[str, list]:
    """The program's own literals as candidates ([internal ref]): each int with its neighbours
    and negation, each float, each short string -- the values its guards compare with.

    Every ``nondet_float_range(a, b)`` call site with literal (or unary-minus-literal)
    bounds also explicitly proposes ``a``, ``b``, their midpoint and the largest value
    ``random.uniform(a, b)`` can return ([internal ref]) as ``float``
    candidates ([internal ref]): the draw itself is always kind ``float``, so an INT-literal
    bound (``nondet_float_range(0, 1)``, common in the SV-Benchmarks set) would
    otherwise never reach the float pool at all -- ``a``/``b`` land in ``ints``
    instead of ``floats`` -- and the endpoints are exactly the values most likely to
    matter once [internal ref] makes an out-of-domain draw invalid rather than silently
    clamped, so the search should try them, not stumble onto them by luck."""
    try:
        tree = ast.parse(read_source(program))
    except (OSError, SyntaxError, ValueError):
        return {}
    # Task metadata is never a value hint: comments are absent from the AST;
    # docstrings and annotations are skipped instead of affecting candidate rank.
    ignored = set()
    for owner in ast.walk(tree):
        if isinstance(owner, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and owner.body:
            first = owner.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ignored.add(id(first.value))
        annotation = (owner.annotation if isinstance(owner, (ast.AnnAssign, ast.arg)) else
                      owner.returns if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) else
                      owner.value if isinstance(owner, getattr(ast, "TypeAlias", ())) else None)
        if annotation is not None:
            ignored.update(id(n) for n in ast.walk(annotation) if isinstance(n, ast.Constant))
    ints: list = []
    floats: list = []
    strs: list = []
    argv_strs: list = []
    for node in ast.walk(tree):
        if id(node) in ignored or not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            continue
        v = node.value
        if isinstance(v, int) and abs(v) < 2**64:
            ints.extend((v, v + 1, v - 1, -v))
        elif isinstance(v, float):
            floats.extend((v, -v))
        elif isinstance(v, str) and len(v) <= 64:
            argv_strs.append(v)
            if admissible("str", v):
                strs.append(v)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 2:
            continue
        callee = node.func
        name = (callee.attr if isinstance(callee, ast.Attribute) else
                callee.id if isinstance(callee, ast.Name) else None)
        if name != "nondet_float_range":
            continue
        a, b = _num_literal(node.args[0]), _num_literal(node.args[1])
        if a is not None:
            floats.append(a)
        if b is not None:
            floats.append(b)
        if a is not None and b is not None:
            floats.append((a + b) / 2.0)
            # [internal ref]: ``b`` itself is usually outside what ``random.uniform(a, b)`` returns; the LARGEST value it can return is the
            # candidate that reaches an upper guard (``assert y < 0.9999``) and it is the nearest producible neighbour of ``b``
            from tharkun.sv import _child

            floats.append(_child.snap_float_range(b, a, b))
    out = {}
    for kind, vals in (("int", ints), ("float", floats), ("str", strs)):
        seen, uniq = set(), []
        for v in vals:
            k = _key(kind, v)
            if k not in seen:
                seen.add(k)
                uniq.append(v)
        out[kind] = uniq[:_HARVEST_CAP]
    out["argv"] = list(argv_candidates(out.get("str", [])))[:80]
    cli = [[os.path.abspath(program), *tail] for tail in out["argv"]]
    full = list(argv_candidates(list(dict.fromkeys(argv_strs))[:_HARVEST_CAP]))
    out["argv_full"] = [list(v) for v in dict.fromkeys(map(tuple, cli + full))][:160]
    # [internal ref]: every int literal at ANY size (``ints`` above stops at 2**64, a bound on what an ``int`` draw is worth trying), with
    # its neighbours and negation (and the negation's neighbours): the values a ``random.randrange(start, stop, step)`` call is compared with, which
    # ``Candidates.value`` turns into the INDEX that makes the call return them (a literal is only a candidate: the child
    # runs the vector and the verdict is what the run does)
    wide: list = []
    for node in ast.walk(tree):
        if id(node) not in ignored and isinstance(node, ast.Constant) and type(node.value) is int:
            v = node.value
            wide.extend((v, v + 1, v - 1, -v, -v + 1, -v - 1))
    out["rindex_values"] = list(dict.fromkeys(wide))[:_RINDEX_VALUES_CAP]
    return out


class Candidates:
    """Per-kind candidate sequences: ``value(kind, r, n)`` is the ``r``-th candidate
    (``None`` past the end of a finite kind).  Unbounded kinds never end ([internal ref])."""

    def __init__(self, harvested: dict | None = None, deepen: bool = True,
                 seeded: dict | None = None):
        #: the program's large int literals (``harvest_constants``): the values a randrange draw is compared with
        self._rindex_values: list = list((harvested or {}).get("rindex_values", []))
        self._seq: dict[str, list] = {}
        self._gen: dict[str, object] = {}
        self._seen: dict[str, set] = {}
        #: where a value first entered its sequence: ``small_first`` (HEAD / BODY),
        #: ``constants`` (a harvested literal, [internal ref]), ``tail`` ([internal ref]), or the component
        #: of the seed stage that proposed it (``seeded``: {component: {kind: values}},
        #: ranked right after the head, [internal ref])
        self._source: dict[tuple, str] = {}
        for kind in UNBOUNDED_KINDS + ("rfloat",):
            harvested_k = (harvested or {}).get(kind, [])
            seeds_k = [(v, comp) for comp, by_kind in (seeded or {}).items()
                       for v in by_kind.get(kind, [])]
            head = ([(v, "small_first") for v in HEAD.get(kind, [])]
                    + seeds_k
                    + [(v, "constants") for v in harvested_k]
                    + [(v, "small_first") for v in BODY.get(kind, [])])
            seq, seen = [], set()
            for v, src in head:
                if not admissible(kind, v):
                    continue            # a seed stage's value the harness would refuse ([internal ref])
                if _key(kind, v) not in seen:
                    seen.add(_key(kind, v))
                    seq.append(v)
                    self._source[_key(kind, v)] = src
            self._seq[kind], self._seen[kind] = seq, seen
            if kind in UNBOUNDED_KINDS and deepen:
                self._gen[kind] = _tail(kind)

    def value(self, kind: str, r: int, n: int | None = None, base: tuple | None = None):
        if kind == "bool":
            return HEAD["bool"][r] if r < 2 else None
        if kind == "rindex":
            n = int(n or 1)
            vals = list(range(min(n, _RINDEX_CAP))) + ([n - 1] if n > _RINDEX_CAP else [])
            if n > _RINDEX_CAP:
                vals += [i for i in rindex_hits(self._rindex_values, n, base) if i not in vals]
            return vals[r] if r < len(vals) else None
        seq = self._seq[kind]
        gen = self._gen.get(kind)
        while r >= len(seq) and gen is not None:
            v = next(gen, None)
            if v is None:           # a finite tail (floats stop short of overflow)
                self._gen.pop(kind, None)
                break
            if _key(kind, v) not in self._seen[kind]:
                self._seen[kind].add(_key(kind, v))
                seq.append(v)
                self._source[_key(kind, v)] = "tail"
        return seq[r] if r < len(seq) else None

    def source(self, entry: dict) -> str:
        """``small_first`` | ``constants`` | ``tail`` | a seed stage's component for ``entry``'s
        value (finite kinds and values never proposed count as ``small_first``)."""
        if entry["kind"] == "rindex":
            # a randrange index that makes the call return a literal of the program ([internal ref]) is a harvested constant
            v, n = entry["value"], int(entry.get("n") or 1)
            if n > _RINDEX_CAP and _RINDEX_CAP <= v != n - 1 and v in rindex_hits(self._rindex_values, n, entry_base(entry)):
                return "constants"
        return self._source.get(_key(entry["kind"], _decode(entry)), "small_first")

    def rank(self, entry: dict, limit: int = 4096) -> int:
        """The rank of ``entry``'s value (``limit`` if it is not among the first ``limit``)."""
        target = _key(entry["kind"], _decode(entry))
        for r in range(limit):
            v = self.value(entry["kind"], r, entry.get("n"), entry_base(entry))
            if v is None:
                break
            if _key(entry["kind"], v) == target:
                return r
        return limit


def rindex_hits(values, n: int, base) -> list:
    """The indices (0 <= i < n) at which ``range(start, ..., step)[i] = start + step * i`` is one of ``values`` ([internal ref]).

    ``base`` is ``(start, step)`` of the ``randrange`` call (logged by the child); a draw with no arithmetic
    (``random.choice``) has none and gets no hit.  Each value is tried at most once and the order is the order of ``values``."""
    if base is None:
        return []
    try:
        start, step = int(base[0]), int(base[1])
    except (TypeError, ValueError, IndexError):
        return []
    if step == 0:
        return []
    out: list = []
    for v in values:
        d = v - start
        if d % step == 0 and 0 <= d // step < n and d // step not in out:
            out.append(d // step)
    return out


def entry_base(entry: dict):
    """``(start, step)`` of a logged randrange draw, or None."""
    if "start" in entry and "step" in entry:
        return entry["start"], entry["step"]
    return None


def domain(entry: dict) -> list:
    """The first candidates of ``entry``'s kind (without program constants), cheapest first."""
    c = Candidates()
    out = []
    for r in itertools.count():
        v = c.value(entry["kind"], r, entry.get("n"), entry_base(entry))
        if v is None or r >= 64:
            return out
        out.append(v)


def rank(entry: dict) -> int:
    return Candidates().rank(entry)


#: run flags that make a raised exception no witness: the run depended on something the model does not control (an
#: undeclared file's content, ``sys.orig_argv``) or on something the model got wrong ([internal ref], [internal ref]; the child's
#: ``_patch_random``)
NEVER_A_WITNESS_FLAGS = ("file_unmodelled", "orig_argv", "model_error", "random_state_unmodelled", "stack_unmodelled",
                         "import_state_unmodelled", "resource_exhausted", "resource_unobserved")


@dataclass
class Outcome:
    status: str                      # clean | exit | raised | timeout | crashed
    exc: str | None = None
    #: the names of the BUILTIN classes in the raised exception's MRO (``_child.py::exception_mro``, [internal ref]): a
    #: program's own ``class IndexError(Exception)`` is not one, ``class E(IndexError)`` has the builtin in it.
    #: ``violated`` matches property classes against this and nothing else.
    mro: list = field(default_factory=list)
    msg: str = ""
    site: dict | None = None
    draws: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    seconds: float = 0.0
    class_hints: dict = field(default_factory=dict)
    invalid_draw: bool = False       # a draw fell outside its documented domain ([internal ref])
    ranges: list = field(default_factory=list)   # [position in draws, a, b] (float.hex, call order) of each nondet_float_range call
    #: module names ``import`` asked for that no finder had in this run (``_child.py::watch_imports``, [internal ref])
    absent_imports: list = field(default_factory=list)

    def violated(self, classes) -> str | None:
            # invalid_draw: matches eregion_sv.replay.violates() -- a run that ever drew
            # outside a documented domain (today, nondet_float_range) is never a
            # violation, no matter what exception actually surfaced, because the
            # grading harness's own replay refuses that draw and never reaches it.
        # file_unmodelled: an undeclared file read whose outcome depends on this
        # host's own filesystem, not the task's model. orig_argv ([internal ref], MG review
        # 2026-09-29): real, unmodelled CPython interpreter metadata that a fresh
        # subprocess replay and a reused forked explore worker genuinely disagree on
        # (fork() never re-execs, so it keeps whatever the SERVER process launched
        # with) -- unlike plain argv (modelled, replayable, fine for FALSE), this
        # channel is not modelled at all, so a run that touched it can never be trusted
        # as a violation either.
        #
        # model_error ([internal ref]): OUR model raised where CPython would not (``len(range(2**100))`` in the random model);
        # random_state_unmodelled ([internal ref]): the run touched ``random.getstate``/``setstate``, which the draws do not
        # model: its outcome is not the program's.  stack_unmodelled ([internal ref]): the program reads the Python stack above its module
        # frame (inspect.stack(), a sys._getframe() walk), which in the child holds the child's own frames.
        if (self.status != "raised" or self.invalid_draw or _resource_exhausted(self)
                or any(f in self.flags for f in NEVER_A_WITNESS_FLAGS)):
            return None
        return next((c for c in classes if c in self.mro), None)


def _cap_memory(limit_bytes: int | None):
    """``preexec_fn`` capping the child's address space ([internal ref]).

    A draw that sizes an allocation (``[None] * n`` with ``n = 2**31``) must fail
    INSIDE the child with ``MemoryError`` -- an exception outside every property, so
    that run simply ends -- instead of letting the machine's (or BenchExec's
    cgroup) OOM killer take down the verifier."""
    if not limit_bytes:
        return None

    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    return apply


def run_once(python: str, program: str, vector: list, *, timeout: float,
             hashseed: int = 0, memory_bytes: int | None = None, files=None,
             observe_fields=None, minimal_env: bool = False) -> Outcome:
    """One run of ``program`` in a fresh interpreter, replaying ``vector``; the child's
    address space is capped at ``memory_bytes`` when given. Validator replay uses
    only PATH and explicit Python runtime controls, without inherited user-site imports."""
    with tempfile.TemporaryDirectory(prefix="tharkun-sv-") as tmp:
        req = os.path.join(tmp, "req.json")
        out = os.path.join(tmp, "out.json")
        try:
            with open(req, "w") as fh:
                json.dump({"program": os.path.abspath(program), "vector": vector, "out": out,
                           "observe_fields": observe_fields,
                           **({"files": files} if files is not None else {})}, fh)
        except (OSError, ValueError):
            return Outcome("crashed", msg="cannot encode draw vector")
        env = ({"PATH": os.environ.get("PATH", os.defpath), "PYTHONNOUSERSITE": "1"}
               if minimal_env else
               {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP")})
        env["PYTHONHASHSEED"] = str(hashseed)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        t0 = time.monotonic()
        try:
            proc = subprocess.run([python, CHILD, req], cwd=tmp, env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=timeout,
                                  preexec_fn=_cap_memory(memory_bytes))
        except subprocess.TimeoutExpired:
            return Outcome("timeout", seconds=time.monotonic() - t0)
        secs = time.monotonic() - t0
        if proc.returncode != 0 or not os.path.exists(out):
            return Outcome("crashed", msg=(proc.stderr or "")[-500:], seconds=secs)
        try:
            with open(out) as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            # a child killed while writing (its memory cap, a huge draw) leaves a partial
            # record: that run crashed; it must not take the verifier down with it
            return Outcome("crashed", msg="unreadable run record", seconds=secs)
    return Outcome(r["status"], r.get("exc"), r.get("mro", []), r.get("msg", ""), r.get("site"),
                   r.get("draws", []), r.get("flags", []), secs, r.get("class_hints", {}),
                   r.get("invalid_draw", False), r.get("ranges", []), absent_imports=r.get("absent_imports", []))


def _resource_exhausted(out: Outcome) -> bool:
    """A caught or terminal interpreter limit is no complete execution."""
    return (any(f in out.flags for f in ("resource_exhausted", "resource_unobserved"))
            or (out.status == "raised" and any(c in out.mro for c in ("MemoryError", "RecursionError"))))


@dataclass
class Verdict:
    verdict: str                     # "true" | "false" | "unknown"
    exception: str | None = None
    reason: str = ""
    witness: dict | None = None
    runs: int = 0
    seconds: float = 0.0
    #: which part of the solver produced the verdict (``deterministic``, ``finite_tree``,
    #: ``proof:<domain>``, ``concolic:guard_solve`` | ``concolic:z3`` | ``concolic:drive``,
    #: ``enum:small_first`` | ``enum:constants`` | ``enum:tail``, ``none`` for UNKNOWN)
    component: str = "none"
    detail: dict = field(default_factory=dict)

    def line(self) -> str:
        if self.verdict == "true":
            return "TRUE"
        if self.verdict == "false":
            return f"FALSE({self.exception})"
        return f"UNKNOWN({self.reason})"


def _witness(program: str, prop_name: str, exc: str, out: Outcome) -> dict:
    return {"format": "sv-py-witness/1", "task": os.path.abspath(program), "property": prop_name,
            "exception": exc, "site": out.site,
            "vector": [{k: v for k, v in d.items() if k not in ("site", "start", "step")} for d in out.draws],
            "producer": "tharkun-sv"}


def _cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def default_jobs() -> int:
    """Children to run side by side: the CPUs this process may use, at most 4 (SV-COMP's cores)."""
    try:
        return max(1, min(4, len(os.sched_getaffinity(0))))
    except AttributeError:
        return max(1, min(4, os.cpu_count() or 1))


def _certify_determinism(program: str, *, env_reads_ok: bool = False, absent_imports=frozenset(),
                         files=None, class_only_exceptions: bool = False):
    """``solve/determinism.py::determinism_census`` of the program (None if unreadable).

    ``env_reads_ok`` ([internal ref] hotfix): excuses ``input()``/``open()``/``sys.argv``/
    ``sys.stdin`` from the census's refusal -- see ``determinism_census``'s own docstring
    for why a caller MUST make this a separate, independent re-run of the whole census, never
    infer it from the ORIGINAL (``env_reads_ok=False``) run's single reported reason.

    ``absent_imports`` ([internal ref]): the module names the runs so far OBSERVED to be absent (``Outcome.absent_imports``): an ``import X``
    of one is an environment probe that raises ``ModuleNotFoundError`` on every run in this environment, so the census resolves it.

    ``class_only_exceptions``: the census judges exceptions by class alone (``determinism_census``'s docstring has the exact
    scope).  Only the deterministic-TRUE route of the scheduler asks for it, and only as a separate re-run after the strict
    census refused; a FALSE witness's confirmation never does."""
    from tharkun.solve.determinism import determinism_census
    try:
        source = read_source(program)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    return determinism_census(source, program, env_reads_ok=env_reads_ok,
                              absent_imports=absent_imports,
                              declared_files=tuple(files or ()),
                              class_only_exceptions=class_only_exceptions)


#: [internal ref] ([internal ref]): how many independent fresh replays of a FALSE witness must
#: ALL reproduce the same exception before the verdict is trusted, for a program
#: ``_certify_determinism`` above cannot certify address-independent. A failure whose
#: *value*, not just its search path, depends on a memory address (``id()``, the default
#: ``object.__hash__``, ``repr``/``str``/set-or-dict order that leaks one) cannot be
#: pinned by any recorded input: unlike ``PYTHONHASHSEED`` for string/bytes hashing,
#: CPython documents no portable, recordable control over ASLR or heap layout, so no
#: value of K can make such a witness truly sound -- see ``determinism.py``'s own
#: module docstring ("the claim is a run ... is a function of the program text") and
#: the plan doc's rule: a failure that depends on memory addresses cannot be pinned; it
#: needs a witness that fails for every address, or it stays unconfirmable. What K/K
#: consecutive confirmations buys is a bounded, justified FALSE-accept probability for
#: exactly this unconfirmable case: the one measured instance driving this constant
#: (rc_id_in_list.py, [internal ref] notes: ``funcs = [id]; x = funcs[0](object()) %
#: 3; print([0, 0][x])``) reproduces on a fresh interpreter only 6 times in 20
#: independent replays (~30%). Requiring K/K in a row bounds the chance of wrongly
#: accepting THAT exact witness at 0.3**K; K=20 matches the sample size already
#: measured for it (0.3**20 ~= 3.5e-11) rather than an arbitrary round number. This
#: path is the FALLBACK, not the common case: it only runs at all when the census
#: above could not certify the program address-independent, which already excludes
#: ordinary deterministic and draw-controlled programs (the vast majority) for free.
#: Each replay still spends only its share of the run's own remaining CPU budget
#: (``run()``'s existing per-call timeout shrink as ``remaining()`` falls), and the
#: loop concedes UNKNOWN, never a guessed FALSE, the moment the budget cannot afford
#: another attempt.
_REPLAY_CONFIRM_K = 20

#: [internal ref] hotfix v2 (MG): v1 tried to recognize an environment-read-only census refusal by
#: STRING-MATCHING the original census's single reported reason ("uses sys.stdin", etc.).
#: That is unsound: ``_Census`` (like most of ``determinism.py``) reports only its FIRST
#: refusal, so a program with an env read AND a genuinely unconfirmable construct elsewhere
#: (h05: stdin read, then a str-set order read; h06: a str-set order read, then an argv read)
#: can report EITHER one first depending on where the walk finds it -- v1 saw "uses
#: sys.stdin"/"uses sys.argv" and wrongly skipped the k-of-k loop even though the OTHER
#: construct was still there, exactly the unconfirmed-FALSE bug [internal ref] exists to prevent.
#: Fixed (see ``found()`` below): a SEPARATE, INDEPENDENT re-run of
#: ``_certify_determinism(program, env_reads_ok=True)`` is required, and ONLY that run's own
#: ``.ok`` decides whether to skip k-of-k -- never the original run's reason text.


def _write_absint_candidates(knowledge, invariants) -> None:
    """[internal ref] SEV1 (owner, 2026-09-29; [internal ref]): absint's own per-point
    invariants are candidates, same as everywhere else -- ``absint/interp.py``'s own
    ``capture_invariant`` docstring: "never trusted by the witness checker". Writes the
    SAME ``candidate_invariant`` Fact shape ``sv/proof.py::finish`` already writes on a
    TRUE verdict; this covers the gap ``finish`` cannot reach -- an UNKNOWN absint run's
    partial progress, thrown away today before this change. Always ``certified=False``;
    nothing downstream may treat this as a proof."""
    if knowledge is None or not invariants:
        return
    from tharkun.facts import Fact
    for inv in invariants:
        knowledge.add(Fact(kind="candidate_invariant",
                           subject=f"{inv.file_name}:{inv.line}:{inv.column}",
                           value=inv.value, source="absint", certified=False))


def _static_proof(program: str, classes, remaining, say, files=None, invariant_locations=(), table=None):
    """Stage 1 of :func:`verify`, nothing runs: a TRUE ``Verdict`` or ``None``.

    The provers are imported lazily, so each is used as soon as it is merged:
    ``tharkun.sv.prove.prove_static`` (the census proofs, [internal ref]) and
    ``tharkun.solve.absint.prove`` (the abstract interpreter, [internal ref]), whose
    per-site records (``status`` safe / may_raise, ``residual``) also tell the concolic
    stages which sites are still open.  A prover that is missing or crashes is skipped.
    (``tharkun.sv.prove.prove`` wraps the same abstract interpreter without its records;
    the engine calls the interpreter itself so it runs once.)

    ``table`` (an :class:`~tharkun.sv.obligation_table.ObligationTable`, [internal ref] step 3a, bd
    [internal ref]) is the store this stage reads and writes instead of passing results
    through its return value and arguments:

    * it READS the first run's observed ``class_hints`` (absint's second, cheaper try is keyed on
      that evidence being in the store -- ``sv/scheduler.py::run``);
    * it WRITES absint's per-site records (``site_record`` Evidence, one generation per run: the
      concolic stages and the seed stages read the latest) and, when absint's own verdict is
      UNKNOWN, its partial ``candidate_invariant`` facts (:func:`_write_absint_candidates`).
      The TRUE path is unchanged: it reaches the store through
      ``verdict.detail["candidate_invariants"]`` -> ``sv/proof.py::finish`` ([internal ref] step 2);
      writing it again here too would only duplicate the same facts under another ``source``.

    Without a ``table`` (a unit test of one prover) the stage reads and writes nothing."""
    try:
        blocker = proof_blocker(read_source(program), TaskEnvironment(tuple(files or [program])), filename=program)
        if blocker:
            say(f"environment: static proof open ({blocker})")
            return None
    except Exception as e:  # noqa: BLE001 -- an unreadable program proves nothing
        say(f"environment: static proof crashed ({type(e).__name__}: {e})")
        return None
    try:
        from tharkun.sv.prove import prove_static
    except (ImportError, AttributeError):
        prove_static = None
    if prove_static is not None:
        try:
            v = prove_static(program, classes, remaining=remaining, log=say, files=files)
        except Exception as e:  # noqa: BLE001 -- a proof that crashes proves nothing
            say(f"prove_static: crashed ({type(e).__name__}: {e})")
            v = None
        if v is not None and v.verdict == "true":
            return v
    try:
        from tharkun.solve.absint import prove as absint_prove
    except (ImportError, AttributeError):
        return None
    try:
        source = read_source(program)
        r = absint_prove(source, tuple(classes), filename=program,
                         time_budget=max(1.0, min(60.0, remaining() * 0.1)),
                         environment=TaskEnvironment(tuple(files or [program])),
                         class_hints=table.class_hints() if table is not None else None,
                         invariant_locations=invariant_locations)
    except Exception as e:  # noqa: BLE001
        say(f"absint: crashed ({type(e).__name__}: {e})")
        return None
    records = [x for x in (getattr(r, "records", None) or ()) if isinstance(x, dict)] or None
    if table is not None:
        table.publish_site_records(records, producer="absint")
    if r.verdict == "TRUE":
        say(f"absint: TRUE ({r.component}), {len(r.sites)} checks")
        lemmas = list(getattr(r, "lemmas", []) or [])
        lem = f"; lemmas {', '.join(lemmas)}" if lemmas else ""
        return Verdict("true", reason=f"absint proof: {len(r.sites)} checks{lem}", component=r.component,
                       detail={"sites": r.sites, "lemmas": lemmas,
                               "candidate_invariants": getattr(r, "invariants", [])})
    say(f"absint: UNKNOWN ({r.reason})")
    if table is not None:
        _write_absint_candidates(table.kb, getattr(r, "invariants", None))
    return None


def verify(program: str, classes, *, prop_name: str = "", python: str | None = None,
           budget_seconds: float = 600.0, run_timeout: float = 120.0, max_runs: int = 100000,
           run_memory_bytes: int | None = None, jobs: int | None = None,
           constants: bool = True, deepen: bool = True, concolic: bool = True,
           concolic_share: float = 0.4, prove: bool = True, canary_seconds: float = 60.0,
           early_stop_seconds: float = 60.0, stages=None, log=None, lazy_vectors: int = 4, lazy_seconds: float = 2.0,
           task: str | None = None, specification: str | None = None, witness_dir=None,
           language_version: str | None = None,
           checker_seconds: float = 3.0, cpu_time_only: bool = False, store_witness: bool = True) -> Verdict:
    """Decide ``! uncaught(E)`` for every ``E`` in ``classes`` on ``program``.

    ``budget_seconds`` is CPU time of the verifier and its children (what BenchExec
    limits), also used as a wall-clock cap unless ``cpu_time_only`` is set; ``jobs``
    children run side by side ([internal ref]; default :func:`default_jobs`). ``task`` names the
    task's own ``.yml`` (declared/required files, [internal ref]'s shared I/O model) -- default
    ``None`` derives it from ``program`` (:class:`~tharkun.solve.environment.
    TaskEnvironment.from_task`).

    ``cpu_time_only`` is a TEST SEAM, not a runtime option ([internal ref]): it exists so a test
    asserting a fully deterministic outcome (one concrete run, genuine CPU cost a
    fraction of a second) is not flaky under host CPU contention that inflates only
    WALL time, never the CPU time actually charged to this process and its children.
    Default ``False`` is PRODUCTION behavior and must stay that way: the SV CLI
    (``tharkun/sv/__main__.py``) never passes it (pinned by
    ``tests/test_sv_cli.py::test_cli_never_sets_cpu_time_only``, which fails the moment
    it does), and BenchExec still enforces the competition's real wall-clock/CPU limits
    externally regardless of this flag either way. A real runaway/hung case is still
    caught even with it set: the per-run ``run_timeout`` kill-switch (a real subprocess
    timeout, unrelated to the overall search budget) and CPU time itself are both
    unaffected -- dropping the wall-clock half of the cap loses no genuine safety margin
    for a check that is supposed to finish in one concrete run.

    The stage order, the CPU-second ledger and every component's contract (static
    proofs, symbolic solve, execution, the portfolio, enumeration -- cheapest-deciding
    first, owner ruling 2026-09-24) are ``sv/scheduler.py::run`` (docs/SV.md "3a", [internal ref]
    step 1). Every proposed FALSE, including an already-fresh or symbolic discovery,
    must then replay with the validator's fixed hash seed, empty stdin and minimal
    environment within the whole call's remaining CPU and wall budget. A failed or
    unaffordable confirmation returns UNKNOWN without an exportable witness.
    """
    from tharkun.sv.scheduler import run as _run
    started, cpu_started = time.monotonic(), _cpu_seconds()
    v = _run(program, classes, prop_name=prop_name, python=python, budget_seconds=budget_seconds,
               run_timeout=run_timeout, max_runs=max_runs, run_memory_bytes=run_memory_bytes, jobs=jobs,
               constants=constants, deepen=deepen, concolic=concolic, concolic_share=concolic_share,
               prove=prove, canary_seconds=canary_seconds, early_stop_seconds=early_stop_seconds,
               stages=stages, log=log, lazy_vectors=lazy_vectors, lazy_seconds=lazy_seconds,
               task=task, specification=specification, witness_dir=witness_dir, language_version=language_version,
               checker_seconds=checker_seconds, cpu_time_only=cpu_time_only, store_witness=store_witness)
    if v.verdict == "true":
        # scheduler.run already withholds these; this keeps any other route to TRUE from bypassing it
        try:
            shadowed = set(shadowed_property_names(read_source(program), classes))
            local_modules = local_import_files(program)
            if local_modules is None:
                shadowed.update(classes)
            for module in local_modules or ():
                shadowed.update(shadowed_property_names(read_source(module), classes))
        except (OSError, UnicodeDecodeError, ValueError):
            shadowed = set(classes)
        if shadowed:
            withhold_true(v, tuple(sorted(shadowed)))
    if v.verdict != "false":
        return v
    left = budget_seconds - (_cpu_seconds() - cpu_started)
    if not cpu_time_only:
        left = min(left, budget_seconds - (time.monotonic() - started))
    reason = "violation witness failed validator replay"
    confirmed = False
    if left > 0 and isinstance(v.witness, dict) and v.exception:
        v.runs += 1
        try:
            confirmed, _ = replay(program, v.witness, (v.exception,), python=python,
                                  timeout=min(run_timeout, left), task=task,
                                  language_version=language_version,
                                  memory_bytes=run_memory_bytes // (jobs or default_jobs())
                                  if run_memory_bytes else None)
        except (OSError, ValueError, TypeError, KeyError):
            pass  # An unusable witness or failed child launch cannot confirm a violation.
    elif left <= 0:
        reason = "violation witness validator replay has no remaining budget"
    v.seconds = time.monotonic() - started
    if confirmed and (_cpu_seconds() - cpu_started >= budget_seconds
                      or not cpu_time_only and v.seconds >= budget_seconds):
        confirmed = False
        reason = "violation witness validator replay exhausted task budget"
    if not confirmed:
        v.verdict, v.exception, v.reason, v.witness, v.component = "unknown", None, reason, None, "none"
        v.detail.pop("unsound_proof", None)
        v.detail.pop("proof_reason", None)
    return v


def _replayed(vector: list, out: Outcome) -> bool:
    """Does ``out`` (a run given ``vector`` as its draw source) actually CONSUME exactly
    ``vector`` -- same length, same kind/value at every position, in order?

    Matches ``eregion_sv.replay.validate``'s own check ([internal ref], folded into
    [internal ref]): the grader marks a witness NOT CONFIRMED whenever ``len(draws) != len(vector)``,
    even if the run happens to raise the claimed exception anyway. A vector shorter than
    what the run actually draws lets the extra draws fall back to their kind's default
    value (``_child.py``'s ``_draw``) instead of what the witness claims; a vector longer
    than what the run draws leaves a tail never consumed at all -- either way, the run is
    not really a replay of the recorded vector, just a coincidence that happened to also
    raise. Tharkun's own self-check (``replay()`` below, and ``sv/scheduler.py::run``'s
    ``found()``) must never be WEAKER than the grader's, or it could hand out a FALSE the
    grader would refuse. Not itself proof against address-dependence ([internal ref]'s separate
    concern) -- a mismatch here is a different failure mode, an outright non-replay."""
    return len(out.draws) == len(vector) and all(
        isinstance(w, dict) and w.get("kind") == d.get("kind") and w.get("value") == d.get("value")
        for w, d in zip(vector, out.draws))


def replay(program: str, witness: dict, classes, *, python: str | None = None,
           timeout: float = 120.0, task: str | None = None,
           language_version: str | None = None, memory_bytes: int | None = None) -> tuple[bool, Outcome]:
    """Validate a witness: does its vector make ``program`` raise a property class, AND does
    the run actually consume exactly that vector (:func:`_replayed`)? Always use a fresh
    child with PYTHONHASHSEED=0, stdin=/dev/null and a minimal environment, including for
    product-proof models and discoveries that already ran in a fresh interpreter."""
    environment, reason = admit_task(program, task=task, python=python, language_version=language_version)
    if reason:
        return False, Outcome("unsupported-runtime", msg=reason)
    vector = witness.get("vector", [])
    if not isinstance(vector, list):
        return False, Outcome("crashed", msg="witness vector is not a list")
    out = run_once(python or sys.executable, program, vector, timeout=timeout,
                   hashseed=0, minimal_env=True, memory_bytes=memory_bytes, files=environment.files)
    return _replayed(vector, out) and out.violated(classes) is not None, out
