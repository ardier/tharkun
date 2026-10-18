"""``determinism_census`` -- is a run of a program a function of its code and its draws? ([internal ref])

SOLVING, by ``docs/LAYERS.md``'s rule: the input is a program's source, the
output is a claim computed from the AST alone.  The claim (``Determinism.ok``)
is that a run of the program, given the values of its draws, is a function of
the program text: neither the hash seed, nor object addresses, nor the clock,
nor the environment, nor any input the replay does not control changes it.
``tharkun.sv`` needs it wherever ONE run stands for an execution: the
``deterministic`` TRUE (a program with no draw), the ``finite_tree`` TRUE and
``proof:stationary`` (one run per draw vector).

It is an ALLOWLIST, not a hunt for known sources (lane F broke the first
denylist version five ways).  A program passes only if:

1. Everything RESOLVES (``solve/pyresolve.py``): every import is in
   :data:`DET_MODULES`, every attribute is one the program binds or an
   allowlisted type or module has, nothing is reflective (``getattr`` with a
   computed name, ``vars``/``globals``/``__builtins__``, dunder attributes,
   ``eval``/``exec``/``__import__``).
2. No builtin or library call outside the deterministic set runs: ``id``,
   ``hash``, ``input``, ``open`` unless its mode is a literal that only writes;
   ``sys`` only for :data:`SYS_OK`; ``random.Random`` only with a literal seed,
   ``random.SystemRandom`` never, ``random.seed()`` only with an argument (the
   module-level ``random`` functions are the run's draws or a seeded generator:
   the run controls them); no ``__del__`` finalizer.
3. No ORDER leaks from a set: a set (display, comprehension, ``set()`` /
   ``frozenset()``, a set operator or method result, a name aliasing one) must
   stay in names whose every use is order-blind (``in``, ``len``, ``==``,
   ``sorted`` without a key, a truth test, the mutating and comparing methods --
   not ``pop``).  A set display of int literals has a fixed order.
4. No ADDRESS leaks through text: every conversion of a value to text
   (``str``, ``repr``, ``ascii``, ``format``, ``%s`` / ``%r`` / ``%a``, an
   f-string field, ``str.format``) whose result the program can read must be
   of a value that cannot be a default ``repr`` -- a literal, a number from
   ``len`` / ``int`` / ``float`` / ..., a class name through ``type()``, or a
   value whose complete provenance is address-free. The flow-insensitive
   analysis joins all bindings, fields, container writes and method returns.
   Output (including stdout) is checked too; unknown provenance refuses.
   ``%d``-like conversions are numeric and exempt.

Defs whose name is never referenced are dead and skipped.
"""

from __future__ import annotations

import ast
import builtins
import re
from collections import defaultdict
from dataclasses import dataclass

from tharkun.solve.pyresolve import Resolver, Top
from tharkun.solve.provenance import Provenance, STRING_CONSTANTS
from tharkun.solve.check_type_hints import closed_hint_effects
from tharkun.solve.set_singleton import singleton_receivers

DET_MODULES = frozenset({
    "_sv_verifier", "math", "cmath", "random", "re", "io", "array", "string", "itertools",
    "functools", "collections", "copy", "heapq", "bisect", "typing", "abc", "dataclasses",
    "enum", "fractions", "decimal", "numbers", "struct", "textwrap", "keyword", "json", "sys",
    "os", "os.path", "types"})

#: ``sys`` attributes whose value does not change from run to run
SYS_OK = frozenset({"setrecursionlimit", "getrecursionlimit", "float_info", "int_info", "maxsize",
                    "version_info", "hexversion", "stdout", "stderr", "exit", "byteorder",
                    "float_repr_style", "maxunicode", "set_int_max_str_digits",
                    "get_int_max_str_digits"})

#: [internal ref] hotfix: ``sys`` names ``determinism_census(..., env_reads_ok=True)`` additionally
#: allows -- a task-declared argv/stdin read, same reasoning as ``check_open``'s exemption.
_ENV_READ_SYS_NAMES = frozenset({"argv", "stdin"})

# [internal ref] (the "os bug"): `os` was refused wholesale (not in DET_MODULES at all) until now --
# every SV-bench task and probe that so much as wrote `import os` (even only to compute a
# sys.path.insert argument, never to call anything non-deterministic) could never be
# deterministic, by construction ([internal ref]; eregion-sv PR 49 had to work around
# exactly this). `os` and `os.path` are now allowlisted, but per-attribute, same shape as
# `sys`/`SYS_OK` above:
#
# * :data:`OS_CONST_OK` -- platform constants (fixed for the process's whole lifetime, same
#   cross-host assumption `SYS_OK`'s `byteorder`/`maxsize` already rely on);
# * :data:`OS_PATH_PURE` -- pure string functions of their arguments (no I/O, no environment
#   read) -- deterministic regardless of whether the exact result is computable;
# * :data:`OS_HAVOC` / :data:`OS_PATH_HAVOC` -- reads of real process/OS/filesystem state
#   (environment variables, cwd, a directory listing, OS entropy, the PID, ...): two runs of
#   the SAME program text can see different values here, so any use refuses unconditionally,
#   the same conservative stance already taken for `os.urandom`/unseeded `random`/`id`/`hash`
#   (this module's own #2) -- not the finer "unless it can never reach output" rule those
#   other checks use elsewhere in this pass; a value this indifferent to its own PROVENANCE
#   is exactly the kind of source lane F was built to catch, so it stays maximally
#   conservative here rather than trying to prove a negative about where it flows;
# * anything else on `os`/`os.path` (``remove``, ``system``, ``exec*``, ``environ``, ...) is
#   default-denied, same as an `os.X` name outside `SYS_OK` is for `sys` -- it is either
#   side-effecting/unmodellable or simply not audited yet, and both refuse the same way here.
OS_CONST_OK = frozenset({"sep", "linesep", "pathsep", "extsep", "curdir", "pardir", "name",
                         "altsep", "devnull"})
OS_PATH_PURE = frozenset({"join", "dirname", "basename", "splitext", "normpath",
                          "abspath", "realpath"})
OS_HAVOC = frozenset({"getenv", "getcwd", "listdir", "urandom", "getpid", "cpu_count"})
OS_PATH_HAVOC = frozenset({"exists", "isfile", "isdir"})

_DENIED_BUILTINS = frozenset({"id", "hash", "input", "open", "exit", "quit", "help", "copyright",
                              "credits", "license"})

_SET_SAFE_METHODS = frozenset({
    "add", "discard", "remove", "clear", "update", "intersection_update", "difference_update",
    "symmetric_difference_update", "issubset", "issuperset", "isdisjoint"})
_SET_RESULT_METHODS = frozenset({"copy", "union", "intersection", "difference",
                                 "symmetric_difference"})

# Other allowlisted modules still resolve, but their entrypoints can hide
# object-to-text conversions (e.g. Template.substitute or UserString). Admit
# those only after a library-specific provenance/effect model is added.
# `os`/`os.path` are address-free: every OS_PATH_PURE result is built only from its own
# string arguments (never a default object repr), and every OS_HAVOC/OS_PATH_HAVOC value
# (a str, a list of str, bytes, an int, a bool) is real data, never a bare object whose
# text conversion reveals its address -- check_os (below) is what refuses THOSE for being
# non-deterministic; check_text's own, narrower job is only the address-leak question.
_ADDRESS_MODULES = frozenset({"_sv_verifier", "math", "cmath", "random", "re", "array", "sys", "io", "types",
                              "os", "os.path"})
_ADDRESS_IO = frozenset({"StringIO", "BytesIO", "DEFAULT_BUFFER_SIZE", "SEEK_SET", "SEEK_CUR", "SEEK_END"})
_CONVERT = frozenset({"str", "repr", "ascii", "format"})
_NUMERIC_SPEC = re.compile(r"%(?:\([^)]*\))?[#0\- +]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[hlL]?([a-zA-Z%])")


@dataclass(frozen=True)
class Determinism:
    ok: bool
    reason: str = ""


class _No(Exception):
    pass


def determinism_census(source: str, filename: str = "<program>", *,
                       env_reads_ok: bool = False, absent_imports=frozenset(),
                       declared_files=(), class_only_exceptions: bool = False) -> Determinism:
    """Is every run of ``source`` (given its draws) the same?  See the module docstring.

    [internal ref] (read-and-recurse): an import ``Resolver`` follows into code bundled with the task
    (``solve/extern.py``) is program code too (condition (d)) -- every file
    ``res.all_resolvers()`` reaches (this program's own, plus every recursively-read one) is
    censused the same way; ANY of them failing refuses the whole determinism claim.

    ``env_reads_ok`` ([internal ref] hotfix, MG's fix for a real gap the string-matching version of
    this hotfix had): treats ``input()``, ``open()`` in ANY mode, ``sys.argv`` and
    ``sys.stdin`` as allowed, exactly like a controlled draw -- [internal ref]'s own I/O model already
    replays a task-declared file/argv/stdin read deterministically from the witness/task, so
    refusing on THOSE constructs alone is a census limitation, not a real non-determinism.
    This must be a SEPARATE, INDEPENDENT re-run of the whole census with those constructs
    excused, not a look at the ORIGINAL run's single reported reason: ``_Census`` (like most
    of this module) reports only its FIRST refusal, so a program with an env read AND a
    genuinely unconfirmable construct elsewhere (set/dict order, id/hash, ...) can report
    EITHER one first depending on where the walk finds it -- a caller that saw "uses
    sys.stdin" and assumed that was the ONLY problem would wrongly trust a program that also,
    say, has a str-set order dependency later on. Calling this a second time with
    ``env_reads_ok=True`` and requiring THAT run's own ``.ok`` is the only sound way to know
    nothing else is wrong.

    ``class_only_exceptions`` (default strict): judge exceptions by their CLASS alone, for a caller whose
    verdict is the observed run's exception class and nothing else (``scheduler.py``'s deterministic
    TRUE, as a separate re-run after the strict census refused).  Two things are then admitted that the
    strict census refuses, both only while NOTHING in the whole program (every file, dead code
    included) can observe an exception in flight (:func:`observer_reason`: no ``try``, ``with``,
    generator, ``__del__``/``__exit__``, ``sys.excepthook``/``exc_info``/``stderr``, traceback, logging,
    atexit, warnings, threading ...), so the only place its text can go is the stderr of the process
    that dies of it:

    * the text of the ``KeyError`` of ``display[key]`` when probe and stored keys are plain
      (``Provenance.class_only_lookup``): the exception is a ``KeyError`` on every path, its text names
      an address only on stderr, and no verdict reads it;
    * heaps of ``(number, identity-only instance)`` tuples (``Provenance._heapq_order_key``): which pair
      of tuples is compared, and whether a tie reaches the instances (a TypeError naming the class), is a
      function of the numbers and of object identity, never of an address.

    With any observer in the program the flag is ignored and the strict census runs."""
    try:
        tree = ast.parse(source, filename)
    except (SyntaxError, ValueError) as e:
        return Determinism(False, f"parse: {e}")
    try:
        res = Resolver(tree, DET_MODULES, filename=filename, absent_imports=absent_imports,
                       _declared_files=declared_files)
        class_only = class_only_exceptions and observer_reason(res.all_resolvers()) is None
        for sub in res.all_resolvers():
            _Census(sub, env_reads_ok=env_reads_ok, class_only=class_only).run()
    except Top as e:
        return Determinism(False, f"does not resolve: {e}")
    except _No as e:
        return Determinism(False, str(e))
    return Determinism(True, "every import, call and attribute resolves to a deterministic entry; "
                             "no set order and no address reaches the program")


#: names through which a program can see, redirect or react to an exception that is still in flight
_OBSERVER_NAMES = frozenset({
    "exc_info", "exception", "excepthook", "unraisablehook", "__excepthook__", "last_exc", "last_value",
    "last_type", "last_traceback", "settrace", "setprofile", "gettrace", "getprofile", "_getframe",
    "_current_frames", "_current_exceptions", "stderr", "__stderr__", "monitoring", "breakpointhook",
    "addaudithook", "with_traceback", "add_note", "__traceback__", "__context__", "__cause__"})
_OBSERVER_MODULES = frozenset({
    "traceback", "logging", "atexit", "warnings", "threading", "_thread", "contextlib", "signal", "inspect",
    "gc", "weakref", "unittest", "faulthandler", "pdb", "bdb", "code", "trace", "sched", "asyncio",
    "concurrent", "multiprocessing", "ctypes", "builtins", "sysconfig"})
_OBSERVER_METHODS = frozenset({"__del__", "__exit__", "__aexit__", "__enter__", "__aenter__"})
_OBSERVER_NODES = tuple(getattr(ast, name) for name in (
    "Try", "TryStar", "ExceptHandler", "With", "AsyncWith", "AsyncFor", "AsyncFunctionDef", "Await", "Yield",
    "YieldFrom", "GeneratorExp") if hasattr(ast, name))


def observer_reason(resolvers) -> str | None:
    """Why some code of the program could see an exception while it is in flight, or None.

    Walks the RAW tree of every file (dead code and unreferenced defs included): a construct that never
    runs cannot observe, but the census does not know which ones run.  The list is deliberately wider
    than what the strict census already refuses -- a ``try`` (handler or ``finally``), a ``with`` (every
    ``__exit__`` receives the exception), anything that suspends (generator, genexp, ``await``), a
    finalizer, the ``sys`` hooks and streams an uncaught exception passes through, the modules that
    register or format them -- so the exemption does not rest on another check staying as it is."""
    for resolver in resolvers:
        for node in ast.walk(resolver.tree):
            where = getattr(node, "lineno", "?")
            if isinstance(node, _OBSERVER_NODES):
                return f"line {where}: {type(node).__name__} can observe an exception in flight"
            if isinstance(node, ast.FunctionDef) and node.name in _OBSERVER_METHODS:
                return f"line {where}: {node.name} receives or outlives an exception"
            if isinstance(node, ast.Attribute) and node.attr in _OBSERVER_NAMES:
                return f"line {where}: .{node.attr} reaches an exception in flight"
            if isinstance(node, ast.Name) and node.id in _OBSERVER_NAMES:
                return f"line {where}: {node.id} reaches an exception in flight"
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module or ""] + [a.name for a in node.names] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                if name.partition(".")[0] in _OBSERVER_MODULES or name in _OBSERVER_NAMES:
                    return f"line {where}: imports {name}"
    return None


def _int_literal_set(node: ast.AST) -> bool:
    def lit(e):
        if isinstance(e, ast.Constant):
            return type(e.value) in (int, bool) or e.value is None
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            return lit(e.operand)
        if isinstance(e, ast.Tuple):
            return all(lit(x) for x in e.elts)
        return False
    return isinstance(node, ast.Set) and all(lit(e) for e in node.elts)


def _literal_check_type_data(node: ast.AST) -> bool:
    """Only literal built-in containers have callback-free iteration/items here."""
    if isinstance(node, ast.Constant):
        return type(node.value) in (type(None), bool, int, float, complex, str, bytes)
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(not isinstance(item, ast.Starred) and _literal_check_type_data(item)
                   for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(key is not None and _literal_check_type_data(key)
                   and _literal_check_type_data(value)
                   for key, value in zip(node.keys, node.values))
    return False


class _Census:
    def __init__(self, res: Resolver, env_reads_ok: bool = False, class_only: bool = False):
        self.r = res
        self.env_reads_ok = env_reads_ok
        self.class_only = class_only
        # print invokes these methods without an explicit attribute reference.
        res.dead.difference_update(id(n) for n in ast.walk(res.tree)
            if isinstance(n, ast.FunctionDef) and n.name in {"write", "flush"})
        self.tree = res.tree
        self.parent = res.parent
        self.user_classes = {n.name for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)}
        self.check_type_hint_nodes = set()
        for candidate in ast.walk(self.tree):
            if (isinstance(candidate, ast.Call) and
                    self.r.call_name(candidate) == "_sv_verifier.check_type" and
                    len(candidate.args) == 2 and not candidate.keywords and
                    not any(isinstance(arg, ast.Starred) for arg in candidate.args) and
                    closed_hint_effects(candidate.args[1], self.r) is not None):
                self.check_type_hint_nodes.update(ast.walk(candidate.args[1]))

    def class_body(self, node):
        child, parent = node, self.parent.get(node)
        while parent is not None:
            if isinstance(parent, ast.ClassDef):
                return parent
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and child is not parent.args:
                return None
            child, parent = parent, self.parent.get(parent)
        return None

    def run(self) -> None:
        r = self.r
        class_bindings = {}
        for node in ast.walk(self.tree):
            owner = self.class_body(node)
            name = (node.id if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                    else node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None)
            if owner is not None and name:
                class_bindings.setdefault(owner, set()).add(name)
        # A module binding is not a local: reads before the binding still fall
        # back to builtins. A flow-insensitive resolver cannot substitute the
        # later user definition for such an address source.
        for name, scopes in r.bindings.items():
            if id(self.tree) in scopes and hasattr(builtins, name):
                raise _No(f"module shadows builtin {name}; binding order is not certified")
        for node in ast.walk(self.tree):
            if not r.live(node):
                continue
            r.check(node)
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id in class_bindings.get(self.class_body(node), set())
                    and node.id in vars(builtins)):
                raise _No(f"line {node.lineno}: class-local builtin alias {node.id}; binding order is not certified")
            if isinstance(node, ast.FunctionDef) and node.name == "__del__":
                raise _No(f"line {node.lineno}: a __del__ finalizer")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and \
                    node.id in _CONVERT and r.is_builtin(node):
                call = self.parent.get(node)
                if node not in self.check_type_hint_nodes and not (isinstance(call, ast.Call) and call.func is node):
                    # ``key=repr``, ``map(str, xs)``: converts values we cannot see
                    raise _No(f"line {node.lineno}: {node.id} used as a value may show an address")
            if isinstance(node, ast.Attribute) and node.attr in ("format", "format_map"):
                call = self.parent.get(node)
                if not (isinstance(call, ast.Call) and call.func is node):
                    raise _No(f"line {node.lineno}: .{node.attr} used as a value")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and \
                    node.id in _DENIED_BUILTINS and r.is_builtin(node):
                if node.id == "open":
                    self.check_open(node)
                elif not (self.env_reads_ok and node.id == "input"):
                    raise _No(f"line {node.lineno}: uses {node.id}")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and \
                    r.imports.modules.get(node.value.id) == "sys" and node.attr not in SYS_OK and \
                    not (self.env_reads_ok and node.attr in _ENV_READ_SYS_NAMES):
                raise _No(f"line {node.lineno}: uses sys.{node.attr}")
            if isinstance(node, ast.ImportFrom) and node.module == "sys":
                allowed = SYS_OK | (_ENV_READ_SYS_NAMES if self.env_reads_ok else frozenset())
                bad = [a.name for a in node.names if a.name not in allowed]
                if bad:
                    raise _No(f"line {node.lineno}: from sys import {bad[0]}")
            if (isinstance(node, ast.Attribute) and node.attr == "buffer_info" or
                    isinstance(node, ast.Call) and r.call_name(node) == "getattr"
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "buffer_info"):
                raise _No(f"line {node.lineno}: buffer_info may expose a memory address")
            self.check_random(node)
            self.check_os(node)
        self.check_sets()
        self.check_text()

    # -- calls -------------------------------------------------------------------
    def check_open(self, node: ast.Name) -> None:
        call = self.parent.get(node)
        if not (isinstance(call, ast.Call) and call.func is node):
            raise _No(f"line {node.lineno}: open used as a value")
        if self.env_reads_ok:
            # [internal ref] hotfix: a read is a task-declared file read ([internal ref]'s own I/O model
            # already replays it deterministically from the witness/task) -- only "open
            # used as a value" above (an unclear, non-call use) still refuses.
            return
        # the shared I/O model's classification ([internal ref], environment.py::open_effect): a constant
        # write-only mode only
        from tharkun.solve.environment import open_effect

        if open_effect(call) != "output":
            raise _No(f"line {node.lineno}: open() that may read")

    def check_random(self, node: ast.AST) -> None:
        r = self.r
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            call = self.parent.get(node)
            called = isinstance(call, ast.Call) and call.func is node
            if node.attr == "seed" and (not called or not call.args or (
                    isinstance(call.args[0], ast.Constant) and call.args[0].value is None)):
                raise _No(f"line {node.lineno}: .seed without a seed (OS entropy)")
            if node.attr == "__init__" and not (
                    (isinstance(node.value, ast.Name) and node.value.id in self.user_classes) or
                    (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                     and node.value.func.id == "super")):
                raise _No(f"line {node.lineno}: __init__ of something not a program class")
            if node.attr in ("fromfile", "fromfd", "open", "open_code", "FileIO", "TextIOWrapper",
                             "BufferedReader", "BufferedRandom", "BufferedRWPair"):
                raise _No(f"line {node.lineno}: .{node.attr} may read outside state")
        name = None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and \
                r.imports.modules.get(node.value.id) == "random":
            name = node.attr
        elif isinstance(node, ast.Name) and node.id in r.imports.names and \
                r.imports.names[node.id][0] == "random":
            name = r.imports.names[node.id][1]
        if name is None:
            return
        call = self.parent.get(node)
        called = isinstance(call, ast.Call) and call.func is node
        if name == "SystemRandom":
            raise _No(f"line {node.lineno}: random.SystemRandom")
        if name == "Random":
            if not (called and len(call.args) == 1 and not call.keywords
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, (int, str, bytes, float))
                    and not isinstance(call.args[0].value, bool)):
                raise _No(f"line {node.lineno}: random.Random without a literal seed")
        if name == "seed" and (not called or not call.args or (
                isinstance(call.args[0], ast.Constant) and call.args[0].value is None)):
            raise _No(f"line {node.lineno}: random.seed without a seed")

    # -- os / os.path ([internal ref]) -----------------------------------------------------------
    def _os_dotted(self, node: ast.AST) -> tuple[str, str] | None:
        """``("os", name)`` / ``("os.path", name)`` for a reference to that attribute --
        via ``import os`` + one or two attribute hops (``os.NAME`` / ``os.path.NAME``),
        ``from os import path`` then ``path.NAME``, or ``from os.path import NAME``
        directly -- else ``None``. The two-hop case (``os.path.NAME``) is the one dotted
        chain this pass understands; `Resolver`/`check_random`'s own single-hop lookup
        (``imports.modules.get(a_name_node.id)``) cannot see it."""
        r = self.r
        if isinstance(node, ast.Name) and node.id in r.imports.names:
            mod, name = r.imports.names[node.id]
            return (mod, name) if mod in ("os", "os.path") else None
        if not isinstance(node, ast.Attribute):
            return None
        base, name = node.value, node.attr
        if isinstance(base, ast.Name):
            mod = r.imports.modules.get(base.id)
            if mod in ("os", "os.path"):
                return (mod, name)
            if r.imports.names.get(base.id) == ("os", "path"):
                return ("os.path", name)          # `from os import path` bound `base`
            return None
        if (isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
                and r.imports.modules.get(base.value.id) == "os" and base.attr == "path"):
            return ("os.path", name)              # `os.path.NAME` via a plain `import os`
        return None

    def check_os(self, node: ast.AST) -> None:
        got = self._os_dotted(node)
        if got is None:
            return
        mod, name = got
        if mod == "os":
            if name in OS_CONST_OK or name == "path":
                return
            if name in OS_HAVOC:
                raise _No(f"line {node.lineno}: uses os.{name} (environment-dependent)")
            raise _No(f"line {node.lineno}: uses os.{name}")
        else:
            if name in OS_PATH_PURE:
                return
            if name in OS_PATH_HAVOC:
                raise _No(f"line {node.lineno}: uses os.path.{name} (queries the filesystem)")
            raise _No(f"line {node.lineno}: uses os.path.{name}")

    # -- set order -------------------------------------------------------------------
    def is_set_source(self, node: ast.AST) -> bool:
        if isinstance(node, (ast.Set, ast.SetComp)):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id in ("set", "frozenset") and self.r.is_builtin(node.func):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr in _SET_RESULT_METHODS and (
                    self.is_tracked(node.func.value) or self.is_set_source(node.func.value)):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitOr, ast.BitAnd,
                                                                  ast.BitXor, ast.Sub)):
            return any(self.is_set_source(x) or self.is_tracked(x) or self.is_view(x)
                       for x in (node.left, node.right))
        return False

    @staticmethod
    def is_view(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
            node.func.attr in ("keys", "items")

    def is_tracked(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.set_names
        if isinstance(node, ast.Attribute):
            return node.attr in self.set_attrs
        return False

    def check_sets(self) -> None:
        self.singleton_receivers = None
        # names and attribute names that may ever hold a set (a fixpoint over aliases too).
        #
        # [internal ref] part 2 (MG): this tracker is flow- AND scope-insensitive by design (a
        # name is matched by its bare string across the whole program, the same way
        # `("attr", name)` is elsewhere in this file) -- so a name must be a set on
        # EVERY one of its bindings, not just any one of them, or a list bound to a
        # set anywhere else (a later real assignment, a dead branch, a `global`
        # rebind inside a function that never runs before the order-sensitive use)
        # wrongly exempts it. This is therefore a GREATEST fixpoint: start with every
        # directly-bound name optimistically a candidate, then repeatedly drop any
        # candidate that has even one binding which is not itself a set-source or an
        # alias of a name that is still a candidate, until stable.
        #
        # [internal ref] part 3 (MG): the fixpoint above only sees Assign/AnnAssign/NamedExpr
        # targets -- every OTHER way Python can bind a name or attribute is a value
        # this tracker cannot see, so if that value is not provably a set it must
        # VETO, never silently pass through as a non-vote (a missed binding that is
        # secretly a set is merely a missed proof -- safe; a missed binding that is
        # secretly NOT a set, while every registered binding IS one, is a wrong TRUE):
        #   - every function/lambda PARAMETER (positional, positional-only, keyword-
        #     only, *args, **kwargs) takes an arbitrary value on each call this
        #     analysis cannot see, so its name is vetoed unconditionally;
        #   - a class-BODY ``items = []`` is read through ``c.items`` (an attribute),
        #     not only through the bare name, so it is registered as an ``("attr",
        #     name)`` binding too (see the target loop below), not vetoed -- it can
        #     still certify when every attribute binding genuinely agrees;
        #   - ``import x as acc`` / ``from m import x as acc`` binds ``acc`` to a
        #     module/class/function object with no Assign node to see -- vetoed;
        #   - a ``match`` capture (``case x:``, ``case [*rest]:``, ``case {**rest}:``)
        #     binds a name to (part of) the subject with no Assign node either --
        #     ``case x:`` in particular captures the WHOLE subject value, so if that
        #     name is also assigned a set elsewhere this tracker must not treat it as
        #     always-a-set on the strength of the Assign alone -- vetoed;
        #   - ``setattr(obj, "literal", value)`` sets an attribute with no Attribute
        #     node to see either -- registered as an ``("attr", literal)`` binding so
        #     a non-set ``value`` here correctly vetoes it via the ordinary fixpoint,
        #     exactly like a `.attr = value` write would.
        # Audited and needing no change: ``global``/``nonlocal`` declarations don't
        # bind a value themselves (the Assign they precede is already walked, by bare
        # name, regardless of scope); ``del`` doesn't bind a value either; starred
        # and tuple/list unpacking targets are already vetoed by ``veto_target``
        # below; ``obj.__dict__[...] = ...`` never resolves (dunder attribute,
        # blocked upstream by ``pyresolve``) so it never reaches this analysis.
        bindings: dict[tuple[str, str], list[ast.expr]] = defaultdict(list)
        vetoed: set[tuple[str, str]] = set()
        # Nothing is tracked yet -- fine here: this only needs to recognize a
        # SYNTACTIC set (a literal, a set()/frozenset() call, a set BinOp), which
        # is self.set_names/set_attrs-independent; is_set_source's alias-dependent
        # branches degrade to False until the fixpoint below fills them in, which
        # only widens (never narrows) what this upfront structural check flags.
        self.set_names = set()
        self.set_attrs = set()

        # [internal ref] part 4 (MG): a hook can be INSTALLED, not only DEFINED --
        # `class C: __getattr__ = some_func` or `= lambda self, n: []` makes
        # `some_func`/the lambda the class's __getattr__ exactly as a `def
        # __getattr__` would, but it is an Assign, not a FunctionDef.
        #
        # [internal ref] part 5 (MG): part 4's fix still enumerated binding FORMS one
        # at a time (Assign/AnnAssign/AugAssign/tuple-unpack, def, setattr)
        # and kept finding another one it missed (a class-body `for
        # __getattr__ in (...):`, a `match` capture). That is whack-a-mole.
        # Every one of those forms -- and del, with-as, walrus, import-as,
        # except-as -- is Python binding a NAME; ``_hook_scope_names`` below
        # enumerates every AST shape that can bind a name, checked against
        # the ast module directly rather than re-derived from memory (see
        # its own docstring), instead of chasing individual statement kinds.
        # A hook can ALSO be installed by mutating a class object dynamically
        # rather than by binding a name in its body at all: 3-argument
        # ``type(name, bases, ns)`` and a ``metaclass=`` keyword can run
        # arbitrary code that sets anything on the resulting class, and
        # ``setattr``/``delattr`` on a class can add or remove ANY attribute,
        # not only the hook names this file happens to enumerate -- all four
        # are handled in the walk below by setting `hook_installed` directly,
        # the same flag the name-binding scan sets. Needing no code here,
        # confirmed unreachable (verified empirically, not assumed):
        # ``import types`` (module not in DET_MODULES); ``vars(x)`` (``vars``
        # is globally reflective in pyresolve.py, blocked for ANY use, not
        # only as a class-mutation vector); ``C.__dict__[...] = ...`` (a
        # non-DUNDER_OK dunder attribute, blocked the same way).
        _ATTR_HOOKS = frozenset({"__getattr__", "__getattribute__", "__get__",
                                 "__setattr__", "__set__", "__init_subclass__",
                                 "__class_getitem__"})
        hook_installed = [False]

        def _hook_scope_names(node: ast.AST):
            """Every name ``node`` itself binds, if any. Two families, covering
            every AST shape Python uses to bind a name (checked against the ast
            module's own node list):

            - ``ast.Name`` with ``ctx`` ``Store`` or ``Del``: CPython gives
              EVERY nested name in a tuple/list/starred unpacking its own
              Store/Del ctx (confirmed via ``ast.dump`` on ``a, b = ...``,
              ``del a, b``, ``with cm() as (a, b):``), so this single check
              -- found by walking the WHOLE tree, not by special-casing each
              enclosing statement -- covers plain/annotated/augmented assign,
              a walrus, a for-loop or comprehension target, a with-item, a
              del target, and a ``type X = ...`` alias (its ``name`` field is
              itself a ``Name`` node, confirmed via ``ast.dump``), unpacked to
              any depth, in one place.
            - the four forms Python represents as a plain ``str`` field
              instead of a ``Name`` node: ``ast.alias`` (import / import-as),
              ``ast.ExceptHandler.name`` (except-as), ``ast.MatchAs.name`` /
              ``ast.MatchStar.name`` / ``ast.MatchMapping.rest`` (a match
              capture), and ``ast.FunctionDef`` / ``ast.AsyncFunctionDef`` /
              ``ast.ClassDef.name`` (def/class).
            """
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                yield node.id
            elif isinstance(node, ast.alias) and node.name != "*":
                yield (node.asname or node.name).split(".")[0]
            elif isinstance(node, ast.ExceptHandler) and node.name:
                yield node.name
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                yield node.name
            elif isinstance(node, ast.MatchMapping) and node.rest:
                yield node.rest
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield node.name

        def veto_target(t: ast.AST) -> None:
            # Unpacking a name out of ANYTHING (a tuple, a starred remainder, a
            # for/comprehension target, a with-item, an except name) never hands
            # that name the whole set/list/etc. it came from -- these names can
            # never be set-tracked, regardless of what the source was.
            if isinstance(t, ast.Name):
                vetoed.add(("name", t.id))
            elif isinstance(t, ast.Attribute):
                vetoed.add(("attr", t.attr))
            elif isinstance(t, (ast.Tuple, ast.List)):
                for e in t.elts:
                    veto_target(e.value if isinstance(e, ast.Starred) else e)

        # One walk of the tree does both jobs (hook detection and the
        # bindings/veto scan below): they inspect disjoint node shapes for the
        # most part, and `self.class_body(node)` -- an O(depth) walk up the
        # parent chain -- is worth computing at most once per node, not twice.
        for node in ast.walk(self.tree):
            in_class_body_node = self.class_body(node) is not None
            if in_class_body_node:
                for _name in _hook_scope_names(node):
                    if _name in _ATTR_HOOKS:
                        hook_installed[0] = True
                if isinstance(node, ast.ClassDef) and any(
                        kw.arg == "metaclass" for kw in node.keywords):
                    # [internal ref] part 5: a metaclass's __new__/__init__/__call__ runs
                    # arbitrary code against the class being created -- it can
                    # set anything on it, so any class defined with one makes
                    # every attribute name unprovable, the same as a hook.
                    hook_installed[0] = True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                    self.r.is_builtin(node.func) and \
                    node.func.id in ("setattr", "delattr") and node.args and \
                    isinstance(node.args[0], ast.Name) and \
                    node.args[0].id in self.user_classes:
                # [internal ref] part 5: setattr/delattr on a CLASS (not an instance,
                # which does not affect the special-method lookup protocol --
                # see the setattr-registration comment below) can add or
                # remove ANY attribute the class has, not only the literal
                # hook names above -- broader than the (attr, literal)
                # registration below, which only covers setattr with a value
                # this tracker can reason about as a binding.
                hook_installed[0] = True
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and \
                    node.id == "type" and self.r.is_builtin(node):
                # [internal ref] part 7 (MG): an ALLOWLIST, not another denylist entry --
                # parts 5 and 6 kept adding one more unsafe SHAPE (a literal
                # 3-arg call, an alias, a saved 1-arg result later called...)
                # and MG kept finding the next one (the *result* of a legal
                # `type(x)` is itself a metaclass when `x` is a class, so
                # `M = type(A); M(...)` builds a class exactly like `type(...)`
                # does, with no further use of the name `type` at all). Instead:
                # class machinery is safe ONLY in a small, explicit list of
                # shapes, checked here in one place; every occurrence of `type`
                # or its result OUTSIDE that list vetoes tracking, with no
                # attempt to keep enumerating the ways it can escape.
                #
                # Safe: `type(x)` -- a call with EXACTLY one positional
                # argument, no keywords, no `*args` -- used DIRECTLY as an
                # operand of is/is not/==/!=/in/not in, as a (direct, not
                # nested-in-a-container) argument to isinstance()/issubclass(),
                # or as the base of a `.__name__` attribute access. A `class`
                # statement with no `metaclass=` keyword is separately safe
                # (checked above, unconditionally -- classes are the norm).
                # Everything else about `type` -- assigned to a name, saved
                # and called later, subscripted out of a container, passed
                # to an arbitrary function, called with any other arity,
                # given keywords or `*args`, or (the safe call itself) used
                # anywhere but the three positions above -- is unsafe.
                call = self.parent.get(node)
                is_safe_call_shape = (
                    isinstance(call, ast.Call) and call.func is node and
                    len(call.args) == 1 and not call.keywords and
                    not any(isinstance(a, ast.Starred) for a in call.args))
                if not is_safe_call_shape:
                    hook_installed[0] = True
                else:
                    use = self.parent.get(call)
                    safe_ops = (ast.Is, ast.IsNot, ast.Eq, ast.NotEq, ast.In, ast.NotIn)
                    is_safe_use = (
                        (isinstance(use, ast.Compare) and
                            ((use.left is call and all(isinstance(o, safe_ops) for o in use.ops))
                             or (call in use.comparators and all(isinstance(o, safe_ops) for o in use.ops))))
                        or (isinstance(use, ast.Call) and isinstance(use.func, ast.Name) and
                            use.func.id in ("isinstance", "issubclass") and self.r.is_builtin(use.func) and
                            (call in use.args or call in (kw.value for kw in use.keywords)))
                        or (isinstance(use, ast.Attribute) and use.value is call and use.attr == "__name__"))
                    if not is_safe_use:
                        hook_installed[0] = True
            if isinstance(node, ast.AugAssign):
                # [internal ref]: `x += s` (also |=, -=, &=, ^=, *=, ...) never RETYPES its
                # target: __iadd__ (list) mutates and returns the same list;
                # __add__/__or__/etc. (str, tuple, frozenset, ...) return a value of
                # the target's OWN type. The RHS being set-shaped tells us nothing
                # about whether the target is a set -- only every real binding
                # agreeing does. Neutral either way: neither a vote nor a veto.
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                value = getattr(node, "value", None)
                if value is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        bindings[("name", t.id)].append(value)
                        if in_class_body_node:
                            # [internal ref] part 3: a class-body `items = []` is reached from
                            # OUTSIDE the class through the attribute `c.items`, not
                            # only through the bare name `items` -- register it as an
                            # ("attr", name) binding too, so an unrelated
                            # `other.items = set()` elsewhere correctly vetoes
                            # `c.items` via the ordinary fixpoint (a class-body
                            # binding this tracker only saw by name, and never by
                            # attribute, used to wrongly let the attribute use pass).
                            bindings[("attr", t.id)].append(value)
                    elif isinstance(t, ast.Attribute):
                        bindings[("attr", t.attr)].append(value)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        if self.is_set_source(value):
                            # Unpacking a set's elements into names is itself an
                            # order hazard (which name gets which element depends
                            # on hash-iteration order) -- refuse outright rather
                            # than merely veto the names involved.
                            raise _No(f"line {node.lineno}: a set unpacked or stored in a container")
                        veto_target(t)
                    elif self.is_set_source(value):
                        # A set stored through a target this tracker does not
                        # bind a name for (a subscript, a slice, ...) -- refuse
                        # rather than silently drop the write's order-hazard.
                        raise _No(f"line {node.lineno}: a set unpacked or stored in a container")
                    # else: an unhandled target (e.g. a subscript) with a value
                    # that is not (syntactically) a set never binds a name, so
                    # it is simply irrelevant to set-name tracking.
            elif isinstance(node, (ast.For, ast.comprehension)):
                veto_target(node.target)
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                veto_target(node.optional_vars)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                vetoed.add(("name", node.name))
            elif isinstance(node, ast.arg):
                # [internal ref] part 3: every parameter kind -- positional, positional-only,
                # keyword-only, *args, **kwargs, a lambda's -- is represented as an
                # `ast.arg` uniformly; each takes an arbitrary value per call.
                vetoed.add(("name", node.arg))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    vetoed.add(("name", (alias.asname or alias.name).split(".")[0]))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        vetoed.add(("name", alias.asname or alias.name))
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                vetoed.add(("name", node.name))
            elif isinstance(node, ast.MatchMapping) and node.rest:
                vetoed.add(("name", node.rest))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                    node.func.id == "setattr" and self.r.is_builtin(node.func) and \
                    len(node.args) >= 3 and isinstance(node.args[1], ast.Constant) and \
                    isinstance(node.args[1].value, str):
                # setattr(obj, "literal", value): the same write as `obj.attr =
                # value` with no Attribute node for the target-walk above to see --
                # register it as the same kind of binding so a non-set `value` here
                # vetoes the attribute exactly as a real `.attr = value` would (a
                # non-literal name already fails to resolve upstream, in pyresolve.py).
                # (part 5's broader setattr/delattr-on-a-class check above already
                # covers a hook name here too; this registration additionally lets
                # a NON-hook literal attr name participate in the ordinary
                # fixpoint, same as a real `.attr = value` write would.)
                bindings[("attr", node.args[1].value)].append(node.args[2])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and \
                    in_class_body_node:
                # [internal ref] part 3 (MG, MG-blocks-PR-80): a class-body METHOD or NESTED
                # CLASS is read through `c.name` (an attribute) exactly like a
                # class-body Name binding is -- `class C: def items(self): return
                # self.store` (a @property or a plain method) or `class C: class
                # items: ...` hands out a value this tracker cannot see at all (no
                # Assign node), so it must veto rather than silently let some
                # OTHER class's unrelated `other.items = set()` (or this part's own
                # class-body-Name-to-attr registration, `class D: items = set()`)
                # stand as the attribute's only visible binding. (Whether `name`
                # is also a hook is decided by the unified scan above.)
                vetoed.add(("attr", node.name))

        # [internal ref] part 3/4/5 (MG): a class defining, installing or dynamically
        # acquiring __getattr__/__getattribute__/__setattr__/__get__/__set__/
        # __init_subclass__/__class_getitem__ (or one whose creation this
        # tracker cannot fully see -- type()/metaclass=) can return or accept
        # ANY value for ANY attribute name on ANY of its instances -- this
        # tracker has no per-class, per-instance model, so ANY of it ANYWHERE
        # in the program makes EVERY attribute name unprovable, not only the
        # ones spelled out. `__slots__` alone needs no extra veto here: an
        # auto-generated slot descriptor is transparent (stores and returns
        # exactly what was assigned) unless paired with one of these dunders,
        # which this same rule already catches.
        unsafe_attrs = hook_installed[0]

        candidates = {key for key in bindings if key not in vetoed}
        if unsafe_attrs:
            candidates = {key for key in candidates if key[0] != "attr"}
        while True:
            self.set_names = {name for kind, name in candidates if kind == "name"}
            self.set_attrs = {name for kind, name in candidates if kind == "attr"}
            dropped = {key for key in candidates
                       if any(not (self.is_set_source(v) or self.is_tracked(v))
                              for v in bindings[key])}
            if not dropped:
                break
            candidates -= dropped
        self.set_names = {name for kind, name in candidates if kind == "name"}
        self.set_attrs = {name for kind, name in candidates if kind == "attr"}
        for node in ast.walk(self.tree):
            if not self.r.live(node):
                continue
            if self.is_set_source(node) and not _int_literal_set(node):
                self.check_set_use(node, "a set")
            elif self.is_tracked(node) and isinstance(getattr(node, "ctx", None), ast.Load):
                self.check_set_use(node, f"set {ast.unparse(node)}")

    def check_set_use(self, node: ast.AST, what: str) -> None:
        par = self.parent.get(node)
        line = getattr(node, "lineno", "?")
        if isinstance(par, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and par.value is node:
            targets = par.targets if isinstance(par, ast.Assign) else [par.target]
            # [internal ref] part 2: "tracked where it lands" only holds when every target
            # is a plain Name/Attribute -- those are exactly the bindings check_sets
            # folds into set_names/set_attrs. A target this tracker does not bind a
            # name for (a subscript, a slice, ...) never gets set-tracked, so the
            # set's order-hazard is NOT carried forward here; check_sets's own
            # upfront pass already refuses a set landing in one of those (raising
            # before this method ever runs), so reaching here with such a target
            # means the value was NOT provably a set at that earlier, alias-blind
            # check -- fall through and let the checks below decide.
            if all(isinstance(t, (ast.Name, ast.Attribute)) for t in targets):
                return                             # tracked where it lands
        if isinstance(par, ast.AugAssign) and node in (par.value, par.target) \
                and self.is_tracked(par.target):
            return  # [internal ref]: only exempt when the TARGET is already a tracked set
                     # (x |= y between two known sets stays order-blind); a target
                     # that is not already a set (or not provably one) makes this
                     # `x += <set>` the order-sensitive case itself, like
                     # list.extend(set) -- fall through and flag it below.
        if isinstance(par, ast.Compare):
            order_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
            if node in par.comparators and all(isinstance(o, (ast.In, ast.NotIn) + order_ops)
                                               for o in par.ops):
                return
            if par.left is node and all(isinstance(o, order_ops) for o in par.ops):
                return
        if isinstance(par, (ast.If, ast.While, ast.IfExp, ast.Assert)) and par.test is node:
            return
        if isinstance(par, ast.UnaryOp) and isinstance(par.op, ast.Not):
            return
        if isinstance(par, ast.BinOp) and self.is_set_source(par):
            return                                 # the result is checked as a set itself
        if isinstance(par, ast.Call) and isinstance(par.func, ast.Name) and node in par.args \
                and self.r.is_builtin(par.func):
            if par.func.id in ("len", "bool"):
                return                             # (not sorted: ties keep the set's order)
            if par.func.id in ("set", "frozenset") and len(par.args) == 1:
                return                             # a copy, checked as a set itself
        if isinstance(par, ast.Call) and isinstance(par.func, ast.Attribute) and node in par.args \
                and par.func.attr in _SET_SAFE_METHODS | _SET_RESULT_METHODS and (
                    self.is_tracked(par.func.value) or self.is_set_source(par.func.value)):
            return                                 # s.update(t), s.union(t), s.issubset(t)
        if isinstance(par, ast.Attribute) and par.value is node:
            gp = self.parent.get(par)
            if (par.attr == "pop" and isinstance(gp, ast.Call) and gp.func is par
                    and not gp.args and not gp.keywords):
                if self.singleton_receivers is None:
                    self.singleton_receivers = singleton_receivers(self.r)
                if node in self.singleton_receivers:
                    return  # exact receiver only, after whole-call/effect closure
            if isinstance(gp, ast.Call) and gp.func is par and (
                    par.attr in _SET_SAFE_METHODS or par.attr in _SET_RESULT_METHODS):
                return
        if isinstance(par, ast.Expr):
            return
        raise _No(f"line {line}: {what} is used where its order can matter "
                  f"({ast.unparse(par)[:60]!r})")

    # -- addresses through text --------------------------------------------------------
    def check_text(self) -> None:
        self.provenance = Provenance(self.r, class_only=self.class_only)
        if self.provenance.deque_calls and not self.provenance.closed_deques():
            raise _No("deque transport or callback effects are not closed")
        counter_calls = {n for n in self.provenance.nodes if self.provenance.empty_counter(n)}
        if counter_calls and self.provenance.deque_calls:
            raise _No("a program with both an active deque and a Counter is not modelled")
        if counter_calls:
            reason = self.provenance.counter_reason(counter_calls)
            if reason:
                raise _No(reason)
        counter_funcs = {node.func for node in counter_calls}
        defaultdict_calls = {n for n in self.provenance.nodes if self.provenance.recognized_defaultdict(n)
                             or self.provenance.defaultdict_candidate(n)}
        if counter_calls and defaultdict_calls:
            raise _No("a program with both an active Counter and defaultdict is not modelled")
        if self.provenance.deque_calls and any(
                self.provenance.lambda_defaultdict(n) for n in defaultdict_calls):
            raise _No("a program with both an active deque and lambda-defaultdict is not modelled")
        if defaultdict_calls:
            reason = self.provenance.defaultdict_reason(defaultdict_calls)
            if reason:
                raise _No(reason)
        defaultdict_funcs = {n.func for n in defaultdict_calls}
        convs = []
        source_call_funcs = {
            node.func for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and self._source_function_call(node.func)
        }
        for node in ast.walk(self.tree):
            if not self.r.live(node):
                continue
            if node in self.check_type_hint_nodes:
                continue
            entry = None
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                entry = self.r.imports.names.get(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                module = self.r.imports.modules.get(node.value.id)
                if module:
                    entry = module, node.attr
            if entry == ("_sv_verifier", "check_type"):
                par = self.parent.get(node)
                if not (isinstance(par, ast.Call) and par.func is node):
                    raise _No(f"line {node.lineno}: check_type used outside a direct call")
            if entry is not None and entry[0] == "heapq":
                par = self.parent.get(node)
                if not (isinstance(par, ast.Call) and par.func is node
                        and self.provenance.heapq_call_safe(par)):
                    raise _No(f"line {node.lineno}: heapq use is outside the ordered direct-call subset")
            if isinstance(node, ast.Call) and self.r.call_name(node) == "_sv_verifier.check_type":
                self.check_type_call(node)
            parent = self.parent.get(node)
            deque_constructor = (entry == ("collections", "deque")
                and isinstance(parent, ast.Call) and parent.func is node
                and parent in self.provenance.deque_calls)
            if node not in source_call_funcs | defaultdict_funcs | counter_funcs and not deque_constructor and entry and entry not in STRING_CONSTANTS and (
                    entry[0] not in _ADDRESS_MODULES | {"heapq"} or
                    entry[0] == "io" and entry[1] not in _ADDRESS_IO):
                raise _No(f"line {node.lineno}: no address-provenance model for {'.'.join(entry)}")
            for operand, where in self.conversions(node):
                if isinstance(operand, ast.Constant):
                    continue            # the text of a literal (``assert x, "msg"``, ``str(5)``) shows no address, whatever else the file does
                convs.append((node, operand, where))
        if not convs:
            return
        if self.provenance.unsafe:
            raise _No("unsupported binding or object hooks in address provenance")
        for node, operand, where in convs:
            if (where == "exception text" and isinstance(node, ast.Subscript)
                    and operand is node.slice and self.provenance.inert_empty_index(operand)):
                continue  # ``d[<empty list>[k]]``: the key raises IndexError before any lookup can show it
            if (self.class_only and where == "exception text" and isinstance(node, ast.Subscript)
                    and operand is node.slice and self.provenance.class_only_lookup(node)):
                continue  # observer-free program: the KeyError's text can only reach stderr, no verdict reads it
            mode = "repr" if where in {"repr()", "ascii()", "%r", "%a"} else "str"
            if where == "an f-string field":
                mode = {97: "repr", 114: "repr", 115: "str"}.get(node.conversion, "format")
            if where == "format()":
                mode = "format"
            if where in {".format", ".format_map", "%-formatting", "exception text"}:
                mode = "any"  # a format string can select str, repr or __format__
            if where == ".format_map" and any(
                    v[0] != "dict" for v in self.provenance.value(operand)):
                raise _No(f"line {node.lineno}: format_map of an unknown mapping")
            if self.provenance.safe(operand, mode):
                continue
            raise _No(f"line {getattr(node, 'lineno', '?')}: {where} of "
                      f"{ast.unparse(operand)[:40]!r} may show an address")

    def _source_function_call(self, func: ast.AST) -> bool:
        """Whether a direct imported callee resolves to an undecorated source def."""
        if isinstance(func, ast.Name):
            module, exported = self.r.imports.names.get(func.id, (None, None))
            child = self.r.externs.get(func.id)
            if child is not None and module == f"$extern:{child.filename}":
                return self._source_export_function(child, exported, set())
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            child = self.r.externs.get(func.value.id)
            imported_module = self.r.imports.modules.get(func.value.id)
            if child is not None and imported_module is not None:
                return self._source_export_function(child, func.attr, set())
        return False

    @classmethod
    def _source_export_function(cls, resolver: Resolver, exported: str,
                                seen: set[tuple[int, str]]) -> bool:
        key = (id(resolver), exported)
        if key in seen or exported not in resolver.public_names():
            return False
        seen.add(key)
        definitions = [node for node in resolver.tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == exported]
        if definitions:
            extra_bindings = any(
                (isinstance(node, ast.AsyncFunctionDef) and node.name == exported) or
                (isinstance(node, ast.ExceptHandler) and node.name == exported) or
                (isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == exported) or
                (isinstance(node, ast.MatchMapping) and node.rest == exported) or
                (isinstance(node, (ast.Global, ast.Nonlocal)) and exported in node.names)
                for node in ast.walk(resolver.tree)
            )
            return (len(definitions) == 1 and not definitions[0].decorator_list
                    and not extra_bindings and not resolver._rebound(exported))
        module, name = resolver.imports.names.get(exported, (None, None))
        child = resolver.externs.get(exported)
        if child is None or module != f"$extern:{child.filename}":
            return False
        return cls._source_export_function(child, name, seen)

    def check_type_call(self, node: ast.Call) -> None:
        """Certify only direct, closed-hint calls whose possible callbacks are absent."""
        if (len(node.args) != 2 or node.keywords or
                any(isinstance(arg, ast.Starred) for arg in node.args)):
            raise _No(f"line {node.lineno}: check_type call shape is not closed")
        effects = closed_hint_effects(node.args[1], self.r)
        if effects is None:
            raise _No(f"line {node.lineno}: check_type hint is not closed")
        value = node.args[0]
        literal_data = _literal_check_type_data(value)
        if not literal_data and (effects or not self.provenance.scalar(self.provenance.value(value))):
            raise _No(f"line {node.lineno}: check_type value effects are not closed")
        if not literal_data and not self.provenance.safe(value, "repr"):
            raise _No(f"line {node.lineno}: check_type value repr may run user code or expose an address")

    def conversions(self, node: ast.AST):
        """(operand, what) for each value ``node`` turns into text."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id in _CONVERT and self.r.is_builtin(node.func):
            if node.args:
                yield node.args[0], f"{node.func.id}()"
            # str's object parameter is keyword-capable (unlike repr/format).
            if node.func.id == "str":
                for keyword in node.keywords:
                    if keyword.arg == "object":
                        yield keyword.value, "str()"
                    elif keyword.arg is None:
                        raise _No(f"line {node.lineno}: str() with unknown keyword expansion")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                self.r.is_builtin(node.func) and isinstance(
                    getattr(builtins, node.func.id, None), type) and \
                issubclass(getattr(builtins, node.func.id), BaseException):
            for a in node.args:
                yield a, "exception text"
        elif (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
              and any(v[0] == "dict" for v in self.provenance.value(node.value))):
            yield node.slice, "exception text"  # KeyError can display its key.
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in {"index", "pop", "remove"}):
            for a in node.args:
                yield a, "exception text"  # lookup failures can repr the item
        elif isinstance(node, ast.Call) and self.r.call_name(node) == "next":
            for a in node.args:
                yield a, "exception text"  # StopIteration may carry a return value
        elif isinstance(node, ast.Assert) and node.msg is not None:
            yield node.msg, "exception text"
        elif isinstance(node, ast.Raise) and node.exc is not None:
            if isinstance(node.exc, ast.Call):
                for a in list(node.exc.args) + [k.value for k in node.exc.keywords]:
                    yield a, "exception text"
            elif not (isinstance(node.exc, ast.Name) and self.r.is_builtin(node.exc)
                      and isinstance(getattr(builtins, node.exc.id, None), type)
                      and issubclass(getattr(builtins, node.exc.id), BaseException)):
                yield node.exc, "exception text"
        elif isinstance(node, ast.Call) and self.r.call_name(node) == "sys.exit":
            for a in node.args:
                yield a, "exception text"
        elif isinstance(node, ast.FormattedValue):
            yield node.value, "an f-string field"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id == "print" and self.r.is_builtin(node.func):
            for a in node.args:                    # text written where the program may read it
                yield a, "print()"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr in ("format", "format_map"):
            fmt = node.func.value
            if not (isinstance(fmt, ast.Constant) and isinstance(fmt.value, str)):
                raise _No(f"line {node.lineno}: .{node.func.attr} on a computed format string")
            if re.search(r"\{[^{}]*[.\[][^{}]*\}", fmt.value):
                raise _No(f"line {node.lineno}: a format field that reaches attributes/items")
            for a in list(node.args) + [k.value for k in node.keywords]:
                yield a, f".{node.func.attr}"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            left = node.left
            if isinstance(left, ast.Constant) and isinstance(left.value, (str, bytes)):
                fmt = left.value.decode("latin1") if isinstance(left.value, bytes) else left.value
                specs = [m.group(1) for m in _NUMERIC_SPEC.finditer(fmt) if m.group(1) != "%"]
                args = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
                if "%(" in fmt or len(specs) != len(args):
                    for a in args:                 # mapping keys or a count we cannot pair
                        yield a, "%-formatting"
                    return
                for spec, a in zip(specs, args):
                    if spec in "sra":
                        yield a, f"%{spec}"
            elif isinstance(left, ast.Constant) or isinstance(left, ast.JoinedStr):
                return
            elif self.provenance.value(left) and all(v[0] in {
                    "int", "float", "complex", "bool"} for v in self.provenance.value(left)):
                return  # Numeric remainder never performs a text conversion.
            else:
                # the left side may be a string: every argument may be %s-formatted
                args = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
                for a in args:
                    if not self.numeric_literal(a):
                        yield a, "%-formatting"

    @staticmethod
    def numeric_literal(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and type(node.value) in (int, float, complex)
