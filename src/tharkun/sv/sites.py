"""Violation sites of an SV task and the harness that exposes its draws ([internal ref] slice 2).

``! uncaught(E)`` is violated where the program can raise ``E``.  This module is
pure AST analysis (it executes nothing; its one tharkun import is the ``call_target`` fact's index,
``solve/call_targets.py::call_target_index``, which says what the entry call calls):

* :func:`violation_sites` lists the EXPLICIT sites -- ``raise X(...)`` with ``X`` a
  property class (or a subclass among the builtins) and, for ``AssertionError``,
  ``assert`` -- with the ``if`` guards that enclose each one.  Implicit sites
  (a failing subscript, a division by zero) are the next slice.
* :func:`build_harness` turns the program's entry code into
  ``def __sv_harness(d0, d1, ...)`` (which code: :func:`_plan`): the ``if __name__ ==
  "__main__":`` block, the body of the one function the program ends by calling
  (``def main(): ...`` then ``main()``, [internal ref]), or the module-level statements when a draw sits
  there.  Every ``_sv_verifier.nondet_*`` call that is
  not inside a lambda or nested def becomes a parameter (a stream, one value per execution, inside a loop or
  comprehension), so
  Tharkun's solvers and ``drive_to_line`` can treat the draws as ordinary
  arguments.  Every draw the harness executes -- a parameter or a default for a
  draw it could not lift -- is logged in execution order by ``__sv_draw``, which is
  exactly the child's draw vector (``_child.py``), so a solved input replays.

Site record (shared with the abstract interpreter, [internal ref]): ``{"line", "col", "exc", "kind",
"function", "guards"}`` where ``kind`` is ``raise`` | ``assert``, ``function`` is the
enclosing def (``"<main>"`` for the main block) and ``guards`` lists the
enclosing ``if`` tests as ``(source, branch)`` with ``branch`` ``"then"``/``"else"``.
Line numbers are the ORIGINAL program's for :func:`violation_sites` and the
harness's for :attr:`Harness.sites`.
"""

from __future__ import annotations

import ast
import builtins
import warnings
from dataclasses import dataclass, field

#: ``_sv_verifier`` functions the harness lifts, and the draw kind each consumes.
LIFTED = {"nondet_int": "int", "nondet_bool": "bool", "nondet_float": "float",
          "nondet_float_range": "float", "nondet_str": "str"}
#: the draw a lifted parameter starts from (the child's default for the kind).
DEFAULTS = {"int": 0, "bool": False, "float": 0.0, "str": "", "argv_full": [], "rindex": 0}
HARNESS_FN = "__sv_harness"


def _raises_class(node: ast.Raise) -> str | None:
    exc = node.exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return exc.attr
    return None


def _matches(name: str | None, classes) -> str | None:
    """The property class ``name`` is (or derives from, among the builtins)."""
    if not name:
        return None
    if name in classes:
        return name
    cls = getattr(builtins, name, None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        for base in cls.__mro__:
            if base.__name__ in classes:
                return base.__name__
    return None


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == "__main__")


def _sites_in(tree: ast.Module, classes) -> list[dict]:
    out: list[dict] = []

    def walk(stmts, fn: str, guards: list):
        for st in stmts:
            if isinstance(st, ast.Raise):
                exc = _matches(_raises_class(st), classes)
                if exc:
                    out.append({"line": st.lineno, "col": st.col_offset, "exc": exc, "kind": "raise",
                                "function": fn, "guards": list(guards)})
            elif isinstance(st, ast.Assert) and "AssertionError" in classes:
                out.append({"line": st.lineno, "col": st.col_offset, "exc": "AssertionError",
                            "kind": "assert", "function": fn, "guards": list(guards)})
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(st.body, st.name, [])
            elif isinstance(st, ast.ClassDef):
                walk(st.body, fn, guards)
            elif isinstance(st, ast.If):
                if _is_main_guard(st) and fn == "<module>":
                    walk(st.body, "<main>", [])
                    walk(st.orelse, "<module>", guards)
                    continue
                src = ast.unparse(st.test)
                walk(st.body, fn, guards + [(src, "then")])
                walk(st.orelse, fn, guards + [(src, "else")])
            else:
                for attr in ("body", "orelse", "finalbody"):
                    inner = getattr(st, attr, None)
                    if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                        walk(inner, fn, guards)
                for h in getattr(st, "handlers", None) or []:
                    walk(h.body, fn, guards)
    walk(tree.body, "<module>", [])
    return out


def violation_sites(source: str, classes) -> list[dict]:
    """The explicit violation sites of ``source`` for the property ``classes``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return _sites_in(tree, tuple(classes))


@dataclass
class Harness:
    source: str                          # the harness module's text
    params: list[str]                    # d0, d1, ... in source order
    kinds: list[str]                     # the draw kind of each parameter
    unlifted: int                        # nondet calls left to the stub (loops, ...)
    sites: list[dict] = field(default_factory=list)   # violation sites, harness lines
    #: which draws feed which function directly: ``{fn: {param: "dK"}}`` for a call
    #: ``fn(a, ...)`` in the main block whose arguments are draws (or names bound to
    #: one) -- the def-use link that lets a site solved inside ``fn`` map back to draws
    feeds: dict = field(default_factory=dict)
    #: which code the harness is (:func:`_plan`), for the log
    entry: str = "main block"
    #: the entry function whose body became the harness body ("" when none did)
    entry_fn: str = ""

    def scoped(self, records):
        """The ``call_target`` records as the harness reads them.  A record is keyed by the scope of its call, and the
        calls of an entry function now sit in the harness, whose scope is the module's (``<main>``); a nested scope
        loses the function's name the same way.  Without this a call to a class in ``main`` has "no call_target fact"."""
        if not self.entry_fn or records is None:
            return records
        out = []
        for r in records:
            scope = r.get("scope") or ""
            if scope == self.entry_fn:
                r = dict(r, scope="<main>")
            elif scope.startswith(self.entry_fn + "."):
                r = dict(r, scope=scope[len(self.entry_fn) + 1:])
            out.append(r)
        return out

    def seed(self) -> dict:
        return {p: ([] if k.startswith("list:") else DEFAULTS[k]) for p, k in zip(self.params, self.kinds)}


class _Lift(ast.NodeTransformer):
    def __init__(self, aliases: set[str], direct: dict[str, str], named: dict | None = None,
                 sys_aliases: set[str] | None = None, argv_names: set[str] | None = None,
                 random_aliases: set[str] | None = None, random_choice_ok: bool = False):
        self.aliases = aliases           # names the _sv_verifier module is bound to
        self.direct = direct             # from-imported nondet names -> their real name
        self.named = named or {}         # id(call) -> the variable it initialises
        self.sys_aliases = sys_aliases or set()
        self.argv_names = argv_names or set()
        self.argv_param = None
        self.random_aliases = random_aliases or set()  # names the random module is bound to
        # whole-module precondition ([internal ref]): ``random.seed`` is never called (qualified
        # form) anywhere in the program, so every ``random.choice`` really does draw from
        # the unseeded, replayable ``rindex`` source _child.py already models (see
        # ``_patch_random`` there) -- NOT a soundness gate (replay always re-derives the
        # truth from the untransformed program, and _child.py's own seeded/unseeded branch
        # is a second independent guard), just a precision one: skip lifting a call this
        # scan cannot show is unseeded, rather than spend a solver round on a candidate
        # _child.py would ignore anyway.
        self.random_choice_ok = random_choice_ok
        self.params: list[str] = []
        self.kinds: list[str] = []
        self.unlifted = 0
        self._opaque = 0                 # inside a lambda / nested def: left to the stub
        self._loop = 0                   # inside a loop / comprehension: an indexed stream

    def _choice_seq(self, call: ast.Call) -> ast.expr | None:
        """``call`` is ``<random>.choice(seq)`` (module-qualified, one positional arg,
        no keywords) -- ``seq`` if so."""
        f = call.func
        if (isinstance(f, ast.Attribute) and f.attr == "choice"
                and isinstance(f.value, ast.Name) and f.value.id in self.random_aliases
                and len(call.args) == 1 and not call.keywords):
            return call.args[0]
        return None

    def _nondet_name(self, call: ast.Call) -> str | None:
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in self.aliases:
            return f.attr
        if isinstance(f, ast.Name) and f.id in self.direct:
            return self.direct[f.id]
        return None

    def _opaque_visit(self, node):
        self._opaque += 1
        try:
            return self.generic_visit(node)
        finally:
            self._opaque -= 1

    def _loop_visit(self, node):
        self._loop += 1
        try:
            return self.generic_visit(node)
        finally:
            self._loop -= 1

    visit_For = visit_While = visit_AsyncFor = _loop_visit
    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _loop_visit
    visit_Lambda = visit_FunctionDef = visit_AsyncFunctionDef = _opaque_visit

    def _argv(self, node):
        if self._opaque:
            self.unlifted += 1
            return node
        if self.argv_param is None:
            self.argv_param = f"d{len(self.params)}"
            self.params.append(self.argv_param)
            self.kinds.append("argv_full")
        return ast.copy_location(ast.Call(func=ast.Name("__sv_argv", ast.Load()),
                                          args=[ast.Name(self.argv_param, ast.Load())], keywords=[]), node)

    def visit_Attribute(self, node: ast.Attribute):
        if (isinstance(node.ctx, ast.Load) and node.attr == "argv"
                and isinstance(node.value, ast.Name) and node.value.id in self.sys_aliases):
            return self._argv(node)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.argv_names:
            return self._argv(node)
        return node

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        seq = self._choice_seq(node)
        if seq is not None:
            # a draw inside a loop would need index-stream support this slice does not
            # add ([internal ref]); an opaque scope is left to the stub like any other draw there;
            # a program this scan could not certify unseeded is left untouched (see
            # ``random_choice_ok``) -- in every skipped case the real ``random.choice``
            # keeps running, exactly as it does today.
            if self._opaque or self._loop or not self.random_choice_ok:
                self.unlifted += 1
                return node
            p = f"d{len(self.params)}"
            self.params.append(p)
            self.kinds.append("rindex")
            return ast.copy_location(
                ast.Call(func=ast.Name("__sv_index", ast.Load()),
                         args=[ast.Constant("rindex"), ast.Name(p, ast.Load()), seq], keywords=[]),
                node)
        name = self._nondet_name(node)
        if name is None:
            return node
        kind = LIFTED.get(name)
        if kind is None or self._opaque or node.keywords:
            self.unlifted += 1
            return node
        k = len(self.params)
        if self._loop:
            # a draw inside a loop is an indexed STREAM: the parameter ``sK`` is the list of
            # values this call site returns, one per execution (the child's default past
            # its end), so a solver sets the i-th iteration's draw as ``sK[i]``
            p = f"s{k}"
            self.params.append(p)
            self.kinds.append("list:" + kind)
            return ast.Call(func=ast.Name("__sv_next", ast.Load()),
                            args=[ast.Constant(kind), ast.Name(p, ast.Load()), *node.args], keywords=[])
        # ``x = nondet_int()`` at the top of the main block: the parameter IS ``x``, so a
        # guard over ``x`` is a guard over a parameter and the guard solvers see it
        # directly; any other draw gets a fresh ``dK``
        p = self.named.get(id(node)) or f"d{k}"
        if p in self.params:
            p = f"d{k}"
        self.params.append(p)
        self.kinds.append(kind)
        return ast.Call(func=ast.Name("__sv_draw", ast.Load()),
                        args=[ast.Constant(kind), ast.Name(p, ast.Load()), *node.args], keywords=[])


_PRELUDE = '''
import math as __sv_math
import _sv_verifier as __sv_v


def __sv_argv(value):
    if "argv" not in __SV_POS:
        __SV_POS["argv"] = 1
        __sv_draw("argv_full", value)
    return value


def __sv_draw(kind, value, *bounds):
    if kind == "float" and len(bounds) == 2:
        a, b = bounds
        value = a if __sv_math.isnan(value) else min(max(value, a), b)
    log = getattr(__sv_v, "LOG", None)
    if log is not None:
        log.append((kind, list(value) if kind == "argv_full" else value))
    return value


def __sv_index(kind, value, seq):
    # the child's own unseeded random.choice(seq): seq[_draw("rindex", n=len(seq))]
    # (see tharkun.sv._child._patch_random) -- same clamp, same log entry, so a
    # solved ``value`` replays to the same element there. An empty ``seq`` clamps to
    # index 0 and the subscript itself raises IndexError -- deliberately NOT a
    # ``raise`` statement here, so violation_sites()'s AST scan of the generated
    # harness (which walks every function, prelude included) never counts this
    # synthetic bookkeeping as a program site the way an explicit ``raise`` would.
    n = len(seq)
    idx = min(max(int(value), 0), max(n - 1, 0))
    log = getattr(__sv_v, "LOG", None)
    if log is not None:
        log.append((kind, idx))
    return seq[idx]


__SV_POS = {}
__SV_DEF = {"int": 0, "bool": False, "float": 0.0, "str": ""}


def __sv_next(kind, stream, *bounds):
    i = __SV_POS.get(id(stream), 0)
    __SV_POS[id(stream)] = i + 1
    return __sv_draw(kind, stream[i] if i < len(stream) else __SV_DEF[kind], *bounds)


def __sv_site(n):
    return n


import operator as __sv_op
__SV_REPR = {int: 1, float: 1.0, bool: True, str: "a", bytes: b"a", list: [1], tuple: (1,),
             type(None): None, dict: {1: 1}}


def __sv_ok_binop(name, a, b):
    """Does ``a <op> b`` accept these operand TYPES?  Decided on representative values
    of the types, never on the operands themselves (no huge ``"a" * n``)."""
    ta, tb = type(a), type(b)
    if ta not in __SV_REPR or tb not in __SV_REPR:
        return True
    try:
        getattr(__sv_op, name)(__SV_REPR[ta], __SV_REPR[tb])
    except TypeError:
        return False
    except Exception:
        return True
    return True
'''


def _imported(tree) -> frozenset:
    names = set()
    for st in ast.walk(tree):
        if isinstance(st, ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in st.names)
        elif isinstance(st, ast.ImportFrom):
            names.update(a.asname or a.name for a in st.names)
    return frozenset(names)


def _extra_checks(records) -> dict:
    """Solve targets from another analysis: every ``may_raise`` record with a residual
    that parses as a Python expression, keyed by its original line."""
    out: dict = {}
    for r in records or ():
        if r.get("status", "may_raise") != "may_raise" or not r.get("residual"):
            continue
        cond = r["residual"]
        scope = [f for f in r.get("scope") or () if _parses(f)]
        if scope:
            cond = " and ".join(f"({f})" for f in scope + [cond])
        if not _parses(cond):
            continue
        out.setdefault(r["line"], []).append((r["exc"], r.get("kind", "region"), cond))
    return out


def _safe(records) -> frozenset:
    """``(line, kind)`` pairs whose every record says ``safe``: no marker is placed there."""
    status: dict = {}
    for r in records or ():
        key = (r.get("line"), r.get("kind"))
        status[key] = status.get(key, True) and r.get("status") == "safe"
    return frozenset(k for k, ok in status.items() if ok)


def _parses(text) -> bool:
    try:
        ast.parse(str(text), mode="eval")
        return True
    except SyntaxError:
        return False


#: the statements that DEFINE (nothing they run reads a draw): the harness leaves them at module level
_HOISTED = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
#: names that read the module's own namespace: a statement moved into a function would read the function's
_NAMESPACE_READERS = frozenset({"globals", "locals", "vars", "dir", "exec", "eval", "__name__", "__file__",
                                "__annotations__"})


@dataclass
class _Plan:
    """Which code of the program becomes ``__sv_harness`` (see :func:`_plan`)."""
    body: list                  # the harness function's statements, in execution order
    rest: list                  # the statements that stay at module level, in order
    moved: list                 # the part of ``body`` that was module-level code: its names are module globals
    keep_global: frozenset      # names a definition reads: a draw assigned to one stays a global, never a parameter
    entry: str                  # what the harness is, for the log: "main block" | "entry main()" | "module level"
    entry_fn: str = ""          # the name of the entry function whose body is in ``body``


def _bound_names(stmts) -> set:
    """Every name ``stmts`` binds or deletes, nested scopes included (over-declaring a name ``global`` is harmless)."""
    out = set()
    for st in stmts:
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                out.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                out.update((a.asname or a.name).split(".")[0] for a in n.names)
    return out


def harness_namespace_refusal(source: str) -> str | None:
    """Original code must not acquire or replace a global injected by the harness."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "the source does not parse"
    prelude = ast.parse(_PRELUDE).body
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    injected = ({HARNESS_FN} | {node.name for node in prelude if isinstance(node, definitions)}
                | _bound_names([node for node in prelude if not isinstance(node, definitions)]))
    names = _bound_names(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    collisions = names & injected
    return f"original source uses generated harness name {sorted(collisions)[0]!r}" if collisions else None


def _own_scope(stmts):
    """The nodes of ``stmts`` that run in THIS function: nested defs, lambdas and classes are other scopes."""
    stack = list(stmts)
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(n))


def _definition_time_loads(st) -> set:
    """Names ``st`` (a def or a class) reads WHILE it is being defined: decorators, defaults, annotations, bases and
    the statements of a class body (a function body runs later, at its calls)."""
    out = set()

    def loads(node):
        out.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))

    def visit(d):
        for x in d.decorator_list:
            loads(x)
        if isinstance(d, ast.ClassDef):
            for x in [*d.bases, *(k.value for k in d.keywords)]:
                loads(x)
            for b in d.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(b)
                else:
                    loads(b)
        else:
            a = d.args
            for x in [*a.defaults, *(k for k in a.kw_defaults if k is not None), d.returns,
                      *(arg.annotation for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
                        if arg is not None)]:
                if x is not None:
                    loads(x)

    visit(st)
    return out


def _draws_in(node, probe: "_Lift") -> bool:
    """``node`` holds a call a harness could lift as a draw."""
    return any(isinstance(n, ast.Call) and probe._nondet_name(n) in LIFTED for n in ast.walk(node))


def _entry(stmts, tree, call_targets, probe: "_Lift"):
    """``(def, call)`` when the program ENDS by calling one function of its own (``main()``) and every condition
    that makes inlining that function's body into the harness exact holds; the reason (a string) when it ends
    with such a call but one condition fails; ``None`` when it does not end by calling a function it defines.

    The conditions: the call is a bare statement with no arguments (its value is dropped); the callee is a plain
    ``def`` of the program, defined once, undecorated, without parameters, not a generator, declaring no ``global``;
    the ``call_target`` fact says the name is a PROVEN ``user_function`` (nothing rebinds it); and the call is the
    only reference to the name in the whole program (a second call, a recursive call or a use as a value would
    each run the body where the harness does not).  The body of the def then IS the rest of the program."""
    if not stmts:
        return None
    last = stmts[-1]
    name = last.value.func.id if (isinstance(last, ast.Expr) and isinstance(last.value, ast.Call)
                                  and isinstance(last.value.func, ast.Name)) else None
    defs = [st for st in stmts[:-1] if isinstance(st, ast.FunctionDef) and st.name == name]
    if not defs:
        # not an entry call.  A function that draws and IS called, somewhere the program does not end with,
        # is the near miss to name: statements follow the call, or its value is used
        called = {n.func.id for st in stmts if not isinstance(st, _HOISTED) for n in ast.walk(st)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for st in stmts:
            if isinstance(st, ast.FunctionDef) and st.name in called and _draws_in(st, probe):
                return (f"the program does not END with the call {st.name}(): statements follow it, "
                        f"or its value is used")
        return None
    call = last.value
    if len(defs) > 1:
        return f"the entry function {name} is defined more than once"
    fn = defs[0]
    if call.args or call.keywords:
        return f"the entry call {name}(...) has arguments"
    a = fn.args
    if a.posonlyargs or a.args or a.kwonlyargs or a.vararg or a.kwarg or fn.decorator_list:
        return f"the entry function {name} takes parameters or is decorated"
    own = list(_own_scope(fn.body))
    if any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await)) for n in own):
        return f"the entry function {name} is a generator"
    if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in own):
        return f"the entry function {name} declares global or nonlocal names"
    from tharkun.solve.call_targets import call_target_index
    if call_targets is None:
        return f"no call_target fact says what {name}() is"
    rec = call_target_index(call_targets).get(("<main>", name))
    if rec is None or rec["target"] != "user_function" or not rec["proven"]:
        why = "no record" if rec is None else rec.get("reason") or rec["target"]
        return f"the call_target fact does not prove {name} a user function ({why})"
    if sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
           and n.func.id == name) > 1:
        return f"the entry function {name} is called more than once"
    if sum(1 for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name) > 1:
        return f"the entry function {name} is referenced besides its one call (recursion or use as a value)"
    return fn, last


def _plan(tree, probe: "_Lift", call_targets):
    """``(_Plan | None, refusal)``: which code of the program becomes the harness function.

    The program is read as the child runs it: ``__name__`` is ``"__main__"``, so the ``if __name__ == "__main__":``
    block is code that runs where it stands.  Then, in this order ([internal ref]):

    * the program ENDS BY CALLING ITS OWN FUNCTION and nothing module-level draws (``def main(): ...`` then
      ``main()``, at module level or under the guard): the harness body is that function's body
      (:func:`_entry` lists what makes this exact), and the rest of the program stays at module level, in order;
    * a statement at MODULE LEVEL draws (``SIZE = 1 + abs(_sv_verifier.nondet_int())``): the draw must be a
      parameter, and a parameter is only read when the harness runs, so every module-level statement that is not
      a definition moves into the harness, in order, after the definitions (which stay).  The names they assign
      are declared ``global``: they are the program's module globals.  Refused, by name, where the move would
      change the program: a definition that READS a moved name while it is being defined (a class body, a
      default, a decorator), a read of the module's own namespace, a star import under a compound statement, an
      entry function whose locals share a name with a moved global;
    * otherwise the ``if __name__ == "__main__":`` block, as before."""
    guard = next((i for i, st in enumerate(tree.body) if _is_main_guard(st)), None)
    refusal = ""
    if guard is None or guard == len(tree.body) - 1:
        prefix = tree.body if guard is None else tree.body[:guard]
        stmts = list(prefix) + ([] if guard is None else tree.body[guard].body)
        entry = _entry(stmts, tree, call_targets, probe)
        fn = entry[0] if isinstance(entry, tuple) else None
        refusal = entry if isinstance(entry, str) else ""
        head = [st for st in (stmts[:-1] if fn is not None else stmts) if st is not fn]
        # a main block with no entry call keeps being lifted in place (below): only its prefix is scanned
        scan = head if fn is not None else prefix
        if any(not isinstance(st, _HOISTED) and _draws_in(st, probe) for st in scan):
            moved = [p for p in (_plain(st) for st in head if not isinstance(st, (_HOISTED, ast.Global)))
                     if p is not None]
            hoisted = [st for st in head if isinstance(st, _HOISTED)]
            refusal = _moved_refusal(moved, hoisted, fn)
            if not refusal:
                keep = frozenset(n.id for st in hoisted for n in ast.walk(st)
                                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
                return _Plan(moved + (fn.body if fn is not None else []), hoisted, moved, keep,
                             f"entry {fn.name}() with module-level draws" if fn is not None
                             else "module level", fn.name if fn is not None else ""), ""
            # refused: the main block (below) is lifted as before, its module-level draws left to the stub
        elif fn is not None:
            return _Plan(fn.body, head, [], frozenset(), f"entry {fn.name}()", fn.name), ""
    if guard is not None:
        main = tree.body[guard]
        return _Plan(main.body, [st for st in tree.body if st is not main], list(main.body), frozenset(),
                     "main block"), refusal
    return None, refusal


def _plain(st):
    """``st`` as a statement of a function: a module-level annotated name cannot be ``global`` there, so ``x: int = v``
    becomes ``x = v`` and a bare ``x: int`` (which binds nothing) goes."""
    if isinstance(st, ast.AnnAssign):
        if st.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[st.target], value=st.value), st)
    return st


def _moved_refusal(moved, hoisted, fn) -> str:
    """The reason moving ``moved`` into the harness would change the program, or ``""``."""
    names = _bound_names(moved)
    for st in hoisted:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bad = _definition_time_loads(st) & names
            if bad:
                return (f"module-level draw: {st.name} reads {sorted(bad)[0]} while it is being defined, "
                        f"and the harness assigns that name later")
    for st in moved:
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and n.id in _NAMESPACE_READERS:
                return f"module-level draw: the program reads the module namespace ({n.id})"
            if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names):
                return "module-level draw: a star import under a compound statement"
    if fn is not None:
        clash = {n.id for n in _own_scope(fn.body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} & names
        if clash:
            return (f"module-level draw: the entry function {fn.name} assigns {sorted(clash)[0]}, "
                    f"a module global the harness also assigns")
    return ""


def _build(source: str, classes, extra_sites=None, call_targets=None):
    """A :class:`Harness`, or the reason there is none (a string)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "the source does not parse"
    aliases, direct, sys_aliases, argv_names, random_aliases = set(), {}, set(), set(), set()
    for st in ast.walk(tree):
        if isinstance(st, ast.Import):
            for a in st.names:
                if a.name == "_sv_verifier":
                    aliases.add(a.asname or a.name)
                elif a.name == "sys":
                    sys_aliases.add(a.asname or a.name)
                elif a.name == "random":
                    random_aliases.add(a.asname or a.name)
        elif isinstance(st, ast.ImportFrom) and st.module == "_sv_verifier":
            for a in st.names:
                direct[a.asname or a.name] = a.name
        elif isinstance(st, ast.ImportFrom) and st.module == "sys":
            argv_names.update(a.asname or a.name for a in st.names if a.name == "argv")
    # [internal ref]: a program this scan cannot show is unseeded (a qualified ``.seed(`` call
    # on the same alias, or a ``from random import ...`` that could bring in ``seed``
    # under another name) keeps its ``random.choice`` calls untouched -- see
    # ``_Lift.random_choice_ok``.
    random_choice_ok = bool(random_aliases) and not any(
        (isinstance(n, ast.ImportFrom) and n.module == "random")
        or (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "seed"
            and isinstance(n.func.value, ast.Name) and n.func.value.id in random_aliases)
        for n in ast.walk(tree))
    probe = _Lift(aliases, direct)
    plan, refusal = _plan(tree, probe, call_targets)
    if plan is None:
        return refusal or "no main block and no module-level draw"
    named, taken = {}, set()
    for st in plan.body:
        if (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                and isinstance(st.value, ast.Call) and probe._nondet_name(st.value) in LIFTED
                and not st.value.keywords):
            n = st.targets[0].id
            if n not in taken and not n.startswith("__") and n not in plan.keep_global:
                named[id(st.value)] = n
            taken.add(n)
    lift = _Lift(aliases, direct, named, sys_aliases, argv_names, random_aliases, random_choice_ok)
    body = [lift.visit(st) for st in plan.body]
    if not lift.params:
        if not _draws_in(tree, probe):
            return "no liftable draw"
        if not refusal and plan.entry.startswith("entry ") and any(
                isinstance(st, ast.FunctionDef) and _draws_in(st, probe) for st in plan.rest):
            return "the draws are inside functions the entry function calls, which the harness leaves to the stub"
        return refusal or "no liftable draw"
    assigned = sorted(_bound_names(plan.moved) - set(lift.params))
    # argv is an initial input, ahead of every per-call draw. Snapshot before
    # target mutation, even when its first source access follows nondet calls.
    boot = ast.parse(f"__sv_argv({lift.argv_param})").body if lift.argv_param else []
    fn_body: list[ast.stmt] = (([ast.Global(names=assigned)] if assigned else [])
                               + ast.parse("__SV_POS.clear()").body + boot + body)
    fn = ast.FunctionDef(
        name=HARNESS_FN,
        # each draw parameter is annotated with its kind (``list[int]`` for a stream), which
        # is what the guard solvers read to pick an integer / float / str domain
        args=ast.arguments(posonlyargs=[], args=[ast.arg(p, ast.parse(_annotation(k), mode="eval").body)
                                                 for p, k in zip(lift.params, lift.kinds)], vararg=None,
                           kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=fn_body, decorator_list=[], returns=None, type_params=[])
    marks: list[dict] = []
    instrumented = _Instrument(tuple(classes), marks, _imported(tree), _extra_checks(extra_sites),
                               typing_scope(tree), _safe(extra_sites)).stmts(plan.rest + [fn])
    module = ast.Module(body=ast.parse(_PRELUDE).body + instrumented, type_ignores=[])
    ast.fix_missing_locations(module)
    text = ast.unparse(module) + "\n"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(text, "<sv_harness>", "exec")
    except (SyntaxError, ValueError) as e:
        return f"the harness does not compile ({type(e).__name__}: {e})"[:200]
    h = Harness(text, lift.params, lift.kinds, lift.unlifted, entry=plan.entry, entry_fn=plan.entry_fn)
    h.sites = violation_sites(text, classes) + _marker_sites(text, marks)
    h.feeds = _feeds(body, {st.name: st for st in plan.rest if isinstance(st, ast.FunctionDef)})
    return h


def build_harness(source: str, classes, extra_sites=None, call_targets=None) -> Harness | None:
    """The draw-parameterised harness of ``source``, or ``None`` when the program has
    no entry code with a draw the harness can lift (:func:`harness_refusal` says why).

    ``extra_sites`` are site records from another analysis (the abstract
    interpreter, :func:`tharkun.solve.absint.prove`), on ORIGINAL lines: a ``may_raise``
    record with a ``residual`` becomes one more marker (a solve target); an implicit
    site whose every record is ``safe`` gets no marker (nothing to solve).  ``call_targets``
    are the program's ``call_target`` records (``solve/call_targets.py``): without them an entry function
    (``def main(): ...`` then ``main()``) is not inlined, since nothing says what ``main`` is."""
    built = _build(source, classes, extra_sites, call_targets)
    return built if isinstance(built, Harness) else None


def harness_refusal(source: str, classes, extra_sites=None, call_targets=None) -> str:
    """Why :func:`build_harness` returned ``None``: the NAMED condition that failed, for the log."""
    built = _build(source, classes, extra_sites, call_targets)
    return built if isinstance(built, str) else ""


def _annotation(kind: str) -> str:
    if kind == "argv_full":
        return "list[str]"
    if kind.startswith("list:"):
        return f"list[{kind[5:]}]"
    return "int" if kind == "rindex" else kind


def _draw_param(node) -> str | None:
    """``dK`` when ``node`` is ``__sv_draw(kind, dK, ...)``."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__sv_draw"
            and len(node.args) >= 2 and isinstance(node.args[1], ast.Name)):
        return node.args[1].id
    return None


def _feeds(body: list, functions: dict) -> dict:
    """Calls in the main block's straight-line statements whose every argument is a
    draw: ``{fn: {param: "dK"}}`` (the first such call of each function wins)."""
    bound: dict[str, str] = {}
    out: dict = {}
    for st in body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            d = _draw_param(st.value)
            if d:
                bound[st.targets[0].id] = d
            else:
                bound.pop(st.targets[0].id, None)
        for node in ast.walk(st):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in functions and not node.keywords):
                continue
            fd = functions[node.func.id]
            params = [a.arg for a in fd.args.args]
            if node.func.id in out or len(params) != len(node.args):
                continue
            m = {}
            for p, a in zip(params, node.args):
                d = _draw_param(a) or (bound.get(a.id) if isinstance(a, ast.Name) else None)
                if d is None:
                    break
                m[p] = d
            else:
                out[node.func.id] = m
    return out



# ---------------------------------------------------------------------------
# implicit sites: a subscript that can miss, a divisor that can be zero
# ---------------------------------------------------------------------------

_PURE = (ast.Name, ast.Constant, ast.Attribute, ast.Subscript, ast.BinOp, ast.UnaryOp, ast.BoolOp,
         ast.Compare, ast.IfExp, ast.Tuple)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp,
           ast.SetComp, ast.DictComp, ast.GeneratorExp)
_CATCH_ALL = {"Exception", "BaseException"}


#: builtins without side effects a site operand may call (``table[len(table)]``)
_PURE_CALLS = {"len", "abs", "min", "max"}


def _pure(node) -> bool:
    """Evaluating ``node`` twice has no effect a second evaluation could see: no call
    other than :data:`_PURE_CALLS`, no walrus, only names, constants, attributes,
    subscripts and operators."""
    return all(isinstance(n, _PURE + (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop,
                                      ast.boolop, ast.Slice))
               or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id in _PURE_CALLS and not n.keywords)
               for n in ast.walk(node))


def _unconditional(expr):
    """``(sub-expression, conditions)`` for every sub-expression of ``expr`` that is
    evaluated when ``expr`` is, with the conditions under which it is: the arms of an
    ``IfExp`` carry its test (or its negation); the operands after the first of a
    short-circuit ``BoolOp`` and the inside of a nested scope are not visited."""
    stack = [(expr, ())]
    while stack:
        n, conds = stack.pop()
        yield n, conds
        if isinstance(n, _SCOPES):
            continue
        if isinstance(n, ast.IfExp):
            stack.append((n.test, conds))
            if _pure(n.test):
                t = f"({ast.unparse(n.test)})"
                stack.append((n.body, conds + (t,)))
                stack.append((n.orelse, conds + (f"not {t}",)))
        elif isinstance(n, ast.BoolOp):
            stack.append((n.values[0], conds))
        else:
            stack.extend((c, conds) for c in ast.iter_child_nodes(n))


def _header_exprs(st):
    """The expressions a statement itself evaluates before any nested block runs."""
    if isinstance(st, (ast.If, ast.While)):
        return [st.test]
    if isinstance(st, (ast.For, ast.AsyncFor)):
        return [st.iter]
    if isinstance(st, (ast.With, ast.AsyncWith)):
        return [i.context_expr for i in st.items]
    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Try)):
        return []
    return [st]


_OPNAME = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv",
           ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow"}
_AST_OPNAME = {ast.Add: "Add", ast.Sub: "Sub", ast.Mult: "Mult", ast.Div: "Div",
               ast.FloorDiv: "FloorDiv", ast.Mod: "Mod", ast.Pow: "Pow"}


_VALUE_KIND = {bool: "num", int: "num", float: "num", str: "str", bytes: "bytes", type(None): "none"}
_CALL_KIND = {"int": "num", "float": "num", "len": "num", "abs": "num", "sum": "num", "str": "str",
              "list": "list", "dict": "dict", "tuple": "tuple", "nondet_int": "num",
              "nondet_bool": "num", "nondet_float": "num", "nondet_float_range": "num",
              "nondet_str": "str", "nondet_list": "list", "nondet_dict": "dict"}


def _value_kind(node) -> str | None:
    if isinstance(node, ast.Constant):
        return _VALUE_KIND.get(type(node.value))
    if isinstance(node, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.Call):
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        return _CALL_KIND.get(name)
    return None


def typing_scope(tree) -> tuple[frozenset, frozenset]:
    """``(poly, nullable)``: names assigned values of two or more Python types somewhere in
    the program (``v = 0`` ... ``v = "tag"``), and names that may hold ``None`` (assigned
    ``None`` or defaulting to it).  Typing sites are only placed on operations over these
    names -- an ``a + b`` over names that only ever hold numbers is no TypeError site."""
    kinds: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            k = _value_kind(n.value)
            if k:
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        kinds.setdefault(t.id, set()).add(k)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            k = _value_kind(n.value)
            if k:
                kinds.setdefault(n.target.id, set()).add(k)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = n.args.posonlyargs + n.args.args
            for a, d in zip(args[len(args) - len(n.args.defaults):], n.args.defaults):
                k = _value_kind(d)
                if k:
                    kinds.setdefault(a.arg, set()).add(k)
            for a, d in zip(n.args.kwonlyargs, n.args.kw_defaults):
                k = _value_kind(d) if d is not None else None
                if k:
                    kinds.setdefault(a.arg, set()).add(k)
    poly = frozenset(v for v, ks in kinds.items() if len(ks) > 1)
    nullable = frozenset(v for v, ks in kinds.items() if "none" in ks)
    return poly, nullable


def implicit_checks(st, classes, modules=frozenset(), typing=(frozenset(), frozenset())):
    """``(exc, kind, condition)`` for every implicit site statement ``st`` evaluates:

    * ``IndexError``: ``c[k]`` with ``c`` a sequence and ``k`` an int out of range;
    * ``KeyError``: ``c[k]`` with ``c`` a dict lacking ``k``;
    * ``ZeroDivisionError``: ``a / b``, ``a // b``, ``a % b`` with ``b == 0``.

    Only operands that are pure (:func:`_pure`) are checked, so evaluating the condition
    before the statement sees exactly the values the statement will use.  Each
    condition is ordinary Python over the program's names, which is what the guard
    solvers read."""
    out = []
    for expr in _header_exprs(st):
        for n, conds in _unconditional(expr):
            pre = "".join(f"{c} and " for c in conds)
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load) \
                    and not isinstance(n.slice, ast.Slice) and _pure(n.value) and _pure(n.slice):
                c, k = f"({ast.unparse(n.value)})", f"({ast.unparse(n.slice)})"
                if "IndexError" in classes:
                    out.append(("IndexError", "index",
                                f"{pre}isinstance({c}, (list, tuple, str)) and isinstance({k}, int) "
                                f"and not -len({c}) <= {k} < len({c})"))
                if "KeyError" in classes:
                    out.append(("KeyError", "key", f"{pre}isinstance({c}, dict) and {k} not in {c}"))
            elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) \
                    and "ZeroDivisionError" in classes and _pure(n.right):
                if isinstance(n.op, ast.Mod) and not (_pure(n.left)):
                    continue
                r = f"({ast.unparse(n.right)})"
                cond = f"{r} == 0"
                if isinstance(n.op, ast.Mod):
                    cond = f"not isinstance(({ast.unparse(n.left)}), str) and {cond}"
                out.append(("ZeroDivisionError", "divide", pre + cond))
            if isinstance(n, ast.BinOp) and "TypeError" in classes and type(n.op) in _OPNAME \
                    and _pure(n.left) and _pure(n.right) \
                    and any(isinstance(x, ast.Name) and x.id in typing[0]
                            for side in (n.left, n.right) for x in ast.walk(side)):
                # a typing site: an operation whose operands' types may not fit
                out.append(("TypeError", "binop",
                            f"{pre}not __sv_ok_binop({_OPNAME[type(n.op)]!r}, ({ast.unparse(n.left)}), "
                            f"({ast.unparse(n.right)}))"))
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and "AttributeError" in classes \
                    and isinstance(n.value, ast.Name) and n.value.id not in modules \
                    and n.value.id in typing[1]:
                out.append(("AttributeError", "attr", f"{pre}({n.value.id}) is None"))
    return out


def _caught(handlers, exc: str) -> bool:
    for h in handlers:
        if h.type is None:
            return True
        names = [t for t in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type])]
        for t in names:
            name = t.id if isinstance(t, ast.Name) else getattr(t, "attr", None)
            if name in _CATCH_ALL or _matches(exc, (name,)) == name:
                return True
    return False


class _Instrument:
    """Insert ``if <condition>: __sv_site(n)`` before every statement that evaluates an
    implicit site, so a site is a LINE ``drive_to_line`` can target.  A site inside a
    ``try`` that catches its exception is not a violation and gets no marker."""

    def __init__(self, classes, marks: list, modules=frozenset(), extra=None,
                 typing=(frozenset(), frozenset()), safe=frozenset()):
        self.classes = classes
        self.marks = marks
        self.modules = modules
        self.typing = typing
        #: ``(line, kind)`` of implicit sites another analysis proved never raise
        self.safe = safe
        #: may-raise regions from another analysis (the abstract interpreter's records) keyed by the
        #: ORIGINAL line of the statement they sit in: ``[(exc, kind, residual)]``
        self.extra = extra or {}

    def stmts(self, body, handlers=(), fn="<module>"):
        out = []
        for st in body:
            line = getattr(st, "lineno", None)
            checks = [c for c in implicit_checks(st, self.classes, self.modules, self.typing)
                      if (line, c[1]) not in self.safe]
            own = {c[1] for c in checks}
            for rec in self.extra.get(line, ()):
                # a region the implicit checks already mark on this statement is not doubled
                if rec[1] not in own and not isinstance(
                        st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.For,
                             ast.While, ast.With, ast.Try)):
                    checks.append(rec)
            for exc, kind, cond in checks:
                if any(_caught(hs, exc) for hs in handlers):
                    continue
                n = len(self.marks)
                self.marks.append({"exc": exc, "kind": kind, "function": fn, "condition": cond})
                check = ast.parse(f"if {cond}:\n    __sv_site({n})").body[0]
                out.append(check)
            out.append(self.stmt(st, handlers, fn))
        return out

    def stmt(self, st, handlers, fn):
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            st.body = self.stmts(st.body, (), st.name)
        elif isinstance(st, ast.ClassDef):
            st.body = self.stmts(st.body, (), fn)
        elif isinstance(st, ast.Try):
            st.body = self.stmts(st.body, handlers + (st.handlers,), fn)
            for h in st.handlers:
                h.body = self.stmts(h.body, handlers, fn)
            st.orelse = self.stmts(st.orelse, handlers, fn)
            st.finalbody = self.stmts(st.finalbody, handlers, fn)
        else:
            for attr in ("body", "orelse"):
                inner = getattr(st, attr, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                    setattr(st, attr, self.stmts(inner, handlers, fn))
        return st


def _marker_sites(text: str, marks: list) -> list[dict]:
    """The harness line of each ``__sv_site(n)`` marker, as site records."""
    out = []
    for node in ast.walk(ast.parse(text)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__sv_site"
                and node.args and isinstance(node.args[0], ast.Constant)):
            m = marks[node.args[0].value]
            out.append({"line": node.lineno, "col": node.col_offset, "exc": m["exc"], "kind": m["kind"],
                        "function": m["function"], "guards": [], "status": "may_raise",
                        "residual": m["condition"], "condition": m["condition"]})
    return sorted(out, key=lambda r: r["line"])


def implicit_sites(source: str, classes) -> list[dict]:
    """The implicit violation sites of ``source`` on ORIGINAL lines (for the proofs, [internal ref]):
    records ``{"line", "col", "exc", "kind": "index"|"key"|"divide", "function", "guards",
    "status": "may_raise", "residual", "condition"}`` (the abstract interpreter's record shape; ``residual`` is the
    predicate under which the site raises, ``condition`` its older name)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []

    def walk(body, fn, handlers):
        for st in body:
            for exc, kind, cond in implicit_checks(st, tuple(classes), modules, typing):
                if not any(_caught(hs, exc) for hs in handlers):
                    out.append({"line": st.lineno, "col": st.col_offset, "exc": exc, "kind": kind,
                                "function": fn, "guards": [], "status": "may_raise",
                                "residual": cond, "condition": cond})
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(st.body, st.name, ())
            elif isinstance(st, ast.Try):
                walk(st.body, fn, handlers + (st.handlers,))
                for h in st.handlers:
                    walk(h.body, fn, handlers)
                walk(st.orelse, fn, handlers)
                walk(st.finalbody, fn, handlers)
            else:
                name = "<main>" if _is_main_guard(st) else fn
                for attr in ("body", "orelse"):
                    inner = getattr(st, attr, None)
                    if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                        walk(inner, name, handlers)
    modules = _imported(tree)
    typing = typing_scope(tree)
    walk(tree.body, "<module>", ())
    return out
