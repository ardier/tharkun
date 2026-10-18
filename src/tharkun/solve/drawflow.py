"""``plan_draws`` -- where each nondet draw of an SV-COMP Python program can reach ([internal ref]).

SOLVING, by ``docs/LAYERS.md``'s rule: the input is a program's source, the
output is a claim about how its ``_sv_verifier.nondet_int()`` draws can
influence the execution, computed from the AST alone, with nothing executed.

The claim this module makes is narrow on purpose.  A draw is classified as

* ``trip`` with offset ``c`` -- its value reaches the program ONLY as the trip
  count ``c + abs(d)`` of one or more ``for _ in range(...)`` loops whose target
  variable is dead (never read).  No other code reads the value: not the loop
  body, not the code after the loop, not a return value, not a container.
* ``quotient`` -- the draw appears only as ``abs(d) % M`` (the modulus ``M``
  mentions no draw), so the program depends on ``d`` only through the residue:
  ``abs(d) % M`` for ``d`` in ``range(abs(M))`` takes every value it can take.

Everything else refuses.  The caller (``tharkun.sv.prove``, the stationarity
proof) turns a ``trip`` claim into "trip counts ``m`` and ``m + 1`` stand for
every trip count" only after it has OBSERVED the loop's state repeat; this
module never claims that part.

The analysis refuses (``DrawPlan.ok`` is False, ``reason`` says why) on
anything it cannot see through, so that a refusal is always the safe answer:

* introspection and dynamic code: ``eval``, ``exec``, ``compile``, ``globals``,
  ``locals``, ``vars``, ``__import__``, ``__builtins__``, frame / code / closure
  attributes (``f_locals``, ``_getframe``, ``__globals__``, ...);
* sources of state or order a replayed run does not control: reads through ``open``,
  ``input``, ``id``, ``hash``, a ``__del__`` finalizer (it runs when the cycle
  collector decides);
* output handles that escape, expose their state, or can be read back;
* an import outside :data:`MODULE_WHITELIST` (a module whose hidden state the
  proof cannot snapshot), a relative import;
* a draw kind other than ``nondet_int`` / ``nondet_bool`` (``bool`` is finite:
  the caller enumerates it), an aliased ``_sv_verifier`` name;
* a draw value flowing anywhere but ``abs(...)`` -> ``+ k`` -> ``range(...)``
  of a ``for`` loop, a plain local/global variable read only in such places,
  or a positional/keyword argument of a module-level function (called by name,
  never rebound, undecorated, no ``*args``/``**kwargs`` involved), or of the ``__init__`` of a class
  the ``call_target`` fact proves is the callee (``solve/call_targets.py``, [internal ref]; a plain class
  with its own undecorated ``__init__``: see ``_Planner._ctor``).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from tharkun.solve.call_targets import call_target_index

#: imported modules whose run-time state the stationarity snapshot covers (``random``:
#: ``getstate()``) or that carry no program-visible state (``re``: only a compile cache).
MODULE_WHITELIST = frozenset({"_sv_verifier", "math", "random", "re", "io", "array", "string"})

#: names whose use makes a run depend on something the proof cannot replay or snapshot
#: ([internal ref] follow-up, MG's PR 108 review: ``setattr``/``delattr`` added -- either can rebind
#: ANY attribute, including a method this pass trusts a resolved definition for, without
#: ever showing up as an ordinary ``ast.Attribute`` store; see ``_Planner._resolve_str_callee``)
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "globals", "locals", "vars", "__import__", "__builtins__",
    "open", "input", "id", "hash", "breakpoint", "memoryview", "setattr", "delattr",
})

#: attributes that reach frames, code, closures or the class graph behind the program's back
#: ([internal ref] follow-up, MG's PR 108 review: ``__dict__`` added -- ``obj.__dict__['name'] = ...``
#: rebinds an attribute exactly like ``setattr`` does, same reasoning; ``__setattr__``/
#: ``__delattr__`` added too -- ``object.__setattr__(c, 'run', bad)``/``c.__setattr__(...)``
#: rebind exactly like the plain ``setattr`` builtin, just spelled as an attribute access
#: instead of a bare name, so ``FORBIDDEN_NAMES``'s own ``setattr``/``delattr`` entries do
#: not see it; ``__class__`` added too -- ``c.__class__ = D`` swaps ``c``'s ENTIRE class
#: after ``_recv_class`` already resolved it from ``c``'s own construction, so a later
#: ``c.run(...)`` can end up calling a method from a class this pass never looked at)
FORBIDDEN_ATTRS = frozenset({
    "_getframe", "f_back", "f_locals", "f_globals", "f_builtins", "f_code", "gi_frame",
    "cr_frame", "ag_frame", "tb_frame", "__globals__", "__code__", "__closure__",
    "__subclasses__", "__builtins__", "cell_contents", "__dict__", "__setattr__",
    "__delattr__", "__class__",
})

#: builtin classes an ``isinstance`` classinfo may name for a trip value (none of them has a
#: metaclass ``__instancecheck__``, so the check runs no user code)
_BUILTIN_CLASSES = frozenset({"int", "float", "str", "bool", "list", "tuple", "dict", "set", "frozenset",
                              "bytes", "bytearray", "complex", "object"})

_FINITE_DRAWS = frozenset({"nondet_bool"})
_INT_DRAWS = frozenset({"nondet_int"})


class Refuse(Exception):
    """The analysis cannot make its claim; the message says why."""


@dataclass(frozen=True)
class TripLoop:
    """A ``for`` loop whose trip count is ``offset + abs(draw)`` for some draw(s)."""
    line: int
    col: int
    dead: tuple[str, ...]      # the loop's target names: never read
    offset: int                # c in ``c + abs(d)``


@dataclass(frozen=True)
class DrawSite:
    """One ``nondet_*()`` call, keyed by the call expression's (line, col)."""
    line: int
    col: int
    kind: str                  # "trip" | "quotient" | "finite"
    offset: int = 0            # trip only
    loops: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class DrawPlan:
    ok: bool
    reason: str = ""
    draws: tuple[DrawSite, ...] = ()
    loops: tuple[TripLoop, ...] = ()
    notes: tuple[str, ...] = field(default=())
    output_files: bool = False


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    out: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
           ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


class _Program:
    def __init__(self, tree: ast.Module):
        self.tree = tree
        self.parent = _parents(tree)
        self.sv_modules: set[str] = set()
        self.sv_funcs: dict[str, str] = {}
        self.funcs: dict[str, ast.FunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.modules: set[str] = set()
        #: every attribute name EVER stored/deleted via a plain ``x.NAME = ...``/``del
        #: x.NAME`` anywhere in the whole program, whatever ``x`` is ([internal ref] follow-up, MG's
        #: PR 108 review) -- `_Planner._resolve_str_callee` refuses to trust a resolved
        #: method whose OWN name is in this set: `c.run = bad`/`C.run = bad2`/
        #: `self.run = bad` (in `__init__` or anywhere else) can all replace the callable a
        #: later `c.run(...)` actually invokes with one this pass never looked at, whatever
        #: `run`'s class-body definition says.
        self.attr_stores: set[str] = set()
        #: how many times each name is bound anywhere in the program (``screen`` fills it);
        #: a builtin name bound even once is never trusted to still be the builtin
        self.bound: dict[str, int] = {}

    # -- scopes ------------------------------------------------------------
    def scope_of(self, node: ast.AST) -> ast.AST:
        p = self.parent.get(node)
        while p is not None and not isinstance(p, (ast.Module,) + _SCOPES):
            p = self.parent.get(p)
        return p if p is not None else self.tree

    def function_of(self, node: ast.AST) -> ast.AST:
        """The innermost def/lambda/module (comprehensions and classes are looked through)."""
        p = self.scope_of(node)
        while not isinstance(p, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            p = self.scope_of(p)
        return p

    # -- the whole-program checks -----------------------------------------
    def screen(self) -> None:
        from .output_effects import output_only
        try:
            self.output_files = output_only(self.tree)
        except ValueError as error:
            raise Refuse(str(error)) from error
        shadows: dict[str, set[int]] = {}
        for node in ast.walk(self.tree):
            for name in _bound_names(node):
                if name in FORBIDDEN_NAMES and not isinstance(self.scope_of(node), ast.ClassDef):
                    shadows.setdefault(name, set()).add(id(self.function_of(node)))
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES \
                    and not self.shadowed(node, shadows.get(node.id, set())) \
                    and not (node.id == "open" and self.output_files):
                raise Refuse(f"line {node.lineno}: uses {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                raise Refuse(f"line {node.lineno}: uses .{node.attr}")
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                self.attr_stores.add(node.attr)
            if isinstance(node, ast.Import):
                for a in node.names:
                    top = a.name.split(".")[0]
                    if top not in MODULE_WHITELIST or "." in a.name:
                        raise Refuse(f"line {node.lineno}: imports {a.name}")
                    if a.name == "_sv_verifier":
                        self.sv_modules.add(a.asname or a.name)
                    else:
                        self.modules.add(a.asname or a.name)
            if isinstance(node, ast.ImportFrom):
                if node.level or (node.module or "").split(".")[0] not in MODULE_WHITELIST:
                    raise Refuse(f"line {node.lineno}: imports from {node.module!r}")
                for a in node.names:
                    if a.name == "*" or a.name.startswith("_"):
                        raise Refuse(f"line {node.lineno}: from {node.module} import {a.name}")
                    if node.module == "_sv_verifier":
                        self.sv_funcs[a.asname or a.name] = a.name
            if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
                raise Refuse(f"line {node.lineno}: async code")
            if isinstance(node, ast.FunctionDef) and node.name == "__del__":
                # a finalizer runs when the collector decides, on state no snapshot holds
                raise Refuse(f"line {node.lineno}: a __del__ finalizer")
        for stmt in self.tree.body:
            if isinstance(stmt, ast.FunctionDef) and not stmt.decorator_list:
                self.funcs.setdefault(stmt.name, stmt)
            # [internal ref] follow-up (interprocedural str-draw tracking, `_Planner._recv_class`):
            # a class this program defines, TOP-LEVEL only (same restriction `funcs` above
            # already applies to functions) -- eligibility (no inheritance, no metaclass, no
            # attribute-redirecting dunder) is checked at USE time, not here, same as a
            # function's arity is checked where it is called, not where it is defined.
            elif isinstance(stmt, ast.ClassDef):
                self.classes.setdefault(stmt.name, stmt)
        # a function/class counts as resolved only if its name is bound exactly once, by
        # that def
        bound: dict[str, int] = {}
        for node in ast.walk(self.tree):
            for name in _bound_names(node):
                bound[name] = bound.get(name, 0) + 1
        self.bound = bound
        self.funcs = {n: f for n, f in self.funcs.items() if bound.get(n, 0) == 1}
        self.classes = {n: c for n, c in self.classes.items() if bound.get(n, 0) == 1}
        for name in ("abs", "range"):
            if bound.get(name):
                raise Refuse(f"the program rebinds {name}")
        for name in set(self.sv_modules) | set(self.sv_funcs):
            if bound.get(name, 0) != 1:
                raise Refuse(f"{name} is rebound")
        # A library module's own state is not in the stationarity snapshot, so the program
        # may only READ its public attributes: no module as a value, no attribute store, no
        # private attribute (``re._cache``).
        for name in self.modules:
            if bound.get(name, 0) != 1:
                raise Refuse(f"module name {name} is rebound")
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and node.id in self.modules:
                attr = self.parent.get(node)
                if not (isinstance(attr, ast.Attribute) and attr.value is node
                        and isinstance(attr.ctx, ast.Load) and not attr.attr.startswith("_")):
                    raise Refuse(f"line {node.lineno}: module {node.id} used other than to read "
                                 f"a public attribute")

    def shadowed(self, node: ast.AST, scopes: set[int]) -> bool:
        """True iff a read of a name at ``node`` resolves to a binding in one of ``scopes``
        (a function enclosing ``node``, or the module) rather than to the builtin."""
        f = self.function_of(node)
        while True:
            if id(f) in scopes:
                return True
            if isinstance(f, ast.Module):
                return False
            f = self.function_of(f)

    def draw_calls(self) -> list[tuple[ast.Call, str]]:
        out = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and node.id in self.sv_funcs:
                call = self.parent.get(node)
                if not (isinstance(call, ast.Call) and call.func is node):
                    raise Refuse(f"line {node.lineno}: {node.id} used as a value")
                out.append((call, self.sv_funcs[node.id]))
            elif isinstance(node, ast.Name) and node.id in self.sv_modules:
                attr = self.parent.get(node)
                call = self.parent.get(attr)
                if not (isinstance(attr, ast.Attribute) and isinstance(call, ast.Call)
                        and call.func is attr):
                    raise Refuse(f"line {node.lineno}: {node.id} used as a value")
                out.append((call, attr.attr))
        return out


def _bound_names(node: ast.AST) -> list[str]:
    """Names (re)bound by ``node`` itself, in whatever scope."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return [node.id]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return list(node.names)
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.ExceptHandler) and node.name:
        return [node.name]
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        return [node.name]
    return []


def _is_call_to(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == name and not node.keywords)


def _str_truthy_or_open_path(use: ast.Name, prog: "_Program") -> bool:
    """[internal ref] follow-up: is this READ of a str-draw variable a truthiness position
    (an ``If``/``While`` test, ``not x``, or the sole argument of ``bool(...)``) or
    the first positional argument of a call to ``open(...)``? Only these two uses
    are covered by plan_draws' "str_truthy" classification -- anything else (an
    f-string, a comparison, a second argument, a method call, ...) is NOT
    recognised here and the caller refuses the whole plan instead of guessing."""
    par = prog.parent.get(use)
    if isinstance(par, (ast.If, ast.While)) and par.test is use:
        return True
    if isinstance(par, ast.UnaryOp) and isinstance(par.op, ast.Not) and par.operand is use:
        return True
    if _is_call_to(par, "bool") and len(par.args) == 1 and par.args[0] is use:
        return True
    if (isinstance(par, ast.Call) and isinstance(par.func, ast.Name) and par.func.id == "open"
            and par.args and par.args[0] is use):
        return True
    return False


#: [internal ref] follow-up: bound recursion for interprocedural str-draw tracking -- a pathological
#: call chain refuses (never silently recurses forever); a real SV-bench task's own call
#: depth for a single draw's path (e.g. bm_raytrace_nondet's args_filename ->
#: bench_raytrace's filename -> canvas.write_ppm's own filename, 2 hops) is far below this.
_MAX_STR_DEPTH = 6


class _Planner:
    def __init__(self, prog: _Program, call_targets=None):
        self.p = prog
        #: ``(scope, callee) -> call_target record`` (``solve/call_targets.py::call_target_index``); empty when no
        #: fact was handed in, and then only a call of a module-level function resolves, as before
        self.targets = call_target_index(call_targets or ())
        self.loops: dict[tuple[int, int], TripLoop] = {}
        self.seen: set[tuple[int, str]] = set()
        self.str_seen: set[tuple[int, str]] = set()

    # -- a trip value: an expression whose value is c + abs(draw) --------------
    def trip_value(self, expr: ast.AST, c: int) -> set[tuple[int, int]]:
        par = self.p.parent[expr]
        if _is_call_to(par, "range") and len(par.args) == 1 and par.args[0] is expr:
            return self.range_object(par, c)
        if isinstance(par, ast.Assign) and par.value is expr:
            if len(par.targets) != 1 or not isinstance(par.targets[0], ast.Name):
                raise Refuse(f"line {par.lineno}: trip value assigned to a pattern")
            return self.trip_variable(par.targets[0].id, self.p.function_of(par), c, par)
        if isinstance(par, ast.Call) and expr in par.args or (
                isinstance(par, ast.keyword)):
            return self.trip_argument(expr, c)
        raise Refuse(f"line {getattr(par, 'lineno', '?')}: a trip value flows into "
                     f"{ast.unparse(par)[:60]!r}")

    def trip_argument(self, expr: ast.AST, c: int) -> set[tuple[int, int]]:
        par = self.p.parent[expr]
        kw = None
        if isinstance(par, ast.keyword):
            kw, par = par, self.p.parent[par]
        if kw is None and self._class_only_call(par, expr):
            return set()
        if not isinstance(par, ast.Call):
            raise Refuse(f"line {getattr(par, 'lineno', '?')}: a trip value is passed to an "
                         f"unresolved callee")
        if isinstance(par.func, ast.Name) and par.func.id in self.p.funcs:
            fn, bound = self.p.funcs[par.func.id], False
        else:
            fn, bound = self._resolved_method_or_ctor(par), True
        if any(isinstance(a, ast.Starred) for a in par.args) or any(
                k.arg is None for k in par.keywords):
            raise Refuse(f"line {par.lineno}: *args/**kwargs at a call with a trip value")
        a = fn.args
        if a.vararg or a.kwarg:
            raise Refuse(f"{fn.name} takes *args/**kwargs")
        positional = [x.arg for x in a.posonlyargs + a.args]
        if kw is not None:
            if kw.arg not in positional + [x.arg for x in a.kwonlyargs] or (bound and kw.arg == positional[0]):
                raise Refuse(f"line {par.lineno}: unknown keyword {kw.arg}")
            param = kw.arg
        else:
            i = par.args.index(expr) + (1 if bound else 0)       # a bound call's first parameter is ``self``
            if i >= len(positional):
                raise Refuse(f"line {par.lineno}: too many arguments")
            param = positional[i]
        return self.trip_variable(param, fn, c, None)

    def _resolved_method_or_ctor(self, call: ast.Call) -> ast.FunctionDef:
        """The ``def`` a call that is not a plain call of a module-level function runs, resolved through the
        ``call_target`` fact: a constructor (:meth:`_ctor`) or a method on a typed receiver (:meth:`_method`);
        anything else ``Refuse`` -- naming a builtin: its result varies with the value (``divmod(n, K)``, ``len``,
        ``ord``: the induction stops tracking a value the moment this module does, see :meth:`_class_only_call`)."""
        callee = ast.unparse(call.func)
        rec = self.targets.get((self._scope_of(call), callee))
        if rec is not None and rec["target"] == "builtin":
            raise Refuse(f"line {call.lineno}: a trip value is passed to the builtin {callee}, whose result varies "
                         f"with the value (only type() and isinstance() against builtin classes are class-only)")
        if rec is not None and rec["target"] == "user_method":
            return self._method(call, rec)
        return self._ctor(call)

    def _method(self, call: ast.Call, rec: dict) -> ast.FunctionDef:
        """The ``def`` of ``recv.m(...)`` where the fact types the receiver variable (``rec["receiver"]``: its one
        binding is ``recv = C(...)``), or ``Refuse``.

        The fact types the NAME; this checks what a name cannot say.  ``C`` must be a top-level class of THIS program
        bound once and :meth:`_plain_class` (no metaclass, decorator, base, ``__new__``, ``__getattr__`` ...: so
        ``C(...)`` builds a ``C`` whose attributes no hook redirects); ``m`` must be bound exactly once in the class
        body by an undecorated ``def`` (a ``staticmethod`` or ``classmethod`` passes no ``self``); and no attribute
        store anywhere in the program may bind ``m`` (``recv.m = f``, ``C.m = f``: it would replace what the call
        runs) -- the same rule :meth:`_resolve_str_callee` keeps."""
        where = f"line {call.lineno}: a trip value is passed to {ast.unparse(call.func)}"
        recv = rec.get("receiver")
        if not rec["proven"] or not isinstance(recv, dict):
            raise Refuse(f"{where}, whose call_target is {rec['target']} (proven={rec['proven']}: {rec['reason']})")
        if not (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                and call.func.value.id == recv["name"]):
            raise Refuse(f"{where}: the receiver is not the variable the fact types")
        cls = self.p.classes.get(recv["class"])
        if cls is None:
            raise Refuse(f"{where}: class {recv['class']} is not a top-level class of this program bound once")
        if not self._plain_class(cls):
            raise Refuse(f"{where}: class {cls.name} has a metaclass, a decorator, a base or a hook that "
                         f"redirects construction or attribute access")
        attr = call.func.attr
        fn = self._class_method(cls, attr)
        if fn is None or rec["name"] != f"{cls.name}.{attr}" or not fn.args.args:
            raise Refuse(f"{where}: {cls.name} has no single undecorated own instance method {attr}")
        if attr in self.p.attr_stores:
            raise Refuse(f"{where}: some attribute store binds .{attr}, which could replace the method")
        return fn

    def _ctor(self, call: ast.Call) -> ast.FunctionDef:
        """The ``__init__`` a call of a user class runs, resolved through the ``call_target`` fact, or ``Refuse``.

        ``Name(...)`` constructs ``C`` only when the fact for ``(scope, callee)`` says ``user_class`` AND ``proven``
        (nothing in the import graph rebinds the name, no local binding hides it) and ``C`` is a top-level class of
        THIS program bound exactly once.  Then the value reaches ``C``'s ``__init__`` and nothing else only if
        :meth:`_plain_class` holds (no metaclass, no decorator, no base: ``__new__`` or a base ``__init__`` could
        read it too) and the class body binds ``__init__`` exactly once, by an undecorated ``def`` that no
        attribute store anywhere in the program (``C.__init__ = f``) can replace."""
        callee = ast.unparse(call.func)
        where = f"line {call.lineno}: a trip value is passed to {callee}"
        rec = self.targets.get((self._scope_of(call), callee))
        if rec is None:
            raise Refuse(f"{where}, an unresolved callee")
        if rec["target"] != "user_class" or not rec["proven"]:
            raise Refuse(f"{where}, whose call_target is {rec['target']} (proven={rec['proven']}: {rec['reason']})")
        cls = self.p.classes.get(rec["name"])
        if cls is None:
            raise Refuse(f"{where}: class {rec['name']} is not a top-level class of this program bound once")
        if not self._plain_class(cls):
            raise Refuse(f"{where}: class {cls.name} has a metaclass, a decorator, a base or a hook that "
                         f"redirects construction or attribute access")
        init = self._class_method(cls, "__init__")
        if init is None or "__init__" in self.p.attr_stores:
            raise Refuse(f"{where}: class {cls.name} has no single undecorated own __init__ (or one is stored "
                         f"into elsewhere)")
        return init

    def _scope_of(self, node: ast.AST) -> str:
        """The ``scope`` key of ``solve/call_targets.py`` for ``node``: the dotted names of the defs and classes
        around it (``"<main>"`` at module level)."""
        names = []
        cur = self.p.parent.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(cur.name)
            cur = self.p.parent.get(cur)
        return ".".join(reversed(names)) if names else "<main>"

    def _class_only_call(self, call: ast.AST, expr: ast.AST) -> bool:
        """Is ``call`` ``type(expr)`` or ``isinstance(expr, <builtin classes>)``, the one kind of
        builtin call a trip value may flow into? ([internal ref], the SOUND part of "known-transparent
        builtins": every other builtin's result varies with the VALUE.)

        Why it is sound: a trip value is ``c + abs(d)``, always an exact ``int``, so the call's
        result is a function of the value's CLASS only -- the same constant for every trip
        count -- and can neither raise nor run user code (an exact int has no hook; the
        classinfo is a builtin class, so no metaclass ``__instancecheck__``). The call can
        therefore be neither a value-dependent check nor a value-dependent effect, wherever
        its result goes. The shapes come from the contracts
        (``contracts/adapters/common.py::result_depends_only_on_class``: a TESTED contract for
        the callable must still exist); a builtin bound anywhere in the program is not trusted.

        ``divmod(n, K)``, ``len``, ``ord``, ``list(range(n))`` and the like do NOT qualify: the
        result varies with n, and the induction stops tracking it the moment this module does
        (W41's never-TRUE probe, ``tests/test_sv_prove.py::MUTANTS['divmod_gate']``)."""
        from tharkun.solve.contracts.adapters.common import result_depends_only_on_class

        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and not call.keywords
                and not any(isinstance(a, ast.Starred) for a in call.args)):
            return False
        name = call.func.id
        if self.p.bound.get(name) or not result_depends_only_on_class(
                name, arg_count=len(call.args), has_keywords=False):
            return False
        if call.args[0] is not expr:
            return False
        if name == "isinstance":
            info = call.args[1]
            names = info.elts if isinstance(info, ast.Tuple) else [info]
            return bool(names) and all(
                isinstance(x, ast.Name) and x.id in _BUILTIN_CLASSES and not self.p.bound.get(x.id)
                for x in names)
        return True

    def trip_variable(self, name: str, scope: ast.AST, c: int,
                      origin: ast.AST | None) -> set[tuple[int, int]]:
        key = (id(scope), name)
        if key in self.seen:
            return set()
        self.seen.add(key)
        loads, stores = self.occurrences(name, scope)
        loops: set[tuple[int, int]] = set()
        for_targets = []
        for st in stores:
            if origin is not None and self.p.parent.get(st) is origin:
                continue
            if isinstance(st, ast.arg):
                continue                          # the parameter binding itself
            par = self.p.parent.get(st)
            if isinstance(par, ast.For) and par.target is st:
                for_targets.append(par)           # ``for loops in range(loops)``
                continue
            raise Refuse(f"line {getattr(st, 'lineno', '?')}: {name} is rebound")
        if origin is None and not any(isinstance(s, ast.arg) for s in stores):
            raise Refuse(f"{name} is not a parameter of its function")
        for use in loads:
            par = self.p.parent[use]
            if _is_call_to(par, "range") and len(par.args) == 1:
                loops |= self.range_object(par, c)
            elif (isinstance(par, ast.Call) and use in par.args) or isinstance(par, ast.keyword):
                loops |= self.trip_argument(use, c)
            else:
                raise Refuse(f"line {use.lineno}: {name} (a trip value) is read by "
                             f"{ast.unparse(par)[:60]!r}")
        for loop in for_targets:
            # ``for v in range(v)``: after the first iteration ``v`` no longer holds the draw,
            # so that range must be the ONLY read of ``v``, and must run once per binding
            if not (_is_call_to(loop.iter, "range") and len(loads) == 1
                    and loop.iter.args[0] is loads[0]):
                raise Refuse(f"line {loop.lineno}: {name} is rebound by a loop and read again")
            if self.inside_loop(loop, scope):
                raise Refuse(f"line {loop.lineno}: {name} is rebound by a nested loop")
        return loops

    def inside_loop(self, node: ast.AST, scope: ast.AST) -> bool:
        p = self.p.parent.get(node)
        while p is not None and p is not scope:
            if isinstance(p, (ast.For, ast.While)):
                return True
            p = self.p.parent.get(p)
        return False

    def range_object(self, rng: ast.Call, c: int) -> set[tuple[int, int]]:
        par = self.p.parent[rng]
        if isinstance(par, ast.For) and par.iter is rng:
            return {self.trip_loop(par, c)}
        if isinstance(par, ast.Assign) and par.value is rng and len(par.targets) == 1 \
                and isinstance(par.targets[0], ast.Name):
            name = par.targets[0].id
            scope = self.p.function_of(par)
            loads, stores = self.occurrences(name, scope)
            if any(self.p.parent.get(s) is not par for s in stores):
                raise Refuse(f"line {par.lineno}: {name} (a trip range) is rebound")
            out = set()
            for use in loads:
                loop = self.p.parent[use]
                if not (isinstance(loop, ast.For) and loop.iter is use):
                    raise Refuse(f"line {use.lineno}: {name} (a trip range) is read by "
                                 f"{ast.unparse(loop)[:60]!r}")
                out.add(self.trip_loop(loop, c))
            return out
        raise Refuse(f"line {rng.lineno}: a trip range flows into {ast.unparse(par)[:60]!r}")

    def trip_loop(self, loop: ast.For, c: int) -> tuple[int, int]:
        if not isinstance(loop.target, ast.Name):
            raise Refuse(f"line {loop.lineno}: loop target is not a plain name")
        t = loop.target.id
        scope = self.p.function_of(loop)
        loads, _ = self.occurrences(t, scope)
        inside_iter = {id(n) for n in ast.walk(loop.iter)}
        live = [u for u in loads if id(u) not in inside_iter]
        if live:
            raise Refuse(f"line {live[0].lineno}: the loop variable {t} is read")
        key = (loop.lineno, loop.col_offset)
        old = self.loops.get(key)
        if old is not None and old.offset != c:
            raise Refuse(f"line {loop.lineno}: trip values with different offsets")
        self.loops[key] = TripLoop(loop.lineno, loop.col_offset, (t,), c)
        return key

    def occurrences(self, name: str, scope: ast.AST) -> tuple[list[ast.Name], list[ast.AST]]:
        """Every read of ``scope``'s variable ``name`` and every binding of it.

        A read belongs to ``scope`` when the innermost function enclosing it that binds
        ``name`` is ``scope`` (Python's lexical resolution; comprehensions count as part
        of their function, which only makes this stricter).  A ``global``/``nonlocal``
        declaration of ``name`` anywhere, or a class body binding it, refuses."""
        owners: set[int] = set()
        stores: list[ast.AST] = []
        for node in ast.walk(self.p.tree):
            if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
                raise Refuse(f"line {node.lineno}: {name} is declared global/nonlocal")
            if name in _bound_names(node):
                if isinstance(self.p.scope_of(node), ast.ClassDef):
                    raise Refuse(f"line {node.lineno}: a class body binds {name}")
                owner = self.p.function_of(node)
                owners.add(id(owner))
                if owner is scope:
                    stores.append(node)
        loads: list[ast.Name] = []
        for node in ast.walk(scope):
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
                f = self.p.function_of(node)
                while id(f) not in owners and not isinstance(f, ast.Module):
                    f = self.p.function_of(f)
                if f is scope:
                    loads.append(node)
        return loads, stores

    # -- [internal ref] follow-up: interprocedural str-draw tracking (bm_raytrace_nondet's
    # args_filename -> bench_raytrace's own `filename` parameter -> canvas.write_ppm's own
    # `filename` parameter -> open()) ---------------------------------------------------
    def _bound_in_class_body(self, cls: ast.ClassDef, name: str) -> list[ast.AST]:
        """Every node directly in ``cls``'s OWN body (recursing into ``if``/``for``/
        ``while``/... at class scope, but NOT into a nested function/lambda/class's own
        separate scope -- a local variable inside a METHOD's own body that happens to
        share the method's name is not a rebinding of the class attribute) that binds
        ``name``."""
        hits: list[ast.AST] = []

        def walk(node: ast.AST) -> None:
            if name in _bound_names(node):
                hits.append(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                return                     # its own name is already counted above
            for child in ast.iter_child_nodes(node):
                walk(child)

        for stmt in cls.body:
            walk(stmt)
        return hits

    def _class_method(self, cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
        """``cls``'s own ``name`` method -- bound EXACTLY ONCE anywhere in the WHOLE class
        body, by that one ``FunctionDef``, undecorated -- None otherwise. MG's PR 108
        review (``n11_class_body_alias.py``): counting only ``FunctionDef`` nodes (the
        previous version of this method) missed a LATER plain rebinding in the same class
        body (``def run(self, s): ...`` followed by ``run = staticmethod(bad)``) -- the
        class body's own later statement is what the class actually ends up with, not the
        def. ``_bound_in_class_body`` catches a rebinding of ANY shape (a plain assignment,
        an ``AugAssign``, one nested inside an ``if``, ...), not just another ``def``."""
        hits = self._bound_in_class_body(cls, name)
        if len(hits) != 1 or not isinstance(hits[0], ast.FunctionDef) or hits[0].decorator_list:
            return None
        return hits[0]

    def _recv_class(self, recv_name: str, call: ast.Call) -> ast.ClassDef | None:
        """The class a local variable ``recv_name`` unambiguously holds an instance of, at
        ``call`` -- bound EXACTLY ONCE (excluding a parameter binding, which this never is:
        a receiver is always a local, never a parameter, for the one call shape this
        resolves) within its own function, directly from ``ClassName(...)`` for a class
        THIS PROGRAM defines (``prog.classes``, top-level only) with no inheritance beyond
        plain ``object``, no metaclass/class-keywords, no decorator, and no dunder that
        could redirect attribute access (``__getattr__``/``__getattribute__``/
        ``__setattr__``) -- else ``None``, never guessed. A single total binding in the
        WHOLE function (not just before this call, positionally) is a stronger, simpler,
        still-sound restriction, the same one ``trip_variable`` already applies to a trip
        value's own variable above."""
        scope = self.p.function_of(call)
        _, stores = self.occurrences(recv_name, scope)
        real_stores = [s for s in stores if not isinstance(s, ast.arg)]
        if len(real_stores) != 1:
            return None
        assign = self.p.parent.get(real_stores[0])
        if not (isinstance(assign, ast.Assign) and len(assign.targets) == 1
                and assign.targets[0] is real_stores[0] and isinstance(assign.value, ast.Call)
                and isinstance(assign.value.func, ast.Name) and not assign.value.keywords):
            return None
        cls = self.p.classes.get(assign.value.func.id)
        if cls is None or not self._plain_class(cls):
            return None
        return cls

    @staticmethod
    def _plain_class(cls: ast.ClassDef) -> bool:
        """No metaclass or class keyword, no decorator, no base beyond plain ``object``, and no dunder that
        redirects construction or attribute access (``__new__`` could hand back another object, ``__getattr__`` /
        ``__getattribute__`` / ``__setattr__`` run code on every access)."""
        if cls.keywords or cls.decorator_list:
            return False
        if cls.bases and not (len(cls.bases) == 1 and isinstance(cls.bases[0], ast.Name)
                              and cls.bases[0].id == "object"):
            return False
        return not any(isinstance(n, ast.FunctionDef)
                       and n.name in ("__new__", "__getattr__", "__getattribute__", "__setattr__")
                       for n in cls.body)

    def _resolve_str_callee(self, call: ast.Call) -> tuple[ast.FunctionDef, list[str]] | None:
        """The ``FunctionDef`` ``call`` invokes and its own plain positional parameter names
        (``self`` already dropped for a method) -- a plain call to a module-level function
        (``prog.funcs``) or a method call on a receiver ``_recv_class`` resolves -- else
        ``None``, never guessed. No ``*args``/``**kwargs``/positional-only params on either
        side: this stays a plain, countable positional mapping throughout."""
        f = call.func
        if isinstance(f, ast.Name):
            fn = self.p.funcs.get(f.id)
            if fn is None or fn.args.vararg or fn.args.kwarg or fn.args.posonlyargs:
                return None
            return fn, [x.arg for x in fn.args.args]
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            # [internal ref] follow-up, MG's PR 108 review: `f.attr`'s CLASS-BODY definition is not
            # necessarily what `call` actually invokes at runtime -- `c.run = bad`,
            # `C.run = bad2`, `self.run = bad` (anywhere, including `__init__`) all replace
            # it, ordinary Python this pass's other checks (a class with no
            # __getattr__/__setattr__, a receiver bound exactly once) do NOT catch, since
            # none of those are reflection and none rebind the RECEIVER. Refuse outright the
            # moment `f.attr` is EVER the target of a plain attribute store/delete anywhere
            # in the whole program (`prog.attr_stores`, populated once in `screen()`),
            # whatever the receiver expression on that store was.
            if f.attr in self.p.attr_stores:
                return None
            cls = self._recv_class(f.value.id, call)
            if cls is None:
                return None
            fn = self._class_method(cls, f.attr)
            if (fn is None or fn.args.vararg or fn.args.kwarg or fn.args.posonlyargs
                    or not fn.args.args):
                return None
            return fn, [x.arg for x in fn.args.args[1:]]     # drop `self`
        return None

    def str_restricted(self, name: str, scope: ast.AST, origin: ast.AST | None,
                       depth: int = 0) -> None:
        """Every read of ``name`` (a str-draw variable, or a parameter that received one
        across a resolvable call) in ``scope`` must be truthiness, ``open()``'s own path
        argument, or itself passed POSITIONALLY to another resolvable call whose matching
        parameter recursively satisfies this SAME rule -- never guessed, bounded by
        :data:`_MAX_STR_DEPTH`. Mirrors ``trip_variable``'s own rebind/origin handling
        exactly (only the "what is a legal read" predicate differs)."""
        key = (id(scope), name)
        if key in self.str_seen:
            return
        self.str_seen.add(key)
        if depth > _MAX_STR_DEPTH:
            raise Refuse(f"{name} (a str draw) flows through too many calls to track")
        loads, stores = self.occurrences(name, scope)
        for s in stores:
            if origin is not None and self.p.parent.get(s) is origin:
                continue
            if isinstance(s, ast.arg):
                continue                                     # the parameter binding itself
            raise Refuse(f"line {getattr(s, 'lineno', '?')}: {name} (a str draw) is rebound")
        if origin is None and not any(isinstance(s, ast.arg) for s in stores):
            raise Refuse(f"{name} is not a parameter of its function")
        for use in loads:
            if _str_truthy_or_open_path(use, self.p):
                continue
            par = self.p.parent.get(use)
            if not (isinstance(par, ast.Call) and use in par.args):
                raise Refuse(f"line {use.lineno}: {name} (a str draw) is used for more than "
                             f"truthiness, open()'s path, or a call this pass can resolve")
            if any(isinstance(a, ast.Starred) for a in par.args) or par.keywords:
                raise Refuse(f"line {use.lineno}: {name} (a str draw) reaches a call with "
                             f"*args/**kwargs")
            resolved = self._resolve_str_callee(par)
            if resolved is None:
                raise Refuse(f"line {use.lineno}: {name} (a str draw) is passed to an "
                             f"unresolved callee")
            fn, params = resolved
            i = par.args.index(use)
            if i >= len(params):
                raise Refuse(f"line {use.lineno}: too many arguments to {fn.name}")
            self.str_restricted(params[i], fn, None, depth + 1)


def plan_draws(source: str, filename: str = "<program>", call_targets=None) -> DrawPlan:
    """Classify every draw of ``source``; refuse unless every draw is ``trip``,
    ``quotient`` or ``finite`` (see the module docstring).  ``call_targets`` are the program's ``call_target``
    records (``solve/call_targets.py``): with them a trip value passed to a proven user class's constructor is
    followed into its ``__init__``."""
    try:
        tree = ast.parse(source, filename)
    except (SyntaxError, ValueError) as e:
        return DrawPlan(False, f"parse: {e}")
    prog = _Program(tree)
    planner = _Planner(prog, call_targets)
    draws: list[DrawSite] = []
    try:
        prog.screen()
        for call, kind in prog.draw_calls():
            if call.args or call.keywords:
                raise Refuse(f"line {call.lineno}: {kind} with arguments")
            if kind in _FINITE_DRAWS:
                draws.append(DrawSite(call.lineno, call.col_offset, "finite"))
                continue
            if kind == "nondet_str":
                # [internal ref] follow-up (bm_raytrace_nondet.py's args_filename): a str draw
                # used ONLY for truthiness and as open()'s own path argument observes
                # nothing about the string beyond "is it empty or not" -- by the time
                # drawflow reaches here, output_effects.py::output_only has ALREADY
                # certified every open() in the whole program as write-mode-only,
                # single-binding, never read back (see solve/output_effects.py's own
                # docstring and docs/SV.md's "Output-file effects" section), so the
                # PATH argument's own exact bytes can never return to the program
                # either -- only its use to name a write target, plus its truthiness,
                # are left to matter. Owner's lesson from [internal ref] applied here on
                # purpose: truthiness IS value-dependent, so this does NOT collapse
                # the draw to "irrelevant" -- it explicitly covers BOTH the empty and
                # non-empty case (prove.py::domain, below, returns a 2-value stand-in
                # domain ``["", "x"]``, mirroring how "bool" maps to
                # ``[False, True]`` -- never a single value standing in for "any
                # string"). The path value itself must flow NOWHERE else, INTERPROCEDURALLY
                # (`_Planner.str_restricted`, below the class): every read of the draw's own
                # variable, or of a parameter it flows into positionally across a call this
                # pass can resolve (a plain function, or a method whose receiver's single
                # class this program defines plainly), recursively, is checked -- one read
                # outside truthiness/open's path/a resolvable call anywhere in that chain
                # refuses the whole plan, same as any other unmodelled use.
                a = prog.parent.get(call)
                if not (isinstance(a, ast.Assign) and len(a.targets) == 1
                        and isinstance(a.targets[0], ast.Name)):
                    raise Refuse(f"line {call.lineno}: nondet_str is not assigned to a plain name")
                name = a.targets[0].id
                scope = prog.function_of(a)
                planner.str_restricted(name, scope, a)
                draws.append(DrawSite(call.lineno, call.col_offset, "str_truthy"))
                continue
            if kind not in _INT_DRAWS:
                raise Refuse(f"line {call.lineno}: {kind} (an unbounded draw kind)")
            a = prog.parent.get(call)
            if not (_is_call_to(a, "abs") and len(a.args) == 1):
                raise Refuse(f"line {call.lineno}: the draw is not taken abs(): a nondet_int used as a value (a "
                             f"precondition-guarded `if not (lo <= n <= hi): return` one included) is no trip count, "
                             f"and stationarity covers trip counts abs(d) only")
            up = prog.parent.get(a)
            if isinstance(up, ast.BinOp) and isinstance(up.op, ast.Mod) and up.left is a:
                if any(isinstance(n, ast.Call) and _mentions_draw(n, prog)
                       for n in ast.walk(up.right)):
                    raise Refuse(f"line {call.lineno}: the modulus draws")
                draws.append(DrawSite(call.lineno, call.col_offset, "quotient"))
                continue
            c, expr = 0, a
            if isinstance(up, ast.BinOp) and isinstance(up.op, ast.Add):
                other = up.right if up.left is a else up.left
                if isinstance(other, ast.Constant) and type(other.value) is int and other.value >= 0:
                    c, expr = other.value, up
            loops = planner.trip_value(expr, c)
            if not loops:
                raise Refuse(f"line {call.lineno}: the draw reaches no loop")
            draws.append(DrawSite(call.lineno, call.col_offset, "trip", c, tuple(sorted(loops))))
    except Refuse as e:
        return DrawPlan(False, str(e))
    if not any(d.kind != "finite" for d in draws):
        return DrawPlan(False, "no int draw to reason about")
    return DrawPlan(True, "", tuple(draws), tuple(sorted(planner.loops.values(),
                                                         key=lambda t: (t.line, t.col))), output_files=prog.output_files)


def _mentions_draw(node: ast.Call, prog: _Program) -> bool:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id in prog.sv_funcs
    return isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and \
        f.value.id in prog.sv_modules
