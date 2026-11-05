"""One ORIGINAL-source exception inventory and its audited census projection.

Sites are a set of physical source locations, not an evaluation schedule.
Unknown dispatch remains open. The census projection retains its closed-program
Resolver/visible-override contract. Complete scope-exit coverage is a separate
absence certificate; a partial typed case cannot widen either proof route.
"""
from __future__ import annotations

import ast
import builtins
import copy
import os
from functools import lru_cache

from tharkun.solve.contracts.adapters import obligations as _contracts
from tharkun.solve.contracts.adapters.common import native_raise_condition, never_raises
from tharkun.solve.contracts.adapters.exception_sites import EXCLUDED_EXCEPTIONS, condition_table, unknown_case
from tharkun.solve.contracts.registry import get
from tharkun.solve.check_type_hints import (check_type_arguments, check_type_contract, closed_hint_effects,
                                             plain_class_hint)
from tharkun.solve.exception_handlers import builtin_exception_chain
from tharkun.solve.fixed_types import FixedTypes
from tharkun.solve.library_sites import LIBRARY_MODULES, is_library_constant, library_may_raise, method_may_raise, type_method_may_raise
from tharkun.solve.source_arguments import bind_source_call
from tharkun.solve.pyresolve import Resolver, Top
from tharkun.solve.terms import PY, and_, canon, from_canon, not_, or_, typeis, var

MODULES = frozenset({"_sv_verifier", "math", "random", "re", "io", "array", "string",
                     "os", "os.path", "typing", "heapq", "bisect", "collections", "itertools", "functools"})


_CONTRACT_MODULES = frozenset({"heapq", "bisect", "collections", "itertools", "functools"})

#: batch-10 functools contracts that store a callable argument and invoke it later: a Name / Attribute
#: argument (live program code) keeps the call an exception site -- scoped to exactly these names, and to
#: nothing in the other contract modules (the regression target of the first batch-10 fix attempt).
_CALLABLE_ARG_SITE_CALLS = frozenset({"functools.wraps", "functools.update_wrapper"})


SITE_KINDS = ("ZeroDivisionError", "FloatingPointError", "AssertionError", "KeyError", "IndexError")


_DYNAMIC = frozenset({"eval", "exec", "compile", "__import__", "globals", "vars", "__builtins__",
                      "breakpoint"})


_NUMERIC_DUNDERS = frozenset({
    "__truediv__", "__rtruediv__", "__itruediv__", "__floordiv__", "__rfloordiv__",
    "__ifloordiv__", "__mod__", "__rmod__", "__imod__", "__divmod__", "__rdivmod__",
    "__pow__", "__rpow__", "__ipow__"})


_ZERO_DIV_CALLS = frozenset({
    "divmod", "pow", "math.log", "log",
    "triangular", "expovariate", "paretovariate", "weibullvariate", "betavariate",
    "gammavariate", "vonmisesvariate", "lognormvariate", "normalvariate", "gauss",
    "binomialvariate", "choices", "sample"}) | _NUMERIC_DUNDERS


_LOOKUP_CALLS = frozenset({
    "pop", "popitem", "remove", "group", "groups", "groupdict", "start", "end", "span",
    "expand", "format", "format_map", "substitute", "choice", "choices", "sample",
    "__getitem__", "__delitem__", "__setitem__", "__missing__", "heappop", "heapreplace",
    "setdefault", "getrandbits", "fromkeys", "vformat", "get_value", "get_field", "sub", "subn",
    "popleft", "move_to_end", "setstate"})       # setstate: random.Random().setstate({}) is KeyError(0) (Opus battery)


_THROW_CALLS = frozenset({"throw", "athrow"})
#: attribute/method names that return an attribute or an item chosen by a runtime string (string.Formatter.get_field / get_value, the
#: attribute-lookup hooks): any use is a named refusal
_NAMED_BY_STRING = frozenset({"get_field", "get_value", "__getattribute__", "__getattr__", "__dict__", "attrgetter", "methodcaller",
                              "itemgetter", "getmembers"})


#: ``_sv_verifier.check_type`` is NOT in these tables: with a closed hint (``visit_call`` refuses any other) its own code raises
#: only what its contract names natively (TypeError), and every other class can only come from user hooks it calls (``__class__``,
#: ``__repr__``, ``__iter__``, ``items`` ...), which are program code the census scans anyway -- dunders and, since check_type is
#: not a closed call, every class method are live (``pyresolve.py::_dead_defs``).  Pinned: ``tests/test_census_check_type_pin.py``.
_CALL_SITES = {
    "ZeroDivisionError": _ZERO_DIV_CALLS,
    "FloatingPointError": frozenset(),
    "AssertionError": frozenset({"binomialvariate", "assert_never"}),   # CPython's own asserts / assert_never; METHOD calls only:
    #                                                  module-level calls are decided by library_sites.LIBRARY_CALLS, not by this table
    "KeyError": _LOOKUP_CALLS,
    "IndexError": _LOOKUP_CALLS,
}


class _Refuse(Exception):
    pass


class _ImportedModules:
    """Membership by root: ``"os.path" in imported`` when ``os`` is imported."""

    def __init__(self, roots: set[str]):
        self.roots = roots

    def __contains__(self, module: str) -> bool:
        return module.split(".")[0] in self.roots


def _builtin_exception(name: str):
    obj = getattr(builtins, name, None)
    return obj if isinstance(obj, type) and issubclass(obj, BaseException) else None


def _bound(node: ast.AST) -> list[str]:
    """Names (re)bound by ``node``."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return [node.id]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.ExceptHandler) and node.name:
        return [node.name]
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        return [node.name]
    return []


def _contract_rules(mod: str, attr: str) -> tuple:
    """Tested exact contract, or the tested descendants admitting a namespace."""
    from tharkun.solve.contracts.rules import all_rules, get_rule
    from tharkun.solve.contracts.visitors import CensusVisitor
    visitor = CensusVisitor()
    rule = get_rule(f"{mod}.{attr}")
    if rule is not None:
        contract = rule.accept(visitor, "admission")
        if contract is not None and getattr(contract, "tested", False):
            return (contract,)
    prefix = f"{mod}.{attr}."
    contracts = []
    for r in all_rules():
        if r.qualname.startswith(prefix):
            contract = r.accept(visitor, "admission")
            if contract is not None and getattr(contract, "tested", False):
                contracts.append(contract)
    return tuple(contracts)


@lru_cache(maxsize=None)
def _has_contract_rule(mod: str, attr: str) -> bool:
    return bool(_contract_rules(mod, attr))


class _CensusProjection:
    def __init__(self, tree: ast.Module, classes: tuple[str, ...], res: Resolver,
                 filename: str = "<program>",
                 user_classes: dict[str, ast.ClassDef] | None = None,
                 rebound: frozenset[str] = frozenset(), fold_constants: bool = False):
        self.tree = tree
        self.rebound = rebound
        self.fold_constants = fold_constants
        self.res = res
        self.filename = filename
        self.classes = classes
        self.want = {c: _builtin_exception(c) for c in classes}
        self.calls = frozenset().union(_THROW_CALLS, *(_CALL_SITES[c] for c in classes))
        self.zero_div = "ZeroDivisionError" in classes
        self.lookup = "KeyError" in classes or "IndexError" in classes
        self.asserts = "AssertionError" in classes
        self.modules: set[str] = set()
        self.from_names: dict[str, str] = {}     # local name -> "module.name"
        # [internal ref]: shared across every file one site_census() call scans (see site_census's own
        # docstring) so a class DEFINED in one recursively-read file and RAISED in another
        # still classifies; defaults to a fresh, single-file table for any other caller
        # (obligation_census) that still scans exactly one tree, unchanged.
        self.user_classes: dict[str, ast.ClassDef] = user_classes if user_classes is not None else {}
        self.sites: list[tuple[int, int, str]] = []
        self.nodes: list[ast.AST] = []
        self.obligations: list[tuple[int, int, "_contracts.Obligation"]] = []

    def _enclosing_function(self, node: ast.AST) -> str:
        """Nearest enclosing ``def``/``async def`` name, or ``"<module>"`` at top
        level -- a location label, not OB's fuller dotted class.method scheme
        ([internal ref] slice 0's ``source_points``), which this doesn't attempt to
        replicate."""
        p = self.parent.get(node)
        while p is not None:
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return p.name
            p = self.parent.get(p)
        return "<module>"

    def _record_obligation(self, node: ast.Call, qualname: str) -> None:
        """A checked, witness-format safety obligation for the call site just
        recorded ([internal ref] step 2 redesign) -- a ``python_expression``
        ``location_invariant``, or an explicit :class:`Gap`, never a dropped or
        English-text claim. Only ``ast.Call`` sites with an exactly resolved
        qualname reach here (a bare name "used as a value" has no call
        arguments to build a safety expression from, so it is skipped, not
        guessed)."""
        ob = _contracts.obligation_for_call(
            qualname, node, file_name=self.filename, function=self._enclosing_function(node),
            classes=self.classes)
        if ob is not None:
            self.obligations.append((node.lineno, node.col_offset, ob))

    def site(self, node: ast.AST, what: str) -> None:
        self.sites.append((getattr(node, "lineno", 0), getattr(node, "col_offset", 0), what))
        self.nodes.append(node)

    # -- tables ----------------------------------------------------------------
    def prepare(self, parent, bound) -> None:
        self.modules = set(self.res.imports.modules)
        self.from_names = {alias: f"{mod}.{attr}" for alias, (mod, attr) in self.res.imports.names.items()
                           if not mod.startswith("$extern:")}
        for alias, (mod, attr) in self.res.imports.names.items():
            if mod in _CONTRACT_MODULES and not _has_contract_rule(mod, attr):
                raise _Refuse(f"module {mod}: no contract for {attr}")
            if mod.startswith("$extern:"):             # `from helpers import assert_never`: what the sibling's name REALLY is
                origin = self.extern_origin(self.res.externs.get(alias), attr)
                if origin is not None:
                    self.from_names[alias] = origin
        self.parent, self.bound = parent, bound
        # A module-level ``K = <number>`` whose name has no other binding in the program holds that
        # number wherever it is read (an unbound read is a NameError, not a division site); ``mod.K = ..`` from
        # another file (``rebound``) rebinds it, as for ``fixed_types.py``.  Only the census PROOF view folds (``fold_constants``): the
        # witness obligations (``obligation_census``) keep one per division site, because the independent checker re-derives the site list.
        self.const_names = {}
        for stmt in (self.tree.body if self.fold_constants else ()):
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                    and bound.get(stmt.targets[0].id) == 1 and stmt.targets[0].id not in self.rebound
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, (int, float, complex)) and not isinstance(stmt.value.value, bool)):
                self.const_names[stmt.targets[0].id] = stmt.value

    def extern_origin(self, sub, name: str, depth: int = 0) -> str | None:
        """The library qualname a name imported from a sibling module really is (``helpers`` re-exports ``typing.assert_never``),
        ``"?extern.<name>"`` when it cannot be followed (a site), or ``None`` when the sibling defines it itself (program code)."""
        if sub is None or depth > 8:
            return f"?extern.{name}"
        imported = sub.imports.names.get(name)
        if imported is None:
            return None
        module, attr = imported
        if module.startswith("$extern:"):
            return self.extern_origin(sub.externs.get(name), attr, depth + 1)
        return f"{module}.{attr}"

    def raises_property(self, name: str, seen=()) -> bool | None:
        """Does class ``name`` derive from a property class?  None: cannot tell."""
        exc = _builtin_exception(name)
        if exc is not None and not self.bound.get(name):
            return any(w is not None and issubclass(exc, w) for w in self.want.values())
        cls = self.user_classes.get(name)
        if cls is None or name in seen or self.bound.get(name) != 1:
            return None
        # svaudit 2026-10-29: a decorator, a metaclass or a ``__new__`` decides what ``name()`` returns --
        # ``def __new__(cls): return ZeroDivisionError()`` makes ``raise name()`` raise a property class
        if cls.keywords or cls.decorator_list or any(
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__new__"
                or isinstance(n, ast.Name) and n.id == "__new__" for n in ast.walk(cls)):
            return None
        verdicts = []
        for b in cls.bases:
            if not isinstance(b, ast.Name):
                return None
            verdicts.append(self.raises_property(b.id, seen + (name,)))
        if any(v is None for v in verdicts):
            return None
        return any(verdicts)

    # -- one node --------------------------------------------------------------
    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Attribute):
            base_mod = None
            if isinstance(node.value, ast.Name):
                base_mod = self.res.imports.modules.get(node.value.id)
            if base_mod in _CONTRACT_MODULES and not _has_contract_rule(base_mod, node.attr):
                raise _Refuse(f"module {base_mod}: no contract for {node.attr}")
        if isinstance(node, ast.Attribute) and node.attr in _NAMED_BY_STRING:
            # an API that returns an attribute or item NAMED BY A RUNTIME STRING (`string.Formatter().get_field("0.__floordiv__", ..)[0]`)
            # makes an unknown callable wherever its result flows (bound to a name, returned from a def, stored in a container): refused
            # at the source, so no flow has to be followed
            raise _Refuse(f"line {node.lineno}: {node.attr} returns an attribute or item named by a runtime string "
                          "(its result is an unknown callable)")
        if isinstance(node, ast.Name) and node.id in _DYNAMIC:
            raise _Refuse(f"line {node.lineno}: uses {node.id}")
        if isinstance(node, ast.Name) and node.id in self.modules:
            par = self.parent.get(node)
            if not (isinstance(par, ast.Attribute) and par.value is node):
                raise _Refuse(f"line {node.lineno}: module {node.id} used as a value")
        if isinstance(node, ast.Raise) and node.exc is not None:
            self.visit_raise(node)
        if self.asserts and isinstance(node, ast.Assert):
            if not (isinstance(node.test, ast.Constant) and node.test.value):
                self.site(node, "assert")
        if self.zero_div and isinstance(node, (ast.BinOp, ast.AugAssign)):
            self.visit_arith(node)
        if self.lookup:
            self.visit_lookup(node)
        if isinstance(node, ast.Call):
            self.visit_call(node)
        elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            # a site-raising callable used as a value can be called from anywhere
            name = self.callable_name(node)
            par = self.parent.get(node)
            if name == "_sv_verifier.check_type" and not (isinstance(par, ast.Call) and par.func is node) \
                    and not (isinstance(par, ast.Attribute) and par.value is node):
                raise _Refuse(f"line {node.lineno}: check_type used as a value (its calls would escape the closed-hint check)")
            split = self.library_split(name) if name else None
            if split is not None and split[0] in self.modules_by_name() and is_library_constant(*split):
                return                              # math.pi, re.IGNORECASE: a value, never called
            if isinstance(node, ast.Attribute) and name is not None and "." not in name \
                    and not (isinstance(par, ast.Call) and par.func is node):
                # a bare attribute load (`self.result`, `xs.pop` not called): only a method of a known builtin/library type that can raise
                # is a site; any other attribute is data
                known = method_may_raise(node.attr)
                if known is not None and known & set(self.classes) and not (isinstance(par, ast.Attribute) and par.value is node):
                    self.site(node, f"{name} used as a value")
                return
            if name is not None and not (isinstance(par, ast.Call) and par.func is node) \
                    and not (isinstance(par, ast.Attribute) and par.value is node) \
                    and self.is_site_callable(name, node):
                self.site(node, f"{name} used as a value")
                # No call arguments exist at a bare reference site, so there is
                # nothing to build a safety expression from -- no obligation call.

    def visit_raise(self, node: ast.Raise) -> None:
        exc = node.exc
        target = exc.func if isinstance(exc, ast.Call) else exc
        if isinstance(target, ast.Name):
            if not isinstance(exc, ast.Call) and self.reraises(node, target.id):
                return                             # re-raising the exception being handled
            verdict = self.raises_property(target.id)
            if verdict is False:
                return
            self.site(node, f"raise {target.id}" + ("" if verdict else " (unknown class)"))
            return
        if isinstance(exc, (ast.Constant, ast.JoinedStr)) and not self.want.get("TypeError"):
            return                                 # raising a str/number raises TypeError
        self.site(node, f"raise {ast.unparse(exc)[:40]} (unknown class)")

    def reraises(self, node: ast.Raise, name: str) -> bool:
        """``raise NAME`` inside the ``except ... as NAME`` handler that binds it, with no
        other binding of ``NAME`` in the program: the exception being handled, re-raised."""
        if self.bound.get(name) != 1:
            return False
        p = self.parent.get(node)
        while p is not None:
            if isinstance(p, ast.ExceptHandler) and p.name == name:
                return True
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                return False
            p = self.parent.get(p)
        return False

    def visit_arith(self, node: ast.AST) -> None:
        op = node.op
        right = node.right if isinstance(node, ast.BinOp) else node.value
        left = node.left if isinstance(node, ast.BinOp) else None
        if isinstance(right, ast.Name):
            right = self.const_names.get(right.id, right)
        num = right.value if isinstance(right, ast.Constant) and not isinstance(
            right.value, (bool, str, bytes)) and right.value is not None else None
        if isinstance(op, (ast.Div, ast.FloorDiv)):
            if not (isinstance(num, (int, float, complex)) and num != 0):
                self.site(node, "division")
        elif isinstance(op, ast.Mod):
            if isinstance(left, (ast.Constant, ast.JoinedStr)) and isinstance(
                    getattr(left, "value", ""), (str, bytes)):
                return                             # string formatting
            if not (isinstance(num, (int, float)) and num != 0):
                self.site(node, "modulo")
        elif isinstance(op, ast.Pow):
            if not (isinstance(num, (int, float)) and num >= 0):
                self.site(node, "power")

    def fixed(self) -> FixedTypes:
        if getattr(self, "_fixed", None) is None:
            self._fixed = FixedTypes(self.tree, lambda call: self.callable_name(call.func), self.res.is_builtin, self.rebound)
        return self._fixed

    def visit_lookup(self, node: ast.AST) -> None:
        # A slice never raises IndexError, but slices are hashable (3.12+): ``d[a:b]`` on a
        # dict raises KeyError -- unless the sliced value is an exact list / tuple / str / bytes.
        if isinstance(node, ast.Subscript):
            if not isinstance(node.slice, ast.Slice):
                self.site(node, "subscript")
            elif "KeyError" in self.classes and self.fixed().kind(node.value) not in ("list", "tuple", "str", "bytes"):
                self.site(node, "subscript")
        op = getattr(node, "op", None)
        if isinstance(op, ast.Mod) and isinstance(node, (ast.BinOp, ast.AugAssign)):
            left = node.left if isinstance(node, ast.BinOp) else None
            if not (isinstance(left, ast.Constant) and not isinstance(left.value, bytes)
                    and "%(" not in str(left.value)):
                # a string on the left may be %-formatting with a mapping key: KeyError -- but an exact int / float
                # left operand never is (int.__mod__ answers NotImplemented to a mapping: TypeError)
                left_node = node.left if isinstance(node, ast.BinOp) else (
                    ast.Name(id=node.target.id, ctx=ast.Load()) if isinstance(node.target, ast.Name) else node.target)
                if self.fixed().kind(left_node) not in ("int", "float"):
                    self.site(node, "modulo (may be %-formatting with a mapping)")

    def callable_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            if node.id in self.from_names:
                return self.from_names[node.id]
            return node.id
        if isinstance(node, ast.Attribute):
            chain, base = [], node
            while isinstance(base, ast.Attribute):
                chain.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name) and base.id in self.modules:      # `os.path.join`, `os.environ.get`: the whole dotted path
                module = self.res.imports.modules.get(base.id, base.id)
                if module.startswith("$extern:") and len(chain) == 1:      # `helpers.assert_never(..)`: through the sibling's own imports
                    origin = self.extern_origin(self.res.externs.get(base.id), chain[0])
                    if origin is not None:
                        return origin
                return ".".join([module, *reversed(chain)])
            prefix = self.callable_name(node.value)
            if prefix is not None and prefix.split(".", 1)[0] in _CONTRACT_MODULES:
                return f"{prefix}.{node.attr}"
            return node.attr
        return None

    def computed_callee_is_site(self, f: ast.AST) -> bool:
        """A call whose callee is not a name or an attribute path (``g(..)(x)``, ``t[k](x)``, ``(a if c else b)(x)``): the callable can be
        one a library call built (``string.Formatter().get_field("0.__floordiv__", ..)[0]``), so it is a site for every class -- except
        where the callee is program code the census scans anyway: an immediate lambda, a subscript of program names, the result of
        calling a program-defined function (a library callable it holds was a site where it was referenced)."""
        if isinstance(f, ast.Lambda):
            return False
        if isinstance(f, ast.Subscript):
            return any(isinstance(n, ast.Call) and not self._is_program_call(n) for n in ast.walk(f.value))
        if isinstance(f, ast.Call):
            return not self._is_program_call(f)
        return True

    def _is_program_call(self, call: ast.Call) -> bool:
        target = call.func
        if isinstance(target, ast.Name):
            return bool(self.bound.get(target.id)) and target.id not in self.from_names
        if isinstance(target, ast.Attribute) and method_may_raise(target.attr) is None:
            # `a.func1()(..)`: a method no builtin/library type has, defined by the program (so program code; a callable it returns
            # that came from a library was a site where it was referenced)
            return bool(self.bound.get(target.attr) or target.attr in self.rebound or target.attr in getattr(self.res, "program_defs", ()))
        return False

    def library_split(self, name: str) -> tuple[str, str] | None:
        """``module, callable`` for a dotted name rooted at a library module of the census table (longest module first)."""
        parts = name.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:cut])
            if module in LIBRARY_MODULES:
                return module, ".".join(parts[cut:])
        return None

    def library_call_is_site(self, name: str) -> bool | None:
        """A call into a non-contract library module is a site iff the table says it can raise a class of this census; a callable
        the table does not know is a site too -- never trusted by name.  ``None``: not a library-module name."""
        split = self.library_split(name)
        if split is None or split[0] not in self.modules_by_name():
            return None
        known = library_may_raise(*split)
        return True if known is None else bool(known & set(self.classes))

    def modules_by_name(self) -> "_ImportedModules":
        """The library modules the program imports: a module is imported when its root is (``import os`` reaches ``os.path``)."""
        roots = {m.split(".")[0] for m in self.res.imports.modules.values()}
        roots |= {v.rsplit(".", 1)[0].split(".")[0] for v in self.from_names.values() if "." in v}
        return _ImportedModules(roots)

    def is_site_callable(self, name: str, node: ast.AST) -> bool:
        if name.startswith("?extern."):
            return True                            # an import of a sibling that cannot be followed to its origin
        mod, _, attr = name.partition(".")
        library = self.library_call_is_site(name)
        if library is not None and mod not in _CONTRACT_MODULES:
            return library
        if mod in _CONTRACT_MODULES:
            contracts = _contract_rules(mod, attr)
            # Only tested never claims exclude a class; unknown and callback claims stay possible.
            if not contracts or any(not never_raises(c, exc) for c in contracts for exc in self.classes):
                return True
        short = name.rsplit(".", 1)[-1]
        if isinstance(node, ast.Name) and "." not in name and name not in dir(builtins):
            return False                           # a program name, not a library callable
        if isinstance(node, ast.Attribute) and "." not in name and self.method_is_site(node.attr, node.value):
            return True
        return name in self.calls or short in self.calls

    def method_is_site(self, attr: str, receiver: ast.AST | None = None) -> bool:
        """``recv.attr(..)`` with a receiver that is not a library module: the census decides by the table of what builtin and
        library objects' methods raise (``library_sites.method_may_raise``), over every known type because the receiver's type is
        not always known.  A method no known type has is unresolved: a site, unless the program defines a method of that name
        (then it is program code, which the census scans).  A name the table does not know never means "cannot raise"."""
        kind = self.fixed().kind(receiver) if receiver is not None else None
        if kind is not None:                       # a receiver FixedTypes resolves: judged as THAT type's method (deque.insert is not list.insert)
            known = type_method_may_raise(kind, attr)
            return True if known is None else bool(known & set(self.classes))
        known = method_may_raise(attr)
        if known is None:
            return not (self.bound.get(attr) or attr in self.rebound              # defined (def / class body) or stored (`self.f = ...`)
                        or attr in getattr(self.res, "program_defs", ()))          # by ANY file of the program
        return bool(known & set(self.classes))

    def visit_call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name) and f.id in ("getattr", "setattr", "delattr", "hasattr"):
            if len(node.args) < 2 or not (isinstance(node.args[1], ast.Constant)
                                          and isinstance(node.args[1].value, str)):
                raise _Refuse(f"line {node.lineno}: {f.id} with a computed name")
            attr_name = node.args[1].value
            target_mod = None
            if isinstance(node.args[0], ast.Name):
                target_mod = self.res.imports.modules.get(node.args[0].id)
                if target_mod in _CONTRACT_MODULES and not _has_contract_rule(target_mod, attr_name):
                    raise _Refuse(f"module {target_mod}: no contract for {attr_name}")
            library = self.library_call_is_site(f"{target_mod}.{attr_name}") if target_mod else None
            if attr_name in self.calls or library or (target_mod in _CONTRACT_MODULES and
                    self.is_site_callable(f"{target_mod}.{attr_name}", node)):
                self.site(node, f"{f.id}(..., {attr_name!r})")
            return
        name = self.callable_name(f)
        if name is None:
            if self.computed_callee_is_site(f):    # `Formatter().get_field("0.__floordiv__", (7,), {})[0](0)`: a callable a library built
                self.site(node, f"call of a computed value ({ast.unparse(f)[:40]})")
            return
        if name == "_sv_verifier.check_type":
            bound = check_type_arguments(node)
            if (check_type_contract() is None or bound is None or
                    closed_hint_effects(bound[1], self.res, lambda n: plain_class_hint(n, self.res)) is None):
                raise _Refuse(f"line {node.lineno}: check_type hint or call shape is not closed")
            contract = check_type_contract()
            for exc in self.classes:
                if native_raise_condition(contract, exc) is not None:
                    self.site(node, f"call {name} (its own code raises {exc})")
                    break
            return
        if name == "math.log" and len(node.args) == 1 and not node.keywords \
                and not isinstance(node.args[0], ast.Starred):
            return                                 # one-argument log: ValueError, never a division
        if self.is_site_callable(name, f):
            self.site(node, f"call {name}")
            self._record_obligation(node, name)
        elif name in _CALLABLE_ARG_SITE_CALLS and "." in name:
            mod, attr = name.rsplit(".", 1)
            # wraps returns a decorator that may execute arbitrary hooks later; keep every call as a site.
            # update_wrapper retains the earlier non-trivial positional argument safeguard.
            if mod == "functools" and _has_contract_rule(mod, attr) \
                    and (name == 'functools.wraps' or any(isinstance(arg, (ast.Name, ast.Attribute)) for arg in node.args)):
                self.site(node, f"contract call {name} with non-trivial argument")
                self._record_obligation(node, name)


def _span(node):
    return [node.lineno, node.col_offset, node.end_lineno, node.end_col_offset]


class _SourceIndex(ast.NodeVisitor):
    """Lexical metadata shared by every inventory projection, not a site walker."""

    def __init__(self, tree, filename):
        self.filename = str(filename)
        self.parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        self.assigned_attributes = {node.attr for node in ast.walk(tree)
                                    if isinstance(node, ast.Attribute)
                                    and isinstance(node.ctx, (ast.Store, ast.Del))}
        self.nonlocal_names = {name for node in ast.walk(tree)
                               if isinstance(node, (ast.Global, ast.Nonlocal)) for name in node.names}
        self.opaque_bindings = any(
            isinstance(node, ast.Name) and node.id in _DYNAMIC or
            isinstance(node, ast.Attribute) and node.attr in {
                "__dict__", "__globals__", "__code__", "__defaults__", "__kwdefaults__",
                "__bases__", "__mro__", "__class__"}
            for node in ast.walk(tree))
        self.opaque_class_identities = any(
            isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name) and node.func.id in {"setattr", "delattr"}
                or isinstance(node.func, ast.Attribute) and node.func.attr in {"__setattr__", "__delattr__"})
            for node in ast.walk(tree))
        self.scopes, self.node_scopes, self.definitions, self.bindings = {}, {}, {}, {}
        self.current = self._scope("<module>", "module", None, [])
        self.visit(tree)

    def _scope(self, function, kind, parent, parameters):
        identity = self.filename + "::" + function
        self.scopes[identity] = {"id": identity, "file_name": self.filename,
                                 "function": function, "kind": kind,
                                 "parameters": parameters, "sites": []}
        self.bindings[identity] = {}
        if not hasattr(self, "scope_parents"):
            self.scope_parents = {}
        self.scope_parents[identity] = parent
        return identity

    def _bind(self, name, value):
        self.bindings[self.current].setdefault(name, []).append(value)

    def visit(self, node):
        self.node_scopes[node] = self.current
        return super().visit(node)

    def generic_visit(self, node):
        for name in _bound(node):
            self._bind(name, node)
        super().generic_visit(node)

    def _name(self, name):
        prefix = self.scopes[self.current]["function"].removesuffix(".<body>")
        return name if prefix == "<module>" else prefix + "." + name

    def visit_FunctionDef(self, node):
        outer = self.current
        self._bind(node.name, node)
        for part in [*node.decorator_list, *node.args.defaults,
                     *(n for n in node.args.kw_defaults if n is not None)]:
            self.visit(part)
        if node.returns is not None:
            self.visit(node.returns)
        args = [*node.args.posonlyargs, *node.args.args]
        if node.args.vararg:
            args.append(node.args.vararg)
        args.extend(node.args.kwonlyargs)
        if node.args.kwarg:
            args.append(node.args.kwarg)
        kind = "method" if self.scopes[outer]["kind"] == "class" else "function"
        scope = self._scope(self._name(node.name), kind, outer, [arg.arg for arg in args])
        self.definitions[node] = scope
        for arg in args:
            if arg.annotation is not None:
                self.visit(arg.annotation)
        self.current = scope
        for arg in args:
            self.node_scopes[arg] = scope
            self._bind(arg.arg, arg)
        for statement in node.body:
            self.visit(statement)
        self.current = outer

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        outer = self.current
        self._bind(node.name, node)
        for part in [*node.decorator_list, *node.bases, *(kw.value for kw in node.keywords)]:
            self.visit(part)
        scope = self._scope(self._name(node.name) + ".<body>", "class", outer, [])
        self.definitions[node] = scope
        self.current = scope
        for statement in node.body:
            self.visit(statement)
        self.current = outer

    def visit_Lambda(self, node):
        outer = self.current
        for part in [*node.args.defaults, *(n for n in node.args.kw_defaults if n is not None)]:
            self.visit(part)
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        args.extend(n for n in (node.args.vararg, node.args.kwarg) if n is not None)
        scope = self._scope(self._name(f"<lambda@{node.lineno}:{node.col_offset}>"),
                            "function", outer, [arg.arg for arg in args])
        self.definitions[node] = scope
        self.current = scope
        for arg in args:
            self._bind(arg.arg, arg)
        self.visit(node.body)
        self.current = outer

    def resolve(self, name, scope):
        if self.opaque_bindings or name in self.nonlocal_names:
            return None  # reflection/nonlocal writes cannot certify a lexical callable binding
        while scope is not None:
            values = self.bindings[scope].get(name, ())
            if values:
                return values[0] if len(values) == 1 else None
            scope = self.scope_parents[scope]
            # A class namespace is not a lexical closure of its methods.
            while scope is not None and self.scopes[scope]["kind"] == "class":
                scope = self.scope_parents[scope]
        return getattr(builtins, name, None)

    def exception_chain(self, name, scope, seen=()):
        value = self.resolve(name, scope)
        if isinstance(value, type) and issubclass(value, BaseException):
            return tuple(cls.__name__ for cls in value.__mro__)
        if not isinstance(value, ast.ClassDef) or value in seen:
            return ()
        outer = self.node_scopes[value]
        chains = [self.exception_chain(base.id, outer, (*seen, value))
                  if isinstance(base, ast.Name) else () for base in value.bases]
        if not chains or any(not chain for chain in chains):
            return ()
        qualified = self.scopes[self.definitions[value]]["function"].removesuffix(".<body>")
        return (qualified, *dict.fromkeys(part for chain in chains for part in chain))

    def legacy_constructor(self, name, scope, seen=()):
        value = self.resolve(name, scope)
        if isinstance(value, type) and issubclass(value, BaseException):
            return not issubclass(value, (OSError, SyntaxError, UnicodeError,
                                          getattr(builtins, "BaseExceptionGroup", OSError)))
        if not isinstance(value, ast.ClassDef) or value in seen:
            return False
        return (not value.decorator_list and not value.keywords and len(value.bases) == 1
                and isinstance(value.bases[0], ast.Name)
                and all(isinstance(n, ast.Pass) or isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
                        for n in value.body)
                and self.legacy_constructor(value.bases[0].id, self.node_scopes[value], (*seen, value)))

    def exception_identity(self, name, scope, seen=()):
        """Class-object ancestry, separate from the result of constructing it."""
        if self.opaque_class_identities:
            return ()
        value = self.resolve(name, scope)
        if isinstance(value, type) and issubclass(value, BaseException):
            return builtin_exception_chain(value.__name__)
        if (not isinstance(value, ast.ClassDef) or value in seen
                or value.decorator_list or value.keywords or getattr(value, "type_params", ())):
            return ()
        outer = self.node_scopes[value]
        chains = []
        for base in value.bases:
            if not isinstance(base, ast.Name):
                return ()
            parent = self.resolve(base.id, outer)
            if isinstance(parent, ast.ClassDef) and any(
                    "__init_subclass__" in _bound(part) for part in ast.walk(parent)):
                return ()  # a source hook can change the newly created class
            chain = self.exception_identity(base.id, outer, (*seen, value))
            if not chain:
                return ()
            chains.append(chain)
        if not chains:
            return ()
        identity = self.definitions[value].removesuffix(".<body>")
        return (identity, *dict.fromkeys(part for chain in chains for part in chain))

    def handlers(self, node):
        result, child = [], node
        scope = self.node_scopes[node]
        parent = self.parents.get(child)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if self.definitions.get(parent) == scope:
                    break
            if isinstance(parent, (ast.Try, getattr(ast, "TryStar", ast.Try))) and child in parent.body:
                for order, handler in enumerate(parent.handlers):
                    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
                    names = [n.id for n in types if isinstance(n, ast.Name)]
                    chains = [self.exception_chain(name, scope) for name in names]
                    identities = [self.exception_identity(name, scope) for name in names]
                    unresolved = handler.type is not None and (
                        isinstance(parent, getattr(ast, "TryStar", ())) or len(names) != len(types)
                        or any(not c for c in identities))
                    result.append({"try_span": _span(parent), "handler_span": _span(handler),
                                   "classes": ["BaseException"] if handler.type is None else [
                                       c[0] if c else name for name, c in zip(names, chains)],
                                   "bare": handler.type is None, "unresolved": unresolved,
                                   "class_ids": ["builtins.BaseException"] if handler.type is None else [
                                       chain[0] if chain else None for chain in identities],
                                   "binding": handler.name, "order": order})
            child, parent = parent, self.parents.get(parent)
        return result


class _InventoryScan:
    """The one physical-node discovery pass, including the audited SV projection."""

    def __init__(self, tree, classes, res, filename="<program>", user_classes=None, *, enrich=False, rebound=frozenset(),
                 fold_constants=False):
        self.tree, self.classes, self.res, self.filename = tree, tuple(classes), res, str(filename)
        self.index = _SourceIndex(tree, filename)
        self.enrich = enrich
        self.condition_tables = {}
        self.records, self.gaps, self.nodes, self.obligations, self.sites = {}, [], [], [], []
        self.refusal = None
        self.projection = _CensusProjection(tree, self.classes, res, filename, user_classes, rebound,
                                              fold_constants) if res else None

    def _record_obligation(self, node, qualname):
        """Preserve the old census plumbing entry without another source walker."""
        if self.projection is None:
            raise _Refuse("obligation recording requires a resolved program")
        self.projection.parent = getattr(self, "parent", self.index.parents)
        self.projection._record_obligation(node, qualname)
        self.obligations = self.projection.obligations

    def _gap(self, record, reason, **fields):
        self.gaps.append({"site": record["id"], "function": record["function"],
                          "line": record["line"], "col": record["col"], "reason": reason, **fields})

    def _record(self, node, kind, operands=None, op=None):
        if node in self.records:
            return self.records[node]
        scope = self.index.node_scopes[node]
        function = self.index.scopes[scope]["function"]
        record = {"id": scope + ":" + ":".join(map(str, _span(node))) + ":" + kind,
                  "file_name": self.filename, "function": function, "scope": scope,
                  "line": node.lineno, "col": node.col_offset,
                  "end_line": node.end_lineno, "end_col": node.end_col_offset,
                  "kind": kind, "classes": [], "class_ids": {}, "conditions": {},
                  "operands": operands or {}, "op": op,
                  "handlers": self.index.handlers(node), "provenance": [], "callee": None}
        self.records[node] = record
        self.index.scopes[scope]["sites"].append(record["id"])
        return record

    def _unknown(self, node, reason, *, domain=None, excluded_classes=(), contract=None):
        record = self._record(node, "unknown")
        if "*" not in record["classes"]:
            record["classes"].append("*")
        record["class_ids"]["*"] = None
        coverage = unknown_case(contract, tuple(record["operands"]), reason,
                                domain=domain, excluded_classes=excluded_classes)
        if self.enrich:
            record["conditions"].setdefault("*", []).append(coverage)
        record["provenance"].append({"kind": "unknown", "reason": reason,
                                      "domain": coverage["domain"],
                                      "excluded_classes": list(excluded_classes)})
        self._gap(record, reason, domain=coverage["domain"], excluded_classes=list(excluded_classes))
        return record

    def _contract(self, node, qualname, operands, op=None, *, applicability=None):
        contract = get(qualname)
        if contract is None:
            self._record(node, "unknown", operands, op)
            return self._unknown(node, "missing contract: " + qualname, contract=qualname, domain=applicability)
        record = self._record(node, "operation", operands, op)
        record["provenance"].append({"kind": "operation_type_contract", "contract": qualname})
        if self.enrich and operands:
            key = (qualname, tuple(operands), op)
            if key not in self.condition_tables:
                self.condition_tables[key] = condition_table(*key)
            # Each physical site still owns independently mutable JSON cases.
            table = copy.deepcopy(self.condition_tables[key])
            for exception, cases in table["conditions"].items():
                for case in cases:
                    if applicability is not None and case.get("unknown"):
                        case = {**case, "domain": canon(and_(applicability, from_canon(case["domain"])))}
                    record["conditions"].setdefault(exception, []).append(case)
            for gap in table["gaps"]:
                fields = {"contract": qualname, "class": gap["class"]}
                if "domain" in gap:
                    domain = from_canon(gap["domain"])
                    fields["domain"] = canon(and_(applicability, domain)) if applicability is not None else gap["domain"]
                self._gap(record, gap["reason"], **fields)
        elif self.enrich:
            self._unknown(node, "zero-operand call condition binding is unresolved", contract=qualname)
        for claim in (*contract.raises, *contract.other_raises):
            if claim.exc not in EXCLUDED_EXCEPTIONS and not (claim.tested and claim.condition == "never"):
                if claim.exc not in record["classes"]:
                    record["classes"].append(claim.exc)
                chain = builtin_exception_chain(claim.exc)
                record["class_ids"][claim.exc] = chain[0] if chain else None
        if contract.may_raise_other:
            # Named families have their own exact-or-unknown coverage above.
            claims = tuple(dict.fromkeys(c.exc for c in (*contract.raises, *contract.other_raises)))
            self._unknown(node, "other exception families remain open: " + qualname,
                          excluded_classes=claims, domain=applicability, contract=qualname)
        return record

    def _callee(self, call, scope):
        func = call.func
        if isinstance(func, ast.Name):
            value = self.index.resolve(func.id, scope)
            if isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef)):
                suspended = isinstance(value, ast.AsyncFunctionDef) or any(
                    isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(value))
                return None if value.decorator_list or suspended else self.index.definitions[value]
            if isinstance(value, ast.ClassDef) and not value.decorator_list and not value.keywords:
                target = self.index.definitions[value].removesuffix(".<body>") + ".__init__"
                return target if target in self.index.scopes else None
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.attr in self.index.assigned_attributes:
                return None
            owner = self.index.resolve(func.value.id, scope)
            if isinstance(owner, ast.ClassDef) and not owner.decorator_list and not owner.keywords:
                target = self.index.definitions[owner].removesuffix(".<body>") + "." + func.attr
                if target in self.index.scopes:
                    method = next((n for n, identity in self.index.definitions.items() if identity == target), None)
                    bindings = self.index.bindings[self.index.definitions[owner]].get(func.attr, ())
                    if (isinstance(method, ast.FunctionDef) and len(bindings) == 1 and bindings[0] is method and all(
                            isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod")
                            and self.index.resolve(d.id, self.index.node_scopes[method]) is getattr(builtins, d.id)
                            for d in method.decorator_list) and not any(
                                isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(method))):
                        return target
        return None

    def _call_contract(self, node, scope):
        if (node.keywords or any(isinstance(n, ast.Starred) for n in node.args)):
            return None, {}
        names = ["receiver", *["argument" if i == 0 else f"argument_{i + 1}"
                              for i in range(max(0, len(node.args) - 1))]]
        operands = dict(zip(names, (ast.unparse(n) for n in node.args)))
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in self.index.assigned_attributes:
                return None, operands
            value = self.index.resolve(func.id, scope)
            if value is getattr(builtins, func.id, None) and value is not None:
                return func.id, operands
            if isinstance(value, ast.ImportFrom) and value.module is not None:
                name = next((a.name for a in value.names if (a.asname or a.name) == func.id), None)
                return value.module + "." + name if name else None, operands
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.attr in self.index.assigned_attributes:
                return None, operands
            value = self.index.resolve(func.value.id, scope)
            if isinstance(value, ast.Import):
                alias = next((a for a in value.names if (a.asname or a.name.split(".")[0]) == func.value.id), None)
                if alias:
                    module = alias.name if alias.asname else alias.name.split(".")[0]
                    return module + "." + func.attr, operands
        return None, operands

    def _call_binding_gap(self, node, scope, callee):
        target = next((n for n, identity in self.index.definitions.items() if identity == callee), None)
        if not isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "source call signature is unresolved"
        if isinstance(node.func, ast.Name) and isinstance(self.index.resolve(node.func.id, scope), ast.ClassDef):
            return "constructor __new__/__init__ return protocol needs a contract adapter"
        _, _, reason = bind_source_call(target, node)
        return "source call binding is unresolved: " + reason if reason else None

    def _discover(self, node):
        scope = self.index.node_scopes.get(node)
        if scope is None:
            return
        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            left, right = ((node.left, node.right) if isinstance(node, ast.BinOp)
                           else (node.target, node.value))
            self._contract(node, "protocol.binary_operator",
                           {"receiver": ast.unparse(left), "argument": ast.unparse(right)}, type(node.op).__name__)
        elif isinstance(node, ast.UnaryOp):
            qualname = "protocol.truthiness" if isinstance(node.op, ast.Not) else "protocol.unary_operator"
            self._contract(node, qualname, {"receiver": ast.unparse(node.operand)},
                           None if isinstance(node.op, ast.Not) else type(node.op).__name__)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            self._contract(node, "protocol.attribute_access", {"receiver": ast.unparse(node.value)},
                           "Attribute:" + node.attr)
        elif isinstance(node, ast.Attribute):
            self._unknown(node, "attribute store/delete dispatch needs a typed contract adapter")
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice) or not isinstance(node.ctx, ast.Load):
                self._unknown(node, "slice/store/delete subscription needs a typed contract adapter")
            else:
                owners = ("list", "tuple", "str", "dict")
                receiver = var("receiver", PY)
                for owner in owners:
                    self._contract(node, owner + ".__getitem__",
                                   {"receiver": ast.unparse(node.value), "argument": ast.unparse(node.slice)},
                                   applicability=typeis(receiver, owner))
                self._unknown(node, "receiver dispatch outside the admitted subscription types",
                              domain=not_(or_(*(typeis(receiver, owner) for owner in owners))))
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                self._unknown(node, "chained comparison needs evaluation-subsite identities")
            elif not isinstance(node.ops[0], (ast.Is, ast.IsNot)):
                contains = isinstance(node.ops[0], (ast.In, ast.NotIn))
                left, right = (node.comparators[0], node.left) if contains else (node.left, node.comparators[0])
                self._contract(node, "protocol.contains_fallback" if contains else "protocol.comparison",
                               {"receiver": ast.unparse(left), "argument": ast.unparse(right)},
                               type(node.ops[0]).__name__)
        elif isinstance(node, ast.BoolOp):
            self._unknown(node, "short-circuit truth tests need evaluation-subsite identities")
        elif isinstance(node, ast.FormattedValue):
            self._contract(node, "syntax.FormattedValue",
                           {"receiver": ast.unparse(node.value),
                            "argument": ast.unparse(node.format_spec) if node.format_spec else repr(""),
                            "argument_2": repr(node.conversion)})
        elif isinstance(node, ast.Call):
            callee = self._callee(node, scope)
            if callee is not None:
                record = self._record(node, "call", {f"argument_{i}": ast.unparse(arg)
                                                     for i, arg in enumerate(node.args)})
                record["operands"].update({"keyword_" + key.arg if key.arg is not None else f"keywords_{i}": ast.unparse(key.value)
                                           for i, key in enumerate(node.keywords)})
                record["callee"] = callee
                record["provenance"].append({"kind": "source_call", "callee": callee})
                gap = self._call_binding_gap(node, scope, callee)
                if gap is not None:
                    self._unknown(node, gap)
            else:
                qualname, operands = self._call_contract(node, scope)
                if qualname is not None:
                    self._contract(node, qualname, operands)
                else:
                    self._record(node, "unknown", {"callable": ast.unparse(node.func), **{
                        f"argument_{i}": ast.unparse(arg) for i, arg in enumerate(node.args)}})
                    self._unknown(node, "call dispatch needs a resolved callee contract (recursive/decorated/rebound or dynamic)")
        elif isinstance(node, ast.Raise):
            record = self._record(node, "raise")
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            chain = self.index.exception_chain(target.id, scope) if isinstance(target, ast.Name) else ()
            if chain and chain[0] not in EXCLUDED_EXCEPTIONS:
                record["classes"].append(chain[0])
                identities = self.index.exception_identity(target.id, scope)
                record["class_ids"][chain[0]] = identities[0] if identities else None
                ordinary = self.index.legacy_constructor(target.id, scope)
                supported = (isinstance(node.exc, ast.Call) and not node.exc.keywords
                             and not any(isinstance(arg, ast.Starred) for arg in node.exc.args)
                             and ordinary)
                record["provenance"].append({"kind": "explicit_raise", "class_chain": list(chain),
                                             "class_chain_ids": list(identities),
                                             "legacy_constructor": supported})
                if not ordinary:
                    self._unknown(node, "exception construction/result class is not statically certified")
            elif chain:
                record["provenance"].append({"kind": "explicit_raise", "excluded": chain[0]})
            else:
                self._unknown(node, "explicit raise class or re-raise binding is unresolved")
            if node.cause is not None:
                self._unknown(node, "exception chaining is deferred in exit summaries v1")
        elif isinstance(node, ast.Assert):
            record = self._record(node, "raise", {"receiver": ast.unparse(node.test)})
            record["classes"] = ["AssertionError"]
            record["class_ids"]["AssertionError"] = "builtins.AssertionError"
            record["provenance"].append({"kind": "explicit_raise", "reason": "assert syntax"})
            self._contract(node, "protocol.truthiness", {"receiver": ast.unparse(node.test)})
        elif isinstance(node, ast.For):
            self._contract(node, "protocol.iteration", {"receiver": ast.unparse(node.iter)})
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self._unknown(node, "comprehension iteration/filter dispatch needs evaluation-subsite identities")
        elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
            self._contract(node, "protocol.truthiness", {"receiver": ast.unparse(node.test)})
        elif isinstance(node, ast.Try) and node.finalbody:
            self._unknown(node, "finally outcome is deferred in exit summaries v1")
        elif isinstance(node, getattr(ast, "TryStar", ())):
            self._unknown(node, "exception-group handling is deferred in exit summaries v1")
        elif isinstance(node, ast.ClassDef):
            self._unknown(node, "class construction/metaclass dispatch is unresolved")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.decorator_list or getattr(node, "type_params", ()):
                self._unknown(node, "definition-time decorator/type-parameter dispatch is unresolved")
        elif isinstance(node, (ast.With, ast.AsyncWith, ast.AsyncFor, ast.Await, ast.Yield, ast.YieldFrom, ast.Match)):
            self._unknown(node, type(node).__name__ + " implicit dispatch/suspension is unresolved")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            self._unknown(node, "import initialization is outside this scope's source inventory")
        elif isinstance(node, ast.Starred) or (
                isinstance(node, (ast.Tuple, ast.List)) and isinstance(node.ctx, ast.Store)):
            self._unknown(node, "unpacking protocol needs a typed contract adapter")
        elif isinstance(node, (ast.Dict, ast.Set)):
            keys = node.keys if isinstance(node, ast.Dict) else node.elts
            if any(not isinstance(key, ast.Constant) for key in keys):
                self._unknown(node, "hashing/unpacking during container construction needs a typed adapter")

    def run(self):
        projection = self.projection
        if projection is not None:
            bound = {}
            for node in ast.walk(self.tree):
                for name in _bound(node):
                    bound[name] = bound.get(name, 0) + 1
                if isinstance(node, ast.ClassDef):
                    projection.user_classes[node.name] = node
            try:
                projection.prepare(self.index.parents, bound)
            except _Refuse as error:
                self.refusal, projection = str(error), None
        for node in ast.walk(self.tree):
            self._discover(node)
            if projection is not None and self.res.live(node):
                try:
                    self.res.check(node)
                    before = len(projection.sites)
                    projection.visit(node)
                    if len(projection.sites) != before and node not in self.records:
                        self._unknown(node, "callable reference counted by the audited closed-program census")
                except (Top, _Refuse) as error:
                    self.refusal = ("does not resolve: " if isinstance(error, Top) else "") + str(error)
                    projection = None
        if self.projection is not None:
            self.sites, self.nodes = self.projection.sites, self.projection.nodes
            self.obligations = self.projection.obligations
        return self.sites


class _Inventory:
    def __init__(self):
        self.report = {"schema_version": 2, "sites": [], "scopes": {}, "gaps": [],
                       "excluded_exceptions": list(EXCLUDED_EXCEPTIONS)}
        self.refusal, self.sites, self.obligations = None, [], []
        self.indices = {}


def _stored_attributes(resolvers) -> frozenset[str]:
    """Attribute names stored, deleted or set by constant name anywhere in a scanned file: ``mod.n = ..`` rebinds a module global
    from outside, so a name spelled like one gets no fixed type (``fixed_types.py``)."""
    names: set[str] = set()
    for sub in resolvers:
        for node in ast.walk(sub.tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.attr)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr") \
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                names.add(node.args[1].value)
    return frozenset(names)


def _task_root(filename: str, declared_files) -> str | None:
    """The program's own directory when the program is the ONE declared file (the grader passes only ``main.py``).

    CPython puts that directory first on ``sys.path`` at run time, so a sibling module the program imports is program code
    (it shadows the stdlib, packages and relative imports resolve from it): the Resolver reads it by the same rules as
    ``determinism.py`` (``_task_root``), and refuses a sibling that collides with an allowed module."""
    if len(declared_files) == 1 and not filename.startswith("<") and (
            os.path.realpath(filename) == os.path.realpath(os.fspath(declared_files[0]))):
        return os.path.dirname(os.path.realpath(filename))
    return None


def _source_inventory(source, filename="<program>", *, declared_files=(), census_classes=None,
                      include_conditions=False):
    inventory = _Inventory()
    try:
        tree = ast.parse(source, filename)
    except (SyntaxError, ValueError) as error:
        inventory.refusal = "parse: " + str(error)
        inventory.report["gaps"].append({"reason": inventory.refusal})
        return inventory
    try:
        # the census view reads the sibling modules CPython would import beside the program; the raw inventory (schema 2) does not
        res = Resolver(tree, MODULES, filename=filename, _declared_files=declared_files,
                       _task_root=_task_root(filename, declared_files) if census_classes is not None else None)
        sources = [(sub.tree, sub, sub.filename) for sub in res.all_resolvers()]
        program_defs = frozenset(n.name for body, _, _ in sources for n in ast.walk(body)
                                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        for _, sub, _ in sources:                  # every file of the program sees every method name the program defines
            sub.program_defs = program_defs
    except Top as error:
        inventory.refusal = "does not resolve: " + str(error)
        sources = [(tree, None, filename)]
    user_classes = {node.name: node for body, _, _ in sources for node in ast.walk(body)
                    if isinstance(node, ast.ClassDef)}
    rebound = _stored_attributes([r for _, r, _ in sources if r is not None]) if census_classes is not None else frozenset()
    for body, resolver, name in sources:
        scan = _InventoryScan(body, SITE_KINDS if census_classes is None else census_classes, resolver, name,
                              user_classes, enrich=census_classes is None or include_conditions, rebound=rebound,
                              fold_constants=census_classes is not None)
        scan.run()
        inventory.indices[str(name)] = scan.index
        inventory.report["sites"].extend(scan.records.values())
        inventory.report["scopes"].update(scan.index.scopes)
        inventory.report["gaps"].extend(scan.gaps)
        inventory.sites.extend(scan.sites)
        inventory.obligations.extend(scan.obligations)
        if scan.refusal is not None:
            inventory.refusal = scan.refusal
    return inventory


def exception_inventory(source: str, filename: str = "<program>", *, declared_files=()) -> dict:
    """Return schema 2's unordered physical-site set from ORIGINAL source text.

    Canonical condition cases retain their operand type assumptions. Unknown
    dispatch is a wildcard site; empty sites or predicates never certify absence.
    Resolved source calls refer to scopes, without evaluation/call-path DFS.
    """
    from tharkun.solve.exception_exits import exit_summaries

    inventory = _source_inventory(source, filename, declared_files=declared_files)
    inventory.report["exit_summaries"] = exit_summaries(inventory)
    return inventory.report
