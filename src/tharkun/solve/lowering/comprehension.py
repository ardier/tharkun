"""Exact comprehension lowering ([internal ref]): list / set / dict comprehensions become loops.

The analyses that do not model a comprehension (``solve/absint``) run on the lowered
program instead.  "Exact" is the whole point -- a wrong TRUE costs 32 points -- so the
rewrite must be the SAME program as far as any observer goes: the same value, the same
exceptions, the same order of every side effect.  What that takes, and what is refused:

* **Scope.**  A comprehension's loop variables live in its own scope: they do not leak
  and do not clobber an enclosing name.  Each loop variable gets a FRESH name, and its
  uses are renamed with a scope-aware renamer (``scopes.rename_names``: a nested
  comprehension or lambda that rebinds the name keeps its own).  A comprehension in a
  class body is left alone: class scope resolves names differently.
* **The first iterable is evaluated once, in the enclosing scope, before the loop.**  A
  ``for`` statement does exactly that; the later iterables are evaluated per iteration
  of the loop above them, where the loop variables are visible (renamed).
* **Evaluation order.**  A comprehension is an expression; its loops are statements.
  Hoisting the loops in front of the statement is exact only if everything that would
  have been evaluated BEFORE the comprehension still is: every earlier operand (and a
  call's callee, as the bound method it would have loaded) that is not provably stable
  is first frozen into a temporary, in order.  A comprehension that is only conditionally
  evaluated (the right of ``and``/``or``, a branch of a conditional expression) becomes
  an ``if`` statement so it is evaluated exactly when it would have been.  Shapes whose
  order this module does not model (``*args``, ``**kw``, chained comparisons with a
  comprehension after the first operator, f-strings, assignment expressions, ``await``,
  ``yield``, a comprehension in a loop test, ``with`` item, default value, decorator,
  augmented assignment, lambda or class body) are REFUSED: the comprehension stays where it
  is, and the analysis that does not model it aborts as it did before.
* **Generator expressions are lazy.**  Lowering one to an eager loop changes WHEN its
  elements are evaluated relative to the code that consumes them (``any``, ``all``,
  ``next``, ``sum``, ``min``, ``max``, ``zip``, ``dict`` and ``set`` interleave with it and
  can stop, raise or have effects part-way).  A generator expression is lowered only as the
  sole argument of a consumer that drains it completely before doing anything else, which
  makes it identical to the list comprehension: ``list(g)``, ``tuple(g)``, ``sorted(g, ...)``
  (the builtins, not rebound anywhere) and ``"...".join(g)`` (``str.join`` builds a
  sequence first).  Every other generator expression is refused.
* **Dict and set comprehensions** evaluate the key, then the value, then insert: the
  lowering keeps that order with temporaries (``acc[k] = v`` would evaluate ``v`` first).

Every node of the result carries the path and position of the node of the ORIGINAL program
it came from (``provenance.py``): a node taken over keeps its own; a node invented here
(the accumulator, a loop, a filter, a temporary) stands for the original node it implements
and is marked with a role.  The caller's findings therefore refer to original sites.

Stdlib only; imports nothing from tharkun outside ``solve/lowering``.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .provenance import PATH_ATTR, annotate, stand_for
from .scopes import COMPREHENSIONS, Names, rename_names, target_names

COMPS = COMPREHENSIONS

#: consumers that drain a generator completely, with nothing observable in between
EAGER_BUILTINS = frozenset({"list", "tuple", "sorted"})


class _Refuse(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Refusal:
    """A comprehension that was left in the program, and why."""

    line: int
    col: int
    kind: str
    reason: str


@dataclass
class Lowered:
    """The result of :func:`lower_comprehensions`."""

    #: the lowered module (a copy; the input is not modified)
    tree: ast.Module
    #: fresh loop-variable name -> the original name it stands for
    renamed: Dict[str, str] = field(default_factory=dict)
    #: fresh names that stand for no original name (accumulators, temporaries)
    synthetic: FrozenSet[str] = frozenset()
    #: fresh loop-variable name -> the path (in the ORIGINAL program) of the comprehension it
    #: is the variable of
    scopes: Dict[str, tuple] = field(default_factory=dict)
    #: how many comprehensions were turned into loops
    lowered: int = 0
    #: the comprehensions that were not
    refused: List[Refusal] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.lowered > 0

    def restore_names(self, text: str) -> str:
        """``text`` (an expression or a message about the lowered program) with every fresh
        LOOP variable replaced by the name it stands for.  Synthetic names are left as they
        are: they name nothing in the original program."""
        import re

        if not self.renamed:
            return text
        pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(self.renamed, key=len, reverse=True)) + r")\b")
        return pattern.sub(lambda m: self.renamed[m.group(1)], text)

    def mentions_synthetic(self, text: str) -> bool:
        return any(name in text for name in self.synthetic)

    def at_site(self, text: str, path) -> Optional[str]:
        """``text`` (a fact, a predicate, an expression about the lowered program's state at
        the node whose ORIGINAL path is ``path``) in terms of the original program's names;
        ``None`` when it cannot be said there.

        A fresh loop variable stands for the original name only INSIDE the comprehension it
        belongs to; at a site elsewhere the original name is some other variable (or none), so
        a fact about the fresh one is not a fact about it.  A synthetic name (accumulator,
        temporary) names nothing in the original program.  Both make the text ``None``."""
        import re

        names = set(self.renamed) | set(self.synthetic)
        if not names:
            return text
        found = {
            m.group(0)
            for m in re.finditer(r"\b_lw\d+_\w*", text)
            if m.group(0) in names
        }
        for name in found:
            if name in self.synthetic:
                return None
            scope = self.scopes.get(name)
            if scope is None or path is None or not _inside_scope(tuple(path), tuple(scope)):
                return None
        return self.restore_names(text)

    def refusal_at(self, line: int) -> Optional[Refusal]:
        for refusal in self.refused:
            if refusal.line == line:
                return refusal
        return None


def _inside_scope(path: tuple, scope: tuple) -> bool:
    """Is the node at ``path`` evaluated where the loop variables of the comprehension at ``scope``
    are bound?  That is the element, the key and value, the conditions and the later iterables:
    strictly INSIDE the comprehension, but not its first iterable (evaluated in the enclosing
    scope) and not a node that stands for the whole comprehension (the loop and accumulator the
    lowering invents carry the comprehension's own path)."""
    if path[: len(scope)] != scope or len(path) == len(scope):
        return False
    return tuple(path[len(scope): len(scope) + 2]) != (("generators", 0), ("iter", None))


def has_comp(node: ast.AST) -> bool:
    return any(isinstance(n, COMPS) for n in ast.walk(node))


def comps_in(node: ast.AST) -> List[ast.AST]:
    return [n for n in ast.walk(node) if isinstance(n, COMPS)]


def lower_comprehensions(tree: ast.Module) -> Lowered:
    """Lower the list / set / dict comprehensions of ``tree`` (see the module docstring).

    ``tree`` is not modified.  A program that can reach its own namespace (``globals()``,
    ``locals()``, ``exec``, ...) is returned unchanged: fresh names would be observable."""
    work = copy.deepcopy(tree)
    annotate(work)
    names = Names(work)
    lowerer = _Lowerer(names)
    if names.reflective:
        for comp in comps_in(work):
            lowerer.refused.append(_refusal(comp, "the program can reach its own namespace"))
        return Lowered(work, refused=lowerer.refused)
    work.body = lowerer.block(work.body, "module")
    ast.fix_missing_locations(work)
    return Lowered(
        work,
        renamed=dict(names.renamed),
        synthetic=frozenset(names.synthetic),
        scopes=dict(names.scope_path),
        lowered=lowerer.lowered,
        refused=lowerer.refused,
    )


def _refusal(comp: ast.AST, reason: str) -> Refusal:
    return Refusal(
        getattr(comp, "lineno", 0), getattr(comp, "col_offset", 0), type(comp).__name__, reason,
    )


class _Lowerer:
    def __init__(self, names: Names):
        self.names = names
        self.lowered = 0
        self.refused: List[Refusal] = []
        self.scope = "module"

    # ------------------------------------------------------------------ bookkeeping
    def _mark(self) -> Tuple[int, int]:
        return self.lowered, len(self.refused)

    def _reset(self, mark: Tuple[int, int]) -> None:
        self.lowered = mark[0]
        del self.refused[mark[1]:]

    def _refuse_all(self, node: ast.AST, reason: str) -> None:
        for comp in comps_in(node):
            self.refused.append(_refusal(comp, reason))

    # ------------------------------------------------------------------ node builders
    def _name(self, ident: str, anchor: ast.AST, role: str, ctx=None) -> ast.Name:
        return stand_for(ast.Name(id=ident, ctx=ctx or ast.Load()), anchor, role)  # type: ignore[return-value]

    def _assign(self, ident: str, value: ast.expr, anchor: ast.AST, role: str) -> ast.Assign:
        stmt = ast.Assign(targets=[self._name(ident, anchor, role, ast.Store())], value=value, type_comment=None)
        return stand_for(stmt, anchor, role)  # type: ignore[return-value]

    def _like(self, node: ast.AST, **changes) -> ast.AST:
        """A shallow copy of ``node`` with some fields replaced: still the original node as
        far as path and position go."""
        new = copy.copy(node)
        for key, value in changes.items():
            setattr(new, key, value)
        return new

    # ------------------------------------------------------------------ statements
    def block(self, stmts: Sequence[ast.stmt], scope: str) -> List[ast.stmt]:
        saved, self.scope = self.scope, scope
        try:
            out: List[ast.stmt] = []
            for stmt in stmts:
                out.extend(self.stmt(stmt))
            return out
        finally:
            self.scope = saved

    def _function(self, fn) -> ast.stmt:
        # decorators, defaults and annotations run in the enclosing scope, at definition time
        header = [*fn.decorator_list, *fn.args.defaults, *[d for d in fn.args.kw_defaults if d is not None]]
        if fn.returns is not None:
            header.append(fn.returns)
        for arg in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs, fn.args.vararg, fn.args.kwarg]:
            if arg is not None and arg.annotation is not None:
                header.append(arg.annotation)
        for node in header:
            self._refuse_all(node, "in a decorator, default value or annotation")
        fn.body = self.block(fn.body, "function")
        return fn

    def _class(self, cls: ast.ClassDef) -> ast.stmt:
        for node in [*cls.decorator_list, *cls.bases, *[k.value for k in cls.keywords]]:
            self._refuse_all(node, "in a class header")
        cls.body = [self._class_stmt(s) for s in cls.body]
        return cls

    def _class_stmt(self, stmt: ast.stmt) -> ast.stmt:
        """A class body keeps its comprehensions (class scope resolves names differently),
        but the functions and classes defined in it are lowered like any others."""
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._function(stmt)  # type: ignore[return-value]
        if isinstance(stmt, ast.ClassDef):
            return self._class(stmt)
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                self._class_stmt(child)
        # (own expressions only: nested defs were handled above)
        for field_name, value in ast.iter_fields(stmt):
            values = value if isinstance(value, list) else [value]
            for v in values:
                if isinstance(v, ast.expr):
                    self._refuse_all(v, "in a class body")
        # nested statement lists keep their structure; their defs were lowered in place
        return stmt

    def _field(self, stmt: ast.stmt, field_name: str) -> List[ast.stmt]:
        """Lower the comprehensions of ``stmt.<field_name>``; return the statements that must
        run before ``stmt``.  On a refusal the statement is left exactly as it was."""
        expr = getattr(stmt, field_name)
        if expr is None or not has_comp(expr):
            return []
        mark = self._mark()
        try:
            pre, new = self.expr(expr)
        except _Refuse as refusal:
            self._reset(mark)
            self._refuse_all(expr, refusal.reason)
            return []
        setattr(stmt, field_name, new)
        return pre

    def stmt(self, s: ast.stmt) -> List[ast.stmt]:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return [self._function(s)]
        if isinstance(s, ast.ClassDef):
            return [self._class(s)]
        if isinstance(s, ast.If):
            pre = self._field(s, "test")
            s.body = self.block(s.body, self.scope)
            s.orelse = self.block(s.orelse, self.scope)
            return pre + [s]
        if isinstance(s, (ast.For, ast.AsyncFor)):
            if has_comp(s.target):
                self._refuse_all(s.target, "in a loop target")
            pre = self._field(s, "iter")
            s.body = self.block(s.body, self.scope)
            s.orelse = self.block(s.orelse, self.scope)
            return pre + [s]
        if isinstance(s, ast.While):
            # the test runs again every iteration: a hoisted prelude would run once
            self._refuse_all(s.test, "in a loop test")
            s.body = self.block(s.body, self.scope)
            s.orelse = self.block(s.orelse, self.scope)
            return [s]
        if isinstance(s, (ast.With, ast.AsyncWith)):
            for item in s.items:
                self._refuse_all(item, "in a with item")
            s.body = self.block(s.body, self.scope)
            return [s]
        if isinstance(s, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            s.body = self.block(s.body, self.scope)
            for handler in s.handlers:
                if handler.type is not None:
                    self._refuse_all(handler.type, "in an except clause")
                handler.body = self.block(handler.body, self.scope)
            s.orelse = self.block(s.orelse, self.scope)
            s.finalbody = self.block(s.finalbody, self.scope)
            return [s]
        if isinstance(s, ast.Match):
            self._refuse_all(s.subject, "in a match statement")
            for case in s.cases:
                if case.guard is not None:
                    self._refuse_all(case.guard, "in a match guard")
                case.body = self.block(case.body, self.scope)
            return [s]
        if isinstance(s, ast.Assign):
            for target in s.targets:
                if has_comp(target):
                    self._refuse_all(s, "in an assignment target")
                    return [s]
            return self._field(s, "value") + [s]
        if isinstance(s, ast.Return):
            return self._field(s, "value") + [s]
        if isinstance(s, ast.Expr):
            return self._field(s, "value") + [s]
        if isinstance(s, ast.Raise):
            if s.cause is not None and has_comp(s.cause):
                self._refuse_all(s, "in a raise ... from clause")
                return [s]
            return self._field(s, "exc") + [s]
        if isinstance(s, ast.Assert):
            if s.msg is not None and has_comp(s.msg):
                self._refuse_all(s, "in an assertion message")
                return [s]
            return self._field(s, "test") + [s]
        # AugAssign, AnnAssign, Delete, Global, ...: not modelled
        if has_comp(s):
            self._refuse_all(s, f"in a {type(s).__name__} statement")
        return [s]

    # ------------------------------------------------------------------ expressions
    def stable(self, e: ast.expr) -> bool:
        """Evaluating ``e`` earlier or later gives the same value and has no effect: nothing
        the code in between can do reaches it."""
        if isinstance(e, ast.Constant):
            return True
        if isinstance(e, ast.Name):
            return isinstance(e.ctx, ast.Load) and (
                e.id in self.names.synthetic or self.names.stable_builtin(e.id)
            )
        if isinstance(e, ast.Attribute):
            # a method of a literal str/bytes, or of a list/dict/set this module made itself
            # (nothing else holds it): the lookup cannot change
            return (
                isinstance(e.value, ast.Constant) and isinstance(e.value.value, (str, bytes))
            ) or (isinstance(e.value, ast.Name) and e.value.id in self.names.synthetic)
        if isinstance(e, ast.UnaryOp):
            return isinstance(e.op, (ast.USub, ast.UAdd)) and isinstance(e.operand, ast.Constant)
        if isinstance(e, ast.Tuple):
            return all(self.stable(x) for x in e.elts)
        return False

    def ordered(self, children: Sequence[ast.expr]) -> Tuple[List[ast.stmt], List[ast.expr]]:
        """Lower sibling expressions that are evaluated in this order.  Everything before the
        last child that needs statements is evaluated, in order, before those statements."""
        results = [self.expr(c) for c in children]
        last = max((i for i, (pre, _) in enumerate(results) if pre), default=-1)
        if last < 0:
            return [], [new for _, new in results]
        prelude: List[ast.stmt] = []
        new_children: List[ast.expr] = []
        for i, (pre, new) in enumerate(results):
            prelude.extend(pre)
            if i < last and not self.stable(new):
                temp = self.names.fresh("t")
                prelude.append(self._assign(temp, new, new, "freeze"))
                new = self._name(temp, new, "freeze")
            new_children.append(new)
        return prelude, new_children

    def expr(self, e: ast.expr) -> Tuple[List[ast.stmt], ast.expr]:
        if not has_comp(e):
            return [], e
        if isinstance(e, (ast.ListComp, ast.SetComp, ast.DictComp)):
            return self.comp(e)
        if isinstance(e, ast.GeneratorExp):
            raise _Refuse(
                "generator expression: lazy, so only the sole argument of list/tuple/sorted or "
                "of str.join on a literal is lowered"
            )
        if isinstance(e, ast.Lambda):
            raise _Refuse("comprehension inside a lambda")
        if isinstance(e, ast.BinOp):
            pre, (left, right) = self.ordered([e.left, e.right])
            return pre, self._like(e, left=left, right=right)  # type: ignore[return-value]
        if isinstance(e, ast.UnaryOp):
            pre, (operand,) = self.ordered([e.operand])
            return pre, self._like(e, operand=operand)  # type: ignore[return-value]
        if isinstance(e, ast.Compare):
            if len(e.ops) > 1 and any(has_comp(c) for c in e.comparators[1:]):
                raise _Refuse("comprehension after the first operator of a chained comparison")
            pre, (left, first) = self.ordered([e.left, e.comparators[0]])
            return pre, self._like(e, left=left, comparators=[first, *e.comparators[1:]])  # type: ignore[return-value]
        if isinstance(e, ast.Call):
            return self.call(e)
        if isinstance(e, ast.Attribute):
            pre, (value,) = self.ordered([e.value])
            return pre, self._like(e, value=value)  # type: ignore[return-value]
        if isinstance(e, ast.Subscript):
            pre, (value, index) = self.ordered([e.value, e.slice])
            return pre, self._like(e, value=value, slice=index)  # type: ignore[return-value]
        if isinstance(e, ast.Slice):
            parts = [(name, getattr(e, name)) for name in ("lower", "upper", "step") if getattr(e, name) is not None]
            pre, new = self.ordered([p for _, p in parts])
            return pre, self._like(e, **{name: value for (name, _), value in zip(parts, new)})  # type: ignore[return-value]
        if isinstance(e, (ast.Tuple, ast.List, ast.Set)):
            if any(isinstance(x, ast.Starred) for x in e.elts):
                raise _Refuse("starred element")
            pre, elts = self.ordered(e.elts)
            return pre, self._like(e, elts=elts)  # type: ignore[return-value]
        if isinstance(e, ast.Dict):
            if any(k is None for k in e.keys):
                raise _Refuse("dict ** unpacking")
            flat = [x for pair in zip(e.keys, e.values) for x in pair]
            pre, new = self.ordered(flat)  # type: ignore[arg-type]
            return pre, self._like(e, keys=new[0::2], values=new[1::2])  # type: ignore[return-value]
        if isinstance(e, ast.BoolOp):
            return self.boolop(e)
        if isinstance(e, ast.IfExp):
            return self.ifexp(e)
        raise _Refuse(f"comprehension inside a {type(e).__name__}")

    # -- calls --------------------------------------------------------------------
    def _eager_consumer(self, call: ast.Call) -> bool:
        if len(call.args) != 1 or not isinstance(call.args[0], ast.GeneratorExp):
            return False
        func = call.func
        if isinstance(func, ast.Name):
            return func.id in EAGER_BUILTINS and self.names.stable_builtin(func.id)
        if isinstance(func, ast.Attribute) and func.attr == "join":
            return isinstance(func.value, ast.Constant) and isinstance(func.value.value, (str, bytes))
        return False

    def call(self, call: ast.Call) -> Tuple[List[ast.stmt], ast.expr]:
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
            raise _Refuse("call with *args / **kwargs")
        args = list(call.args)
        if self._eager_consumer(call):
            gen = args[0]
            listcomp = ast.ListComp(elt=gen.elt, generators=gen.generators)  # type: ignore[attr-defined]
            ast.copy_location(listcomp, gen)
            setattr(listcomp, PATH_ATTR, getattr(gen, PATH_ATTR, ()))
            args[0] = listcomp
        children = [call.func, *args, *[k.value for k in call.keywords]]
        pre, new = self.ordered(children)
        n = len(args)
        keywords = [self._like(k, value=v) for k, v in zip(call.keywords, new[1 + n:])]
        return pre, self._like(call, func=new[0], args=new[1:1 + n], keywords=keywords)  # type: ignore[return-value]

    # -- conditionally evaluated operands -----------------------------------------
    def boolop(self, e: ast.BoolOp) -> Tuple[List[ast.stmt], ast.expr]:
        return self._bool_chain(e, list(e.values))

    def _bool_chain(self, e: ast.BoolOp, values: List[ast.expr]) -> Tuple[List[ast.stmt], ast.expr]:
        pre, first = self.expr(values[0])
        if len(values) == 1:
            return pre, first
        if not any(has_comp(v) for v in values[1:]):
            return pre, self._like(e, values=[first, *values[1:]])  # type: ignore[return-value]
        temp = self.names.fresh("b")
        stmts = [*pre, self._assign(temp, first, first, "cond")]
        rest_pre, rest = self._bool_chain(e, values[1:])
        probe: ast.expr = self._name(temp, e, "cond")
        if isinstance(e.op, ast.Or):
            probe = stand_for(ast.UnaryOp(op=ast.Not(), operand=probe), e, "cond")  # type: ignore[assignment]
        branch = stand_for(
            ast.If(test=probe, body=[*rest_pre, self._assign(temp, rest, rest, "cond")], orelse=[]), e, "cond",
        )
        stmts.append(branch)  # type: ignore[arg-type]
        return stmts, self._name(temp, e, "cond")

    def ifexp(self, e: ast.IfExp) -> Tuple[List[ast.stmt], ast.expr]:
        pre, test = self.expr(e.test)
        if not has_comp(e.body) and not has_comp(e.orelse):
            return pre, self._like(e, test=test)  # type: ignore[return-value]
        temp = self.names.fresh("c")
        body_pre, body = self.expr(e.body)
        else_pre, orelse = self.expr(e.orelse)
        branch = stand_for(
            ast.If(
                test=test,
                body=[*body_pre, self._assign(temp, body, body, "cond")],
                orelse=[*else_pre, self._assign(temp, orelse, orelse, "cond")],
            ),
            e,
            "cond",
        )
        return [*pre, branch], self._name(temp, e, "cond")  # type: ignore[list-item]

    # -- the comprehension itself -------------------------------------------------
    def comp(self, e) -> Tuple[List[ast.stmt], ast.expr]:
        if self.scope == "class":  # not reached: class bodies are not lowered
            raise _Refuse("comprehension in a class body")
        gens = e.generators
        for gen in gens:
            if gen.is_async:
                raise _Refuse("async comprehension")
            if target_names(gen.target) is None:
                raise _Refuse("comprehension target is not a plain name pattern")
        for node in ast.walk(e):
            if isinstance(node, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
                raise _Refuse(f"{type(node).__name__} inside a comprehension")
        if isinstance(e, ast.SetComp) and not self.names.stable_builtin("set"):
            raise _Refuse("set comprehension: `set` is rebound or the program reaches the builtins")

        mapping: Dict[str, str] = {}
        for gen in gens:
            for name in sorted(target_names(gen.target) or ()):
                if name not in mapping:
                    mapping[name] = self.names.fresh(
                        name, renames=name, scope=getattr(e, PATH_ATTR, ()),
                    )

        acc = self.names.fresh("acc")
        if isinstance(e, ast.ListComp):
            empty: ast.expr = stand_for(ast.List(elts=[], ctx=ast.Load()), e, "acc")  # type: ignore[assignment]
        elif isinstance(e, ast.DictComp):
            empty = stand_for(ast.Dict(keys=[], values=[]), e, "acc")  # type: ignore[assignment]
        else:
            empty = stand_for(
                ast.Call(func=self._name("set", e, "acc"), args=[], keywords=[]), e, "acc",
            )  # type: ignore[assignment]
        init = self._assign(acc, empty, e, "acc")

        if isinstance(e, ast.DictComp):
            key_t, val_t = self.names.fresh("k"), self.names.fresh("v")
            key = rename_names(e.key, mapping)
            value = rename_names(e.value, mapping)
            store = stand_for(
                ast.Assign(
                    targets=[stand_for(
                        ast.Subscript(
                            value=self._name(acc, e, "acc"),
                            slice=self._name(key_t, e, "key"),
                            ctx=ast.Store(),
                        ),
                        e, "acc",
                    )],
                    value=self._name(val_t, e, "value"),
                    type_comment=None,
                ),
                e, "acc",
            )
            body: List[ast.stmt] = [
                self._assign(key_t, key, e.key, "key"),  # type: ignore[arg-type]
                self._assign(val_t, value, e.value, "value"),  # type: ignore[arg-type]
                store,  # type: ignore[list-item]
            ]
        else:
            method = "append" if isinstance(e, ast.ListComp) else "add"
            elt = rename_names(e.elt, mapping)
            call = stand_for(
                ast.Call(
                    func=stand_for(
                        ast.Attribute(value=self._name(acc, e, "acc"), attr=method, ctx=ast.Load()), e, "acc",
                    ),
                    args=[elt],  # type: ignore[list-item]
                    keywords=[],
                ),
                e, "acc",
            )
            body = [stand_for(ast.Expr(value=call), e, "acc")]  # type: ignore[list-item]

        for index in range(len(gens) - 1, -1, -1):
            gen = gens[index]
            for cond in reversed(gen.ifs):
                body = [stand_for(
                    ast.If(test=rename_names(cond, mapping), body=body, orelse=[]), cond, "filter",
                )]  # type: ignore[list-item]
            iterable = gen.iter if index == 0 else rename_names(gen.iter, mapping)
            target = rename_names(gen.target, mapping)
            body = [stand_for(
                ast.For(target=target, iter=iterable, body=body, orelse=[], type_comment=None),  # type: ignore[arg-type]
                gen.target, "loop",
            )]  # type: ignore[list-item]

        lowered = self.block([init, *body], self.scope)
        self.lowered += 1
        return lowered, self._name(acc, e, "acc")
