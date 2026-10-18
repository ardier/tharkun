"""Names, scopes and fresh variables for the lowerings ([internal ref]).

Three facts a lowering needs and must get exactly right:

* **Which names a program binds.**  A name no statement of the program ever binds
  (no assignment, ``def``, ``class``, import, parameter, ``except ... as``, match
  capture, ``global``/``nonlocal``, walrus) and that names a builtin is a STABLE
  read: it cannot be unbound, rebound or shadowed, so evaluating it earlier or later
  in a statement is the same.  Anything else is not stable, and a lowering that
  would move other code across its evaluation copies it into a temporary first.
* **Scope-aware renaming.**  A comprehension's loop variables live in the
  comprehension's own scope: they do not leak and do not clobber an enclosing name.
  A loop lowering gets that by giving each loop variable a fresh name, and renaming
  its uses -- but only the uses that are in the comprehension's scope.  A nested
  comprehension or a lambda that binds the same name has its own variable, and the
  FIRST iterable of a nested comprehension is evaluated in the enclosing scope.
* **Fresh names** that cannot collide with anything in the program.

Stdlib only; imports nothing from tharkun.
"""

from __future__ import annotations

import ast
import builtins
import copy
from typing import Dict, Iterable, Optional, Set

#: Reading one of these inspects or depends on the scope it runs in, so code that
#: names them is never moved into another scope or frame.
REFLECTIVE_NAMES = frozenset(
    {
        "super", "locals", "vars", "globals", "dir", "eval", "exec", "compile",
        "__import__", "__builtins__", "__class__", "breakpoint", "__locals__",
    }
)

COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

#: Attributes that hand a program the frame or the namespace it runs in.
FRAME_ATTRS = frozenset(
    {"f_locals", "f_globals", "f_back", "_getframe", "currentframe", "__globals__", "gi_frame",
     "cr_frame", "tb_frame"}
)

_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def bound_names(tree: ast.AST) -> Set[str]:
    """Every name some construct of ``tree`` binds or declares, in any scope."""
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            for param in getattr(node, "type_params", ()) or ():
                names.add(param.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif type(node).__name__ == "TypeAlias" and isinstance(getattr(node, "name", None), ast.Name):
            names.add(node.name.id)
    return names


def _own_scope_locals(node: ast.AST) -> Set[str]:
    """The names a ``def`` / ``lambda`` binds LOCALLY: its parameters and the names its own body
    binds (not a nested function, lambda or class body; a comprehension binds only its own
    variables, except for an assignment expression, which binds in the function), minus the names
    it declares ``global``.  A load of such a name is never a builtin."""
    args = node.args  # type: ignore[attr-defined]
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg:
        params.append(args.vararg)
    if args.kwarg:
        params.append(args.kwarg)
    names: Set[str] = {a.arg for a in params}
    declared_global: Set[str] = set()
    body = [node.body] if isinstance(node, ast.Lambda) else list(node.body)  # type: ignore[attr-defined]
    stack = body
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
            continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, COMPREHENSIONS):
            for sub in ast.walk(n):
                if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                    names.add(sub.target.id)
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            names.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            names.add(n.rest)
        elif isinstance(n, ast.Global):
            declared_global.update(n.names)
        stack.extend(ast.iter_child_nodes(n))
    return names - declared_global


def _comprehension_locals(node: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for gen in node.generators:  # type: ignore[attr-defined]
        for n in ast.walk(gen.target):
            if isinstance(n, ast.Name):
                out.add(n.id)
    return out


def reaches_the_namespace(tree: ast.AST) -> bool:
    """Can the program read its own scope, or the builtins, so that a fresh name added by a
    lowering would be observable?  The names that do that (``locals``, ``globals``, ``vars``,
    ``dir``, ``eval``, ``exec``, ...) are a problem when a load of one can resolve to the
    BUILTIN: a load inside a function whose own scope binds the name (a parameter, a loop
    variable) is a local read and is not.  A binding at module level does not exempt a load: a
    function can run before it executes.  Frame attributes (``f_locals``, ``_getframe``, ...)
    are a problem wherever they appear, and so is a ``global`` declaration of such a name."""
    hot = REFLECTIVE_NAMES - {"super"}
    stack = [(tree, ())]
    while stack:
        node, scopes = stack.pop()
        if isinstance(node, ast.Name):
            if node.id in hot and isinstance(node.ctx, ast.Load) and not any(node.id in s for s in scopes):
                return True
        elif isinstance(node, ast.Attribute):
            if node.attr in FRAME_ATTRS:
                return True
        elif isinstance(node, ast.Global):
            if hot & set(node.names):
                return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            # defaults and annotations run where the def is written; the body in its own scope
            outer = [d for d in a.defaults] + [d for d in a.kw_defaults if d is not None]
            outer += [x.annotation for x in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
                      if x is not None and x.annotation is not None]
            if not isinstance(node, ast.Lambda):
                outer += list(node.decorator_list)
                if node.returns is not None:
                    outer.append(node.returns)
            stack.extend((c, scopes) for c in outer)
            inner = scopes + (_own_scope_locals(node),)
            body = [node.body] if isinstance(node, ast.Lambda) else node.body
            stack.extend((c, inner) for c in body)
            continue
        if isinstance(node, COMPREHENSIONS):
            gens = node.generators
            stack.append((gens[0].iter, scopes))  # evaluated in the enclosing scope
            inner = scopes + (_comprehension_locals(node),)
            for i, gen in enumerate(gens):
                stack.append((gen.target, inner))
                if i > 0:
                    stack.append((gen.iter, inner))
                stack.extend((c, inner) for c in gen.ifs)
            if isinstance(node, ast.DictComp):
                stack.append((node.key, inner))
                stack.append((node.value, inner))
            else:
                stack.append((node.elt, inner))
            continue
        stack.extend((c, scopes) for c in ast.iter_child_nodes(node))
    return False


def identifiers(tree: ast.AST) -> Set[str]:
    """Every identifier appearing in ``tree`` as a name (for choosing fresh ones)."""
    out = bound_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
    return out


class Names:
    """The name facts of one program: bound names, stability, fresh names."""

    def __init__(self, tree: ast.AST):
        self.bound = bound_names(tree)
        self.taken = identifiers(tree)
        self._star = any(
            isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
            for n in ast.walk(tree)
        )
        # a program that can reach or rewrite the builtins namespace has no stable builtin
        self.reflective = self._star or "builtins" in self.taken or reaches_the_namespace(tree)
        self._counter = 0
        #: fresh name -> the original name it renames (loop variables only)
        self.renamed: Dict[str, str] = {}
        #: fresh names that rename nothing (accumulators and temporaries)
        self.synthetic: Set[str] = set()
        #: fresh loop name -> the path of the ORIGINAL comprehension whose variable it is: the
        #: only part of the original program where the name it stands for is that variable
        self.scope_path: Dict[str, tuple] = {}

    def stable_builtin(self, name: str) -> bool:
        """True when reading ``name`` is the same read anywhere in the program: it is
        a builtin no construct of the program rebinds or can reach."""
        return (
            not self.reflective
            and name not in self.bound
            and name not in REFLECTIVE_NAMES
            and hasattr(builtins, name)
        )

    def fresh(self, hint: str, *, renames: Optional[str] = None, scope: tuple = ()) -> str:
        """A name not appearing anywhere in the program.  ``renames`` records the
        original name a loop variable stands for, ``scope`` the path of the comprehension
        it is the variable of."""
        clean = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in hint)
        while True:
            self._counter += 1
            # single underscore: a name with a leading double underscore is mangled inside a class body
            name = f"_lw{self._counter}_{clean}"
            if name not in self.taken:
                break
        self.taken.add(name)
        if renames is None:
            self.synthetic.add(name)
        else:
            self.renamed[name] = renames
            self.scope_path[name] = scope
        return name


def target_names(target: ast.AST) -> Optional[Set[str]]:
    """The names an assignment target binds, or ``None`` when it is not a plain
    name / tuple / list / starred pattern of names (an attribute or subscript target
    evaluates sub-expressions, which a loop lowering does not handle)."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out: Set[str] = set()
        for elt in target.elts:
            sub = target_names(elt)
            if sub is None:
                return None
            out |= sub
        return out
    if isinstance(target, ast.Starred):
        return target_names(target.value)
    return None


class _Renamer(ast.NodeTransformer):
    """Rename names in place, honouring scopes (see the module docstring)."""

    def __init__(self, mapping: Dict[str, str]):
        self.stack = [dict(mapping)]

    @property
    def current(self) -> Dict[str, str]:
        return self.stack[-1]

    def _without(self, names: Iterable[str]) -> Dict[str, str]:
        drop = set(names)
        return {k: v for k, v in self.current.items() if k not in drop}

    def visit_Name(self, node: ast.Name):
        new = self.current.get(node.id)
        if new is not None:
            node.id = new
        return node

    def visit_Lambda(self, node: ast.Lambda):
        args = node.args
        # defaults and annotations are evaluated where the lambda is written
        for default in list(args.defaults) + [d for d in args.kw_defaults if d is not None]:
            self.visit(default)
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg:
            params.append(args.vararg)
        if args.kwarg:
            params.append(args.kwarg)
        for param in params:
            if param.annotation is not None:
                self.visit(param.annotation)
        self.stack.append(self._without(p.arg for p in params))
        self.visit(node.body)
        self.stack.pop()
        return node

    def _comprehension(self, node):
        generators = node.generators
        # the FIRST iterable is evaluated in the enclosing scope
        self.visit(generators[0].iter)
        bound: Set[str] = set()
        for gen in generators:
            names = target_names(gen.target)
            if names is None:
                names = {n.id for n in ast.walk(gen.target) if isinstance(n, ast.Name)}
            bound |= names
        self.stack.append(self._without(bound))
        for i, gen in enumerate(generators):
            self.visit(gen.target)
            if i > 0:
                self.visit(gen.iter)
            for cond in gen.ifs:
                self.visit(cond)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.stack.pop()
        return node

    visit_ListComp = visit_SetComp = visit_GeneratorExp = visit_DictComp = _comprehension


def rename_names(node: ast.AST, mapping: Dict[str, str]) -> ast.AST:
    """A copy of ``node`` with the names in ``mapping`` renamed in the scope the node
    is evaluated in.  ``node`` itself is not modified."""
    if not mapping:
        return copy.deepcopy(node)
    return _Renamer(mapping).visit(copy.deepcopy(node))
