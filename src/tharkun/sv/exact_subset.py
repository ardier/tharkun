"""tharkun.sv.exact_subset: the audited ALLOWLIST a program must sit inside before
:mod:`tharkun.sv.product_proof` may call a symbolic run a PROOF ([internal ref], PR #114).

Why an allowlist and not a list of attacks. :mod:`tharkun.sv._symexec` is a search
aid: a skipped statement, a name recognised by string, a value approximated by its
truth are all fine there, because every model it produces is replayed on the real
program. A proof has no replay to catch it: one construct the interpreter handles
approximately, or silently drops, turns UNSAT into a wrong TRUE. MG's two review
rounds on PR #114 found nine such holes (a rebound builtin via an ``if``-nested
import, an attribute store on ``__builtins__``, a nested ``def`` redefinition, a
user ``__sv_site``, ``x or 7`` evaluated as a truth value, ...), each patched by a
new top-level name check and each missed by the next shape. So the rule is inverted:

    EXACTNESS BY CONSTRUCTION. Every AST node of the whole module, at every depth, is
    checked against the table below; every name it reads, binds or calls is checked
    too. Anything not listed is refused (:func:`audit` returns the reason, the caller
    answers UNKNOWN). Nothing is "probably fine".

What is allowed, and why each entry is exact in :mod:`tharkun.sv._symexec` (the
semantics were audited against the interpreter, and are re-checked by
``tests/test_sv_product_proof.py`` plus a randomised differential run against native
Python):

* values: ``int`` and ``bool`` (``Constant`` of those two types, and the
  ``nondet_int()`` / ``nondet_bool()`` draws, lifted to z3 ``Int`` / ``Bool`` by
  :mod:`tharkun.sv.product_proof`), and -- [internal ref] PR 3 -- alias-free LISTS of them (see
  the lists entry below). No float, str, bytes, None, dict, set, tuple value, object,
  lambda, comprehension, f-string: so no TypeError path, no IEEE-754, no string
  theory, no aliasing, no identity.
* expressions: ``Name`` (Load), ``UnaryOp`` (``-``, ``+``, ``not``), ``BinOp``
  (``+ - * // % << >>``), ``BoolOp`` (``and`` / ``or``, operand semantics, later
  operands evaluated only where the earlier ones let them), ``Compare`` (``== != < <=
  > >=``; a CHAIN only over TOTAL operands, see :func:`_total`: a chain short-circuits
  natively but the interpreter evaluates every link, which is the same program only
  when no operand can raise or constrain the path), ``IfExp``, ``Subscript`` (Load) of a
  LIST NAME by an int expression (see lists), and ``Call``
  with positional arguments only of exactly: a user ``def`` of this module (inlined),
  ``abs(x)``, ``int(x)`` (the identity on an int; ``_symexec.py`` refuses a bool operand),
  ``min(a, b)``, ``max(a, b)``, ``len(xs)`` of a list name.
* lists ([internal ref] PR 3, ``_Audit._type_lists``, ``_list_expr``, ``_subscript_target``): a list is built by a
  literal of scalar expressions (at most 256), ``[...] * K`` with ``K`` a literal or a module constant that folds to
  0..256 (``exact_unroll.py::fold_int``), or ``[e for i in range(<constant>)]``, which
  :mod:`tharkun.sv.exact_unroll` has already expanded to the literal of its elements before the audit runs.
  Which names hold a list is settled per scope BEFORE any statement is checked (a name assigned a list
  expression; a parameter annotated ``list[int]`` / ``list[bool]``, indexed or measured in its def, or handed a
  list at some call: a fixpoint over the call sites), and then a list NAME may only be indexed (``xs[i]``, a
  Load), measured (``len(xs)``), stored into (``xs[i] = v`` / ``xs[i] += v``, only a list assigned in THIS scope
  that is not a parameter) or passed to a def whose parameter is a list. It is never a value: no alias ``ys =
  xs``, no ``return xs``, no comparison, operand, truth test, tuple element, slice, nested list, ``del``,
  method call or iteration; a list parameter is never rebound; a list name is never assigned a scalar or
  unpacked into. So no two names ever hold the same list (a callee reads the caller's list and cannot write it;
  ``_symexec.py::_Runner._store_list`` replaces the list instead of mutating it), every list has a fixed length
  known when it is built, and every cell is an int or a bool. ``xs[i]`` and ``xs[i] = v`` are Python's rule
  exactly (``_Runner.ev_Subscript`` / ``_store_list``): ``-len <= i < len`` or the run ends in ``IndexError`` (a
  path end, not the property), a negative ``i`` counts from the end, a symbolic ``i`` selects the cell it equals
  under that path condition; ``xs[i] += v`` evaluates ``i`` once (``_Exec.st``); two arms that leave a list with
  the cells changed merge elementwise (``_ite``) and two arms that leave it with different lengths poison the
  name (``_merge_if``), so the run is inexact, never a wrong proof.
* statements: ``Assign`` (one ``Name`` target, or -- [internal ref] -- a tuple of distinct names unpacking the call of a
  TUPLE FUNCTION, see below, or ``xs[i]`` for a list of this scope), ``AugAssign`` (``Name`` or ``xs[i]`` target, same
  operators), ``Assert`` (no message), ``If``, ``Raise`` of a builtin exception class
  (bare or called with constants; no ``from``), ``Return`` (with a value, or bare in a
  PROCEDURE, see below, inside a def), ``Pass``, an ``Expr`` that is a constant or a
  call, and, at module level only, ``import _sv_verifier`` / ``from _sv_verifier
  import ...`` and plain ``def``. A call ``_sv_verifier.nondet_int()`` / ``nondet_bool()``
  inside a def, wherever an expression may stand, is a call the caller marked (``fresh_calls``,
  no argument): the proof gives every EXECUTION of it a fresh term.
* defs: top level only; plain positional parameters; no decorator, default,
  ``*args``, ``**kwargs``, keyword-only or positional-only parameter; a parameter or
  return annotation only if it is the builtin name ``int``, ``bool`` or ``object`` (``-> None``
  too; a parameter may also carry ``list[int]`` / ``list[bool]``, and then it must be a list parameter, see
  lists), which no program binding can shadow (a builtin name is never bound); and
  EITHER a FUNCTION (every ``return`` has a value and every control path ends in
  ``return`` or ``raise``: it never falls off the end into an implicit ``None``) OR a
  PROCEDURE (no ``return`` has a value; it may return early with a bare ``return`` or
  fall off its end), whose call is a bare ``Expr`` statement and nothing else, so the
  ``None`` it produces is dropped on the spot and no value of another type ever
  exists. A def that returns a value on some paths only is refused. A TUPLE FUNCTION ([internal ref]) is a function whose every
  ``return`` is a tuple expression of the same arity (at least 2, no starred element, elements audited like any
  expression); its value exists only as the right side of ``a, b = f(...)`` with exactly that many distinct names (the
  unpacking can never raise), or dropped by a bare call statement: a tuple is never stored, compared, passed, indexed or
  operated on, so no tuple value outlives its unpacking and the interpreter's elementwise ``ite`` merge of two returned
  tuples (``_symexec.py::_ite``) is exact for the int/bool elements the subset holds.
* names: nothing is bound that starts with ``__`` or is a builtin (``abs``,
  ``__builtins__``, ``__sv_site``, ...), or that is an alias of ``_sv_verifier``
  (which may appear only in the lifted draw statements); a draw name and a def name
  are bound exactly once in the whole module; a function's assigned locals never share a name
  with a module-level name, and no local takes a def's name (a PARAMETER may shadow
  a module variable: the call frame is a fresh env holding exactly the parameters); every ``Name`` read
  is bound where it is read; a def name is only ever called; a builtin name is only
  ever called, with the right arity.

Refused outright (each is an ``Unsupported`` or an approximation in the interpreter,
or a way to rebind what it recognises by string): loops (a ``for`` over ``range`` of constants and a COUNTER ``while``
-- ``i = 0`` just before it, ``while i < K``, ``i += 1`` last in the body, [internal ref] PR 4 -- are unrolled BEFORE the audit by
:mod:`tharkun.sv.exact_unroll`, exactly, with ``K`` a module constant or a parameter every call passes the same constant;
any other ``for`` or ``while`` is refused by name), ``try``, ``with``,
``class``, ``lambda``, comprehensions (a list comprehension over ``range`` of constants is expanded BEFORE
the audit by :mod:`tharkun.sv.exact_unroll`, exactly; any other stays one and is refused), ``global`` /
``nonlocal`` / ``del``, nested ``def`` or ``import`` (also under ``if``), any ``Attribute`` (so no attribute
store, no ``__builtins__.abs = ...``), any ``Subscript`` that is not a list name indexed by an int expression
(slices, tuples as keys, a subscript of a subscript), tuples and lists as values, dicts, sets, walrus,
annotated assignment (``product_proof.py`` rewrites ``x: int = v`` and ``xs: list[int] = v`` to ``x = v`` /
``xs = v`` first), ``**``, ``/``, ``~``, ``& | ^``, ``is`` / ``in``, strings.
"""
from __future__ import annotations

import ast
import builtins

from tharkun.sv.exact_unroll import fold_int, module_constants


def _builtin_arity(name: str) -> int | None:
    """The only positional arity at which the builtin ``name`` is admitted (its ``_symexec`` semantics on int/bool operands were
    audited exact: ``abs(x)``, ``int(x)``, ``min(a, b)``, ``max(a, b)``), or ``None``.  The answer is the RULE's ([internal ref] step 2:
    ``solve/contracts/visitors.py::ProductVisitor.EXACT_ARITY``, given while the contract is TESTED), not a table of names here."""
    from tharkun.solve.contracts.rules import Refused, get_rule
    from tharkun.solve.contracts.visitors import ProductVisitor

    rule = get_rule(name)
    if rule is None:
        return None
    arity = rule.accept(ProductVisitor())
    return None if isinstance(arity, Refused) else arity
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.LShift, ast.RShift)
_UNARYOPS = (ast.USub, ast.UAdd, ast.Not)
_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
#: the only annotations a def may carry: builtin value types of the subset (no binding can shadow them, see
#: ``_name_bindable``), so evaluating one, now or lazily, runs nothing
ANNOTATIONS = ("int", "bool", "object")
#: operators whose result exists for every int/bool operand and adds nothing to the path: no divisor, no shift
_TOTAL_BINOPS = (ast.Add, ast.Sub, ast.Mult)


class _Refuse(Exception):
    pass


def _no(why: str, node: ast.AST | None = None):
    line = getattr(node, "lineno", None)
    raise _Refuse(f"{why}" + (f" (line {line})" if line else ""))


def _name_bindable(name: str, reserved: set, node=None) -> None:
    if name.startswith("__") or hasattr(builtins, name) or name in reserved:
        _no(f"binds {name!r}, a dunder / builtin / _sv_verifier name", node)


def _store_names(stmts, out: list, *, into_defs: bool = False) -> None:
    """Every ``Name`` stored by ``stmts`` (Assign / AugAssign targets), through nested
    ``if`` arms, not into defs."""
    for st in stmts:
        if isinstance(st, ast.Assign):
            for t in st.targets:
                if isinstance(t, ast.Name):
                    out.append(t.id)
                elif isinstance(t, ast.Tuple):
                    out.extend(e.id for e in t.elts if isinstance(e, ast.Name))
        elif isinstance(st, ast.AugAssign):
            if isinstance(st.target, ast.Name):
                out.append(st.target.id)
        elif isinstance(st, ast.If):
            _store_names(st.body, out)
            _store_names(st.orelse, out)


def _returns_in(stmts):
    """Every ``Return`` of a def body, through nested ``if`` arms (the only compound statement a def holds)."""
    for st in stmts:
        if isinstance(st, ast.Return):
            yield st
        elif isinstance(st, ast.If):
            yield from _returns_in(st.body)
            yield from _returns_in(st.orelse)


def _list_annotation(n) -> bool:
    """``list[int]`` / ``list[bool]``: builtin names no program binding can shadow, run nowhere (a parameter annotation)."""
    return (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "list"
            and isinstance(n.slice, ast.Name) and n.slice.id in ("int", "bool"))


def _annotation_ok(n, none_ok: bool) -> bool:
    if isinstance(n, ast.Name):
        return isinstance(n.ctx, ast.Load) and n.id in ANNOTATIONS
    return none_ok and isinstance(n, ast.Constant) and n.value is None


def _total(n) -> bool:
    """``n`` has a value for EVERY int/bool binding of its names and constrains nothing: a name, an int/bool
    constant, ``-``/``+``/``not`` of a total operand, ``+ - *`` of total operands.  No division, modulo, shift,
    call or ``and``/``or``/conditional.  A chain ``a < b < c`` only evaluates ``c`` when ``a < b`` holds; the
    interpreter evaluates every operand, so the two agree exactly when evaluating an operand cannot raise,
    cannot end the path and cannot add to the path condition: exactly the total operands."""
    if isinstance(n, ast.Name):
        return isinstance(n.ctx, ast.Load)
    if isinstance(n, ast.Constant):
        return type(n.value) in (int, bool)
    if isinstance(n, ast.UnaryOp):
        return isinstance(n.op, _UNARYOPS) and _total(n.operand)
    if isinstance(n, ast.BinOp):
        return isinstance(n.op, _TOTAL_BINOPS) and _total(n.left) and _total(n.right)
    return False


def _always_ends(stmts) -> bool:
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.If):
        return _always_ends(last.body) and bool(last.orelse) and _always_ends(last.orelse)
    return False


class _Audit:
    def __init__(self, tree: ast.Module, lifted_ids: set, fresh_calls: frozenset = frozenset()):
        self.tree = tree
        self.lifted_ids = lifted_ids
        self.fresh_calls = fresh_calls       # id() of the draw calls the caller lets sit inside defs
        self.reserved: set = set()           # aliases of _sv_verifier
        self.defs: dict = {}                 # name -> FunctionDef
        self.procedures: set = set()         # the defs that return no value (bare return / fall off the end)
        self.tuple_defs: dict = {}           # name -> arity of the defs whose every return is a tuple ([internal ref])
        self.module_names: set = set()       # readable module-level variables (not defs)
        self.consts = module_constants(tree)  # names bound once to a foldable int (``K = 4``): the size of ``[0] * K``
        self.module_lists: set = set()       # module-level names holding a list ([internal ref] PR 3)
        self.def_lists: dict = {}            # def name -> its local names holding a list (parameters included)
        self.list_params: dict = {}          # def name -> indexes of the parameters that are lists
        self.cur_lists: set = set()          # the list names of the scope being audited
        self.cur_params: set = set()         # ... of which the parameters (read-only)
        self.visible_lists: set = set()      # ... plus the module's lists (read-only inside a def)

    # -- module level -----------------------------------------------------------
    def run(self) -> None:
        body = []
        draws = []
        for st in self.tree.body:
            if isinstance(st, ast.Import):
                for a in st.names:
                    if a.name != "_sv_verifier":
                        _no(f"imports {a.name!r}", st)
                    self._alias(a.asname or a.name, st)
            elif isinstance(st, ast.ImportFrom):
                if st.module != "_sv_verifier" or st.level:
                    _no("imports from a module other than _sv_verifier", st)
                for a in st.names:
                    if a.name == "*":
                        _no("star import", st)
                    self._alias(a.asname or a.name, st)
            elif id(st) in self.lifted_ids:
                draws.append(st.targets[0].id)
            elif isinstance(st, ast.FunctionDef):
                if st.name in self.defs:
                    _no(f"def {st.name} twice", st)
                self.defs[st.name] = st
                body.append(st)
            else:
                body.append(st)
        # every module-level binding, counted
        stores: list = list(draws)
        _store_names([s for s in body if not isinstance(s, ast.FunctionDef)], stores)
        counts: dict = {}
        for n in stores + list(self.defs):
            counts[n] = counts.get(n, 0) + 1
        for name, c in counts.items():
            _name_bindable(name, self.reserved)
            if (name in self.defs or name in draws) and c != 1:
                _no(f"{name!r} (a draw or a def) is bound {c} times")
        self.module_names = set(counts) - set(self.defs)
        for fd in self.defs.values():
            self._classify(fd)
        module_body = [s for s in body if not isinstance(s, ast.FunctionDef)]
        self._type_lists(module_body)
        for fd in self.defs.values():
            self._def(fd)
        self.cur_lists, self.cur_params, self.visible_lists = set(self.module_lists), set(), set(self.module_lists)
        self._stmts(module_body, self.module_names, False)

    def _alias(self, name: str, node) -> None:
        if hasattr(builtins, name):
            _no(f"imports _sv_verifier names as the builtin {name!r}", node)   # ``int`` must stay the builtin
        self.reserved.add(name)

    # -- defs ---------------------------------------------------------------------
    def _classify(self, fd: ast.FunctionDef) -> None:
        """A FUNCTION returns a value on every path; a PROCEDURE never returns one (see the module docstring)."""
        rets = list(_returns_in(fd.body))
        bare = [r for r in rets if r.value is None]
        if bare and len(bare) != len(rets):
            _no(f"def {fd.name} returns a value on some paths only", fd)
        tuples = [r for r in rets if isinstance(r.value, ast.Tuple)]
        if tuples:
            arities = {len(r.value.elts) for r in tuples}
            if len(tuples) != len(rets) or len(arities) != 1:
                _no(f"def {fd.name} returns a tuple on some paths only, or tuples of different arity", fd)
            if next(iter(arities)) < 2 or any(isinstance(e, ast.Starred) for r in tuples for e in r.value.elts):
                _no(f"def {fd.name} returns a tuple of fewer than two elements or a starred one", fd)
            self.tuple_defs[fd.name] = next(iter(arities))
        if bare or (not rets and not _always_ends(fd.body)):
            self.procedures.add(fd.name)
        elif not _always_ends(fd.body):
            _no(f"def {fd.name} can fall off its end (implicit None)", fd)

    def _def(self, fd: ast.FunctionDef) -> None:
        a = fd.args
        if fd.decorator_list or a.defaults or a.kw_defaults or a.kwonlyargs or a.vararg or a.kwarg \
                or a.posonlyargs or getattr(fd, "type_params", None):
            _no(f"def {fd.name} is not a plain positional function", fd)
        if (fd.returns is not None and not _annotation_ok(fd.returns, True)) \
                or any(x.annotation is not None and not (_annotation_ok(x.annotation, False) or _list_annotation(x.annotation))
                       for x in a.args):
            _no(f"def {fd.name} has an annotation other than int / bool / object / list[int] / list[bool]", fd)
        params = [x.arg for x in a.args]
        listed = {params[j] for j in self.list_params[fd.name]}
        self.cur_params = listed
        self.cur_lists = set(self.def_lists[fd.name]) | listed
        # a parameter named like a module list is the parameter (the call frame holds exactly the parameters)
        self.visible_lists = self.cur_lists | (self.module_lists - set(params))
        for j, x in enumerate(a.args):
            if x.annotation is not None and _list_annotation(x.annotation) != (j in self.list_params[fd.name]):
                _no(f"parameter {x.arg} of {fd.name} is annotated as a list but is not passed one, or the reverse", fd)
        local_stores: list = []
        _store_names(fd.body, local_stores)
        for n in set(params) | set(local_stores):
            _name_bindable(n, self.reserved, fd)
            if n in self.defs or (n in self.module_names and n not in params):
                # a parameter may shadow a module VARIABLE (the interpreter's call
                # frame is a fresh env holding exactly the parameters); an ASSIGNED
                # local may not (read-before-assign would see the global in the
                # interpreter, raise UnboundLocalError natively), and nothing local
                # may take a def's name (the interpreter's call test is `not in env`)
                _no(f"{n!r} is both local to {fd.name} and a module-level name", fd)
        scope = set(params) | set(local_stores) | self.module_names
        self._stmts(fd.body, scope, True)

    # -- lists ([internal ref] PR 3) ---------------------------------------------------------
    def _is_list_expr(self, v) -> bool:
        return isinstance(v, ast.List) or (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult)
                                           and (isinstance(v.left, ast.List) or isinstance(v.right, ast.List)))

    def _type_lists(self, module_body) -> None:
        """Which names hold a LIST, per scope, before any statement is audited: a name assigned a list expression is a list; a
        parameter is a list when it is annotated ``list[int]`` / ``list[bool]``, is indexed or measured with ``len`` in its def,
        or receives a list at some call (a fixpoint over the call sites).  The statements then enforce what a list may do."""
        def assigned_lists(stmts):
            return {st.targets[0].id for s in stmts for st in ast.walk(s)
                    if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                    and self._is_list_expr(st.value)}

        self.module_lists = assigned_lists(module_body)
        for fd in self.defs.values():
            params = [x.arg for x in fd.args.args]
            used = set()
            for n in (m for s in fd.body for m in ast.walk(s)):
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in params:
                    used.add(n.value.id)
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len" \
                        and len(n.args) == 1 and isinstance(n.args[0], ast.Name) and n.args[0].id in params:
                    used.add(n.args[0].id)
            self.list_params[fd.name] = {j for j, x in enumerate(fd.args.args)
                                         if x.arg in used or (x.annotation is not None and _list_annotation(x.annotation))}
            self.def_lists[fd.name] = assigned_lists(fd.body)
        scopes = [(None, module_body)] + [(fd.name, fd.body) for fd in self.defs.values()]
        changed = True
        while changed:
            changed = False
            for owner, body in scopes:
                names = set(self.module_lists) if owner is None else (
                    self.module_lists | self.def_lists[owner] | {x.arg for j, x in enumerate(self.defs[owner].args.args)
                                                                  if j in self.list_params[owner]})
                for n in (m for s in body for m in ast.walk(s)):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.defs:
                        for j, a in enumerate(n.args):
                            if isinstance(a, ast.Name) and a.id in names and j < len(self.defs[n.func.id].args.args) \
                                    and j not in self.list_params[n.func.id]:
                                self.list_params[n.func.id].add(j)
                                changed = True

    def _list_expr(self, v, scope: set) -> None:
        """A list literal of scalar expressions, or ``[...] * <constant>`` (the size folds, 0..256): a FRESH list."""
        if isinstance(v, ast.List):
            if len(v.elts) > 256:
                _no("list literal of more than 256 elements", v)
            for e in v.elts:
                self._expr(e, scope)
            return
        lst, count = (v.left, v.right) if isinstance(v.left, ast.List) else (v.right, v.left)
        size = fold_int(count, self.consts)
        if size is None or not 0 <= size <= 256:
            _no("list repetition whose count is not a constant between 0 and 256", v)
        self._list_expr(lst, scope)

    def _subscript_target(self, t, scope: set) -> None:
        """``xs[i] = ...`` / ``xs[i] += ...``: ``xs`` is a list of THIS scope that is not a parameter (a callee never mutates a
        caller's list, so aliasing cannot exist), the index is an int expression."""
        if not (isinstance(t.value, ast.Name) and t.value.id in self.cur_lists - self.cur_params and t.value.id in scope):
            _no("subscript store into something that is not a list local to this scope (a parameter or another scope's list "
                "would be aliased)", t)
        if isinstance(t.slice, (ast.Slice, ast.Tuple)):
            _no("slice or tuple subscript", t)
        self._expr(t.slice, scope)

    # -- statements ---------------------------------------------------------------
    def _stmts(self, stmts, scope: set, in_def: bool) -> None:
        for st in stmts:
            self._stmt(st, scope, in_def)

    def _stmt(self, st, scope: set, in_def: bool) -> None:
        if isinstance(st, ast.Assign):
            if len(st.targets) == 1 and isinstance(st.targets[0], ast.Tuple):
                self._unpack(st, scope)
                return
            if len(st.targets) == 1 and isinstance(st.targets[0], ast.Subscript):
                self._subscript_target(st.targets[0], scope)
                self._expr(st.value, scope)
                return
            if len(st.targets) != 1 or not isinstance(st.targets[0], ast.Name):
                _no("assignment target is not a single name", st)
            name = st.targets[0].id
            if self._is_list_expr(st.value):
                if name in self.cur_params:
                    _no(f"list parameter {name} is rebound", st)
                self._list_expr(st.value, scope)
            else:
                if name in self.cur_lists:
                    _no(f"{name!r} holds a list and is assigned something else", st)
                self._expr(st.value, scope)
        elif isinstance(st, ast.AugAssign):
            if isinstance(st.target, ast.Subscript) and isinstance(st.op, _BINOPS):
                self._subscript_target(st.target, scope)
                self._expr(st.value, scope)
                return
            if not isinstance(st.target, ast.Name) or not isinstance(st.op, _BINOPS):
                _no("augmented assignment shape", st)
            if st.target.id not in scope:
                _no("augmented assignment to an unbound name", st)
            if st.target.id in self.visible_lists:
                _no("augmented assignment to a list", st)
            self._expr(st.value, scope)
        elif isinstance(st, ast.Assert):
            if st.msg is not None:
                _no("assert with a message", st)
            self._expr(st.test, scope)
        elif isinstance(st, ast.If):
            self._expr(st.test, scope)
            self._stmts(st.body, scope, in_def)
            self._stmts(st.orelse, scope, in_def)
        elif isinstance(st, ast.Raise):
            self._raise(st)
        elif isinstance(st, ast.Return):
            if not in_def:
                _no("return outside a def", st)
            if isinstance(st.value, ast.Tuple):      # a tuple function's return (``_classify`` checked the shape)
                for e in st.value.elts:
                    self._expr(e, scope)
            elif st.value is not None:               # a bare return only exists in a procedure (``_classify``)
                self._expr(st.value, scope)
        elif isinstance(st, ast.Pass):
            pass
        elif isinstance(st, ast.Expr):
            if isinstance(st.value, ast.Call):
                self._call(st.value, scope, statement=True)     # a procedure may be called here and only here
            elif not isinstance(st.value, ast.Constant):
                self._expr(st.value, scope)
        elif isinstance(st, ast.For):
            _no("for loop that is not range(<constant>) with at most 64 trips and no break / continue / else "
                "(sv/exact_unroll.py unrolls those before the audit)", st)
        elif isinstance(st, ast.While):
            _no("while loop that is not a counter loop (v = <constant> just before it, while v < <constant>, v += <constant> "
                "last in the body, no other store of v, no break / continue / else, at most 64 trips; sv/exact_unroll.py "
                "unrolls those before the audit): a loop whose trip count the audit cannot see is not a proof", st)
        else:
            _no(f"statement kind {type(st).__name__}", st)

    def _unpack(self, st: ast.Assign, scope: set) -> None:
        """``a, b = f(...)``: ``f`` is a tuple function of exactly that arity and the targets are distinct plain names, so the
        unpacking cannot raise and the tuple is consumed on the spot."""
        target, value = st.targets[0], st.value
        names = [e.id for e in target.elts if isinstance(e, ast.Name)]
        if len(names) != len(target.elts) or len(set(names)) != len(names):
            _no("tuple assignment target is not distinct plain names", st)
        if any(n in self.visible_lists for n in names):
            _no("tuple assignment into a name that holds a list", st)
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and self.tuple_defs.get(value.func.id) == len(names)):
            _no("tuple assignment whose value is not the call of a tuple function of that arity", st)
        self._call(value, scope, unpack=True)

    def _raise(self, st: ast.Raise) -> None:
        exc = st.exc
        if st.cause is not None or exc is None:
            _no("raise shape", st)
        if isinstance(exc, ast.Call):
            if exc.keywords or not all(isinstance(a, ast.Constant) for a in exc.args):
                _no("raise with a non-constant argument", st)
            exc = exc.func
        cls = getattr(builtins, exc.id, None) if isinstance(exc, ast.Name) else None
        if not (isinstance(cls, type) and issubclass(cls, BaseException)):
            _no("raise of something that is not a builtin exception class", st)

    # -- expressions ----------------------------------------------------------------
    def _expr(self, n, scope: set) -> None:
        if isinstance(n, ast.Constant):
            if type(n.value) not in (int, bool):
                _no(f"constant of type {type(n.value).__name__}", n)
        elif isinstance(n, ast.Name):
            if not isinstance(n.ctx, ast.Load) or n.id not in scope:
                _no(f"name {n.id!r} is not a bound variable here", n)
            if n.id in self.visible_lists:
                _no(f"the list {n.id} is used as a value (it can only be indexed, measured with len or passed to a list parameter)", n)
        elif isinstance(n, ast.Subscript):
            if not (isinstance(n.ctx, ast.Load) and isinstance(n.value, ast.Name) and n.value.id in self.visible_lists
                    and n.value.id in scope):
                _no("subscript of something that is not a list", n)
            if isinstance(n.slice, (ast.Slice, ast.Tuple)):
                _no("slice or tuple subscript", n)
            self._expr(n.slice, scope)
        elif isinstance(n, ast.UnaryOp):
            if not isinstance(n.op, _UNARYOPS):
                _no("unary operator", n)
            self._expr(n.operand, scope)
        elif isinstance(n, ast.BinOp):
            if not isinstance(n.op, _BINOPS):
                _no(f"binary operator {type(n.op).__name__}", n)
            self._expr(n.left, scope)
            self._expr(n.right, scope)
        elif isinstance(n, ast.BoolOp):
            for v in n.values:
                self._expr(v, scope)
        elif isinstance(n, ast.Compare):
            if not all(isinstance(op, _CMPOPS) for op in n.ops):
                _no("comparison shape (is, in)", n)
            if len(n.ops) > 1 and not all(_total(x) for x in [n.left, *n.comparators]):
                _no("comparison chain over an operand that can raise or constrain the path", n)
            self._expr(n.left, scope)
            for c in n.comparators:
                self._expr(c, scope)
        elif isinstance(n, ast.IfExp):
            self._expr(n.test, scope)
            self._expr(n.body, scope)
            self._expr(n.orelse, scope)
        elif isinstance(n, ast.Call):
            self._call(n, scope)
        else:
            _no(f"expression kind {type(n).__name__}", n)

    def _call(self, n: ast.Call, scope: set, statement: bool = False, unpack: bool = False) -> None:
        if id(n) in self.fresh_calls:
            return                           # a draw: the caller checked its shape (no argument, int / bool)
        if n.keywords or not isinstance(n.func, ast.Name):
            _no("call shape", n)
        name = n.func.id
        if name in self.defs:
            if len(n.args) != len(self.defs[name].args.args):
                _no(f"call of {name} with the wrong arity", n)
            if name in self.procedures and not statement:
                _no(f"the value of {name}, which returns none, is used", n)
            if name in self.tuple_defs and not (statement or unpack):
                _no(f"the tuple {name} returns is used as a value (it can only be unpacked into names)", n)
            for j, a in enumerate(n.args):
                if j in self.list_params[name]:
                    if not (isinstance(a, ast.Name) and a.id in self.visible_lists and a.id in scope):
                        _no(f"argument {j} of {name} must be a list (the parameter is one)", n)
                else:
                    self._expr(a, scope)
            return
        elif name == "len":                  # the builtin: no program binding may take its name (``_name_bindable``)
            if not (len(n.args) == 1 and isinstance(n.args[0], ast.Name) and n.args[0].id in self.visible_lists
                    and n.args[0].id in scope):
                _no("len of something that is not a list", n)
            return
        elif not name.startswith("__") and _builtin_arity(name) is not None:
            if len(n.args) != _builtin_arity(name):
                _no(f"call of {name} with the wrong arity", n)
        else:
            _no(f"call of {name!r}", n)
        for a in n.args:
            self._expr(a, scope)


def audit(tree: ast.Module, lifted_ids: set, fresh_calls: frozenset = frozenset()) -> str | None:
    """``None`` when the WHOLE module sits inside the allowlist (see the module
    docstring), else the reason it does not. ``lifted_ids``: ``id()`` of the top-level
    ``name = _sv_verifier.nondet_TYPE()`` statements the caller already lifted;
    ``fresh_calls``: ``id()`` of the ``_sv_verifier.nondet_TYPE()`` calls inside defs, which the
    caller checked have that exact shape (no argument, kind ``int`` / ``bool``)."""
    try:
        _Audit(tree, lifted_ids, fresh_calls).run()
    except _Refuse as e:
        return str(e)
    return None
