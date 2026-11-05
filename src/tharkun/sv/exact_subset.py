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
  the lists entry below), plus the restricted string tags below. No float, bytes,
  None, dict, set, tuple value, object, escaping lambda, comprehension or f-string.
* string tags: module names assigned printable ASCII literals without backslashes,
  used only as direct ==/!= operands. Branch merges select those exact literals;
  equality over their Z3 encodings is Python equality. Backslashes are excluded
  because StringVal interprets Unicode escape text. Every name ever assigned a
  string tag is refused in numeric operations, truth tests, calls, returns, aliases
  and list positions, including the int-only check_type success rewrite. Function
  locals cannot receive string literals. Mixed int/string branch values retain
  their types in the interpreter's Union; no string draw or general string
  operation is admitted in an integer-only proof. When the caller has authenticated
  a string draw, every literal uses exact_strings' literal code-point sequence;
  string values may be assigned, returned, concatenated, compared for equality or
  tested for emptiness. The interpreter refuses every other string transfer.
  Identity, methods, ordering, comparison chains and the int-only check_type
  rewrite remain refused for this mode. Its draw alphabet is a sound superset of
  native string.printable; only UNSAT is decisive, never a decoded string model.
* direct lambda calls: plain positional parameters, no captured names/defaults or
  escaping lambda values; the scalar body and arguments pass the same expression
  audit, and each call inlines through the existing _FuncVal path.
  Closed one-parameter literal callback arguments at once-only module/conditional
  sites normalize to fresh defs and reuse the closed-def path below. Captures,
  defaults, callable metadata/escapes and repeated function-body sites refuse.
* closed callable flow: scalar-only programs may assign source defs to aliases,
  select them conditionally, pass them to plain function parameters, call them
  with exact positional/keyword binding, and compare their identities. The finite
  flow analysis checks every candidate binding and every callable use; a callable
  cannot escape as a scalar, container element, return value, hint or arbitrary
  object. The interpreter preserves guarded values and source-def identity, and
  evaluates arguments once in native written order before dispatch. Containers
  and lambdas remain outside this separate mode. Existing scalar argument goals,
  type-check obligations, draw independence and whole-run exactness still apply.
* pure factories: zero-argument module functions whose sole statement returns a
  known module def or another pure factory result. Their signatures still pass
  the ordinary def audit. Results may only be called or dropped, never used as
  scalar values (including the int-checker success erasure).
* named lambdas: ``f = lambda ...: e`` at module level, ``f`` bound once and not read before that statement, is the def
  ``f`` with one return (``exact_callables.py::lift_named_lambdas``, before the audit): same function value, and the
  differences (``__name__``, ``repr``) are unobservable here.
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
  finite literal of scalar expressions, ``[...] * K`` with ``K`` a literal or a module constant that folds to
  an integer (``exact_unroll.py::fold_int``), or ``[e for i in range(<constant>)]``, which
  :mod:`tharkun.sv.exact_unroll` has already expanded to the literal of its elements before the audit runs.
  Which names hold a list is settled per scope BEFORE any statement is checked (a name assigned a list
  expression; a parameter annotated ``list[int]`` / ``list[bool]``, indexed or measured in its def, or handed a
  list at some call: a fixpoint over the call sites), and then a list NAME may only be indexed (``xs[i]``, a
  Load), measured (``len(xs)``), stored into (``xs[i] = v`` / ``xs[i] += v``, only a list assigned in THIS scope
  that is not a parameter) or passed to a def whose parameter is a list. It is never a value: no alias ``ys =
  xs``, no ``return xs``, no general comparison, operand, truth test, tuple element, slice, nested list, ``del``,
  method call or iteration; a list parameter is never rebound; a list name is never assigned a scalar or
  unpacked into. So no two names ever hold the same list (a callee reads the caller's list and cannot write it;
  ``_symexec.py::_Runner._store_list`` replaces the list instead of mutating it), every list has a fixed length
  known when it is built, and every cell is an int or a bool. ``xs[i]`` and ``xs[i] = v`` are Python's rule
  exactly (``_Runner.ev_Subscript`` / ``_store_list``): ``-len <= i < len`` or the run ends in ``IndexError`` (a
  path end, not the property), a negative ``i`` counts from the end, a symbolic ``i`` selects the cell it equals
  under that path condition; ``xs[i] += v`` evaluates ``i`` once (``_Exec.st``); two arms that leave a list with
  the cells changed merge elementwise (``_ite``) and two arms that leave it with different lengths poison the
  name (``_merge_if``), so the run is inexact, never a wrong proof.
  Known finite scalar lists additionally admit scalar builtin folds, membership and elementwise equality.
  No-key symbolic sorting uses generic stable comparisons over their represented int/bool elements. Boolean singleton identity is limited to direct or singly assigned any/all results.
* statements: ``Assign`` (one ``Name`` target, or -- [internal ref] -- a tuple of distinct names unpacking the call of a
  TUPLE FUNCTION, see below, or ``xs[i]`` for a list of this scope), ``AugAssign`` (``Name`` or ``xs[i]`` target, same
  operators), ``Assert`` (no message), ``If``, ``Raise`` of a builtin exception class
  (bare or called with constants; no ``from``), ``Return`` (with a value, or bare in a
  PROCEDURE, see below, inside a def), ``Pass``, an ``Expr`` that is a constant or a
  call, and, at module level only, ``import _sv_verifier`` / ``from _sv_verifier
  import ...`` and plain ``def``. A call ``_sv_verifier.nondet_int()`` / ``nondet_bool()``
  at module level or inside a def, wherever an expression may stand, is a call the caller marked (``fresh_calls``,
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
  ``return`` is a tuple expression of the same arity (no starred element, elements audited like any
  expression); its value exists only as the right side of ``a, b = f(...)`` with exactly that many distinct names (the
  unpacking can never raise), or dropped by a bare call statement: a tuple is never stored, compared, passed, indexed or
  operated on, so no tuple value outlives its unpacking and the interpreter's elementwise ``ite`` merge of two returned
  tuples (``_symexec.py::_ite``) is exact for the int/bool elements the subset holds.
* an unused `_sv_verifier.check_type(value, int)` call, through a resolved import:
  its value expression is audited as an exact int/bool scalar, so the tested
  success contract applies without hooks or mismatch formatting. The caller must
  preserve evaluation of that expression when removing the successful check.
  A separate guarded-return mode admits literal-string returns with an int hint:
  it retains the call, uses code-point literals and type-preserving return joins,
  and requires every reaching path's value to satisfy the tested int/bool success
  contract. A possible mismatch refuses the proof; it is never erased or promoted
  into another property class. A separate AssertionError-only fallback models
  exact primitive mismatches as path ends after argument evaluation; every
  other member type is refused. An unused builtin bool hint uses these same
  strict obligations and type-preserving joins, with success ONLY for bool;
  int continues to accept int and bool. It cannot execute user hooks or catch an exit,
  and does not weaken the success-only route's obligations. String draws keep
  their existing refusal here.
* builtin type identity only as ``type(a) is type(b)`` or ``is not``, with audited
  scalar expressions and a tested type contract. Type-observing proofs preserve
  bool/int distinctions through joins, returns, operands and list cells; min/max
  and escaping type values stay refused. Other proofs/search retain their prior
  numeric merges. No general identity operation or new property class is admitted.
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
``class``, escaping ``lambda``, comprehensions (a list comprehension over ``range`` of constants is expanded BEFORE
the audit by :mod:`tharkun.sv.exact_unroll`, exactly; any other stays one and is refused), ``global`` /
``nonlocal`` / ``del``, nested ``def`` or ``import`` (also under ``if``), any ``Attribute`` (so no attribute
store, no ``__builtins__.abs = ...``), any ``Subscript`` that is not a list name indexed by an int expression
(slices, tuples as keys, a subscript of a subscript), tuples and lists as values, dicts, sets, walrus,
annotated assignment (``product_proof.py`` rewrites ``x: int = v`` and ``xs: list[int] = v`` to ``x = v`` /
``xs = v`` first), powers other than a literal integer exponent 0..8 (the range the symbolic evaluator expands exactly), ``/``, ``~``, ``& | ^``,
identity/membership outside the bounded forms above, and strings outside the tag subset.
"""
from __future__ import annotations

import ast
import builtins

from tharkun.sv.exact_unroll import EXACT_POW_MAX, _bindings, fold_int, module_constants


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


def type_identity_args(node):
    """Arguments of the one admitted type observation: type(a) is/is not type(b)."""
    if not (isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1
            and isinstance(node.ops[0], (ast.Is, ast.IsNot))):
        return None
    calls = [node.left, node.comparators[0]]
    if all(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "type"
           and len(c.args) == 1 and not c.keywords for c in calls):
        return calls[0].args[0], calls[1].args[0]
    return None


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
    def __init__(self, tree: ast.Module, lifted_ids: set, fresh_calls: frozenset = frozenset(), integer_checks=None,
                 string_draws=False, guarded_type_checks=None, closed_callables=None):
        self.tree = tree
        self.lifted_ids = lifted_ids
        self.fresh_calls = fresh_calls       # id() of the draw calls evaluated at their own sites
        self.reserved: set = set()           # aliases of _sv_verifier
        self.verifier_modules: set = set()
        self.check_type_functions: set = set()
        self.integer_checks = integer_checks if integer_checks is not None else set()
        self.guarded_type_checks = guarded_type_checks
        self.closed_callables = closed_callables
        self.cur_function = None
        self.string_draws = string_draws
        self.type_observed = guarded_type_checks is not None or any(type_identity_args(n) is not None for n in ast.walk(tree))
        self.defs: dict = {}                 # name -> FunctionDef
        self.procedures: set = set()         # the defs that return no value (bare return / fall off the end)
        self.factory_targets: dict = {}    # pure zero-argument factories -> returned module def
        self.tuple_defs: dict = {}           # name -> arity of the defs whose every return is a tuple ([internal ref])
        self.string_names: set = set()       # module-level literal tags; every other use is refused
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
        if self.closed_callables is not None and any(isinstance(n, (
                ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Lambda)) for n in ast.walk(self.tree)):
            _no("closed callable flow is scalar-only, without containers or lambdas")
        body = []
        draws = []
        for st in self.tree.body:
            if isinstance(st, ast.Import):
                for a in st.names:
                    if a.name != "_sv_verifier":
                        _no(f"imports {a.name!r}", st)
                    self._alias(a.asname or a.name, st)
                    self.verifier_modules.add(a.asname or a.name)
            elif isinstance(st, ast.ImportFrom):
                if st.module != "_sv_verifier" or st.level:
                    _no("imports from a module other than _sv_verifier", st)
                for a in st.names:
                    if a.name == "*":
                        _no("star import", st)
                    self._alias(a.asname or a.name, st)
                    if a.name == "check_type":
                        self.check_type_functions.add(a.asname or a.name)
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
        self.string_names = {n.targets[0].id for st in body if not isinstance(st, ast.FunctionDef)
                             for n in ast.walk(st) if isinstance(n, ast.Assign) and len(n.targets) == 1
                             and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)
                             and type(n.value.value) is str}
        if self.string_draws:
            self.string_names.clear()  # run_exact uses the audited code-point model for every string literal.
        changed = True
        while changed:
            changed = False
            for fd in self.defs.values():
                if fd.name in self.factory_targets or fd.args.args or len(fd.body) != 1 \
                        or not isinstance(fd.body[0], ast.Return):
                    continue
                value = fd.body[0].value
                target = (value.id if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load)
                          and value.id in self.defs else self._factory_result(value))
                if target is not None:
                    self.factory_targets[fd.name] = target
                    changed = True
        for fd in self.defs.values():
            self._classify(fd)
        module_body = [s for s in body if not isinstance(s, ast.FunctionDef)]
        self._type_lists(module_body)
        for fd in self.defs.values():
            self._def(fd)
        self.cur_function = None
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
            if any(isinstance(e, ast.Starred) for r in tuples for e in r.value.elts):
                _no(f"def {fd.name} returns a starred tuple", fd)
            self.tuple_defs[fd.name] = next(iter(arities))
        if bare or (not rets and not _always_ends(fd.body)):
            self.procedures.add(fd.name)
        elif not _always_ends(fd.body):
            _no(f"def {fd.name} can fall off its end (implicit None)", fd)

    def _def(self, fd: ast.FunctionDef) -> None:
        self.cur_function = fd.name
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
        # Pure factory bodies were checked above; their results may only be callees.
        if fd.name not in self.factory_targets:
            self._stmts(fd.body, scope, True)

    # -- lists ([internal ref] PR 3) ---------------------------------------------------------
    def _is_list_expr(self, v) -> bool:
        return isinstance(v, ast.List) or (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult)
                                           and (isinstance(v.left, ast.List) or isinstance(v.right, ast.List))) or (
            isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "sorted")

    def _sole_assignment(self, name):
        if _bindings(self.tree).get(name) != 1:
            return None
        return next((n.value for n in ast.walk(self.tree) if isinstance(n, ast.Assign)
                     and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == name), None)

    def _finite_list_expr(self, value, scope):
        """An existing finite scalar-list representation; lengths and values are not bounded."""
        if self.string_draws:
            _no("finite-list consumers require integer/bool scalars", value)
        if isinstance(value, ast.Name) and value.id in self.visible_lists and value.id in scope:
            return
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "sorted":
            if len(value.args) == 1 and not value.keywords:
                self._finite_list_expr(value.args[0], scope)
                return
        elif self._is_list_expr(value):
            self._list_expr(value, scope)
            return
        _no("consumer requires a known finite scalar list", value)

    def _truth_fold(self, value):
        if isinstance(value, ast.Name):
            value = self._sole_assignment(value.id)
        return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id in {"any", "all"} and len(value.args) == 1 and not value.keywords)

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
                elif not self.string_draws and isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len" \
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
        """A fresh scalar list: finite literal, known-count repetition, or a fresh sorted list."""
        if isinstance(v, ast.Call):
            self._call(v, scope)
            return
        if isinstance(v, ast.List):
            for e in v.elts:
                self._expr(e, scope)
            return
        lst, count = (v.left, v.right) if isinstance(v.left, ast.List) else (v.right, v.left)
        size = fold_int(count, self.consts)
        if size is None:
            _no("list repetition whose count is not a known constant integer", v)
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
                callable_value = self.closed_callables is not None and bool(
                    self.closed_callables.name_targets(name, self.cur_function))
                self._expr(st.value, scope, strings=not in_def and isinstance(st.value, ast.Constant),
                           callables=callable_value)
        elif isinstance(st, ast.AugAssign):
            if isinstance(st.target, ast.Subscript) and isinstance(st.op, _BINOPS):
                self._subscript_target(st.target, scope)
                self._expr(st.value, scope)
                return
            if not isinstance(st.target, ast.Name) or not isinstance(st.op, _BINOPS):
                _no("augmented assignment shape", st)
            if st.target.id not in scope:
                _no("augmented assignment to an unbound name", st)
            if st.target.id in self.string_names:
                _no("augmented assignment to a string tag", st)
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
            _no("for loop that is not range(<constant>) without break / continue / else within the AST work budget "
                "(sv/exact_unroll.py unrolls those before the audit)", st)
        elif isinstance(st, ast.While):
            _no("while loop that is not a counter loop (v = <constant> just before it, while v < <constant>, v += <constant> "
                "last in the body, no other store of v, no break / continue / else, within the AST work budget; sv/exact_unroll.py "
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
    def _expr(self, n, scope: set, *, strings: bool = False, callables: bool = False) -> None:
        if callables:
            if self.closed_callables is None or not self.closed_callables.targets(n, self.cur_function):
                _no("callable value is not a closed source def", n)
            if isinstance(n, ast.Name) and (n.id in scope or n.id in self.defs):
                return
            if isinstance(n, ast.IfExp):
                self._expr(n.test, scope)
                self._expr(n.body, scope, callables=True)
                self._expr(n.orelse, scope, callables=True)
                return
            _no("unsupported callable value expression", n)
        if isinstance(n, ast.Name) and self.closed_callables is not None \
                and self.closed_callables.name_targets(n.id, self.cur_function):
            _no("closed callable used as a scalar", n)
        if isinstance(n, ast.Constant):
            if self.string_draws and type(n.value) is str:
                return
            if strings and type(n.value) is str:
                # shortcut: printable ASCII without backslashes only; audit Z3's encoding before widening.
                if any(not 32 <= ord(c) <= 126 or c == "\\" for c in n.value):
                    _no("string tag outside the exact literal encoding", n)
                return
            if type(n.value) not in (int, bool):
                _no(f"constant of type {type(n.value).__name__}", n)
        elif isinstance(n, ast.Name):
            if not isinstance(n.ctx, ast.Load) or n.id not in scope:
                _no(f"name {n.id!r} is not a bound variable here", n)
            if n.id in self.string_names and not strings:
                _no("string tag used outside equality or inequality", n)
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
            if isinstance(n.op, ast.Pow):
                if (self.string_draws or not isinstance(n.right, ast.Constant)
                        or type(n.right.value) is not int or not 0 <= n.right.value <= EXACT_POW_MAX):
                    _no(f"exact integer power requires a literal exponent in 0..{EXACT_POW_MAX}", n)
                self._expr(n.left, scope)
                return
            if not isinstance(n.op, _BINOPS):
                _no(f"binary operator {type(n.op).__name__}", n)
            self._expr(n.left, scope)
            self._expr(n.right, scope)
        elif isinstance(n, ast.BoolOp):
            for v in n.values:
                self._expr(v, scope)
        elif isinstance(n, ast.Compare):
            if len(n.ops) == 1:
                right, op = n.comparators[0], n.ops[0]
                if isinstance(op, (ast.In, ast.NotIn)):
                    self._expr(n.left, scope)
                    self._finite_list_expr(right, scope)
                    return
                if isinstance(op, (ast.Eq, ast.NotEq)) and (
                        isinstance(n.left, ast.List) or isinstance(n.left, ast.Name)
                        and n.left.id in self.visible_lists):
                    self._finite_list_expr(n.left, scope)
                    self._finite_list_expr(right, scope)
                    return
                if (isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant)
                        and type(right.value) is bool and self._truth_fold(n.left)):
                    self._expr(n.left, scope)
                    return
            if (self.closed_callables is not None and len(n.ops) == 1
                    and isinstance(n.ops[0], (ast.Is, ast.IsNot))
                    and self.closed_callables.targets(n.left, self.cur_function)
                    and self.closed_callables.targets(n.comparators[0], self.cur_function)):
                self._expr(n.left, scope, callables=True)
                self._expr(n.comparators[0], scope, callables=True)
                return
            observed = type_identity_args(n)
            if observed is not None:
                if _builtin_arity("type") != 1:
                    _no("no tested builtin type contract", n)
                for arg in observed:
                    self._expr(arg, scope)
                return
            if not all(isinstance(op, _CMPOPS) for op in n.ops):
                _no("comparison shape (is, in)", n)
            if self.string_draws and len(n.ops) > 1:
                # The eager chain interpreter's totality argument assumes int/bool operands.
                _no("comparison chain with string draws needs short-circuit modelling", n)
            if len(n.ops) > 1 and not all(_total(x) for x in [n.left, *n.comparators]):
                _no("comparison chain over an operand that can raise or constrain the path", n)
            tags = all(isinstance(op, (ast.Eq, ast.NotEq)) for op in n.ops)
            self._expr(n.left, scope, strings=tags)
            for c in n.comparators:
                self._expr(c, scope, strings=tags)
        elif isinstance(n, ast.IfExp):
            self._expr(n.test, scope)
            self._expr(n.body, scope)
            self._expr(n.orelse, scope)
        elif isinstance(n, ast.Call):
            self._call(n, scope)
        else:
            _no(f"expression kind {type(n).__name__}", n)

    def _factory_result(self, call):
        if not isinstance(call, ast.Call) or call.args or call.keywords:
            return None
        name = call.func.id if isinstance(call.func, ast.Name) else self._factory_result(call.func)
        return self.factory_targets.get(name)

    def _call(self, n: ast.Call, scope: set, statement: bool = False, unpack: bool = False, callee: bool = False) -> None:
        if id(n) in self.fresh_calls:
            return                           # a draw: the caller checked its shape (no argument, int / bool)
        if isinstance(n.func, ast.Attribute) and n.func.attr in {"startswith", "endswith"}:
            from tharkun.solve.contracts.rules import get_rule
            from tharkun.solve.contracts.visitors import ProductVisitor
            rule = get_rule("str." + n.func.attr)
            if (n.keywords or not 1 <= len(n.args) <= 3 or rule is None
                    or rule.accept(ProductVisitor()) is not True):
                _no("string edge requires a tested positional contract", n)
            self._expr(n.func.value, scope, strings=True)
            def prefix(value):
                if isinstance(value, ast.Tuple):
                    for item in value.elts:
                        prefix(item)
                else:
                    self._expr(value, scope, strings=True)
            prefix(n.args[0])
            for value in n.args[1:]:
                if not (isinstance(value, ast.Constant) and value.value is None):
                    self._expr(value, scope)
            return
        if isinstance(n.func, ast.Lambda):
            lam = n.func
            args = lam.args
            if n.keywords or args.defaults or args.kw_defaults or args.kwonlyargs or args.posonlyargs \
                    or args.vararg or args.kwarg or len(n.args) != len(args.args):
                _no("direct lambda call shape", n)
            params = [a.arg for a in args.args]
            if len(params) != len(set(params)):
                _no("duplicate lambda parameter", lam)
            for name in params:
                _name_bindable(name, self.reserved, lam)
                if name in self.defs:
                    _no("lambda parameter shadows a def", lam)
            free = {x.id for x in ast.walk(lam.body) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
            if free - set(params):
                _no("lambda body reads a non-parameter name", lam)
            self._expr(lam.body, set(params))
            for value in n.args:
                self._expr(value, scope)
            return
        is_check_type = (isinstance(n.func, ast.Name) and n.func.id in self.check_type_functions) or (
            isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
            and n.func.value.id in self.verifier_modules and n.func.attr == "check_type")
        if is_check_type:
            if (statement and not n.keywords and len(n.args) == 2
                    and isinstance(n.args[1], ast.Name) and n.args[1].id == "str"):
                if self.guarded_type_checks is not None:
                    self._expr(n.args[0], scope)
                    self.guarded_type_checks.add(id(n))
                    return  # keep every reaching value's string-success obligation
                # Authenticate a string from straight-line source; preserve argument evaluation.
                def is_string(value, names, active=frozenset()):
                    if isinstance(value, ast.Constant):
                        return type(value.value) is str
                    if isinstance(value, ast.Name):
                        return names.get(value.id, False)
                    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                        return is_string(value.left, names, active) and is_string(value.right, names, active)
                    if not isinstance(value, ast.Call) or value.keywords:
                        return False
                    func = value.func
                    draw = ((isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                             and func.value.id in self.verifier_modules and func.attr == "nondet_str")
                            or (isinstance(func, ast.Name) and any(
                                isinstance(st, ast.ImportFrom) and st.module == "_sv_verifier"
                                and any(a.name == "nondet_str" and (a.asname or a.name) == func.id for a in st.names)
                                for st in self.tree.body)))
                    if draw:
                        return not value.args
                    if not isinstance(func, ast.Name) or func.id not in self.defs or func.id in active:
                        return False
                    fd = self.defs[func.id]
                    if len(fd.body) != 1 or not isinstance(fd.body[0], ast.Return) \
                            or len(value.args) != len(fd.args.args):
                        return False
                    params = {a.arg: is_string(v, names, active) for a, v in zip(fd.args.args, value.args)}
                    return is_string(fd.body[0].value, params, active | {fd.name})

                names = {}
                for st in self.tree.body:
                    if isinstance(st, ast.Expr) and st.value is n:
                        if not is_string(n.args[0], names):
                            _no("check_type str value is not source-proven primitive string", n)
                        self._expr(n.args[0], scope)
                        self.integer_checks.add(id(n))
                        return
                    if isinstance(st, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
                        continue
                    if (isinstance(st, ast.Assign) and len(st.targets) == 1
                            and isinstance(st.targets[0], ast.Name)):
                        names[st.targets[0].id] = is_string(st.value, names)
                    else:
                        break
                _no("check_type str requires straight-line module source", n)
            if (len(n.args) == 2 and isinstance(n.args[1], ast.Name) and n.args[1].id == "bool"):
                if not statement or n.keywords or self.guarded_type_checks is None:
                    _no("bool check_type requires an unused value and strict success obligations", n)
                from tharkun.solve.contracts.rules import get_rule
                from tharkun.solve.contracts.visitors import ProductVisitor
                rule = get_rule("_sv_verifier.check_type")
                if rule is None or rule.accept(ProductVisitor()) is not True:
                    _no("no tested check_type success contract", n)
                self._expr(n.args[0], scope)
                self.guarded_type_checks.add(id(n))
                return
            if self.guarded_type_checks is not None:
                if (not statement or n.keywords or len(n.args) != 2
                        or not isinstance(n.args[1], ast.Name) or n.args[1].id != "int"):
                    _no("guarded check_type requires an unused scalar value and builtin int hint", n)
                from tharkun.solve.contracts.rules import get_rule
                from tharkun.solve.contracts.visitors import ProductVisitor
                rule = get_rule("_sv_verifier.check_type")
                if rule is None or rule.accept(ProductVisitor()) is not True:
                    _no("no tested check_type success contract", n)
                self._expr(n.args[0], scope)
                self.guarded_type_checks.add(id(n))
                return
            if self.string_draws:
                _no("int-only check_type rewrite cannot admit string draws", n)
            if (not statement or n.keywords or len(n.args) != 2
                    or not isinstance(n.args[1], ast.Name) or n.args[1].id != "int"):
                _no("exact check_type requires an unused scalar value and builtin int hint", n)
            from tharkun.solve.contracts.rules import get_rule
            from tharkun.solve.contracts.visitors import ProductVisitor
            rule = get_rule("_sv_verifier.check_type")
            if rule is None or rule.accept(ProductVisitor()) is not True:
                _no("no tested check_type success contract", n)
            self._expr(n.args[0], scope)  # this subset admits only exact int/bool scalar values
            self.integer_checks.add(id(n))
            return
        targets = self.closed_callables.targets(n.func, self.cur_function) if self.closed_callables else ()
        if targets:
            from tharkun.sv.exact_callables import binding
            self._expr(n.func, scope, callables=True)
            arguments = [*n.args, *(k.value for k in n.keywords)]
            for name in targets:
                if name in self.factory_targets and not (statement or callee):
                    _no("callable factory result used as a scalar", n)
                positions = binding(self.defs[name], n)
                if positions is None:
                    _no("unsupported closed callable argument binding", n)
                if name in self.procedures and not statement:
                    _no("closed procedure result used as a value", n)
                for param, index in zip(self.defs[name].args.args, positions):
                    self._expr(arguments[index], scope, callables=bool(
                        self.closed_callables.values[name].get(param.arg)))
            return
        if n.keywords:
            _no("call shape", n)
        if isinstance(n.func, ast.Call):
            name = self._factory_result(n.func)
            if name is None:
                _no("callee is not a known factory return", n)
            self._call(n.func, scope, callee=True)
        elif isinstance(n.func, ast.Name):
            name = n.func.id
        else:
            _no("call shape", n)
        if name in self.factory_targets and not (statement or callee):
            _no("callable factory result used as a scalar", n)
        if self.type_observed and name in {"min", "max"}:
            _no("type-observed min/max need an audited tie-preserving value model", n)
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
            if self.string_draws and len(n.args) == 1:
                self._expr(n.args[0], scope)
                return
            if not (len(n.args) == 1 and isinstance(n.args[0], ast.Name) and n.args[0].id in self.visible_lists
                    and n.args[0].id in scope):
                if len(n.args) != 1:
                    _no("len of something that is not a list", n)
                self._finite_list_expr(n.args[0], scope)
            return
        elif name == "type":
            _no("type values may only be compared directly by builtin type identity", n)
        elif name in {"sum", "min", "max", "sorted", "any", "all"} and len(n.args) == 1:
            if name == "sorted" and not any(isinstance(st, ast.Assign) and st.value is n
                    and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                    for st in ast.walk(self.tree)):
                _no("sorted result must be a fresh named list or finite-list comparison", n)
            self._finite_list_expr(n.args[0], scope)
            return
        elif not name.startswith("__") and _builtin_arity(name) is not None:
            if len(n.args) != _builtin_arity(name):
                _no(f"call of {name} with the wrong arity", n)
        else:
            _no(f"call of {name!r}", n)
        for a in n.args:
            self._expr(a, scope)


def audit(tree: ast.Module, lifted_ids: set, fresh_calls: frozenset = frozenset(), *, integer_checks=None,
          string_draws=False, guarded_type_checks=None, closed_callables=None) -> str | None:
    """``None`` when the WHOLE module sits inside the allowlist (see the module
    docstring), else the reason it does not. ``lifted_ids``: ``id()`` of the top-level
    ``name = _sv_verifier.nondet_TYPE()`` statements the caller already lifted;
    ``fresh_calls``: ``id()`` of the ``_sv_verifier.nondet_TYPE()`` calls evaluated at their own sites, which the
    caller checked have that exact shape. ``string_draws`` means the caller uses
    proof-only code-point sequences for every string value; it is not permission
    to use the search executor's string theory. ``guarded_type_checks`` collects
    authenticated int/bool-hint calls whose path-sensitive success obligations the
    caller must discharge before proving the original property."""
    try:
        auditor = _Audit
        if any(isinstance(n, ast.ClassDef) for n in tree.body):
            from tharkun.sv.exact_classes import ClassAudit
            auditor = ClassAudit
        auditor(tree, lifted_ids, fresh_calls, integer_checks, string_draws, guarded_type_checks, closed_callables).run()
    except _Refuse as e:
        return str(e)
    return None
