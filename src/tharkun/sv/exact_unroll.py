"""tharkun.sv.exact_unroll: exact unrolling of ``for i in range(<constant>)`` and ``[e for i in range(<constant>)]`` for the
product proof ([internal ref] PR 2 and PR 3).

The product proof (:mod:`tharkun.sv.product_proof`) runs a program through :mod:`tharkun.sv._symexec` and calls the run a
PROOF only when the whole module sits inside the audited allowlist of :mod:`tharkun.sv.exact_subset`; a loop is not in it
(an interpreter that unrolls to a bound can only say "no modelled run fires the site", never "none does").  A ``for`` over
``range`` of CONSTANTS has a trip count known before the program runs, so it can be replaced by the statements it stands
for, and the proof then runs on straight-line code the audit already knows:

    for i in range(3):              i = 0
        body              ==>       body
                                    i = 1
                                    body
                                    i = 2
                                    body

The replacement is exact, not a bound: the iterations are the whole loop.  What makes it exact (each point is pinned by
``tests/test_sv_product_proof_subset.py``):

* the iterable is ``range(stop)``, ``range(start, stop)`` or ``range(start, stop, step)`` with every argument a literal
  int or a NAME that is bound exactly once in the whole module (any scope, any binding form) to a foldable int
  expression (``K = 4``, ``INT_BOUND = 1 << 30``) or -- since PR 4 -- as a def's parameter every call passes the same
  foldable int and whose loads are all inside that def's body (:func:`_param_constants`), so ``range`` is evaluated once,
  to one fixed sequence, and the name
  cannot mean anything else inside a function (``range`` itself is the builtin: the audit never lets a program bind a
  builtin name);
* the target is a plain name, assigned before EVERY copy of the body (a body that rebinds ``i`` does not change the next
  iteration's value, exactly as in Python) and left at the last value afterwards (or untouched when the range is empty);
* no ``else`` clause and no ``break`` / ``continue`` anywhere in the body (a ``return`` is fine: it is a statement of the
  copy, and the interpreter's early-return merge handles it like any other); a step of ``0`` is not unrolled (``range``
  raises ``ValueError``: the audit refuses the remaining ``for``);
* at most :data:`MAX_TRIPS` iterations and :data:`MAX_NODES` AST nodes in all, so a program cannot make the proof's
  input explode;
* every copy is a deep copy that keeps the ORIGINAL line numbers, so the violation sites (``violation_sites`` is read from
  the original source) are the same lines in every copy, and each copy's draw calls are distinct nodes, so each iteration
  draws a fresh term (``product_proof.py::_lift`` marks them per node).

A ``for`` this module does not unroll stays a ``for``, and the audit refuses it by name.

A COUNTER ``while`` ([internal ref] PR 4) is unrolled the same way: the statement right before it binds the counter to a constant
(``i = 0``), the test is ``i < | <= | > | >= <constant>`` with the counter on the left, the LAST statement of the body
steps it by a constant (``i += 1``, ``i = i + 1``, ``i = i - 2``) in the direction the test is read in (``<`` / ``<=``
with a positive step, ``>`` / ``>=`` with a negative one), nothing else in the body stores the counter, no ``break`` /
``continue`` / ``else``.  The trip count is then ``range(start, stop, step)`` (``stop + 1`` for ``<=``, ``stop - 1`` for
``>=``), decided before the program runs, and the loop is replaced by that many deep copies of its body (the step
included, so the counter is right in every copy and left at the first value failing the test, exactly as in Python; a
``return`` in the body is a statement of its copy, the interpreter's early-return merge handles it; an empty trip count
removes the loop, the init stays).  The constants are those of :func:`module_constants`: module-level names bound once to a
foldable int AND, since PR 4, a def's PARAMETER every call passes the same foldable int (``all_not_none(xs, N)`` for
``def all_not_none(xs, n)``: the name is bound exactly once in the whole module, every load is inside the owning def's
body, the def never assigns it, and its runtime value is the folded one at every call).  Any other ``while`` stays, and the
audit refuses it by name.

A LIST COMPREHENSION over the same kind of ``range`` (one generator, no ``if``, a plain-name target) is replaced by the list
literal of its elements, ``[e(0), e(1), e(2)]``: the target is the comprehension's own name, so each copy of ``e`` has the
trip's value written in for it (a ``Name`` load of the target becomes the constant) and nothing is left bound.  An element that
holds a lambda, another comprehension or a walrus is not expanded (they could bind the target's name again); every copy is a
deep copy, so each draw call in ``e`` is a distinct node and each element draws a fresh value, as in Python.  Whether the list
the expansion leaves is something the proof may use is the audit's call (``exact_subset.py``).
"""
from __future__ import annotations

import ast
import copy
import operator

#: the most iterations of one loop
MAX_TRIPS = 64
#: the most AST nodes the unrolled copies of one loop may hold
MAX_NODES = 20000
_FOLD = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
         ast.Mod: operator.mod, ast.LShift: operator.lshift, ast.RShift: operator.rshift}
_BIG = 1 << 62


def fold_int(node, consts: dict):
    """The int value of ``node`` (literals, names of ``consts``, ``- + * // % << >>``), or ``None``."""
    if isinstance(node, ast.Constant):
        return node.value if type(node.value) is int else None
    if isinstance(node, ast.Name):
        return consts.get(node.id) if isinstance(node.ctx, ast.Load) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = fold_int(node.operand, consts)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp) and type(node.op) in _FOLD:
        a, b = fold_int(node.left, consts), fold_int(node.right, consts)
        if a is None or b is None or abs(a) > _BIG or abs(b) > _BIG:
            return None
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and b == 0:
            return None
        if isinstance(node.op, (ast.LShift, ast.RShift)) and not 0 <= b <= 64:
            return None
        v = _FOLD[type(node.op)](a, b)
        return v if abs(v) <= _BIG else None
    return None


def _bindings(tree: ast.Module) -> dict:
    """``{name: number of bindings}`` over the WHOLE module: stores and deletes, parameters, defs, classes, imports,
    ``except ... as``, ``global`` / ``nonlocal`` (counted twice: they make the name another scope's)."""
    count: dict = {}

    def bump(name, n=1):
        count[name] = count.get(name, 0) + n

    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bump(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bump(n.name)
        elif isinstance(n, ast.arg):
            bump(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bump((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bump(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            for name in n.names:
                bump(name, 2)
    return count


def _param_constants(tree: ast.Module, consts: dict, count: dict) -> None:
    """Add to ``consts`` the def PARAMETERS every call passes the same foldable int (``all_not_none(xs, N)`` for
    ``def all_not_none(xs, n)`` makes ``n`` the constant ``N``): the def is a top-level plain positional ``def`` bound once
    and called at least once, the parameter's name is bound exactly once in the whole module (as that parameter, and the def
    never assigns it), every load is inside that def's body, and every call of the def is positional with the
    argument at that place folding to the one same value.  To a fixpoint, since an argument may be another constant name."""
    defs = {st.name: st for st in tree.body if isinstance(st, ast.FunctionDef) and count.get(st.name) == 1
            and not (st.args.posonlyargs or st.args.kwonlyargs or st.args.vararg or st.args.kwarg or st.args.defaults)}
    calls: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defs:
            calls.setdefault(n.func.id, []).append(n)
    changed = True
    while changed:
        changed = False
        for name, fd in defs.items():
            body_nodes = {id(n) for st in fd.body for n in ast.walk(st)}
            outside_loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
                             and isinstance(n.ctx, ast.Load) and id(n) not in body_nodes}
            for j, a in enumerate(fd.args.args):
                if a.arg in consts or count.get(a.arg) != 1 or not calls.get(name) or a.arg in outside_loads:
                    continue
                vals = {fold_int(c.args[j], consts) if j < len(c.args) and not c.keywords else None for c in calls[name]}
                if len(vals) == 1 and None not in vals:
                    consts[a.arg] = vals.pop()
                    changed = True


def module_constants(tree: ast.Module) -> dict:
    """Names bound exactly once in the whole module to a foldable int: by a top-level ``name = <expression>``, or as a def's
    parameter every call passes the same value (:func:`_param_constants`)."""
    count = _bindings(tree)
    consts: dict = {}
    for st in tree.body:
        if (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                and count.get(st.targets[0].id) == 1):
            v = fold_int(st.value, consts)
            if v is not None:
                consts[st.targets[0].id] = v
    _param_constants(tree, consts, count)
    return consts


def _trips(start: int, stop: int, step: int) -> list | None:
    """The exact range values, or ``None`` when their count exceeds :data:`MAX_TRIPS`."""
    values = range(start, stop, step)
    try:
        if len(values) > MAX_TRIPS:
            return None
    except OverflowError:
        return None
    return list(values)


def _range_args(node: ast.For, consts: dict):
    it = node.iter
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
            and not it.keywords and 1 <= len(it.args) <= 3):
        return None
    vals = [fold_int(a, consts) for a in it.args]
    if any(v is None for v in vals):
        return None
    start, stop, step = ((0, vals[0], 1) if len(vals) == 1 else (vals[0], vals[1], 1) if len(vals) == 2 else vals)
    return None if step == 0 else (start, stop, step)


_STEP_OPS = {ast.Add: 1, ast.Sub: -1}
#: the test operators of a counter loop, with what ``range`` needs added to the bound
_TEST_OPS = {ast.Lt: 0, ast.LtE: 1, ast.Gt: 0, ast.GtE: -1}


def _counter_step(st, name: str, consts: dict):
    """The signed constant ``st`` adds to ``name`` (``name += c``, ``name = name + c``, ``name = name - c``), or ``None``."""
    if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and st.target.id == name \
            and type(st.op) in _STEP_OPS:
        op, c = st.op, fold_int(st.value, consts)
    elif (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
          and st.targets[0].id == name and isinstance(st.value, ast.BinOp) and type(st.value.op) in _STEP_OPS
          and isinstance(st.value.left, ast.Name) and st.value.left.id == name):
        op, c = st.value.op, fold_int(st.value.right, consts)
    else:
        return None
    return None if c is None else c * _STEP_OPS[type(op)]


def _counter_trips(init, loop: ast.While, consts: dict):
    """The values the counter of ``loop`` takes when ``init`` (the statement just before it) and ``loop`` are a counter
    loop (see the module docstring), else ``None``."""
    if loop.orelse or not loop.body or not (isinstance(init, ast.Assign) and len(init.targets) == 1
                                            and isinstance(init.targets[0], ast.Name)):
        return None
    name = init.targets[0].id
    start = fold_int(init.value, consts)
    t = loop.test
    if start is None or not (isinstance(t, ast.Compare) and len(t.ops) == 1 and type(t.ops[0]) in _TEST_OPS
                             and isinstance(t.left, ast.Name) and t.left.id == name):
        return None
    stop = fold_int(t.comparators[0], consts)
    step = _counter_step(loop.body[-1], name, consts)
    if stop is None or step is None or step == 0 or (step > 0) != (type(t.ops[0]) in (ast.Lt, ast.LtE)) \
            or not _body_leaves_the_counter(loop.body[:-1], name):
        return None
    return _trips(start, stop + _TEST_OPS[type(t.ops[0])], step)


def _body_leaves_the_counter(stmts, name: str) -> bool:
    """No statement of ``stmts`` (the body before its step) stores, deletes or declares ``name``, and none breaks or
    continues: the counter is then exactly ``start``, ``start + step``, ... at the top of each copy."""
    for st in stmts:
        for n in ast.walk(st):
            if isinstance(n, (ast.Break, ast.Continue)) \
                    or (isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load)) \
                    or (isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names):
                return False
    return True


def _fuse_counter_loops(stmts: list, consts: dict) -> list:
    """``stmts`` with every counter ``while`` (its init is the statement before it in this very list) replaced by the
    copies of its body."""
    out: list = []
    for st in stmts:
        if isinstance(st, ast.While) and out:
            values = _counter_trips(out[-1], st, consts)
            if values is not None:
                size = sum(1 for s in st.body for _ in ast.walk(s))
                if len(values) <= MAX_TRIPS and size * len(values) <= MAX_NODES:
                    for _ in values:
                        out.extend(copy.deepcopy(s) for s in st.body)
                    continue
        out.append(st)
    return out


class _Unroll(ast.NodeTransformer):
    def __init__(self, consts: dict):
        self.consts = consts

    def generic_visit(self, node):
        """Children first (an inner loop is plain statements by then), then every statement list of ``node`` has its
        counter ``while`` loops fused (``_fuse_counter_loops``: a ``while`` and the init just before it)."""
        super().generic_visit(node)
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(x, ast.stmt) for x in value):
                setattr(node, field, _fuse_counter_loops(value, self.consts))
        return node

    def visit_For(self, node: ast.For):
        self.generic_visit(node)             # inner loops first: an unrolled inner loop is plain statements here
        if not isinstance(node.target, ast.Name) or node.orelse:
            return node
        args = _range_args(node, self.consts)
        if args is None:
            return node
        if any(isinstance(n, (ast.Break, ast.Continue)) for st in node.body for n in ast.walk(st)):
            return node
        values = _trips(*args)
        if values is None:
            return node
        size = sum(1 for st in node.body for _ in ast.walk(st)) + 3
        if len(values) > MAX_TRIPS or size * len(values) > MAX_NODES:
            return node
        out: list = []
        for k in values:
            bind = ast.Assign(targets=[ast.Name(node.target.id, ast.Store())], value=ast.Constant(k))
            out.append(ast.copy_location(ast.fix_missing_locations(bind), node))
            out.extend(copy.deepcopy(st) for st in node.body)
        return out

    def visit_ListComp(self, node: ast.ListComp):
        self.generic_visit(node)             # a comprehension inside the element (or the bounds) first
        return _expand_comprehension(node, self.consts) or node


class _Substitute(ast.NodeTransformer):
    """The constant ``value`` for every load of the name ``name`` (the element of a comprehension, for one trip)."""

    def __init__(self, name: str, value: int):
        self.name, self.value = name, value

    def visit_Name(self, node: ast.Name):
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(self.value), node)
        return node


_REBINDING = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.NamedExpr)


def _expand_comprehension(node: ast.ListComp, consts: dict):
    """The ``ast.List`` ``node`` stands for, or ``None`` when it is not exactly expandable."""
    if len(node.generators) != 1:
        return None
    gen = node.generators[0]
    if gen.is_async or gen.ifs or not isinstance(gen.target, ast.Name):
        return None
    args = _range_args(gen, consts)
    if args is None or any(isinstance(n, _REBINDING) for n in ast.walk(node.elt)):
        return None
    values = _trips(*args)
    if values is None:
        return None
    size = sum(1 for _ in ast.walk(node.elt)) + 2
    if len(values) > MAX_TRIPS or size * len(values) > MAX_NODES:
        return None
    elts = [_Substitute(gen.target.id, k).visit(copy.deepcopy(node.elt)) for k in values]
    return ast.copy_location(ast.List(elts=elts, ctx=ast.Load()), node)


def unroll_for_loops(tree: ast.Module) -> None:
    """Replace, in place, every ``for``, counter ``while`` and list comprehension this module can unroll exactly (see the
    module docstring)."""
    _Unroll(module_constants(tree)).visit(tree)
