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
* integer folding uses a cumulative :data:`FOLD_WORK_BUDGET` bit-work allowance, shared with the later source audit;
  exhaustion leaves the operation unexpanded rather than truncating a value or renewing work for another expression;
* at most :data:`MAX_NODES` generated AST nodes per expansion, costed before materializing the range
  or copying its body; this is a work budget, not a bound on sequence lengths or integer values;
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

#: the most AST nodes the unrolled copies of one loop may hold
MAX_NODES = 20000
# The largest constant integer exponent the symbolic evaluator (`_symexec.ev_BinOp`) expands EXACTLY (repeated multiplication); the audit admits
# exactly the powers the evaluator can decide, not one spelling of them.
EXACT_POW_MAX = 8
_FOLD = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
         ast.Mod: operator.mod, ast.LShift: operator.lshift, ast.RShift: operator.rshift}
#: Cumulative integer bit-work per normalization and its subsequent source audit.
FOLD_WORK_BUDGET = 1_000_000
_DEFAULT_INT = 1 << 62


class _FoldWork:
    def __init__(self):
        self.remaining = FOLD_WORK_BUDGET

    def spend(self, cost):
        if cost > self.remaining:
            self.remaining = 0
            return False
        self.remaining -= cost
        return True


def _fold_observed_names(tree):
    """Bindings reaching a range, counter condition or repetition count, including direct-call parameters."""
    nodes = tuple(ast.walk(tree))
    reads = lambda node: {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    wanted, dependencies = set(), {}
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    dependencies.setdefault(target.id, set()).update(reads(node.value))
        if isinstance(node, ast.While):
            wanted.update(reads(node.test))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if isinstance(node.left, ast.List):
                wanted.update(reads(node.right))
            if isinstance(node.right, ast.List):
                wanted.update(reads(node.left))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == 'range':
                for argument in node.args:
                    wanted.update(reads(argument))
            if node.func.id in defs:
                for param, arg in zip(defs[node.func.id].args.args, node.args):
                    dependencies.setdefault(param.arg, set()).update(reads(arg))
    pending = list(wanted)
    while pending:
        for name in dependencies.get(pending.pop(), set()) - wanted:
            wanted.add(name)
            pending.append(name)
    return wanted


class _FoldConstants(dict):
    def __init__(self, tree, values=()):
        super().__init__(values)
        self.tree = tree
        self.work = getattr(tree, '_tharkun_fold_work', None)
        self.binding = None

    def wide_work(self):
        if self.tree is not None and self.binding is not None:
            observed = getattr(self.tree, '_tharkun_fold_observed', None)
            if observed is None:
                self.tree._tharkun_fold_observed = observed = _fold_observed_names(self.tree)
            if self.binding not in observed:
                return None  # No modeled count or bound can observe this optional constant.
        if self.work is None:
            self.work = _FoldWork()
            if self.tree is not None:
                self.tree._tharkun_fold_work = self.work
        return self.work


def _wide_fold(a, b, op, consts):
    work = consts.wide_work()
    if work is None:
        return None
    if op in (ast.UAdd, ast.USub):
        cost = max(1, a.bit_length())
    elif op is ast.LShift:
        cost = a.bit_length() + b if a else 1
    elif op is ast.RShift:
        cost = max(1, a.bit_length() - b + 1)
    else:
        cost = a.bit_length() + b.bit_length() + 1
    if not work.spend(max(1, cost)):
        return None
    return (+a if op is ast.UAdd else -a) if b is None else _FOLD[op](a, b)


def fold_int(node, consts: dict):
    """Keep primitive folding cheap; allocate wide context only for compound expressions."""
    try:
        if isinstance(node, ast.Constant):
            return node.value if type(node.value) is int or type(node.value) is bool else None
        if isinstance(node, ast.Name):
            return consts.get(node.id) if isinstance(node.ctx, ast.Load) else None
        if type(consts) is dict:
            consts = _FoldConstants(None, consts)
        if isinstance(node, ast.UnaryOp):
            op = type(node.op)
            if op not in (ast.USub, ast.UAdd):
                return None
            value = fold_int(node.operand, consts)
            if value is None:
                return None
            if op is ast.UAdd:
                return +value
            return -value if abs(value) <= _DEFAULT_INT else _wide_fold(value, None, op, consts)
        if isinstance(node, ast.BinOp):
            op = type(node.op)
            operation = _FOLD.get(op)
            if operation is None:
                return None
            a, b = fold_int(node.left, consts), fold_int(node.right, consts)
            if a is None or b is None:
                return None
            if op in (ast.FloorDiv, ast.Mod) and b == 0:
                return None
            shift = op is ast.LShift or op is ast.RShift
            if shift and b < 0:
                return None
            if op is ast.LShift and a == 0:
                return 0
            if op is ast.RShift and abs(a) <= _DEFAULT_INT and b > 64:
                return -1 if a < 0 else 0
            if abs(a) > _DEFAULT_INT or abs(b) > _DEFAULT_INT or shift and b > 64:
                return _wide_fold(a, b, op, consts)
            value = operation(a, b)
            return value if abs(value) <= _DEFAULT_INT else _wide_fold(a, b, op, consts)
    except (ArithmeticError, MemoryError, RecursionError):
        if type(consts) is dict:
            consts = _FoldConstants(None, consts)
        work = consts.wide_work()
        if work is not None:
            work.remaining = 0
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
    never assigns it), every load is inside that def's body, the def is only referenced as a direct callee,
    and every call of the def is positional with the
    argument at that place folding to the one same value.  To a fixpoint, since an argument may be another constant name."""
    direct = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    # A returned function can be called with arguments the named-call census never sees.
    escaped = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
               and id(n) not in direct}
    defs = {st.name: st for st in tree.body if isinstance(st, ast.FunctionDef) and count.get(st.name) == 1
            and st.name not in escaped
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
                consts.binding = a.arg
                vals = {fold_int(c.args[j], consts) if j < len(c.args) and not c.keywords else None for c in calls[name]}
                consts.binding = None
                if len(vals) == 1 and None not in vals:
                    consts[a.arg] = vals.pop()
                    changed = True


def module_constants(tree: ast.Module) -> dict:
    """Names bound exactly once in the whole module to a foldable int: by a top-level ``name = <expression>``, or as a def's
    parameter every call passes the same value (:func:`_param_constants`)."""
    count = _bindings(tree)
    consts = _FoldConstants(tree)
    for st in tree.body:
        if (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                and count.get(st.targets[0].id) == 1):
            consts.binding = st.targets[0].id
            v = fold_int(st.value, consts)
            if v is not None:
                consts[st.targets[0].id] = v
    consts.binding = None
    _param_constants(tree, consts, count)
    return consts


def _trips(start: int, stop: int, step: int, *, node_cost: int = 1, node_overhead: int = 0) -> list | None:
    """Exact values, or ``None`` before materialization would exhaust the expansion work budget."""
    values = range(start, stop, step)
    try:
        if node_overhead + len(values) * node_cost > MAX_NODES:
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
    size = sum(1 for st in loop.body for _ in ast.walk(st))
    return _trips(start, stop + _TEST_OPS[type(t.ops[0])], step, node_cost=size)


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
                if size * len(values) <= MAX_NODES:
                    for _ in values:
                        out.extend(copy.deepcopy(s) for s in st.body)
                    continue
        out.append(st)
    return out


class _Unroll(ast.NodeTransformer):
    def __init__(self, consts: dict):
        self.consts = consts

    def visit_Assign(self, node: ast.Assign):
        return _expand_symbolic_range_unpack(node) or self.generic_visit(node)

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
        size = sum(1 for st in node.body for _ in ast.walk(st)) + 4
        values = _trips(*args, node_cost=size)
        if values is None:
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


def _affine(node):
    """``(coefficients by name, constant)`` of a pure integer expression built from names, int literals, ``+``, ``-``, unary signs and a
    product with an int literal; ``None`` for anything else (calls, attributes, bool or float literals, non-literal products)."""
    if isinstance(node, ast.Constant):
        return ({}, node.value) if type(node.value) is int else None
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return {node.id: 1}, 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _affine(node.operand)
        if inner is None:
            return None
        sign = 1 if isinstance(node.op, ast.UAdd) else -1
        return {k: sign * v for k, v in inner[0].items()}, sign * inner[1]
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left, right = _affine(node.left), _affine(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            for scale, other in ((left, right), (right, left)):
                if not scale[0]:
                    return {k: scale[1] * v for k, v in other[0].items()}, scale[1] * other[1]
            return None
        sign = 1 if isinstance(node.op, ast.Add) else -1
        merged = dict(left[0])
        for name, coefficient in right[0].items():
            merged[name] = merged.get(name, 0) + sign * coefficient
        return merged, left[1] + sign * right[1]
    return None


def _constant_difference(start, stop):
    """``stop - start`` when linear arithmetic over the same names makes it an int constant, else ``None``."""
    a, b = _affine(start), _affine(stop)
    if a is None or b is None:
        return None
    names = set(a[0]) | set(b[0])
    return b[1] - a[1] if all(a[0].get(n, 0) == b[0].get(n, 0) for n in names) else None


def _expand_symbolic_range_unpack(node: ast.Assign):
    """Immediate fixed-cardinality scalar generator; complete the RHS before target stores."""
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple) \
            or not isinstance(node.value, ast.GeneratorExp):
        return None
    names = [n.id for n in node.targets[0].elts if isinstance(n, ast.Name)]
    count = len(names)
    if count != len(node.targets[0].elts) or len(set(names)) != count:
        return None
    expr = node.value
    if len(expr.generators) != 1:
        return None
    gen = expr.generators[0]
    call = gen.iter
    if gen.is_async or gen.ifs or not isinstance(gen.target, ast.Name) \
            or not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) \
            or call.func.id != 'range' or call.keywords or len(call.args) not in (2, 3):
        return None
    start, stop = call.args[:2]
    # The category is "a range whose trip count is a compile-time constant equal to the number of targets": the stop minus the start is a
    # constant by linear arithmetic over the same names, however it is spelled (`start + 3`, `3 + start`, `start + 1 + 2`, `n * 2 + 3` ...).
    step = 1
    if len(call.args) == 3:
        literal = _affine(call.args[2])
        if literal is None or literal[0] or literal[1] == 0:
            return None
        step = literal[1]
    difference = _constant_difference(start, stop)
    if difference is None or len(range(0, difference, step)) != count:
        return None
    allowed = (ast.Name, ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
               ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub)
    nodes = list(ast.walk(expr.elt))
    generated_size = 14 + count * (len(nodes) + 3 * sum(isinstance(n, ast.Name) for n in nodes) + 2)
    if generated_size > MAX_NODES or any(not isinstance(n, allowed)
            or isinstance(n, ast.Name) and n.id != gen.target.id
            or isinstance(n, ast.Constant) and type(n.value) is not int for n in nodes):
        return None
    # Whole-module admission still authenticates range and rejects generated-name collisions.
    helper = f'tharkun_genexpr_{node.lineno}_{node.col_offset}'
    param = 'tharkun_origin'
    class Bind(ast.NodeTransformer):
        def __init__(self, offset):
            self.offset = offset

        def visit_Name(self, n):
            return ast.copy_location(ast.BinOp(ast.Name(param, ast.Load()), ast.Add(),
                                              ast.Constant(self.offset)), n)
    values = [Bind(i * step).visit(copy.deepcopy(expr.elt)) for i in range(count)]
    args = ast.arguments(posonlyargs=[], args=[ast.arg(param)], vararg=None,
                         kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    fd = ast.FunctionDef(helper, args, [ast.Return(ast.Tuple(values, ast.Load()))],
                         decorator_list=[], returns=None, type_params=[])
    assignment = ast.Assign(copy.deepcopy(node.targets), ast.Call(ast.Name(helper, ast.Load()),
                                                                 [copy.deepcopy(start)], []))
    return [ast.fix_missing_locations(ast.copy_location(st, node)) for st in (fd, assignment)]


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
    nodes = list(ast.walk(node.elt))
    size = len(nodes) - sum(isinstance(n, ast.Name) and n.id == gen.target.id
                            and isinstance(n.ctx, ast.Load) for n in nodes)
    values = _trips(*args, node_cost=size, node_overhead=2)
    if values is None:
        return None
    elts = [_Substitute(gen.target.id, k).visit(copy.deepcopy(node.elt)) for k in values]
    return ast.copy_location(ast.List(elts=elts, ctx=ast.Load()), node)


def unroll_for_loops(tree: ast.Module) -> None:
    """Replace, in place, every ``for``, counter ``while`` and list comprehension this module can unroll exactly (see the
    module docstring)."""
    _Unroll(module_constants(tree)).visit(tree)
