"""Loop-body guards solved through their local aliases (``is_palindrome``: ``head_char != tail_char``).

A guard such as ``head_char != tail_char`` has no parameter in it: the names are locals that the enclosing function
binds before the guard, ``string_len = len(input_string)``, ``for index in range(string_len)``,
``head_char = input_string[index]``, ``tail_char = input_string[string_len - index - 1]`` and, under ``if ignore_case:``,
``head_char = head_char.lower()``.  ``condition._solve_single_expr`` treats such names as free variables and answers
``{'head_char': 0, 'tail_char': 0}``, which is no input of the function.

Here the aliases are substituted back (straight-line ``name = expr`` assignments, ``if flag: name = expr`` rebinding in
both arms, pre-loop ``if C: return`` guards required False), which leaves an index-affine relation between two characters
of one parameter, ``E1(P[p]) op E2(P[q])`` with ``p, q`` affine in the loop index ``k`` and in ``len(P)`` and ``E1, E2``
chains of zero-argument str methods.  It is SOLVED, not searched: the length and positions come from an integer
optimisation (z3, minimal length with in-bounds, non-wrapping indices), the characters from the str-method preimages
(``solve/preimage``) of the value the other side produces.  Every witness is then checked by evaluating the path guards and
the aliased guard (the caller's ``check``).  The first iteration (``k = 0``) is solved; later iterations (a mismatch after
matching pairs) are not.  Anything outside this shape returns ``None`` and the caller keeps its previous route."""
from __future__ import annotations

import ast
import itertools
from typing import Any, Callable

_STR_METHODS = frozenset({"lower", "upper", "casefold", "swapcase", "title", "capitalize", "strip", "lstrip", "rstrip"})
_MAX_COND_REBINDS = 3


class _Subst(ast.NodeTransformer):
    def __init__(self, env: dict[str, ast.expr]):
        self.env = env

    def visit_Name(self, node: ast.Name):
        return self.env.get(node.id, node) if isinstance(node.ctx, ast.Load) else node


def _subst(expr: ast.expr, env: dict[str, ast.expr]) -> ast.expr:
    import copy
    return _Subst(env).visit(copy.deepcopy(expr))


def _locate(tree: ast.Module, lineno: int):
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for loop in ast.walk(fn):
            if not (isinstance(loop, ast.For) and isinstance(loop.target, ast.Name)
                    and isinstance(loop.iter, ast.Call) and isinstance(loop.iter.func, ast.Name)
                    and loop.iter.func.id == "range" and len(loop.iter.args) == 1 and not loop.orelse):
                continue
            if loop.lineno == lineno:
                return fn, loop, loop
            for stmt in loop.body:
                if stmt.lineno == lineno and isinstance(stmt, ast.If):
                    return fn, loop, stmt
    return None


def _assigns_only(stmts: list[ast.stmt]) -> dict[str, ast.expr] | None:
    out: dict[str, ast.expr] = {}
    for s in stmts:
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)):
            return None
        out[s.targets[0].id] = s.value
    return out


def _index_z3(expr: ast.expr, z3, k, length, pname: str, kname: str):
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return z3.IntVal(expr.value)
    if isinstance(expr, ast.Name) and expr.id == kname:
        return k
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "len" and len(expr.args) == 1 \
            and isinstance(expr.args[0], ast.Name) and expr.args[0].id == pname:
        return length
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)):
        a = _index_z3(expr.left, z3, k, length, pname, kname)
        b = _index_z3(expr.right, z3, k, length, pname, kname)
        if a is None or b is None:
            return None
        return a + b if isinstance(expr.op, ast.Add) else a - b
    return None


def _char_term(expr: ast.expr):
    """``m2(m1(P[idx]))`` -> (P name, idx expr, [m1, m2]) or None."""
    methods: list[str] = []
    while isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and not expr.args and not expr.keywords \
            and expr.func.attr in _STR_METHODS:
        methods.insert(0, expr.func.attr)
        expr = expr.func.value
    if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name) and not isinstance(expr.slice, ast.Slice):
        return expr.value.id, expr.slice, methods
    return None


def _apply(methods: list[str], text: str) -> str:
    for m in methods:
        text = getattr(text, m)()
    return text


def _chars_with_value(methods: list[str], target: str) -> list[str]:
    """One-character strings c with ``chain(c) == target``, by composing the str-method preimages."""
    from tharkun.solve.contracts.rules import Refused
    from tharkun.solve.preimage import Eq, HOLE, preimages
    level = [target]
    for m in reversed(methods):
        nxt: list[str] = []
        for value in level:
            got = preimages("str." + m, Eq(value), (HOLE,), budget=4)
            if not isinstance(got, Refused):
                nxt.extend(g for g in got if isinstance(g, str))
        level = list(dict.fromkeys(nxt))
    return [c for c in level if len(c) == 1 and _apply(methods, c) == target]


def solve_loop_guard(tree: ast.Module, lineno: int, desired: str, solve_guard: Callable[[int, str], dict[str, Any] | None],
                     check: Callable[[ast.expr, dict[str, Any]], bool]) -> list[dict[str, Any]] | None:
    """Witnesses (full parameter dicts) for the guard at ``lineno`` inside ``for k in range(...)``, or ``None``.

    *solve_guard(line, 'then'|'else')* is the ordinary solver for a guard over parameters (first witness or None);
    *check(expr, env)* evaluates a guard expression over *env* (the helper-inlining native evaluator)."""
    try:
        import z3
    except ImportError:
        return None
    found = _locate(tree, lineno)
    if found is None:
        return None
    fn, loop, node = found
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    kname = loop.target.id
    # -- straight-line prefix of the function, up to the loop
    aliases: dict[str, ast.expr] = {}
    paths: list[tuple[ast.If, bool]] = []          # (if node, taken)
    for stmt in fn.body:
        if stmt is loop or any(n is loop for n in ast.walk(stmt)):
            break
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            aliases[stmt.targets[0].id] = _subst(stmt.value, aliases)
        elif isinstance(stmt, ast.If) and not stmt.orelse and stmt.body and isinstance(stmt.body[-1], (ast.Return, ast.Raise)):
            paths.append((stmt, False))
        elif isinstance(stmt, ast.If) and not stmt.orelse and _assigns_only(stmt.body) is not None:
            paths.append((stmt, False))            # the rebinding arm needs an inverse of its RHS: not taken here
        else:
            return None
    # -- loop-body prefix up to the guard
    body_pre: list[ast.stmt] = []
    for stmt in ([] if node is loop else loop.body):
        if stmt is node:
            break
        body_pre.append(stmt)
    steps: list[tuple[str, Any]] = []              # ("assign", (name, expr)) | ("cond", (if node, {name: expr}))
    for stmt in body_pre:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            steps.append(("assign", (stmt.targets[0].id, stmt.value)))
        elif isinstance(stmt, ast.If) and not stmt.orelse and _assigns_only(stmt.body) is not None:
            steps.append(("cond", (stmt, _assigns_only(stmt.body))))
        else:
            return None
    conds = [p[0] for kind, p in steps if kind == "cond"]
    if len(conds) > _MAX_COND_REBINDS:
        return None
    # the loop bound
    bound_src = loop.iter.args[0]
    out: list[dict[str, Any]] = []
    want_then = desired == "then"
    for taken in itertools.product((False, True), repeat=len(conds)):
        env = dict(aliases)
        arms = iter(taken)
        for kind, payload in steps:
            if kind == "assign":
                env[payload[0]] = _subst(payload[1], env)
            elif next(arms):
                for n, v in payload[1].items():
                    env[n] = _subst(v, env)
        arm_paths = paths + list(zip(conds, taken))
        witness: dict[str, Any] = {}
        ok = True
        for if_node, arm in arm_paths:
            w = solve_guard(if_node.lineno, "then" if arm else "else")
            if w is None or any(k in witness and witness[k] != v for k, v in w.items()):
                ok = False
                break
            witness.update(w)
        if not ok:
            continue
        bound = _subst(bound_src, env)
        guard = _subst(node.test, env) if isinstance(node, ast.If) else None
        results = _solve_arm(z3, node, guard, bound, kname, params, want_then, witness, check, arm_paths, solve_guard)
        out.extend(r for r in results if r not in out)
    return out or None


def _solve_arm(z3, node, guard, bound, kname, params, want_then, witness, check, arm_paths, solve_guard):
    pname = None
    term = None
    if isinstance(node, ast.If) and isinstance(guard, ast.Compare) and len(guard.ops) == 1 \
            and isinstance(guard.ops[0], (ast.Eq, ast.NotEq)):
        a, b = _char_term(guard.left), _char_term(guard.comparators[0])
        if a is None or b is None or a[0] != b[0] or a[0] not in params:
            return []
        pname, term = a[0], (a, b)
    elif isinstance(node, ast.If):
        # a guard over parameters (``if ignore_case``): its own witness plus entry into the loop
        w = solve_guard(node.lineno, "then" if want_then else "else")
        if w is None:
            return []
        witness = {**witness}
        if any(k in witness and witness[k] != v for k, v in w.items()):
            return []
        witness.update(w)
    elif not want_then:
        return []                                   # the loop's empty arm needs the rebinding arms: not solved here
    # the length parameter: find the indexed parameter name (P in len(P)), from the loop bound or the guard
    if pname is None:
        names = [n.args[0].id for n in ast.walk(bound) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "len" and isinstance(n.args[0], ast.Name) and n.args[0].id in params]
        if not names:
            return []
        pname = names[0]
    length, k = z3.Int("L"), z3.Int("K")
    bound_z = _index_z3(bound, z3, k, length, pname, kname)
    if bound_z is None:
        return []
    opt = z3.Optimize()
    opt.add(length >= 1, k == 0, bound_z >= 1)
    if term is None:
        opt.minimize(length)
        if opt.check() != z3.sat:
            return []
        n = opt.model()[length].as_long()
        text = "a" * n
        return _finish(witness, pname, text, kname, bound, node, check, arm_paths, want_then)
    (a, b) = term
    pz, qz = (_index_z3(a[1], z3, k, length, pname, kname), _index_z3(b[1], z3, k, length, pname, kname))
    if pz is None or qz is None:
        return []
    is_eq = isinstance(guard.ops[0], ast.Eq)
    equal_arm = (is_eq == want_then)               # the guard holds <=> the two characters are equal
    out: list[dict[str, Any]] = []
    variants = ([True, False] if equal_arm else [False])    # same position (equal by construction) / distinct positions
    for same in variants:
        o = z3.Optimize()
        o.add(length >= 1, k == 0, bound_z >= 1, pz >= 0, pz < length, qz >= 0, qz < length)
        o.add(pz == qz if same else pz != qz)
        o.minimize(length)
        if o.check() != z3.sat:
            continue
        m = o.model()
        n, p, q = m[length].as_long(), m.eval(pz).as_long(), m.eval(qz).as_long()
        c1 = "a"
        v1 = _apply(a[2], c1)
        if same:
            seconds = [c1]
        elif equal_arm:
            seconds = _chars_with_value(b[2], v1)
        else:
            seconds = []
            for step in range(1, 9):
                seconds = _chars_with_value(b[2], chr(ord(v1[-1]) + step))
                if seconds:
                    break
            if not seconds:                         # no preimage of the next code point (``upper`` never gives 'b'): a
                # neighbouring character whose image under the chain differs
                seconds = [x for x in (chr(ord(c1) + d) for d in range(1, 9)) if _apply(b[2], x) != v1][:1]
        for c2 in seconds:
            text = [c1] * n
            text[p], text[q] = c1, c2
            out.extend(_finish(witness, pname, "".join(text), kname, bound, node, check, arm_paths, want_then,
                               guard))
    return out


def _holds(check, expr: ast.expr, env: dict[str, Any], value: bool) -> bool:
    """Does *expr* evaluate to *value* over *env*?  (``check`` answers False for an evaluation failure, so a False
    is asked as ``not expr``: a failed evaluation never passes either way.)"""
    return check(expr if value else ast.UnaryOp(op=ast.Not(), operand=expr), env)


def _finish(witness, pname, text, kname, bound, node, check, arm_paths, want_then, guard=None):
    env = {**witness, pname: text}
    if not all(_holds(check, if_node.test, env, arm) for if_node, arm in arm_paths):
        return []                                  # pre-loop path guards and the chosen rebinding arms hold natively
    if not check(ast.Compare(left=bound, ops=[ast.GtE()], comparators=[ast.Constant(1)]), env):
        return []                                  # the loop is entered
    if guard is not None and not _holds(check, guard, {**env, kname: 0}, want_then):
        return []
    return [env]
