"""``guard_facts`` / ``guard_domains`` -- the solving core's answer for a guard
anchor, as sound value-set claims (#0611, plan #0600).

The request is a predicate: the guard at ``(source_path, lineno)``, the branch
the caller wants (``then`` / ``else``) and optional prior facts.  The answer is
a claim about the guard's free variables' value sets: the prior facts refined
by every atom the interval / set arithmetic models (a comparison of a plain
name against a constant, membership in a literal collection), joined across
``or`` branches, with anything else left untouched.  That makes every claim a
**sound superset** by construction: no atom ever narrows a variable it does
not model.  No observed output, no target execution -- the source is read
and parsed, nothing more.

A concrete **witness** (one satisfying assignment) is available on request;
it comes from the static solver's search and may rely on ``hypothesis`` regex
strategies, so it is reported separately and never as a domain.  By default
the witness search runs under ``no_targets()`` too.

Every answer carries :class:`Provenance` (backend, strategy, elapsed ms) so a
consumer can trace and compare it against its own answer.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from tharkun.solve.fact_env import FactEnv, VarFacts, join, refine
from tharkun.solve.guard import find_branch_at_line, free_names, try_extract_constant
from tharkun.solve.normalize import normalize_guard

BACKEND = "tharkun.solve"
STRATEGY_STATIC = "static_intervals"

_CMP_OPS: Dict[type, str] = {
    ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!=",
}
_FLIP = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "==": "==", "!=": "!="}


@dataclass(frozen=True)
class Provenance:
    """Who answered, how, and how long it took."""

    backend: str = BACKEND
    strategy: str = STRATEGY_STATIC
    elapsed_ms: float = 0.0


def _scalar_type(v: Any) -> str | None:
    """The domain type of one value, or ``None`` when it has no usable domain."""
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return None if v != v else "float"   # NaN breaks set equality
    if isinstance(v, (str, bytes)):
        return type(v).__name__
    return None


def _same_type(values) -> bool:
    kinds = {_scalar_type(v) for v in values}
    return len(kinds) == 1 and None not in kinds


def _typed_set(members) -> frozenset | None:
    """``members`` as an exact set when they share one scalar type, else ``None``."""
    if not members:
        return None
    return frozenset(members) if _same_type(members) else None


@dataclass(frozen=True)
class IntDomain:
    """A value-set claim for one variable: bounds, an exact set, exclusions."""

    lo: int | float | None = None
    hi: int | float | None = None
    values: frozenset | None = None      # exact set: ints, or one non-int scalar type
    excluded: frozenset = frozenset()    # exclusions, same typing rule

    @property
    def kind(self) -> str:
        """``point`` / ``finite_set`` / ``interval`` / ``empty`` / ``unknown``."""
        if self.values is not None:
            n = len(self.values - self.excluded)
            return "empty" if n == 0 else ("point" if n == 1 else "finite_set")
        if self.lo is not None and self.hi is not None:
            if self.lo > self.hi:
                return "empty"
            width = self.hi - self.lo + 1
            if width - len({v for v in self.excluded if self.lo <= v <= self.hi}) <= 0:
                return "empty"
            return "point" if width == 1 else "interval"
        return "interval" if (self.lo is not None or self.hi is not None) else "unknown"

    @classmethod
    def of(cls, vf: VarFacts) -> "IntDomain":
        """The claim these facts support.

        An exact set is carried whenever it is HOMOGENEOUS -- every member the
        same scalar type.  Ints were the original case; ``str`` / ``bytes`` /
        ``float`` / ``bool`` sets are carried too, because a guard like
        ``s == "POST"`` states a value set exactly as much as ``x == 2`` does
        and refusing it made the claim ``unknown`` where the fact was known.

        Two shapes are declined rather than guessed at.  A MIXED set has no
        single domain to describe.  A float set containing NaN cannot be
        compared by set equality at all (``nan != nan``), which every consumer
        of ``values`` -- ``kind``, ``materialize``, the verdict comparison --
        relies on.  Bounds stay int-only: ``lo``/``hi`` are inclusive integer
        bounds by the contract every consumer shares.
        """
        values = _typed_set(vf.eq)
        excluded = _typed_set(vf.neq) or frozenset()
        # Exclusions only make sense against the same type as the values.
        if values is not None and excluded and not _same_type(values | excluded):
            excluded = frozenset()
        return cls(lo=vf.lo, hi=vf.hi, values=values, excluded=excluded)


@dataclass(frozen=True)
class GuardClaim:
    """The answer to a guard-anchor request."""

    env: FactEnv                                  # the prior facts, refined
    domains: Dict[str, IntDomain] = field(default_factory=dict)
    feasible: bool = True                         # False: the refined facts contradict
    sound_superset: bool = True                   # the domains are supersets of the true sets
    provenance: Provenance = field(default_factory=Provenance)
    guard_source: str | None = None
    unmodelled_atoms: int = 0                     # how many atoms narrowed nothing
    # WHICH atoms those were, with their free names and shape.  The count above
    # is a path total and cannot answer "was it my variable?" -- #0666.
    unmodelled: tuple = ()                        # tuple[UnmodelledAtom, ...]
    witness: Dict[str, Any] | None = None         # one satisfying assignment, if asked
    witness_strategy: str | None = None


def _load_guard(
    source_path: str | None,
    lineno: int,
    assume_calls: Sequence[str] = (),
    tree: ast.AST | None = None,
) -> Optional[ast.expr]:
    """The predicate at ``lineno``: an ``if`` / ``while`` test, an ``assert``
    test, or -- when ``assume_calls`` names them -- the first argument of a
    bare call statement such as ``assume(cond)`` (the shape lowering passes
    leave behind for a taken branch).  ``tree`` (an already-parsed module)
    takes precedence over ``source_path`` -- the caller's own AST is the one
    whose line numbers it holds (#0622)."""
    if tree is None:
        if not source_path:
            return None
        try:
            tree = ast.parse(Path(source_path).read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError, ValueError):
            return None
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != lineno:
            continue
        if isinstance(node, ast.Assert):
            return node.test
        if (
            assume_calls
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in assume_calls
            and len(node.value.args) == 1
        ):
            return node.value.args[0]
        if isinstance(node, ast.For):
            return None  # ``for`` yields its iterable, not a predicate
    return find_branch_at_line(tree, lineno)


def _load_assignment(
    source_path: str | None, lineno: int, tree: ast.AST | None = None,
) -> Optional[ast.stmt]:
    """The ``Assign`` / ``AnnAssign`` statement at ``lineno`` whose targets are
    plain names (or a tuple of them), else ``None``."""
    if tree is None:
        if not source_path:
            return None
        try:
            tree = ast.parse(Path(source_path).read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError, ValueError):
            return None
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != lineno:
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and _target_names(node.targets[0]) is not None:
            return node
        if isinstance(node, ast.AnnAssign) and node.value is not None and _target_names(node.target) is not None:
            return node
    return None


def _target_names(target: ast.expr) -> Optional[list[str]]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Name) for e in target.elts):
        return [e.id for e in target.elts]
    return None


def _apply_assignment(env: FactEnv, stmt: ast.stmt) -> FactEnv:
    """``env`` after the assignment: every target loses its facts (the old
    value is gone), and a constant RHS pins the target to that constant --
    the same discipline the caller's own fact env uses on a path."""
    target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
    names = _target_names(target) or []
    facts = {k: v for k, v in env.facts.items() if k not in names}
    env = FactEnv(facts=facts, possible=env.possible)
    value = stmt.value
    if value is None:
        return env
    if len(names) == 1:
        const, ok = try_extract_constant(value)
        if ok:
            return refine(env, names[0], "==", const)
        return env
    if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(names):
        for name, elt in zip(names, value.elts):
            const, ok = try_extract_constant(elt)
            if ok:
                env = refine(env, name, "==", const)
    return env


def _parse_predicate(predicate: str | ast.expr) -> Optional[ast.expr]:
    """``predicate`` as an expression node: Python source text is parsed
    (``None`` when it is not an expression); a node is taken as is."""
    if isinstance(predicate, ast.expr):
        return predicate
    try:
        return ast.parse(str(predicate).strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None


def _parse_assignment(assignment: str | ast.stmt) -> Optional[ast.stmt]:
    """``assignment`` as an ``Assign`` / ``AnnAssign`` to plain names (source
    text is parsed as one statement); ``None`` for anything else."""
    stmt: Any = assignment
    if not isinstance(assignment, ast.stmt):
        try:
            body = ast.parse(str(assignment).strip()).body
        except (SyntaxError, ValueError):
            return None
        stmt = body[0] if len(body) == 1 else None
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and _target_names(stmt.targets[0]) is not None:
        return stmt
    if isinstance(stmt, ast.AnnAssign) and stmt.value is not None and _target_names(stmt.target) is not None:
        return stmt
    return None


def _assignment_claim(prior: FactEnv, stmt: ast.stmt, started: float) -> GuardClaim:
    try:
        env = _apply_assignment(prior, stmt)
    except Exception:
        env = prior
    return GuardClaim(
        env=env,
        domains={name: IntDomain.of(vf) for name, vf in env.facts.items()},
        feasible=env.possible,
        provenance=Provenance(strategy="assignment", elapsed_ms=_ms(started)),
        guard_source=ast.unparse(stmt),
    )


_ARITH_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_BITWISE_OPS = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)


def _operand_shape(node: ast.expr) -> Optional[str]:
    """The reason ``_atom`` cannot use *node* as a comparison operand, or
    ``None`` when it can (a plain name, or a foldable constant)."""
    if isinstance(node, ast.Name):
        return None
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, _ARITH_OPS):
            return "binop-arith"
        if isinstance(node.op, _BITWISE_OPS):
            return "binop-bitwise"
        return "binop-other"
    if isinstance(node, ast.Subscript):
        return "subscript"
    if isinstance(node, ast.Call):
        return "call"
    if isinstance(node, ast.Attribute):
        return "attribute"
    _v, ok = try_extract_constant(node)
    return None if ok else "other"


def _atom_shape(expr: ast.expr) -> str:
    """Why ``_atom`` declined *expr* -- a small closed vocabulary, so a
    consumer can COUNT the shapes it is losing instead of re-deriving them by
    replaying the guard through a private function (#0666).

    The solver knows why it abstained; saying so is part of answering.  A
    caller forced to reconstruct this is a caller re-implementing ``_atom``.
    """
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return _atom_shape(expr.operand)
    if isinstance(expr, ast.Name):
        return "truthiness"          # ``if x:`` -- no comparator at all
    if isinstance(expr, ast.Call):
        return "call"
    if not isinstance(expr, ast.Compare):
        return "other"
    if len(expr.ops) != 1:
        return "chained-compare"     # ``0 <= x <= 9`` if normalisation left it
    left, op, right = expr.left, expr.ops[0], expr.comparators[0]
    if isinstance(op, (ast.In, ast.NotIn)):
        if not isinstance(left, ast.Name):
            return _operand_shape(left) or "other"
        return "membership-nonliteral"
    lshape, rshape = _operand_shape(left), _operand_shape(right)
    if lshape is not None:
        return lshape
    if rshape is not None:
        return rshape
    # Both sides are usable operands but neither folds to a constant: the only
    # remaining shape is Name-vs-Name, which has no per-variable answer.
    return "name-vs-name"


@dataclass(frozen=True)
class UnmodelledAtom:
    """One atom the arithmetic could not model, and why.

    ``names`` is what lets a caller ask the question that actually matters --
    "was it MY variable that got dropped, or someone else's?".  A path-total
    COUNT cannot tell a real vocabulary gap from an unmodelled atom elsewhere
    on the path; both look identical, and a diagnosis built on the count
    misattributes one as the other (#0666)."""

    source: str
    names: frozenset = frozenset()
    shape: str = "other"


def _unmodelled(expr: ast.expr) -> UnmodelledAtom:
    try:
        source = ast.unparse(expr)
    except Exception:
        source = "<unparseable>"
    try:
        names = frozenset(free_names(expr))
    except Exception:
        names = frozenset()
    return UnmodelledAtom(source=source, names=names, shape=_atom_shape(expr))


def _atom(expr: ast.expr) -> Optional[tuple[str, str, Any]]:
    """``(var, op, value)`` for a comparison of a plain name against a
    constant (or a literal collection for ``in`` / ``not in``), else ``None``."""
    if not isinstance(expr, ast.Compare) or len(expr.ops) != 1:
        return None
    left, op, right = expr.left, expr.ops[0], expr.comparators[0]
    if isinstance(op, (ast.In, ast.NotIn)):
        if isinstance(left, ast.Name) and isinstance(right, (ast.Tuple, ast.List, ast.Set)):
            values = []
            for elt in right.elts:
                val, ok = try_extract_constant(elt)
                if not ok:
                    return None
                values.append(val)
            try:
                return left.id, ("in" if isinstance(op, ast.In) else "not in"), set(values)
            except TypeError:
                return None
        return None
    if isinstance(op, (ast.Is, ast.IsNot)):
        if isinstance(left, ast.Name) and isinstance(right, ast.Constant) and right.value is None:
            return left.id, ("==" if isinstance(op, ast.Is) else "!="), None
        return None
    sym = _CMP_OPS.get(type(op))
    if sym is None:
        return None
    # len(x) == K / len(x) != K: a container-length fact (eregion parity, [internal ref])
    if sym in ("==", "!="):
        for side, other in ((left, right), (right, left)):
            if (isinstance(side, ast.Call) and isinstance(side.func, ast.Name) and side.func.id == "len"
                    and len(side.args) == 1 and not side.keywords and isinstance(side.args[0], ast.Name)):
                val, ok = try_extract_constant(other)
                if ok and isinstance(val, int) and not isinstance(val, bool):
                    return side.args[0].id, ("container_len_eq" if sym == "==" else "container_len_ne"), val
    if isinstance(left, ast.Name):
        val, ok = try_extract_constant(right)
        if ok:
            return left.id, sym, val
    if isinstance(right, ast.Name):
        val, ok = try_extract_constant(left)
        if ok:
            return right.id, _FLIP[sym], val
    return None


def _refine_expr(env: FactEnv, expr: ast.expr) -> tuple[FactEnv, tuple[UnmodelledAtom, ...]]:
    """``env`` refined by the normalised guard ``expr``; also every atom the
    arithmetic could not model (each one narrows nothing), with its free names
    and its shape -- see :class:`UnmodelledAtom` for why a count is not enough."""
    if isinstance(expr, ast.BoolOp):
        if isinstance(expr.op, ast.And):
            unmodelled: tuple[UnmodelledAtom, ...] = ()
            for value in expr.values:
                env, found = _refine_expr(env, value)
                unmodelled += found
                if not env.possible:
                    break
            return env, unmodelled
        branches = [_refine_expr(env, value) for value in expr.values]
        out = branches[0][0]
        for branch_env, _found in branches[1:]:
            out = join(out, branch_env)
        return out, tuple(a for _e, found in branches for a in found)
    atom = _atom(expr)
    if atom is None:
        return env, (_unmodelled(expr),)
    var, op, value = atom
    return refine(env, var, op, value), ()


def guard_facts(
    source_path: str | None,
    lineno: int | None,
    *,
    facts: FactEnv | None = None,
    desired_branch: str = "then",
    witness: bool = False,
    allow_target_loading: bool = False,
    assume_calls: Sequence[str] = (),
    tree: ast.AST | None = None,
    assignments: bool = False,
    predicate: str | ast.expr | None = None,
    assignment: str | ast.stmt | None = None,
) -> GuardClaim:
    """The sound value-set claim for the guard at ``lineno`` of ``source_path``
    (or of ``tree``, an already-parsed module -- the caller's own AST, whose
    line numbers it holds; ``source_path`` may then be ``None``).

    ``facts`` are prior facts to refine (a :class:`FactEnv`; default empty).
    ``desired_branch`` picks the polarity (``"else"`` negates the guard).
    ``assume_calls`` names bare call statements whose first argument is an
    assumed predicate (``assume(cond)``), so a lowered branch condition is a
    guard too; ``assert`` statements always are.
    ``assignments=True`` also accepts an assignment statement at ``lineno``:
    its targets lose their facts and a constant right-hand side pins the
    target (strategy ``"assignment"``) -- so a caller can replay a whole
    path, guards and writes in order, and a guard on a variable that is
    later reassigned never contradicts the write.
    ``predicate`` hands the guard over directly -- Python source text or an
    ``ast.expr`` -- and ``assignment`` a write to replay the same way
    (``"q = 2"``): neither needs a path, a tree or a line (``lineno`` is
    then display only).  That is the anchor for a caller whose statements
    share line numbers, as a lowering pass copying one location onto
    everything it injects leaves them.
    ``witness=True`` additionally asks the static solver for one satisfying
    assignment (reported in ``witness`` / ``witness_strategy``, never folded
    into the domains; needs ``source_path`` and ``lineno``); the search runs
    under ``no_targets()`` unless ``allow_target_loading`` is set.  Never
    raises.
    """
    started = time.perf_counter()
    prior = facts if facts is not None else FactEnv()
    guard: Optional[ast.expr] = None
    if predicate is not None:
        guard = _parse_predicate(predicate)
    elif assignment is not None:
        stmt = _parse_assignment(assignment)
        if stmt is not None:
            return _assignment_claim(prior, stmt, started)
    elif lineno is not None:
        guard = _load_guard(source_path, lineno, assume_calls, tree)
        if guard is None and assignments:
            stmt = _load_assignment(source_path, lineno, tree)
            if stmt is not None:
                return _assignment_claim(prior, stmt, started)
    if guard is None:
        return GuardClaim(
            env=prior,
            domains={name: IntDomain.of(vf) for name, vf in prior.facts.items()},
            feasible=prior.possible,
            provenance=Provenance(strategy="no_guard", elapsed_ms=_ms(started)),
        )
    expr: ast.expr = guard if desired_branch != "else" else ast.UnaryOp(op=ast.Not(), operand=guard)
    try:
        normalised = normalize_guard(ast.fix_missing_locations(expr))
        env, unmodelled = _refine_expr(prior, normalised)
    except Exception:
        # The whole guard is unmodelled, and saying so is the honest record --
        # a bare count of 1 here used to be indistinguishable from one dropped
        # atom inside a guard that otherwise parsed (#0666).
        env, unmodelled = prior, (_unmodelled(expr),)
    names = set(env.facts) | free_names(guard)
    domains = {name: IntDomain.of(env.facts[name]) for name in sorted(names) if name in env.facts}
    claim = GuardClaim(
        env=env,
        domains=domains,
        feasible=env.possible,
        sound_superset=True,
        provenance=Provenance(strategy=STRATEGY_STATIC, elapsed_ms=_ms(started)),
        guard_source=ast.unparse(guard),
        unmodelled_atoms=len(unmodelled),
        unmodelled=unmodelled,
    )
    if not witness or not env.possible or not source_path or lineno is None:
        return claim
    return _with_witness(claim, source_path, lineno, desired_branch, allow_target_loading, started)


def guard_domains(
    source_path: str | None,
    lineno: int | None,
    *,
    facts: FactEnv | None = None,
    desired_branch: str = "then",
    assume_calls: Sequence[str] = (),
    tree: ast.AST | None = None,
    predicate: str | ast.expr | None = None,
) -> Dict[str, IntDomain]:
    """Just the per-variable domains of :func:`guard_facts` (``{}`` if infeasible)."""
    claim = guard_facts(
        source_path, lineno, facts=facts, desired_branch=desired_branch,
        assume_calls=assume_calls, tree=tree, predicate=predicate,
    )
    return claim.domains if claim.feasible else {}


def _with_witness(claim: GuardClaim, source_path: str, lineno: int, desired_branch: str,
                  allow_target_loading: bool, started: float) -> GuardClaim:
    import contextlib
    import dataclasses

    from tharkun.solve.condition import solve_condition
    from tharkun.solve.loader import no_targets

    guard = contextlib.nullcontext() if allow_target_loading else no_targets()
    try:
        with guard:
            sol = solve_condition(source_path, lineno, facts=claim.env, desired_branch=desired_branch)
        witness = dict(sol.satisfying_inputs[0]) if sol.satisfying_inputs else None
        strategy = sol.strategy
    except Exception:
        witness, strategy = None, "error"
    return dataclasses.replace(
        claim,
        witness=witness,
        witness_strategy=strategy,
        provenance=Provenance(strategy=f"{STRATEGY_STATIC}+witness:{strategy}", elapsed_ms=_ms(started)),
    )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


__all__ = ["BACKEND", "GuardClaim", "IntDomain", "Provenance", "guard_domains", "guard_facts"]
