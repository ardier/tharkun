"""Phase 7: Candidate value generation.

For each blocking atom, propose candidate values for its leaf dependencies.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from tharkun.minleaf.types import (
    LeafFactsIndex,
    Path,
    StateIndex,
)


@dataclass(frozen=True)
class LeafCandidates:
    path: Path
    current_value: Any
    candidates: tuple[Any, ...]
    facts_injected: int = 0
    facts_pruned: int = 0


def propose_leaf_candidates(
    *,
    atom_expr: ast.expr,
    atom_value: bool,
    dependency_paths: frozenset[Path],
    state_index: StateIndex,
    env: dict[str, object],
    leaf_facts_index: LeafFactsIndex | None,
    max_per_leaf: int,
) -> dict[Path, LeafCandidates]:
    """Propose candidate values for leaf dependencies of a blocking atom."""
    result: dict[Path, LeafCandidates] = {}

    for p in sorted(dependency_paths):
        if p not in state_index.leaves:
            continue

        leaf_ref = state_index.leaves[p]
        current_value = leaf_ref.getter()
        leaf_type = leaf_ref.typ

        raw: list[Any] = []
        facts_injected = 0
        facts_pruned = 0

        # Stage 1: Fact-injected candidates
        if leaf_facts_index is not None and leaf_facts_index.has_eq(p):
            lf = leaf_facts_index.get(p)
            if lf is not None and lf.eq is not None:
                for v in sorted(lf.eq, key=repr):
                    raw.append(v)
                    facts_injected += 1

        # Stage 2: Pattern-derived candidates
        pattern_cands = _extract_pattern_candidates(
            atom_expr, atom_value, p, state_index, env
        )
        raw.extend(pattern_cands)

        # Stage 3: E1 combined constraint -- handled externally by patch_search
        # (not in per-atom proposal, but we handle modulo pattern here)

        # Stage 4: Fallback candidates (when no pattern matched and no fact eq)
        if not raw:
            raw.extend(_fallback_candidates(current_value, leaf_type, p))

        # Stage 5: Fact-based filtering
        if leaf_facts_index is not None:
            filtered: list[Any] = []
            for v in raw:
                # Bounds clamping for numeric types
                if isinstance(v, (int, float)):
                    lf = leaf_facts_index.get(p)
                    if lf is not None:
                        if lf.lo is not None and v < lf.lo:
                            v = lf.lo
                        if lf.hi is not None and v > lf.hi:
                            v = lf.hi

                if leaf_facts_index.is_excluded(p, v):
                    facts_pruned += 1
                    continue
                if not leaf_facts_index.in_bounds(p, v):
                    facts_pruned += 1
                    continue
                filtered.append(v)
            raw = filtered

        # Stage 6: Dedup + current-value exclusion + truncation
        seen: list[Any] = []
        seen_set: set = set()
        for v in raw:
            try:
                key = (type(v), v)
                if key in seen_set:
                    continue
                seen_set.add(key)
            except TypeError:
                # unhashable type -- use linear search
                if any(type(v) == type(s) and v == s for s in seen):
                    continue
            if v == current_value and type(v) == type(current_value):
                continue
            seen.append(v)

        candidates = tuple(seen[:max_per_leaf])

        result[p] = LeafCandidates(
            path=p,
            current_value=current_value,
            candidates=candidates,
            facts_injected=facts_injected,
            facts_pruned=facts_pruned,
        )

    return result


def _extract_pattern_candidates(
    atom_expr: ast.expr,
    atom_value: bool,
    target_path: Path,
    state_index: StateIndex,
    env: dict[str, object],
) -> list[Any]:
    """Extract pattern-derived candidates from a Compare atom."""
    candidates: list[Any] = []

    # Check for modulo pattern: (leaf % m) == r
    mod_cands = _check_modulo_pattern(atom_expr, atom_value, target_path, env)
    if mod_cands:
        candidates.extend(mod_cands)

    # Check for length pattern: len(leaf) OP n
    len_cands = _check_len_pattern(
        atom_expr, atom_value, target_path, state_index, env
    )
    if len_cands:
        candidates.extend(len_cands)

    if not isinstance(atom_expr, ast.Compare):
        return candidates
    if len(atom_expr.ops) != 1:
        return candidates

    op = atom_expr.ops[0]
    left = atom_expr.left
    right = atom_expr.comparators[0]

    # Determine leaf side and const side
    leaf_side, const_side, leaf_is_left = _identify_sides(
        left, right, target_path, state_index, env
    )
    if leaf_side is None or const_side is None:
        return candidates

    const_val = _resolve_const(const_side, env)
    if const_val is None:
        return candidates

    # Generate candidates based on op and direction
    if not atom_value:
        # Atom is False, we want True
        candidates.extend(_false_to_true_candidates(op, const_val, leaf_is_left, env))
    else:
        # Atom is True, we want False
        candidates.extend(_true_to_false_candidates(op, const_val, leaf_is_left, env))

    return candidates


def _required_lengths(op: ast.cmpop, n: int, atom_value: bool, len_is_left: bool) -> list[int]:
    """Lengths that flip ``len(leaf) OP n`` to the desired truth value."""
    # Normalise so the length is always on the left (n < len(x)  ->  len(x) > n).
    if not len_is_left:
        op = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}.get(
            type(op), type(op)
        )()
    want_true = not atom_value
    if isinstance(op, ast.Eq):
        return [n] if want_true else [n + 1]
    if isinstance(op, ast.NotEq):
        return [n + 1] if want_true else [n]
    if isinstance(op, ast.Gt):
        return [n + 1] if want_true else [n]
    if isinstance(op, ast.GtE):
        return [n] if want_true else [max(n - 1, 0)]
    if isinstance(op, ast.Lt):
        return [max(n - 1, 0)] if want_true else [n]
    if isinstance(op, ast.LtE):
        return [n] if want_true else [n + 1]
    return []


def _resize_like(current: Any, length: int) -> Any:
    """A value of the same kind as *current* resized to *length*.

    Growing reuses the existing elements (cycling) so the new container keeps the
    element *type* the program expects; an empty one falls back to a benign filler.
    Shrinking truncates. This is the structural mutation minleaf otherwise declines —
    admitted here only for the length itself, which is what a ``len()`` guard reads.
    """
    if length < 0:
        return None
    if isinstance(current, str):
        base = current or "x"
        return (base * (length // len(base) + 1))[:length]
    if isinstance(current, (bytes, bytearray)):
        base = bytes(current) or b"x"
        out = (base * (length // len(base) + 1))[:length]
        return bytearray(out) if isinstance(current, bytearray) else out
    if isinstance(current, (list, tuple, set, frozenset)):
        items = list(current)
        if length <= len(items):
            out = items[:length]
        else:
            filler = items if items else [0]
            out = items + [filler[i % len(filler)] for i in range(length - len(items))]
        if isinstance(current, tuple):
            return tuple(out)
        if isinstance(current, (set, frozenset)):
            # A set cannot be padded with duplicates; only a truncation is meaningful.
            return None if length > len(items) else type(current)(out)
        return out
    if isinstance(current, dict):
        items = list(current.items())
        if length <= len(items):
            return dict(items[:length])
        out = dict(items)
        i = 0
        while len(out) < length:
            key = f"__k{i}__"
            if key not in out:
                out[key] = items[0][1] if items else 0
            i += 1
        return out
    return None


def _check_len_pattern(
    atom_expr: ast.expr,
    atom_value: bool,
    target_path: Path,
    state_index: StateIndex,
    env: dict[str, object],
) -> list[Any]:
    """Candidates for ``len(<chain>) OP n``.

    The leaf side is a ``Call`` rather than an access chain, so ``_identify_sides``
    finds nothing and the atom yields no candidates at all — the reason a guard like
    ``len(cfg["tags"]) > 2`` blocks. Here we look through ``len(...)``, resolve the
    argument as the target path, and propose the container resized to a satisfying
    length.
    """
    if not isinstance(atom_expr, ast.Compare) or len(atom_expr.ops) != 1:
        return []

    def _len_arg(node: ast.expr) -> ast.expr | None:
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "len" and len(node.args) == 1):
            return node.args[0]
        return None

    left, right = atom_expr.left, atom_expr.comparators[0]
    arg = _len_arg(left)
    len_is_left = True
    if arg is None:
        arg = _len_arg(right)
        len_is_left = False
        if arg is None:
            return []

    if _try_build_chain(arg) != target_path:
        return []
    n = _resolve_const_value(right if len_is_left else left, env)
    if not isinstance(n, int) or isinstance(n, bool):
        return []

    leaf_ref = state_index.leaves.get(target_path)
    if leaf_ref is None:
        return []
    current = leaf_ref.getter()

    out: list[Any] = []
    for want in _required_lengths(atom_expr.ops[0], n, atom_value, len_is_left):
        resized = _resize_like(current, want)
        if resized is not None:
            out.append(resized)
    return out


def _check_modulo_pattern(
    atom_expr: ast.expr,
    atom_value: bool,
    target_path: Path,
    env: dict[str, object],
) -> list[Any]:
    """Check for (leaf % m) == r pattern."""
    if not isinstance(atom_expr, ast.Compare):
        return []
    if len(atom_expr.ops) != 1:
        return []
    if not isinstance(atom_expr.ops[0], ast.Eq):
        return []

    left = atom_expr.left
    right = atom_expr.comparators[0]

    # Check if left is BinOp(leaf, Mod, m)
    if isinstance(left, ast.BinOp) and isinstance(left.op, ast.Mod):
        m_val = _resolve_const_value(left.right, env)
        r_val = _resolve_const_value(right, env)
        if m_val is not None and r_val is not None and isinstance(m_val, int) and isinstance(r_val, int):
            if not atom_value:
                # Want (leaf % m) == r to be True -> candidate = r (or r + m if r < 0)
                cand = r_val if r_val >= 0 else r_val + m_val
                return [cand]

    return []


def _identify_sides(
    left: ast.expr,
    right: ast.expr,
    target_path: Path,
    state_index: StateIndex,
    env: dict[str, object],
) -> tuple[ast.expr | None, ast.expr | None, bool]:
    """Identify which side of a Compare is the leaf and which is the constant.

    Returns (leaf_side, const_side, leaf_is_left).
    """
    left_chain = _try_build_chain(left)
    right_chain = _try_build_chain(right)

    if left_chain == target_path:
        return left, right, True
    if right_chain == target_path:
        return right, left, False
    return None, None, True


def _try_build_chain(node: ast.expr) -> tuple | None:
    """Try to build an access chain from an AST node."""
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _try_build_chain(node.value)
        if parent:
            return parent + (node.attr,)
    if isinstance(node, ast.Subscript):
        parent = _try_build_chain(node.value)
        if parent:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, (str, int)):
                return parent + (node.slice.value,)
    return None


def _resolve_const(node: ast.expr, env: dict[str, object]) -> Any:
    """Resolve a constant from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        members = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                members.append(elt.value)
        return members if members else None
    if isinstance(node, ast.Name):
        if node.id in env:
            val = env[node.id]
            if isinstance(val, (int, bool, float, str, bytes, type(None))):
                return val
    # UnaryOp(USub, Constant)
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)):
        return -node.operand.value
    return None


def _resolve_const_value(node: ast.expr, env: dict[str, object]) -> Any:
    """Resolve a constant value (scalar only)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        val = env[node.id]
        if isinstance(val, (int, bool, float, str)):
            return val
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)):
        return -node.operand.value
    return None


def _false_to_true_candidates(
    op: ast.cmpop,
    const_val: Any,
    leaf_is_left: bool,
    env: dict[str, object],
) -> list[Any]:
    """Generate candidates to make a False Compare True."""
    cands: list[Any] = []

    if isinstance(op, ast.Eq):
        if isinstance(const_val, list):
            cands.extend(const_val)
        else:
            cands.append(const_val)
    elif isinstance(op, ast.NotEq):
        if isinstance(const_val, int) and not isinstance(const_val, bool):
            cands.append(const_val + 1)
    elif isinstance(op, ast.Lt):
        if isinstance(const_val, int) and not isinstance(const_val, bool):
            if leaf_is_left:
                # x < const is False -> x >= const. Want True -> x = const - 1
                cands.append(const_val - 1)
            else:
                # const < x is False -> const >= x. Want True -> x = const + 1
                cands.append(const_val + 1)
    elif isinstance(op, ast.LtE):
        if isinstance(const_val, int) and not isinstance(const_val, bool):
            if leaf_is_left:
                cands.append(const_val)
            else:
                cands.append(const_val)
    elif isinstance(op, ast.Gt):
        if isinstance(const_val, int) and not isinstance(const_val, bool):
            if leaf_is_left:
                # x > const is False -> want True -> x = const + 1
                cands.append(const_val + 1)
            else:
                cands.append(const_val - 1)
    elif isinstance(op, ast.GtE):
        if isinstance(const_val, int) and not isinstance(const_val, bool):
            if leaf_is_left:
                cands.append(const_val)
            else:
                cands.append(const_val)
    elif isinstance(op, ast.In):
        if isinstance(const_val, list):
            cands.extend(const_val)
    elif isinstance(op, ast.NotIn):
        if isinstance(const_val, list):
            int_members = [m for m in const_val if isinstance(m, int) and not isinstance(m, bool)]
            str_members = [m for m in const_val if isinstance(m, str)]
            if int_members:
                cands.append(max(int_members) + 1)
            elif str_members:
                cands.append("__X__")
            else:
                cands.append("__X__")

    return cands


def _true_to_false_candidates(
    op: ast.cmpop,
    const_val: Any,
    leaf_is_left: bool,
    env: dict[str, object],
) -> list[Any]:
    """Generate candidates to make a True Compare False."""
    cands: list[Any] = []

    if isinstance(op, ast.Eq):
        if isinstance(const_val, int) and not isinstance(const_val, bool):
            cands.append(const_val + 1)
        elif isinstance(const_val, bool):
            cands.append(not const_val)
        elif isinstance(const_val, str):
            cands.append(const_val + "x")
        elif isinstance(const_val, list):
            # eq with container -- shouldn't happen often
            pass
        else:
            pass
    elif isinstance(op, ast.NotEq):
        if isinstance(const_val, list):
            cands.extend(const_val)
        else:
            cands.append(const_val)
    elif isinstance(op, ast.Lt):
        if isinstance(const_val, int) and not isinstance(const_val, bool):
            if leaf_is_left:
                cands.append(const_val)
            else:
                cands.append(const_val)
    elif isinstance(op, ast.LtE):
        if isinstance(const_val, int) and not isinstance(const_val, bool):
            if leaf_is_left:
                cands.append(const_val + 1)
            else:
                cands.append(const_val - 1)
    elif isinstance(op, ast.Gt):
        if isinstance(const_val, int) and not isinstance(const_val, bool):
            if leaf_is_left:
                cands.append(const_val)
            else:
                cands.append(const_val)
    elif isinstance(op, ast.GtE):
        if isinstance(const_val, int) and not isinstance(const_val, bool):
            if leaf_is_left:
                cands.append(const_val - 1)
            else:
                cands.append(const_val + 1)
    elif isinstance(op, ast.In):
        if isinstance(const_val, list):
            int_members = [m for m in const_val if isinstance(m, int) and not isinstance(m, bool)]
            str_members = [m for m in const_val if isinstance(m, str)]
            if int_members:
                cands.append(max(int_members) + 1)
            elif str_members:
                cands.append("__X__")
            else:
                cands.append("__X__")
    elif isinstance(op, ast.NotIn):
        if isinstance(const_val, list):
            cands.extend(const_val)

    return cands


def _fallback_candidates(
    current_value: Any,
    leaf_type: type,
    path: Path,
) -> list[Any]:
    """Generate fallback candidates when no pattern matched."""
    if leaf_type is bool or isinstance(current_value, bool):
        return [not current_value]

    if leaf_type is int or isinstance(current_value, int):
        # Check if byte range (0-255)
        if isinstance(current_value, int) and 0 <= current_value <= 255:
            # Check if this is a byte path (parent is bytes)
            # We can't easily tell, so provide both standard and byte fallbacks
            return [0, 1, -1, current_value + 1, current_value - 1]
        return [0, 1, -1, current_value + 1, current_value - 1]

    if leaf_type is float or isinstance(current_value, float):
        return [0.0, 1.0, -1.0, current_value + 1.0, current_value - 1.0]

    if leaf_type is str or isinstance(current_value, str):
        return ["", "A", "test"]

    if leaf_type is bytes or isinstance(current_value, bytes):
        return [b"", b"\x00", b"\xff"]

    return [0, 1, -1]


def propose_combined_candidates(
    *,
    atom_exprs: list[tuple[ast.expr, bool]],  # (atom_ast, atom_value)
    target_path: Path,
    state_index: StateIndex,
    env: dict[str, object],
) -> list[Any]:
    """Stage 3/E1: Combined-constraint candidates.

    When multiple blocking atoms constrain the same leaf with compatible patterns
    (e.g., x > L and x % m == r), synthesize combined candidate.
    """
    # Collect constraints
    lower_bound: int | None = None
    modulo_spec: tuple[int, int] | None = None  # (m, r)

    for atom_expr, atom_value in atom_exprs:
        if not atom_value and isinstance(atom_expr, ast.Compare) and len(atom_expr.ops) == 1:
            op = atom_expr.ops[0]
            left = atom_expr.left
            right = atom_expr.comparators[0]

            left_chain = _try_build_chain(left)
            right_chain = _try_build_chain(right)

            if left_chain == target_path:
                const_val = _resolve_const_value(right, env)
                if const_val is not None and isinstance(const_val, int):
                    if isinstance(op, ast.Gt):
                        lower_bound = const_val + 1
                    elif isinstance(op, ast.GtE):
                        lower_bound = const_val

            # Check modulo: (leaf % m) == r
            if (isinstance(op, ast.Eq) and isinstance(left, ast.BinOp)
                    and isinstance(left.op, ast.Mod)):
                m_val = _resolve_const_value(left.right, env)
                r_val = _resolve_const_value(right, env)
                if (m_val is not None and r_val is not None
                        and isinstance(m_val, int) and isinstance(r_val, int)):
                    modulo_spec = (m_val, r_val)

    if lower_bound is not None and modulo_spec is not None:
        m, r = modulo_spec
        # Find smallest x >= lower_bound where x % m == r
        x = lower_bound
        remainder = x % m
        if remainder != r:
            diff = (r - remainder) % m
            x += diff
        return [x]

    return []
