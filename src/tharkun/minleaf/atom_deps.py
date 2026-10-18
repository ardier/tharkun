"""Phase 5: Atom dependency extraction.

Given an atom AST and a StateIndex, determine which leaf paths and container
paths the atom depends on, and whether the atom is structural.
"""

from __future__ import annotations

import ast

from tharkun.minleaf.types import AtomDeps, Path, StateIndex

# C5 iteration helper function names
_C5_FUNCS = {"any", "all", "sum", "sorted", "min", "max"}


def atom_dependencies(atom_expr: ast.expr, state_index: StateIndex) -> AtomDeps:
    """Compute dependencies of a single atom on the projected state.

    No env parameter. All classification is syntactic + prefix-of-leaf reachability.
    """
    # Step 1: Extract access chains
    chains: list[Path] = []
    _extract_chains(atom_expr, chains)

    # Step 2: Classify chains
    leaf_paths: set[Path] = set()
    unresolved_chains: set[Path] = set()

    for chain in chains:
        if chain in state_index.leaves:
            leaf_paths.add(chain)
        else:
            unresolved_chains.add(chain)

    # Step 3: Container-context detection (C1-C5)
    container_paths: set[Path] = set()
    structural_reason: str | None = None

    # C1: Truthiness -- atom expression is exactly the chain (bare Name/Attribute as boolean)
    top_chain = _try_build_chain(atom_expr)
    if top_chain is not None and top_chain in unresolved_chains:
        if _is_reachable_container(top_chain, state_index):
            container_paths.add(top_chain)
            if structural_reason is None:
                structural_reason = "truthiness"

    # C2: len() argument
    if isinstance(atom_expr, ast.Compare):
        # Check both sides of the compare for len() calls
        _check_len_calls(atom_expr.left, unresolved_chains, state_index,
                         container_paths)
        for comp in atom_expr.comparators:
            _check_len_calls(comp, unresolved_chains, state_index, container_paths)
        if container_paths and structural_reason is None:
            structural_reason = "len"
    _check_len_calls(atom_expr, unresolved_chains, state_index, container_paths)
    if container_paths and structural_reason is None:
        structural_reason = "len"

    # C3: Membership container -- Compare(x, [In|NotIn], [chain])
    if isinstance(atom_expr, ast.Compare) and len(atom_expr.ops) == 1:
        if isinstance(atom_expr.ops[0], (ast.In, ast.NotIn)):
            for comp in atom_expr.comparators:
                comp_chain = _try_build_chain(comp)
                if (comp_chain is not None and comp_chain in unresolved_chains
                        and _is_reachable_container(comp_chain, state_index)):
                    container_paths.add(comp_chain)
                    if structural_reason is None:
                        structural_reason = "membership_container"

    # C4: Dynamic indexing -- Subscript(value=chain, slice=<non-Constant>)
    _check_dynamic_index(atom_expr, unresolved_chains, state_index, container_paths)
    if container_paths and structural_reason is None and _has_dynamic_index(atom_expr, unresolved_chains, state_index):
        structural_reason = "dynamic_index"

    # C5: Iteration helper -- Call(func=Name(f), args=[chain]) where f in C5_FUNCS
    _check_iteration_helpers(atom_expr, unresolved_chains, state_index, container_paths)
    if container_paths and structural_reason is None and _has_iteration_helper(atom_expr, unresolved_chains, state_index):
        structural_reason = "iteration_helper"

    # Step 4: Structural classification
    structural = len(container_paths) > 0

    return AtomDeps(
        leaf_paths=frozenset(leaf_paths),
        container_paths=frozenset(container_paths),
        structural=structural,
        structural_reason=structural_reason if structural else None,
    )


def _extract_chains(node: ast.expr, chains: list[Path]) -> None:
    """Walk the atom AST recursively, extracting access chains."""
    chain = _try_build_chain(node)
    if chain is not None:
        chains.append(chain)
        # Don't recurse into the chain components -- they're part of this chain
        # But if we have a Subscript with non-constant slice, we need to recurse
        # into the slice and the value separately
        return

    # Recurse into children to find more chains
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            _extract_chains(child, chains)


def _try_build_chain(node: ast.expr) -> Path | None:
    """Try to build an access chain from an AST node.

    Returns None if the node can't form a complete chain.
    """
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _try_build_chain(node.value)
        if parent is not None:
            return parent + (node.attr,)
        return None
    if isinstance(node, ast.Subscript):
        parent = _try_build_chain(node.value)
        if parent is not None:
            key = _try_constant_key(node.slice)
            if key is not None:
                return parent + (key,)
        return None
    return None


def _try_constant_key(node: ast.expr) -> str | int | None:
    """Try to extract a constant key from a subscript slice."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return node.value
    # Python 3.8 compat: ast.Index wraps the actual value
    if isinstance(node, ast.Index):  # type: ignore[attr-defined]
        return _try_constant_key(node.value)  # type: ignore[attr-defined]
    # Negative index: UnaryOp(USub, Constant(int))
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)):
        return -node.operand.value
    return None


def _is_reachable_container(chain: Path, state_index: StateIndex) -> bool:
    """True if any leaf path starts with chain as a strict prefix."""
    return any(
        lp[:len(chain)] == chain and len(lp) > len(chain)
        for lp in state_index.leaves
    )


def _check_len_calls(
    node: ast.expr,
    unresolved: set[Path],
    state_index: StateIndex,
    container_paths: set[Path],
) -> None:
    """Check for len(chain) calls and add container deps."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            if node.args:
                arg_chain = _try_build_chain(node.args[0])
                if (arg_chain is not None and arg_chain in unresolved
                        and _is_reachable_container(arg_chain, state_index)):
                    container_paths.add(arg_chain)
    # Recurse into children
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr) and child is not node:
            _check_len_calls(child, unresolved, state_index, container_paths)


def _check_dynamic_index(
    node: ast.expr,
    unresolved: set[Path],
    state_index: StateIndex,
    container_paths: set[Path],
) -> None:
    """Check for dynamic indexing and add container deps."""
    if isinstance(node, ast.Subscript):
        chain = _try_build_chain(node.value)
        key = _try_constant_key(node.slice)
        if chain is not None and key is None:
            # Non-constant slice -> dynamic indexing
            if chain in unresolved and _is_reachable_container(chain, state_index):
                container_paths.add(chain)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            _check_dynamic_index(child, unresolved, state_index, container_paths)


def _has_dynamic_index(
    node: ast.expr,
    unresolved: set[Path],
    state_index: StateIndex,
) -> bool:
    """Check if node has a dynamic index context."""
    if isinstance(node, ast.Subscript):
        chain = _try_build_chain(node.value)
        key = _try_constant_key(node.slice)
        if chain is not None and key is None:
            if chain in unresolved and _is_reachable_container(chain, state_index):
                return True
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            if _has_dynamic_index(child, unresolved, state_index):
                return True
    return False


def _check_iteration_helpers(
    node: ast.expr,
    unresolved: set[Path],
    state_index: StateIndex,
    container_paths: set[Path],
) -> None:
    """Check for C5 iteration helper calls and add container deps."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _C5_FUNCS:
            if node.args:
                arg_chain = _try_build_chain(node.args[0])
                if (arg_chain is not None and arg_chain in unresolved
                        and _is_reachable_container(arg_chain, state_index)):
                    container_paths.add(arg_chain)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            _check_iteration_helpers(child, unresolved, state_index, container_paths)


def _has_iteration_helper(
    node: ast.expr,
    unresolved: set[Path],
    state_index: StateIndex,
) -> bool:
    """Check if node has an iteration helper context."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _C5_FUNCS:
            if node.args:
                arg_chain = _try_build_chain(node.args[0])
                if (arg_chain is not None and arg_chain in unresolved
                        and _is_reachable_container(arg_chain, state_index)):
                    return True
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            if _has_iteration_helper(child, unresolved, state_index):
                return True
    return False
