"""Phase 0: Data model for minleaf guard slicer.

All shared types. No logic beyond trivial computed properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from tharkun.solve.fact_env import FactEnv, VarFacts  # noqa: F401 -- moved to tharkun.solve (#0606); re-exported
from tharkun.solve.guard_eval import AtomEval, GuardTrace  # noqa: F401 -- moved to tharkun.solve (#0606); re-exported

PathSeg = str | int
"""Attribute name (str) or index/key (int)."""

Path = tuple[PathSeg, ...]
"""Fully-qualified path from named root to leaf. First segment is always the root name (str)."""


@dataclass(frozen=True)
class LeafRef:
    """Reference to a single patchable leaf in the projected state."""
    path: Path
    typ: type
    getter: Callable[[], Any]
    setter: Callable[[Any], None]


@dataclass(frozen=True)
class SetOp:
    """A single leaf assignment."""
    path: Path
    old: Any
    new: Any


@dataclass(frozen=True)
class Patch:
    """Ordered collection of leaf assignments. Ops sorted by path for determinism."""
    ops: tuple[SetOp, ...]

    def touched_paths(self) -> frozenset[Path]:
        return frozenset(op.path for op in self.ops)

    def cost(self) -> int:
        return len(self.touched_paths())


@dataclass(frozen=True)
class AtomDeps:
    """Dependencies of a single atom on the projected state."""
    leaf_paths: frozenset[Path]
    container_paths: frozenset[Path]             # C1-C5 contexts only
    structural: bool
    structural_reason: str | None
    # structural_reason values: "truthiness" | "len" | "membership_container" |
    #   "dynamic_index" | "iteration_helper" | None


@dataclass(frozen=True)
class LeafFacts:
    """Per-leaf-path fact constraints from upstream analysis. All fields advisory."""
    eq: frozenset[Any] | None = None
    neq: frozenset[Any] = field(default_factory=frozenset)
    lo: int | float | None = None
    hi: int | float | None = None
    typ: type | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class LeafFactsIndex:
    """Lookup table: leaf path -> LeafFacts."""
    facts: dict[Path, LeafFacts]
    global_possible: bool = True

    def get(self, path: Path) -> LeafFacts | None:
        return self.facts.get(path)

    def has_eq(self, path: Path) -> bool:
        lf = self.facts.get(path)
        return lf is not None and lf.eq is not None and len(lf.eq) > 0

    def is_excluded(self, path: Path, value: Any) -> bool:
        lf = self.facts.get(path)
        return lf is not None and value in lf.neq

    def in_bounds(self, path: Path, value: Any) -> bool:
        lf = self.facts.get(path)
        if lf is None:
            return True
        if lf.lo is not None and value < lf.lo:
            return False
        if lf.hi is not None and value > lf.hi:
            return False
        return True


@dataclass(frozen=True)
class StateIndex:
    leaves: dict[Path, LeafRef]
    bytes_policy: str
    roots: dict[str, object]


@dataclass(frozen=True)
class SolveDiagnostics:
    guard_src: str
    guard_value_before: bool
    trace: GuardTrace
    blocking_atoms: tuple[AtomEval, ...]
    dependency_leaf_paths: tuple[Path, ...]       # sorted
    dependency_container_paths: tuple[Path, ...]  # sorted
    structural_blockers: tuple[str, ...]
    facts_used: bool
    facts_injected_candidates: int
    facts_pruned_candidates: int
    tried_patches: int
    depth_reached: int
    retrace_rounds_used: int
    failure_reason: str | None


@dataclass(frozen=True)
class SolveResult:
    success: bool
    patch: Patch | None
    diagnostics: SolveDiagnostics
