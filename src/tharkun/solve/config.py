"""Centralized configuration for tharkun solver parameters.

All tunable thresholds, confidence values, timeouts, and limits are
defined here with sensible defaults. Override by modifying the module-level
``Config`` instance or by calling ``configure()``.

Usage::

    from tharkun.solve.config import config

    # Read a value
    timeout = config.snapshot_timeout_s

    # Override at runtime
    from tharkun.solve.config import configure
    configure(snapshot_timeout_s=5.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """Tharkun solver configuration.

    All values have sensible defaults. Override as needed.
    """

    # --- Confidence scores ---
    # Confidence when all free variables have directly-extracted values
    confidence_direct: float = 0.9
    # Confidence when type_inferred inputs pass guard verification
    confidence_verified: float = 0.85
    # Confidence when or-branch has some direct values
    confidence_or_branch_direct: float = 0.8
    # Confidence for type_inferred without direct extraction
    confidence_type_inferred: float = 0.5
    # Confidence for or-branch without direct values
    confidence_or_branch_inferred: float = 0.4
    # Confidence for snapshot-resolved (all vars known from runtime)
    confidence_snapshot_resolved: float = 0.95
    # Confidence for snapshot-simplified (guard reduced then solved)
    confidence_snapshot_simplified: float = 0.85
    # Confidence for snapshot-assisted (type hints from snapshot)
    confidence_snapshot_assisted: float = 0.7

    # --- Timeouts ---
    # Runtime snapshot capture timeout
    snapshot_timeout_s: float = 2.0

    # --- Limits ---
    # Max guard decomposition recursion depth
    max_decompose_depth: int = 5

    # --- Numeric ranges ---
    # Default bounds for int constraint intersection
    int_range_default: tuple[int, int] = (-10000, 10000)
    # Hypothesis int/float strategy bounds
    hypothesis_int_range: tuple[int, int] = (-1000, 1000)
    # Window around narrowed constraint values for Hypothesis
    narrowed_window: int = 1000

    # --- Stdlib contracts ([internal ref] step 2) ---
    # 2026-09-28 owner direction: no feature flags. Every engine that consults
    # ``tharkun.solve.contracts`` (through its thin adapter, ``solve/contracts/
    # adapters/``) does so unconditionally -- an adapter is either correct and
    # wired in, or it is not in the tree yet. There is no flag left to flip.


# Module-level singleton
config = Config()


def configure(**kwargs: object) -> None:
    """Override configuration values.

    Example::

        configure(snapshot_timeout_s=5.0, confidence_direct=0.95)
    """
    for key, value in kwargs.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(config, key, value)


def reset() -> None:
    """Reset all configuration to defaults (mutates in-place)."""
    defaults = Config()
    for f in Config.__dataclass_fields__:
        setattr(config, f, getattr(defaults, f))
