"""Hybrid concolic fuzzer: Atheris in the loop, tharkun unblocking its frontiers.

The loop: run the real OSS Atheris harness in the background; when its coverage
plateaus on a hard guard, tharkun finds the high-value frontier, solves it, and recodes
the solution into a byte seed Atheris reloads and runs past the guard.

    from tharkun.hybrid import HybridFuzzer, HarnessSpec

    spec = HarnessSpec(
        harness_path="fuzz_harness.py",   # unmodified OSS harness (TestOneInput + FDP)
        target_path="mylib/parser.py",    # the code whose branches we want covered
        target_func="parse",              # what the harness exercises
        corpus_dir="./corpus",
    )
    stats = HybridFuzzer(spec).run(duration_s=300)   # atheris bg + assist on plateau

Everything is gated on the optional ``atheris`` dependency (tested on CPython 3.14.4); import
:func:`atheris_available` to check.  The frontier finder and the orchestration logic are
pure and usable without atheris.
"""

from __future__ import annotations

from .assist import AssistEngine, AssistResult, Frontier, HarnessSpec, find_frontiers
from .backend import AtherisBackend, FakeBackend
from .fdp_codec import (
    FieldRecord,
    RecodeRejection,
    RecordingFDP,
    atheris_available,
    encode_overrides,
    field_bounds,
    record_schedule,
    replay_values,
)
from .fuzzer import HybridFuzzer, HybridStats, should_assist

__all__ = [
    # orchestration
    "HybridFuzzer",
    "HybridStats",
    "HarnessSpec",
    "should_assist",
    # backends
    "AtherisBackend",
    "FakeBackend",
    # assist engine
    "AssistEngine",
    "AssistResult",
    "Frontier",
    "find_frontiers",
    # FDP codec
    "FieldRecord",
    "RecodeRejection",
    "RecordingFDP",
    "atheris_available",
    "encode_overrides",
    "field_bounds",
    "record_schedule",
    "replay_values",
]
