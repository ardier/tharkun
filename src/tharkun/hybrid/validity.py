"""Is a solution one the ORIGINAL harness could have produced?

Hand-writing a harness is a convenience, not a licence. The point of the hybrid is to
supply inputs *smartly* — never to widen the API surface so that new code becomes
reachable. Those are different claims, and only the first is a coverage result:

* **legitimate** — the original harness could produce this input; we merely found it in
  seconds instead of never.
* **a lie** — the original harness could *never* produce this input, and the "new
  coverage" is an artifact of the rewrite.

The risk is concrete, not theoretical. A real OSS-Fuzz harness reading
``ConsumeUnicodeNoSurrogates(256)`` excludes surrogate codepoints *by construction* and
caps length; a naive hand-written ``decode`` that does ``data.decode('utf-16', 'replace')``
or omits the cap can emit values outside that envelope and reach guards the original
cannot. Nothing about hand-writing prevents this — so it has to be checked.

The check here is deliberately strong: rather than reason about the envelope, we
**construct the receipt**. Run the solution back through the original harness's own FDP
codec; if it encodes to bytes that the *real* ``FuzzedDataProvider`` decodes to exactly
that solution, then the original harness demonstrably could have produced it, and the
bytes are the proof. If it cannot, we report that plainly rather than quietly counting
the win.

    v = check_reachable_by_original(orig_harness, ["password"], {"password": "..."})
    if v.valid:
        open("witness", "wb").write(v.witness)   # feed the ORIGINAL harness this
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidityVerdict:
    """Whether the original harness could produce a solution, and the proof."""

    valid: bool
    reason: str
    witness: bytes | None = None      # bytes the ORIGINAL harness decodes to the solution
    decoded: dict[str, Any] | None = None   # what those bytes actually decode to

    def __bool__(self) -> bool:
        return self.valid


def check_reachable_by_original(
    original_harness_path: str,
    param_names: list[str],
    solution: dict[str, Any],
    *,
    base_seed: bytes = b"\x00" * 64,
    harness_func: str | None = None,
) -> ValidityVerdict:
    """Prove (or refute) that the original harness can produce *solution*.

    Encodes the solution through the original harness's own recorded FDP schedule and
    verifies the result against the real provider. A success returns the witness bytes —
    hand those to the unmodified harness and it reproduces the solved input, which is
    what makes the coverage claim honest.
    """
    from tharkun.branch_solver import _load_module_for_execution
    from tharkun.hybrid.codec import FdpCodec
    from tharkun.hybrid.fdp_codec import atheris_available

    if not atheris_available():
        return ValidityVerdict(False, "atheris unavailable — cannot check the original harness")

    mod, _ = _load_module_for_execution(os.path.realpath(original_harness_path))
    if mod is None:
        return ValidityVerdict(False, f"could not load {original_harness_path}")

    fn = getattr(mod, harness_func, None) if harness_func else None
    if fn is None:
        fn = getattr(mod, "TestOneInput", None) or getattr(mod, "TestInput", None)
    if fn is None:
        return ValidityVerdict(False, "original harness exposes no TestOneInput/TestInput")

    codec = FdpCodec(fn, list(param_names))
    try:
        witness = codec.encode(base_seed, dict(solution))
    except Exception as exc:                       # noqa: BLE001 — report, never raise
        return ValidityVerdict(False, f"encoding through the original raised {type(exc).__name__}")

    if witness is None:
        return ValidityVerdict(
            False,
            "the original harness cannot represent this input — it lies outside the "
            "envelope its own decoder can produce, so the coverage would not be real",
        )

    decoded = codec.decode(witness)
    for name, want in solution.items():
        if name in decoded and decoded[name] != want:
            return ValidityVerdict(
                False,
                f"round-trip disagreed on {name!r}: original decodes {decoded[name]!r}",
                witness=witness, decoded=decoded,
            )

    return ValidityVerdict(
        True,
        "the original harness produces this input from the witness bytes",
        witness=witness, decoded=decoded,
    )


def describe_domain(harness_path: str, *, samples: int = 512,
                    harness_func: str | None = None) -> dict[str, Any]:
    """Sample what a harness's decoder actually emits, to compare two envelopes.

    A coarse but useful sanity read when the strong check is unavailable: the maximum
    length observed, whether surrogates ever appear, and whether values stay ASCII. A
    hand-written replacement that exceeds any of these is suspect.

    Reads the raw FDP consumption schedule rather than mapped parameters, because the
    envelope is a property of the harness's decoders and must not depend on how fields
    happen to be named or mapped.
    """
    import random

    from tharkun.branch_solver import _load_module_for_execution
    from tharkun.hybrid.fdp_codec import atheris_available, record_schedule

    out: dict[str, Any] = {"max_len": 0, "surrogates": False, "non_ascii": False,
                           "fields": set(), "sampled": 0}
    if not atheris_available():
        return out
    mod, _ = _load_module_for_execution(os.path.realpath(harness_path))
    if mod is None:
        return out
    fn = getattr(mod, harness_func, None) if harness_func else None
    if fn is None:
        fn = getattr(mod, "TestOneInput", None) or getattr(mod, "TestInput", None)
    if fn is None:
        return out

    rng = random.Random(0)
    for _ in range(samples):
        n = rng.randint(0, 64)
        data = bytes(rng.randrange(256) for _ in range(n))
        try:
            # Read the raw consumption schedule: the envelope is a property of the
            # harness's own decoders, independent of any param-name mapping.
            values = [r.value for r in record_schedule(fn, data)]
        except Exception:      # noqa: BLE001 — a decoder that throws tells us nothing
            continue
        out["sampled"] += 1
        out["fields"].add(len(values))
        for v in values:
            if isinstance(v, (str, bytes, list, dict)):
                out["max_len"] = max(out["max_len"], len(v))
            if isinstance(v, str):
                if any(0xD800 <= ord(c) <= 0xDFFF for c in v):
                    out["surrogates"] = True
                if not v.isascii():
                    out["non_ascii"] = True
    out["fields"] = sorted(out["fields"])
    return out
