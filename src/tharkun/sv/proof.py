"""Common TRUE finalizer: candidate witness, independent check, explicit gaps."""
from __future__ import annotations

import ast
import hashlib
import os
import tempfile
import time
import subprocess
from functools import lru_cache
from pathlib import Path

from tharkun.solve.obligations import Invariant, Obligation, source_points, witness
from tharkun.sv.validate import Validation, check_candidates
from tharkun.sv.witness import dumps


def finish(verdict, source, filename, specification, census, *, directory=None, seconds=3.0, python=None,
          knowledge=None, table=None, shadow_seconds=0.1):
    """All TRUE routes pass here. A TRUE from this path is never downgraded
    (owner ruling: no feature flags -- there is no runtime switch that turns an
    unchecked TRUE into UNKNOWN; see ``_downgrade_if_unchecked`` below for the
    internal, test-only equivalent of the old ``--strict`` behavior).

    The whole finalizer -- reusing ``census``, the independent checker, the
    ``python -I`` interpreter probe and writing witness.yml/obligations.yml --
    shares one wall-clock budget (``seconds``), [internal ref] review: previously only
    the checker's own solver calls were bounded, so a task with many open
    obligations could cost far more than ``seconds``. Once the deadline
    passes, remaining work is skipped and reported as unchecked/not written,
    never as a verdict change: writing the witness (or failing to) must never
    affect the verdict either ([internal ref] review: a read-only proof directory used
    to turn TRUE into UNKNOWN).

    ``knowledge`` ([internal ref] step 2, [internal ref]'s fact-kind taxonomy addendum),
    when given, gets one ``kind="candidate_invariant"`` :class:`~tharkun.facts.Fact`
    per candidate this call is ABOUT to check -- ``certified=False``: a candidate is a
    proposal, not a decision, whether or not the check below goes on to prove it.
    ``check_candidates`` below still does the SAME proving it always did; this only
    records what was proposed. Optional and additive: every existing caller (the
    ``finish()`` unit tests, anything not yet passing a store) keeps working unchanged.
    """
    deadline = time.monotonic() + max(0, seconds)
    # [internal ref]: input_file_hashes must bind the file's actual bytes, not a UTF-8
    # re-encoding of the (possibly non-UTF-8) decoded `source` -- read once,
    # falling back to that re-encoding if the file is somehow gone by now (it
    # existed when verify() read it; the witness/tamper-detection design
    # already treats a genuine source change as its own UNKNOWN, not this).
    try:
        from tharkun.sv.engine import read_source_bytes
        source_bytes = read_source_bytes(filename)
    except OSError:
        source_bytes = None
    facts = verdict.detail.pop("candidate_invariants", [])
    # A census adapter gap is not an invariant. The independent source checker
    # may discharge it using a different sufficient condition (e.g. list vs KeyError).
    candidates = [o.invariant for o in census.obligations if not o.gap] + facts
    if not candidates:
        tree = ast.parse(source, filename)
        loc = source_points(tree, source, filename)[tree.body[0]][1] if tree.body else (filename, 1, 1, "<module>")
        candidates = [Invariant(*loc, "True")]
    candidates = list(dict.fromkeys(candidates))
    if knowledge is not None:
        from tharkun.facts import Fact
        for inv in candidates:
            knowledge.add(Fact(kind="candidate_invariant",
                               subject=f"{inv.file_name}:{inv.line}:{inv.column}",
                               value=inv.value, source="finish", certified=False))
    # [internal ref] review: check the candidates directly -- no YAML round trip. ``doc``
    # (for ``correctness_witness`` / the eventual file write) is still built
    # from the exact same list, but is not needed to check it: check_candidates
    # re-proves the Invariant objects already in hand, instead of the old path
    # (serialize to YAML text, then have validate() immediately re-parse that
    # same text back into the same Invariant objects).
    doc = witness(source, filename, specification, candidates, source_bytes=source_bytes)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        result = Validation(False, census.obligations, "unchecked (budget)")
    elif python is not None and not _python314(python):
        result = Validation(False, census.obligations, "target interpreter is not CPython 3.14")
    else:
        result = check_candidates(source, filename, specification, census, candidates,
                                  seconds=deadline - time.monotonic())
    opened = result.open_obligations
    if not result.confirmed and not opened:
        opened = (Obligation(hashlib.sha256(source.encode("utf-8")).hexdigest(), "$checker", "coverage",
                             Invariant(filename, 1, 1, "<module>", "False"), result.reason),)
    verdict.detail.update(proof_checked=result.confirmed, proof_reason=result.reason,
                          open_obligations=[o.invariant.entry() for o in opened],
                          correctness_witness=doc)
    out = None
    if directory is not None:
        if time.monotonic() >= deadline:
            verdict.detail["witness_write_error"] = "budget exhausted before export"
        else:
            try:
                text = dumps(doc) if result.confirmed else None
                Path(directory).mkdir(parents=True, exist_ok=True)
                # Each run owns its directory, including simultaneous properties of one source.
                out = Path(tempfile.mkdtemp(prefix=Path(filename).stem + "-", dir=directory))
                if result.confirmed:
                    _write(out / "witness.yml", text)
                # The open set is a subset of the witness candidates already built above;
                # only re-serialize it (a second, possibly-large YAML pass) if there is
                # still budget for it ([internal ref] review).
                if time.monotonic() < deadline:
                    _write(out / "obligations.yml",
                          dumps(witness(source, filename, specification, [o.invariant for o in opened],
                                       source_bytes=source_bytes)))
                    verdict.detail["obligations_path"] = str(out / "obligations.yml")
                if result.confirmed:
                    verdict.detail["witness_path"] = str(out / "witness.yml")
            except (OSError, ValueError) as e:
                # ValueError: witness.py's dumps() self-validates via loads(),
                # which raises ValueError("witness exceeds 4 MiB") for a task
                # with roughly 13k+ open obligations (MG's [internal ref] review note).
                # A witness this large not being written must never fail the
                # verdict either -- same as any other write failure here.
                verdict.detail["witness_write_error"] = f"{type(e).__name__}: {e}"
    # Extra work starts only after the frozen checker and every legacy export.
    # ``shadow_seconds=None`` (``--no-store-witness``) skips it: no store projection, no store-witness.yml.
    if table is not None and shadow_seconds is not None:
        try:
            from tharkun.sv.store_witness import shadow
            verdict.detail.update(shadow(table, source, filename, specification,
                                         directory=out, source_bytes=source_bytes, seconds=shadow_seconds))
        except Exception as exc:
            verdict.detail["store_error"] = f"{type(exc).__name__}: {exc}"
    return verdict


def _downgrade_if_unchecked(verdict):
    """The old ``--strict`` behavior, kept ONLY as an internal function tests
    call directly (owner ruling 2026-09-28: no feature flags -- this is
    unreachable from ``verify()``, the CLI, or any runtime parameter). Turns an
    unchecked TRUE (as ``finish()`` above reports it, in ``verdict.detail``)
    into UNKNOWN with its open obligations in the reason, exactly as the old
    ``strict=True`` path did.
    """
    if not verdict.detail.get("proof_checked"):
        opened = verdict.detail.get("open_obligations", ())
        verdict.verdict = "unknown"
        verdict.reason = f"unchecked proof: {len(opened)} open obligation(s); {verdict.detail.get('proof_reason', '')}"
    return verdict


def _write(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=8)
def _python314(python):
    try:
        result = subprocess.run([python, "-I", "-c", "import sys; print(sys.implementation.name, *sys.version_info[:2])"],
                                capture_output=True, text=True, timeout=2, check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return False
    return result.stdout.strip() == "cpython 3 14"
