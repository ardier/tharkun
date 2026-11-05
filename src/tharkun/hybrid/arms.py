"""Three ways to search one program, measured the same way: generator alone, free fuzzer, domain-limited and steered.

Owner 2026-10-23 (limit-then-steer): a generator (a ``tharkun.strategies`` constraint dict) LIMITS the input domain, the
hybrid fuzzer searches INSIDE it and feedback STEERS.  A :class:`Subject` is one program plus its domain, and each arm
searches it once with one seed:

``generator``  draws members of the domain (Hypothesis, ``@seed``), runs the program, no feedback of any kind.
``targeted``   ``generator`` plus Hypothesis's own ``target()`` on the distance (an ablation: steering without Atheris).
``free``       Atheris over raw bytes decoded by the harness the way an OSS harness decodes them (no domain), coverage only.
``domain``     Atheris over bytes decoded into the domain (:class:`~tharkun.hybrid.domain_codec.DomainCodec`), coverage only.
``free_fb``    ``free`` plus the distance feedback (an ablation: is it the domain or the feedback that helps?).
``compound``   ``domain`` plus the distance feedback (:class:`~tharkun.hybrid.steer.Ladder`), seeded with generator draws.

Every arm reports whether the target was hit (the target line fired AND the wanted result y was produced), the executions
and seconds to the first hit, and the share of its executions whose input was a member of the domain.  Each run is one
child process (``python -m tharkun.hybrid.arms``) so the arms are isolated and a crash costs one run.

Determinism (owner 2026-10-23, "set the seed"): the seed goes to Hypothesis (``@seed``) and to libFuzzer (``-seed=``),
the child runs with ``PYTHONHASHSEED=0``, and the same seed gives the same executions-to-hit.  A run that ends on the
wall-clock budget ends where the clock says, so only a run that hits is exactly reproducible; ``max_execs`` makes the
budget deterministic when it binds first.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

ARMS = ("generator", "targeted", "free", "domain", "free_fb", "compound")
_DOMAIN_ARMS = ("domain", "compound")
_FEEDBACK_ARMS = ("free_fb", "compound")
_HYPOTHESIS_ARMS = ("generator", "targeted")


@dataclass
class Subject:
    """One program under search, its domain, and how to tell a hit."""

    name: str
    target: str                                   # what a hit means, one line
    domains: Mapping[str, dict]                   # the LIMIT: one strategies.make_strategy dict per parameter
    packages: tuple[str, ...]                     # top-level modules Atheris instruments (coverage of the program only)
    load: Callable[[], Callable[..., Any]]        # import the program and return ``call(**args)``
    free: Callable[[Any], dict]                   # the byte-level harness: FuzzedDataProvider -> args, no domain
    in_domain: Callable[[dict], bool]             # membership of ``domains``, written independently of the generator
    hit: Callable[[dict, Any], bool] | None = None      # (args, result) -> y was produced; None: the line alone is the hit
    line: tuple[str, str] | None = None           # (module, text of the target line); must fire for a hit
    distance: Callable[[dict, Any], float | None] | None = None   # (args, result) -> distance from y (the feedback)
    distance_kind: str = "log"
    distance_scale: float = 1.0
    gate_distance: bool = True                    # count the distance only when the target ``line`` fired
    paths: tuple[str, ...] = ()                   # added to sys.path before ``load``


def load_subject(ref: str) -> Subject:
    """``path/to/file.py:NAME``: the Subject (or a zero-argument factory of one) called NAME in that file."""
    path, _, name = ref.rpartition(":")
    spec = importlib.util.spec_from_file_location("_tharkun_subjects", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tharkun_subjects"] = mod
    spec.loader.exec_module(mod)
    obj = getattr(mod, name)
    return obj() if callable(obj) and not isinstance(obj, Subject) else obj


def _line_number(module: str, text: str) -> tuple[str, int]:
    filename = sys.modules[module].__file__
    rows = [i for i, row in enumerate(Path(filename).read_text().splitlines(), 1) if row.strip() == text.strip()]
    if len(rows) != 1:
        raise ValueError(f"{text!r} is on {len(rows)} lines of {filename}, not exactly one")
    return filename, rows[0]


class _Done(Exception):
    """Leaves Hypothesis's loop once a run is over (the fuzzing arms leave through ``os._exit``).  Never raised inside the
    program's own call, so its ``except Exception`` cannot swallow it."""


def run_worker(subject: Subject, arm: str, seed: int, *, max_execs: int, max_seconds: float, out_path: str,
               seeds: int = 8) -> None:
    """Search ``subject`` with ``arm`` and write the result JSON to ``out_path``.  Runs in the child process."""
    if arm not in ARMS:
        raise ValueError(f"arm is one of {ARMS}, not {arm!r}")
    for p in subject.paths:
        sys.path.insert(0, p)
    fuzzing = arm not in _HYPOTHESIS_ARMS
    if fuzzing:
        import atheris

        with atheris.instrument_imports(include=list(subject.packages)):
            call = subject.load()
    else:
        call = subject.load()

    from tharkun.hybrid.domain_codec import DomainCodec
    from tharkun.hybrid.steer import Ladder, LineWatch

    codec = DomainCodec(subject.domains) if arm in _DOMAIN_ARMS + _HYPOTHESIS_ARMS else None
    ladder = Ladder() if arm in _FEEDBACK_ARMS and subject.distance is not None else None
    watch = LineWatch(*_line_number(*subject.line)) if subject.line else None
    n = {"execs": 0, "calls": 0, "rejected": 0, "valid": 0, "accepted": 0}
    res: dict[str, Any] = {"subject": subject.name, "arm": arm, "seed": seed, "hit": False, "reason": None}
    t0 = time.perf_counter()

    def finish(reason: str) -> None:
        if res["reason"] is None:     # Hypothesis replays its failing example once more: the first finish stands
            res.update(n, reason=reason, seconds=round(time.perf_counter() - t0, 4))
            res["valid_rate"] = round(n["valid"] / n["execs"], 4) if n["execs"] else 0.0
            Path(out_path).write_text(json.dumps(res))
        if fuzzing:
            os._exit(0)               # Atheris treats an exception out of the callback as a crash
        raise _Done()

    def execute(args: dict) -> float | None:
        """Run the program on ``args``; returns the distance from y (None: undefined), and ends the run on a hit."""
        n["calls"] += 1
        n["valid"] += bool(subject.in_domain(args))
        if watch:
            watch.reset()
        try:
            result, failed = call(**args), False
        except Exception:  # noqa: BLE001 -- a program rejecting an input is data, not an error
            result, failed = None, True
        n["accepted"] += not failed
        d = None
        if subject.distance is not None:
            # the distance counts only through the target line: an input that never reaches it is not close to y
            gated = subject.gate_distance and watch is not None and not watch.fired
            d = None if failed or gated else subject.distance(args, result)
            if d is not None and d < res.get("best_distance", float("inf")):
                res["best_distance"] = d
            if ladder is not None:
                ladder.mark(d, kind=subject.distance_kind, scale=subject.distance_scale)
        if not failed and (watch is None or watch.fired) and (subject.hit is None or subject.hit(args, result)):
            res.update(hit=True, execs_to_hit=n["execs"], seconds_to_hit=round(time.perf_counter() - t0, 4),
                       input=repr(args)[:300], result=repr(result)[:100])
            finish("hit")
        return d

    def over_budget() -> None:
        if n["execs"] >= max_execs:
            finish("budget_execs")
        if time.perf_counter() - t0 > max_seconds:
            finish("budget_seconds")

    try:
        if arm in _HYPOTHESIS_ARMS:
            _generate(codec, seed, n, res, execute, over_budget, targeted=arm == "targeted")
            finish("exhausted")
        else:
            import atheris

            def TestOneInput(data: bytes) -> None:
                over_budget()
                n["execs"] += 1
                if codec is not None:
                    args = codec.decode(data)
                else:
                    try:
                        args = subject.free(atheris.FuzzedDataProvider(data))
                    except Exception:  # noqa: BLE001 -- a short buffer the harness cannot decode
                        args = None
                if args is None:
                    n["rejected"] += 1
                    return
                execute(args)

            corpus = tempfile.mkdtemp(prefix="corpus-")
            if arm == "compound" and codec is not None:
                for i, buf in enumerate(codec.seeds(seeds, seed)):
                    Path(corpus, f"seed{i:02d}").write_bytes(buf)
            atheris.Setup([sys.argv[0], corpus, f"-seed={seed}", "-rss_limit_mb=0", *os.environ.get("THARKUN_LIBFUZZER_ARGS", "").split()], TestOneInput)
            t0 = time.perf_counter()
            atheris.Fuzz()
    except _Done:
        pass
    finally:
        if watch:
            watch.close()
    os._exit(0)   # libFuzzer's Fuzz() does not return control; the result file is already written


def _generate(codec: Any, seed: int, n: dict, res: dict, execute: Callable[[dict], float | None],
              over_budget: Callable[[], None], *, targeted: bool) -> None:
    """Hypothesis draws members of the domain.  Alone, nothing is fed back; ``targeted`` feeds the distance to ``target()``."""
    from hypothesis import HealthCheck, Phase, Verbosity, given, settings, target
    from hypothesis import seed as hypothesis_seed

    # Hypothesis mixes the integer/string literals of every loaded LOCAL module (here: the program under test) into its
    # draws; tharkun.strategies.examples switches that off ("a program's literals reach a strategy only as `prefer`"),
    # and so does this arm, or the generator would read the answer off the program's source.
    from tharkun.strategies import make_strategy, without_local_constants

    @hypothesis_seed(seed)
    @settings(database=None, deadline=None, derandomize=False, phases=[Phase.generate, Phase.target] if targeted else [Phase.generate],
              max_examples=10 ** 9,
              suppress_health_check=list(HealthCheck), verbosity=Verbosity.quiet, report_multiple_bugs=False)
    @given(make_strategy(codec.domain))
    def draw(values: tuple) -> None:
        if res["reason"] is not None:
            raise _Done()
        over_budget()
        n["execs"] += 1
        d = execute(dict(zip(codec.names, values)))
        if targeted:
            target(-(1e18 if d is None else float(d)), label="distance from y")

    with without_local_constants():
        draw()


# --- parent side --------------------------------------------------------------------------------------------------

_COV = re.compile(rb"cov:\s*(\d+)")


def run_arm(spec_ref: str, arm: str, seed: int, *, workdir: str, max_execs: int = 50_000_000, max_seconds: float = 20.0,
            python: str | None = None) -> dict:
    """Run one arm in a child process; return its result dict (``error`` set when the child died without one)."""
    out = tempfile.NamedTemporaryFile(dir=workdir, prefix="arm-", suffix=".json", delete=False).name
    err = out + ".log"
    src = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [python or sys.executable, "-m", "tharkun.hybrid.arms", "--spec", spec_ref, "--arm", arm, "--seed", str(seed),
           "--max-execs", str(max_execs), "--max-seconds", str(max_seconds), "--out", out]
    started = time.perf_counter()
    try:
        with open(err, "wb") as log:
            subprocess.run(cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           timeout=max_seconds + 90)
        raw = Path(out).read_text() if Path(out).stat().st_size else ""
        result = json.loads(raw) if raw else {"error": "child wrote no result", "tail": Path(err).read_bytes()[-400:].decode(errors="replace")}
    except subprocess.TimeoutExpired:
        result = {"error": "child timed out"}
    covs = _COV.findall(Path(err).read_bytes())
    result.update(subject=spec_ref.rpartition(":")[2], arm=arm, seed=seed, cov=int(covs[-1]) if covs else None,
                  wall_seconds=round(time.perf_counter() - started, 2))
    for p in (out, err):
        Path(p).unlink(missing_ok=True)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="one arm, one seed, one subject (the child of run_arm)")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-execs", type=int, default=50_000_000)
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run_worker(load_subject(a.spec), a.arm, a.seed, max_execs=a.max_execs, max_seconds=a.max_seconds, out_path=a.out)


if __name__ == "__main__":
    main()
