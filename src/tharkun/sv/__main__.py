"""``python -m tharkun.sv``: tharkun as an SV-COMP Python verifier.

Usage::

    python -m tharkun.sv PROGRAM.py|TASK.yml --property P.prp [--python PY]
        [--budget-seconds S] [--canary-seconds S] [--run-timeout S] [--witness-out W.json]
    python -m tharkun.sv PROGRAM.py --property P.prp --replay W.json

The last line printed is the verdict: ``TRUE``, ``FALSE(<Exception>)`` or
``UNKNOWN(<reason>)`` (``--replay`` prints ``CONFIRMED`` / ``NOT CONFIRMED``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tharkun.sv.engine import admit_task, read_source, read_source_bytes, replay, verify
from tharkun.sv.spec import read_property

#: the package version (pyproject.toml, ``tharkun.__version__``), as the witness producer names it too;
#: a literal because ``tharkun.sv`` may not import the top-level package (tests/test_layering.py);
#: tests/test_version_sync.py keeps the three in step
VERSION = "tharkun-sv 0.2.0"


def _stage_names(a):
    from tharkun.sv import stages
    names = list(stages.DEFAULT) if a.stages is None else [n for n in a.stages.split(",") if n]
    return [n for n in names if n not in a.no_stage]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tharkun.sv", description=__doc__.split("\n")[0])
    ap.add_argument("task", nargs="?", help="the program (.py) or its task definition (.yml)")
    ap.add_argument("--property", help="the .prp file")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter the program runs under (the track's language version)")
    ap.add_argument("--language-version", help="task language declaration (default: Python 3.14)")
    ap.add_argument("--budget-seconds", type=float, default=600.0)
    ap.add_argument("--canary-seconds", type=float, default=60.0,
                    help="CPU seconds the search may keep trying to refute a proof's TRUE (at most a tenth of the "
                         "budget): 60 is the shipped behaviour, the competition's and the regression runs'; 0 returns "
                         "the TRUE the moment a proof decides it (development campaigns)")
    ap.add_argument("--run-timeout", type=float, default=120.0)
    ap.add_argument("--max-runs", type=int, default=100000)
    ap.add_argument("--jobs", type=int, default=0,
                    help="child runs side by side (0: the usable CPUs, at most 4)")
    ap.add_argument("--run-memory-mb", type=int, default=0,
                    help="address space the child runs may use together, split evenly over "
                         "--jobs (0: no cap); keep it below the verifier's own memory limit")
    ap.add_argument("--no-constants", action="store_true",
                    help="do not try the program's own literals first (#1028; for ablations)")
    ap.add_argument("--no-deepening", action="store_true",
                    help="stop at the fixed candidate sample (#1030; for ablations)")
    ap.add_argument("--no-concolic", action="store_true",
                    help="skip solving for violating draws (#1028 slice 2; for ablations)")
    ap.add_argument("--no-prove", action="store_true",
                    help="skip the static proof stages (sv/prove.py, #1021; for ablations)")
    ap.add_argument("--stages", default=None,
                    help="comma-separated portfolio stages to run (tharkun.sv.stages; default: "
                         "its DEFAULT)")
    ap.add_argument("--no-stage", action="append", default=[], metavar="NAME",
                    help="drop one portfolio stage (repeatable; for ablations)")
    ap.add_argument("--witness-out", help="write the FALSE witness (sv-py-witness/1 JSON) here")
    ap.add_argument("--proof-dir", default="proofs", help="parent directory for per-run witness.yml and obligations.yml")
    ap.add_argument("--no-store-witness", action="store_true",
                    help="do not write the experimental store-derived witness (store-witness.yml) next to witness.yml")
    ap.add_argument("--validate-witness", help="independently check this YAML correctness witness")
    ap.add_argument("--replay", help="validate this witness instead of searching")
    ap.add_argument("--version", action="store_true")
    a = ap.parse_args(argv)
    if a.version:
        print(VERSION)
        return 0
    if not a.task or not a.property:
        ap.error("task and --property are required")
    from tharkun.solve.environment import TaskEnvironment, TaskMetadataError

    metadata_error = None
    try:
        environment = TaskEnvironment.from_task(a.task)
    except TaskMetadataError as error:
        environment, metadata_error = TaskEnvironment(error.files), str(error)
    except (OSError, ValueError) as error:
        environment, metadata_error = TaskEnvironment(()), f"unsupported task metadata: {error}"
    if not environment.files and metadata_error is None:
        ap.error("cannot read input_files")
    if a.witness_out and environment.files and not (a.replay or a.validate_witness):
        witness_out = Path(a.witness_out)
        inputs = (*environment.files, a.task, a.property)
        if witness_out.resolve() in {Path(path).resolve() for path in inputs}:
            ap.error("--witness-out must differ from the task, its inputs and the property")
        # A witness belongs only to this invocation, including runs that terminate early.
        witness_out.unlink(missing_ok=True)
    if metadata_error:
        if a.replay:
            print(f"tharkun-sv: replay -> {metadata_error}")
            print("NOT CONFIRMED")
            return 1
        print(f"UNKNOWN({metadata_error})")
        return 1 if a.validate_witness else 0
    program = environment.files[0]
    language_version = a.language_version
    if language_version is None and not a.task.endswith((".yml", ".yaml")):
        language_version = "3.14"
    prop_name, classes = read_property(a.property)
    with open(a.property, encoding="utf-8", newline="") as fh:
        specification = fh.read()
    if a.validate_witness:
        _, reason = admit_task(program, task=a.task, python=a.python, language_version=language_version)
        if reason:
            print(f"UNKNOWN({reason})")
            return 1
        from tharkun.sv.validate import validate
        source = read_source(program)
        source_bytes = read_source_bytes(program)
        with open(a.validate_witness, encoding="utf-8") as fh:
            result = validate(source, program, specification, fh.read(), source_bytes=source_bytes)
        print("CONFIRMED" if result.confirmed else f"UNKNOWN({result.reason})")
        return 0 if result.confirmed else 1
    if a.replay:
        with open(a.replay) as fh:
            w = json.load(fh)
        ok, out = replay(program, w, classes, python=a.python, timeout=a.run_timeout, task=a.task,
                         language_version=language_version)
        print(f"tharkun-sv: replay -> {out.status} {out.msg or out.exc or ''} at {out.site}")
        print("CONFIRMED" if ok else "NOT CONFIRMED")
        return 0 if ok else 1
    print(f"{VERSION}: {program} ! uncaught({', '.join(classes)})", flush=True)
    v = verify(program, classes, prop_name=prop_name, python=a.python,
               budget_seconds=a.budget_seconds, canary_seconds=a.canary_seconds, run_timeout=a.run_timeout,
               max_runs=a.max_runs,
               run_memory_bytes=a.run_memory_mb * 2**20 or None, jobs=a.jobs or None,
               constants=not a.no_constants, deepen=not a.no_deepening, concolic=not a.no_concolic,
               prove=not a.no_prove,
               stages=_stage_names(a), task=a.task, specification=specification, witness_dir=a.proof_dir,
               language_version=language_version, store_witness=not a.no_store_witness,
               log=lambda m: print(f"tharkun-sv: {m}", flush=True))
    print(f"tharkun-sv: {v.runs} runs in {v.seconds:.1f}s; {v.reason}")
    if v.witness is not None:
        print(f"tharkun-sv: witness {json.dumps(v.witness)}")
        if a.witness_out:
            with open(a.witness_out, "w") as fh:
                json.dump(v.witness, fh, indent=1)
    if "proof_checked" in v.detail:
        # Tagged "proof: " (not "checked_witness=...") so the SV harness's
        # diagnostic-line grammar recognizes this line as non-verdict output;
        # otherwise "checked=False" reads as an embedded FALSE verdict token.
        print(f"tharkun-sv: proof: checked_witness={v.detail['proof_checked']} "
              f"open_obligations={len(v.detail.get('open_obligations', []))} "
              f"witness={v.detail.get('witness_path', '')}")
        if v.detail.get("witness_write_error"):
            print(f"tharkun-sv: proof: witness not written: {v.detail['witness_write_error']}")
    calls = v.detail.get("solver_calls")
    print(f"tharkun-sv: component={v.component}" + (f" solver_calls={calls}" if calls is not None else ""))
    print(v.line(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
