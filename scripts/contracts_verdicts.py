"""Pinned RW/SO/float verdict+reason comparison. Run through testq with -n $TESTQ_JOBS."""
import argparse, concurrent.futures, hashlib, json, os, pathlib, subprocess, sys, tempfile

a = argparse.ArgumentParser()
a.add_argument("-n", type=int, required=True)
a.add_argument("--source", required=True)
a.add_argument("--svb", required=True)
a.add_argument("--solve-only", required=True)
a.add_argument("--harness", required=True)
a.add_argument("--python", default=sys.executable)
a.add_argument("--out", required=True)
a.add_argument("--engine", choices=["absint", "pipeline"], required=True)
args = a.parse_args()
sys.path.insert(0, args.harness)
from eregion_sv.tasks import read_set, read_task
src = pathlib.Path(args.source)
py = args.python
env = dict(os.environ, PYTHONPATH=str(src), OPENBLAS_NUM_THREADS="1")
jobs = []
for suite, svb, setname in [
    ("rw", args.svb, "Real-world.set"),
    ("so", args.solve_only, "Solve-only.set"),
]:
    for yml in read_set(os.path.join(svb, setname)):
        task = read_task(yml)
        for prop, expected, _ in task.properties:
            jobs.append((suite, task.name, task.program, prop))
abs_code = """
import json, pathlib, sys
from tharkun.solve.absint import prove
from tharkun.sv.spec import read_property
program, prop = sys.argv[1:]
r = prove(pathlib.Path(program).read_text(), read_property(prop)[1], filename=program, time_budget=120)
print(json.dumps([r.verdict, r.reason]))
"""
def one(job):
    suite, name, program, prop = job
    with tempfile.TemporaryDirectory(prefix="rul-equality-") as tmp:
        if args.engine == "absint":
            cmd = [py, "-c", abs_code, program, prop]
        else:
            cmd = [py, "-m", "tharkun.sv", program, "--property", prop, "--budget-seconds", "120",
                   "--jobs", "1", "--proof-dir", tmp]
        run = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=480)
    if args.engine == "absint":
        result = json.loads(run.stdout)
    else:
        lines = run.stdout.strip().splitlines()
        result = [lines[-1] if lines else "ERROR(no output)",
                  next((line.split("=",1)[1].split()[0] for line in reversed(lines)
                        if line.startswith("tharkun-sv: component=")), "-")]
    return (suite + "/" + name + "/" + pathlib.Path(prop).name, result)

with concurrent.futures.ThreadPoolExecutor(args.n) as pool:
    rows = dict(pool.map(one, jobs))
for suite in ["rw", "so", "float"]:
    subset = {k:v for k,v in rows.items()
              if k.startswith(suite+"/") or (suite=="float" and "/bm_float_" in k)}
    data = (json.dumps(subset, sort_keys=True, indent=2)+"\n").encode()
    path = pathlib.Path(args.out) / f"{args.engine}-{suite}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(args.engine, suite, len(subset), hashlib.sha256(data).hexdigest(), flush=True)
