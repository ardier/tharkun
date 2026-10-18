"""scripts/absint_kind_matrix.py -- run the kind matrix (tests/test_absint_kind_matrix.py) over EVERY program (the tests run a slice).

Run from the repository root: ``PYTHONPATH=src python scripts/absint_kind_matrix.py [WORKERS]``; results in ``$MATRIX_OUT`` (default ``.``):
matrix_wrong.txt, matrix_crash.txt, matrix_skipped.txt.  Run it through the queue (``testq``), never as a fan-out of your own.
``MATRIX_FILTER=<regex>`` runs only the programs whose label (``<value> | <consumer> [w=<other>]``) matches: the negative controls.

Run the kind matrix over every program: bounded workers (min($TESTQ_JOBS, MAX_WORKERS)), each capped in address space,
each program's native run a capped child (tests/test_absint_kind_matrix.py::native).  A program that hits a cap is SKIPPED and
listed, never passed."""
import collections, multiprocessing as mp, os, resource, sys, time, traceback
sys.path.insert(0, ".")
from tests import test_absint_kind_matrix as T

OUT = os.environ.get("MATRIX_OUT", ".")


def _init():
    resource.setrlimit(resource.RLIMIT_AS, (T.WORKER_BYTES, T.WORKER_BYTES))


def one(args):
    label, setup, consumer = args
    try:
        got = T.check(label, setup, consumer)
    except MemoryError:
        return label, None, "MemoryError in absint (worker cap)"
    except BaseException as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)[-1]
        return label, [], f"{type(e).__name__}: {str(e)[:100]} @ {tb.name}:{tb.lineno}"
    if got is T.SKIPPED:
        return label, None, None
    return label, got, None


if __name__ == "__main__":
    t0 = time.time()
    workers = max(1, min(int(sys.argv[1]) if len(sys.argv) > 1 else 8, T.MAX_WORKERS))
    wrongs, crashes, skipped = [], [], []
    programs = T.PROGRAMS
    if os.environ.get("MATRIX_FILTER"):                  # negative controls: only the programs whose label matches this regex
        import re

        pattern = re.compile(os.environ["MATRIX_FILTER"])
        programs = [p for p in programs if pattern.search(p[0])]
    done = 0
    with mp.get_context("fork").Pool(workers, initializer=_init) as pool:
        for label, w, crash in pool.imap_unordered(one, programs, chunksize=64):
            done += 1
            if w is None:
                skipped.append((label, crash))
            else:
                wrongs += w
            if crash and w is not None:
                crashes.append((label, crash))
            if done % 50000 == 0:
                print(f"  ... {done}/{len(programs)} {time.time()-t0:.0f}s", flush=True)
    print(f"programs={len(programs)} checked={len(programs)-len(skipped)} skipped={len(skipped)} wrong_true={len(wrongs)} crashes={len(crashes)} "
          f"workers={workers} secs={time.time()-t0:.0f}")
    open(os.path.join(OUT, "matrix_wrong.txt"), "w").write("\n=====\n".join(wrongs))
    open(os.path.join(OUT, "matrix_crash.txt"), "w").write("\n".join(f"{l}  ->  {c}" for l, c in crashes))
    open(os.path.join(OUT, "matrix_skipped.txt"), "w").write("\n".join(f"{l}  ->  {c}" for l, c in skipped))
    print("crash sites:", collections.Counter(c.split(" @ ")[-1] for _, c in crashes).most_common(10))
    print("skipped by consumer:", collections.Counter(l.split(" | ")[1].split(" w=")[0] for l, _ in skipped).most_common(15))
    print("skipped by value:", collections.Counter(l.split(" | ")[0] for l, _ in skipped).most_common(15))
