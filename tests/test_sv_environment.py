"""Two small programs per input channel, including both soundness directions."""
import sys

import pytest

from tharkun.sv import replay, run_once, verify
from tharkun.solve.environment import TaskEnvironment

PYTHON = sys.executable


def program(tmp_path, source):
    p = tmp_path / "program.py"
    p.write_text(source)
    return str(p)


def test_argv_draw_is_a_replayable_list_and_is_never_an_exhaustive_true(tmp_path):
    """The proposed task domain is the whole list; no sample exhausts it."""
    p = program(tmp_path, "import sys\nif len(sys.argv) == 2 and sys.argv[1] == 'boom': raise IndexError\n")
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, jobs=1, concolic=False)
    assert v.verdict == "false", v
    assert v.witness["vector"][0] == {"kind": "argv_full", "value": [p, "boom"]}
    assert replay(p, v.witness, ("IndexError",), python=PYTHON)[0]
    typed = program(tmp_path, "import sys\nif type(sys.argv) is list and len(sys.argv) == 1: raise IndexError\n")
    v = verify(typed, ("IndexError",), python=PYTHON, budget_seconds=60, jobs=1, concolic=False)
    assert v.verdict == "false" and replay(typed, v.witness, ("IndexError",), python=PYTHON)[0]
    changed = program(tmp_path, "import sys\nif len(sys.argv) == 1:\n    sys.argv.append('changed')\n    raise IndexError\n")
    v = verify(changed, ("IndexError",), python=PYTHON, budget_seconds=60, jobs=1, concolic=False)
    assert v.verdict == "false" and len(v.witness["vector"][0]["value"]) == 1
    assert replay(changed, v.witness, ("IndexError",), python=PYTHON)[0]
    invalid = {"vector": [{"kind": "argv", "value": [1]}]}
    ok, out = replay(changed, invalid, ("IndexError",), python=PYTHON)
    assert not ok and out.status == "invalid-draw"
    for vector in ([{"kind": "argv"}], [None]):
        ok, out = replay(program(tmp_path, "pass\n"), {"vector": vector}, ("KeyError",), python=PYTHON)
        assert not ok and out.status == "invalid-draw"
    # Negative control for TRUE: the sampled empty tail is safe, but another tail fails.
    p = program(tmp_path, "import sys\nif sys.argv[1:] == ['not-sampled']: raise IndexError\n")
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=8, jobs=1,
               concolic=False, prove=False)
    assert v.verdict != "true"


def test_full_argv_covers_empty_list_and_arbitrary_index_zero_and_keeps_legacy_tails(tmp_path):
    for source in ("import sys\nassert sys.argv[0] != ''\n",
                   "import sys\nassert sys.argv != []\n",
                   "import sys, os\nassert os.path.basename(sys.argv[0]) != ''\n"):
        p = program(tmp_path, source)
        v = verify(p, ("AssertionError",), python=PYTHON, budget_seconds=60, jobs=1, concolic=False)
        assert v.verdict == "false", v
        assert v.witness["vector"][0]["kind"] == "argv_full"
        assert replay(p, v.witness, ("AssertionError",), python=PYTHON)[0]
    p = program(tmp_path, "import sys, os\nassert os.path.realpath(sys.argv[0]) == os.path.realpath(__file__)\n")
    out = run_once(PYTHON, p, [{"kind": "argv", "value": ["irrelevant"]}], timeout=30)
    assert out.status == "clean", out


def test_orig_argv_is_flagged_and_never_a_witness(tmp_path):
    """MG review, [internal ref], 2026-09-29: sys.orig_argv is real, unmodelled CPython
    interpreter metadata -- a fresh-subprocess replay and a reused forked "explore"
    worker give genuinely DIFFERENT values (fork() never re-execs), so a run that
    reads it can never be trusted as a witness, and a static read of it must not
    license a TRUE either."""
    p = program(tmp_path, "import sys\na = sys.orig_argv\nassert len(a) <= 3\n")
    out = run_once(PYTHON, p, [], timeout=30)
    assert "orig_argv" in out.flags
    v = verify(p, ("AssertionError",), python=PYTHON, budget_seconds=60, jobs=1, concolic=False)
    assert v.verdict != "false", v
    from tharkun.solve.environment import proof_blocker
    blocker = proof_blocker("import sys\nx = sys.orig_argv\n")
    assert blocker is not None and "orig_argv" in blocker


def test_empty_stdin_is_admissible_for_false_but_arbitrary_for_true(tmp_path):
    # A nonempty stdin would violate; the FALSE mode may not invent it.
    p = program(tmp_path, "import sys\nif sys.stdin.read() == 'boom': raise IndexError\n")
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3, jobs=1,
               concolic=False)
    assert v.verdict == "unknown"
    p = program(tmp_path, "import sys\nif sys.__stdin__.read() == 'boom': raise IndexError\n")
    assert verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3, jobs=1,
                  concolic=False).verdict == "unknown"
    p = program(tmp_path, "import sys\ntry: sys.stdin.fileno()\nexcept OSError: raise IndexError\n")
    assert verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3,
                  jobs=1, concolic=False).verdict != "false"
    p = program(tmp_path, "try: open(0, closefd=False)\nexcept OSError: raise IndexError\n")
    assert verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3,
                  jobs=1, concolic=False).verdict != "false"
    p = program(tmp_path, "try:\n    input()\nexcept EOFError:\n    raise IndexError\n")
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, jobs=1, concolic=False)
    assert v.verdict == "false" and replay(p, v.witness, ("IndexError",), python=PYTHON)[0]
    for source in (
        "import sys\nif sys.stdin.writable(): raise IndexError\n",
        "import sys\ntry: sys.stdin.buffer.raw\nexcept AttributeError: raise IndexError\n",
    ):
        p = program(tmp_path, source)
        assert run_once(PYTHON, p, [], timeout=30).status == "clean"
        assert verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, jobs=1,
                      concolic=False).verdict != "false"


def test_fixed_declared_files_and_generated_bytes(tmp_path):
    data = tmp_path / "fixtures" / "data.txt"
    data.parent.mkdir()
    data.write_text("fixed")
    p = program(tmp_path, "from pathlib import Path\nassert Path('fixtures/data.txt').exists()\n"
                          "assert Path('fixtures/data.txt').is_file()\nassert Path('fixtures').is_dir()\n"
                          "with open('fixtures/data.txt') as f:\n    assert f.read() == 'fixed'\n")
    task = tmp_path / "task.yml"
    task.write_text("input_files: program.py\nrequired_files:\n  - fixtures/data.txt\n")
    env = TaskEnvironment.from_task(str(task))
    assert env.read_effect(str(data)) == "fixed"
    out = run_once(PYTHON, p, [], timeout=30, files=env.files)
    assert out.status == "clean", out
    wrong_path = program(tmp_path, "open('data.txt').read()\n")
    assert run_once(PYTHON, wrong_path, [], timeout=30, files=env.files).exc == "FileNotFoundError"
    for source in ("import os\nif 'fixtures' not in os.listdir('.'): raise IndexError\n",
                   "import os\ntry: os.open('fixtures/data.txt', os.O_RDONLY)\n"
                   "except FileNotFoundError: raise IndexError\n"):
        candidate = program(tmp_path, source)
        v = verify(candidate, ("IndexError",), python=PYTHON, task=str(task),
                   budget_seconds=60, max_runs=2, jobs=1, concolic=False)
        assert v.verdict == "unknown"
    for path, error in ((str(data.parent), "IsADirectoryError"),
                        (str(data / "child"), "NotADirectoryError")):
        candidate = program(tmp_path, f"try: open({path!r}, 'w')\n"
                                      f"except {error}: raise IndexError\n")
        v = verify(candidate, ("IndexError",), python=PYTHON, task=str(task),
                   budget_seconds=60, jobs=1, concolic=False)
        assert v.verdict == "false" and replay(candidate, v.witness, ("IndexError",),
                                                python=PYTHON, task=str(task))[0]
    # The file is made and then read during the same execution.
    p = program(tmp_path, "from pathlib import Path\nwith open('out.txt', 'w') as f: print('made', file=f)\n"
                          "if Path('out.txt').exists() and Path('out.txt').read_text() == 'made\\n': raise IndexError\n")
    # [internal ref]: deterministic (no nondet draw). Repeatedly bumping verify()'s WALL-CLOCK
    # budget (5s -> 20s -> 60s -> 120s) kept failing under heavy campaign CPU contention;
    # cpu_time_only=True (a CPU-time budget instead) still flaked under a real -n-heavy
    # parallel suite run, where the genuine CPU cost of importing/running under that much
    # contention can itself grow, not just wall-clock scheduling delay. This check does
    # not need verify()'s search machinery at all -- the program takes no draws, so a
    # single concrete run (run_once, the same helper the two checks above already use)
    # settles it directly, with only a plain subprocess timeout to reason about.
    out = run_once(PYTHON, p, [], timeout=45)
    assert out.violated(("IndexError",)) is not None, out
    assert replay(p, {"vector": []}, ("IndexError",), python=PYTHON, timeout=45)[0]


def test_undeclared_file_never_supplies_a_false_witness_or_a_true_proof(tmp_path):
    (tmp_path / "missing.txt").write_text("invented")
    p = program(tmp_path, "from pathlib import Path\nassert not Path('missing.txt').exists()\n"
                          "try:\n    x = open('missing.txt').read()\nexcept OSError:\n    x = ''\n"
                          "if x == 'invented': raise IndexError\n")
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3, jobs=1,
               concolic=False)
    assert v.verdict == "unknown"
    p = program(tmp_path, "from pathlib import Path\nif Path('missing.txt').exists(): raise IndexError\n")
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3, jobs=1,
               concolic=False)
    assert v.verdict == "unknown"
    p = program(tmp_path, f"from pathlib import Path\nif Path({str(tmp_path / 'missing.txt')!r}).exists(): raise IndexError\n")
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3, jobs=1,
               concolic=False)
    assert v.verdict == "unknown"
    # The other direction: an undeclared file may have exactly these bytes.
    assert TaskEnvironment.from_task(p).read_effect(str(tmp_path / "missing.txt")) == "unknown"


def test_low_level_writes_cannot_escape_scratch_or_make_false(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"fixed")
    for source in (
        f"import io\ntry: io.FileIO({str(outside)!r}, 'wb')\nexcept OSError: raise IndexError\n",
        f"import os\ntry: os.open({str(outside)!r}, os.O_WRONLY | os.O_TRUNC)\n"
        "except OSError: raise IndexError\n",
    ):
        p = program(tmp_path, source)
        v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=2,
                   jobs=1, concolic=False)
        assert v.verdict != "false"
        assert outside.read_bytes() == b"fixed"


def test_output_paths_keep_native_parent_errors_and_names(tmp_path):
    p = program(tmp_path, "try: open('absent-parent/out.txt', 'w').close()\n"
                          "except FileNotFoundError: raise IndexError('parent missing')\n")
    # the first run raises it, so the budget is only a ceiling: 3 s was starved into UNKNOWN on a loaded host
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, jobs=1, concolic=False)
    assert v.verdict == "false" and replay(p, v.witness, ("IndexError",), python=PYTHON)[0]
    p = program(tmp_path, "with open('out.txt', 'w') as f:\n"
                          "    if f.name[-1] in '01234567': raise IndexError\n")
    assert run_once(PYTHON, p, [], timeout=30).status == "clean"
    assert verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, jobs=1,
                  concolic=False).verdict != "false"


def test_declared_file_directly_in_task_base_does_not_block_false_witnesses(tmp_path):
    # A declared file sitting directly in the task's own directory (the common
    # case -- no fixtures/ subdirectory) must not be treated as touching an
    # unmodelled path merely because opening it happens to stat its own parent
    # directory. If it were, every program that reads its declared input and
    # later raises would wrongly stay stuck at UNKNOWN forever.
    data = tmp_path / "data.txt"
    data.write_text("fixed")
    p = program(tmp_path, "with open('data.txt') as f:\n"
                          "    if f.read() == 'fixed': raise IndexError\n")
    task = tmp_path / "task.yml"
    task.write_text("input_files: program.py\nrequired_files:\n  - data.txt\n")
    v = verify(p, ("IndexError",), python=PYTHON, task=str(task), budget_seconds=60, jobs=1, concolic=False)
    assert v.verdict == "false", v
    assert replay(p, v.witness, ("IndexError",), python=PYTHON, task=str(task))[0]


def test_stat_on_unmodelled_absolute_path_cannot_supply_a_false_witness(tmp_path):
    # os.stat (and anything built on it: Path.stat/exists/is_file/is_dir) on an
    # undeclared, non-scratch, non-infrastructure path is host-dependent in
    # BOTH directions: whether it raises is exactly as unmodelled as what it
    # returns on success. A raise must not slip through as if it were a
    # modelled, replayable outcome.
    p = program(tmp_path, "import os\ntry:\n    os.stat('/definitely-does-not-exist-xyz-123')\n"
                          "except FileNotFoundError:\n    raise IndexError\n")
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3, jobs=1, concolic=False)
    assert v.verdict == "unknown", v


# --- [internal ref]: no stand-in for sys.stdin, ever (owner ruling) ------------------
#
# REV (soundness review of PR #38 @977bb37) found the previous dynamic
# _Stdin wrapper gave 4 wrong FALSEs: the wrapper had no .mode, was never
# `sys.__stdin__`, and had the wrong type, so any target checking any of
# those observed something that could never happen under real execution.
# Repro: [host]:[local path] (tasks a04/b01/b02/b06). Fixed
# by never replacing sys.stdin at all; "stdin touched" is now a static,
# conservative source scan (environment.references_stdin) instead.

def test_real_stdin_identity_type_and_mode_never_produce_a_wrong_false(tmp_path):
    # a04_stdin_mode: a stand-in TextIOWrapper built without a mode kwarg has
    # no .mode at all -- AttributeError natively never raised.
    p = program(tmp_path, "import sys\ntry:\n    sys.stdin.mode\nexcept AttributeError:\n    raise IndexError\n")
    assert verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3, jobs=1,
                  concolic=False).verdict != "false"
    # b01_stdin_identity: a fresh wrapper object is never `sys.__stdin__`.
    p = program(tmp_path, "import sys\nif sys.stdin is not sys.__stdin__: raise IndexError\n")
    assert verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3, jobs=1,
                  concolic=False).verdict != "false"
    # b02_stdin_type: a subclass is never exactly io.TextIOWrapper.
    p = program(tmp_path, "import io, sys\nif type(sys.stdin) is not io.TextIOWrapper: raise IndexError\n")
    assert verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3, jobs=1,
                  concolic=False).verdict != "false"
    # b06_stdin_identity_v: identity must still hold even after a read.
    p = program(tmp_path, "import sys\nsys.stdin.read()\nif sys.stdin is not sys.__stdin__: raise IndexError\n")
    assert verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=3, jobs=1,
                  concolic=False).verdict != "false"
    # The static pre-scan still flags stdin as touched (TRUE-disqualifying),
    # matching the old dynamic wrapper's effect on this signal.
    from tharkun.solve.environment import references_stdin
    assert references_stdin(b"import sys\nsys.stdin.read()\n")
    assert references_stdin(b"input()\n")
    assert not references_stdin(b"x = 1\n")


def test_declared_relative_file_witness_replays_end_to_end(tmp_path):
    # a06_declared_rel: the ACTUAL fix is on the eregion-sv (validator) side
    # (declared/required files are staged unconditionally from the task's own
    # .yml, never from a witness draw) -- this pins Tharkun's own half: a
    # genuine FALSE witness for a program that reads a declared, relative
    # file must actually replay through tharkun.sv's own `replay()`, which
    # uses the same `install_files`/`TaskEnvironment` this branch didn't
    # change.
    data = tmp_path / "a06_data.txt"
    data.write_text("hello\n")
    p = program(tmp_path, "with open('a06_data.txt') as f:\n"
                          "    s = f.read()\nassert s != 'hello\\n'\n")
    task = tmp_path / "task.yml"
    task.write_text("input_files: program.py\nrequired_files:\n  - a06_data.txt\n")
    v = verify(p, ("AssertionError",), python=PYTHON, task=str(task), budget_seconds=60, jobs=1,
              concolic=False)
    assert v.verdict == "false", v
    assert replay(p, v.witness, ("AssertionError",), python=PYTHON, task=str(task))[0]


def test_argv_and_draw_driven_false_still_confirm(tmp_path):
    """Negative controls for the stdin fix ([internal ref]): the two OTHER FALSE
    channels -- an argv-driven violation, and a plain _sv_verifier
    draw-driven one -- must still confirm exactly as before; the stdin fix
    must not have disturbed either."""
    p = program(tmp_path, "import sys\nif sys.argv[1:] == ['boom']: raise IndexError\n")
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, jobs=1, concolic=False)
    assert v.verdict == "false", v
    assert replay(p, v.witness, ("IndexError",), python=PYTHON)[0]
    p = program(tmp_path, "import _sv_verifier\nif _sv_verifier.nondet_int() == 7: raise IndexError\n")
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, jobs=1, concolic=False)
    assert v.verdict == "false", v
    assert replay(p, v.witness, ("IndexError",), python=PYTHON)[0]


def test_sys_path_insert_import_of_an_external_package_is_not_an_undeclared_file_read(tmp_path):
    """[internal ref] regression ([internal ref], W65's GUUT-harness evidence): a target
    program that does ``sys.path.insert(0, <external dir>)`` then imports a
    plain module AND a package (``__init__.py``) from it must run exactly like
    any other clean import -- CPython's import machinery reading that source
    off disk is not a program file read under the I/O model, whether or not
    the external directory happens to sit under the task's own base directory.
    Before the fix, the very first such read raised FileNotFoundError
    ("undeclared file"), matching W65's report of codetiming/dataclasses-json/
    python-string-utils breaking under the GUUT harness's gen_harness.py.

    Deterministic (no nondet draw): checked with a single concrete run
    (run_once/replay), not verify()'s wall-clock search -- an earlier version
    of this test used a `_sv_verifier.nondet_int() == 7` condition and
    verify(budget_seconds=5), which needs a search to find 7 and flaked under
    a loaded host (MG review of PR 115, 2026-09-29: 13/20 unknown instead of
    false under -n 64 heavy contention, matching the load-sensitive family
    already documented elsewhere in this file)."""
    extlib = tmp_path / "extlib"
    (extlib / "pkg").mkdir(parents=True)
    (extlib / "plain_mod.py").write_text("VALUE = 42\n")
    (extlib / "pkg" / "__init__.py").write_text("X = 7\n")
    (extlib / "pkg" / "leaf.py").write_text("def f():\n    return 2\n")
    p = program(tmp_path, f"import sys\nsys.path.insert(0, {str(extlib)!r})\n"
                          "import plain_mod\nimport pkg\nfrom pkg.leaf import f\n"
                          "if plain_mod.VALUE == 42 and pkg.X * f() == 14:\n"
                          "    raise IndexError\n")
    out = run_once(PYTHON, p, [], timeout=30, files=(p,))
    assert out.status == "raised" and out.exc == "IndexError" and not out.flags, out
    assert out.violated(("IndexError",)) is not None, out
    assert replay(p, {"vector": []}, ("IndexError",), python=PYTHON)[0]


def test_sys_path_insert_import_raising_inside_the_packages_own_init_still_confirms_false(tmp_path):
    """[internal ref] regression, second half: an exception raised INSIDE the external
    package's own __init__.py at import time (not by the program's own code)
    must still confirm FALSE. Before the fix this stayed flagged
    "file_unmodelled" even though the raise itself classified correctly: the
    HARNESS's own traceback.extract_tb(e.__traceback__), used to classify
    where the run died, calls linecache.updatecache to annotate the __init__
    frame, which os.stat()s that undeclared external path along the way."""
    extlib = tmp_path / "extlib"
    (extlib / "pkg").mkdir(parents=True)
    (extlib / "pkg" / "__init__.py").write_text("_denom = 0\n_boom = 1 / _denom\n")
    p = program(tmp_path, f"import sys\nsys.path.insert(0, {str(extlib)!r})\nimport pkg\n")
    out = run_once(PYTHON, p, [], timeout=30, files=(p,))
    assert out.status == "raised" and out.exc == "ZeroDivisionError" and not out.flags, out
    assert out.violated(("ZeroDivisionError",)) is not None, out
    assert replay(p, {"vector": []}, ("ZeroDivisionError",), python=PYTHON)[0]


def test_sys_path_insert_does_not_exempt_the_programs_own_open_of_a_sibling_data_file(tmp_path):
    """Negative control: the import-machinery exemption must be scoped to
    CPython's own mechanical source read, never to "anything under a
    directory the target put on sys.path". A plain open() of a sibling,
    non-imported file in that same external directory -- the program's own
    file access, not the import system's -- must still be refused."""
    extlib = tmp_path / "extlib"
    (extlib / "pkg").mkdir(parents=True)
    (extlib / "pkg" / "__init__.py").write_text("X = 1\n")
    (extlib / "secret.txt").write_text("invented")
    p = program(tmp_path, f"import sys\nsys.path.insert(0, {str(extlib)!r})\nimport pkg\n"
                          f"with open({str(extlib / 'secret.txt')!r}) as fh:\n"
                          "    data = fh.read()\n"
                          "if data == 'invented': raise IndexError\n")
    out = run_once(PYTHON, p, [], timeout=30, files=(p,))
    assert out.status == "raised" and out.exc == "FileNotFoundError", out
    assert "file_unmodelled" in out.flags, out
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=2, jobs=1, concolic=False)
    assert v.verdict == "unknown", v


def test_a_loader_driven_directly_cannot_read_an_undeclared_file(tmp_path):
    """MG's review of PR 115 (2026-09-29): the first version of the import-machinery
    exemption checked only the IMMEDIATE caller frame, so it also exempted a program
    driving a loader directly on an arbitrary, undeclared path with no real import
    involved -- importlib.machinery.SourceFileLoader(name, path).get_data(path) reaches
    _io.open/_io.FileIO exactly the way a real import's get_data() does. Fixed by
    requiring _find_and_load specifically (the one frame every REAL import -- a bare
    `import`, __import__(), or importlib.import_module() -- passes through) somewhere
    in the walk, which a direct get_data() call never has above it."""
    outside = tmp_path.parent / (tmp_path.name + "_secret.txt")
    outside.write_text("TOP-SECRET-CONTENT")
    p = program(tmp_path, "import importlib.machinery\n"
                          f"loader = importlib.machinery.SourceFileLoader('x', {str(outside)!r})\n"
                          "data = loader.get_data(loader.path)\n"
                          "if b'TOP-SECRET' in data: raise IndexError('leaked: ' + data.decode())\n")
    out = run_once(PYTHON, p, [], timeout=30, files=(p,))
    assert out.status == "raised" and out.exc == "FileNotFoundError", out
    assert out.violated(("IndexError",)) is None, out
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=2, jobs=1, concolic=False)
    assert v.verdict != "false", v
    outside.unlink()


def test_linecache_driven_directly_cannot_supply_a_false_witness(tmp_path):
    """MG's review of PR 115 (2026-09-29): a program calling linecache directly
    (linecache.getline(path, n)) on an undeclared path must never contribute a FALSE.
    linecache.updatecache's own os.stat() of that path is NOT exempted (unlike the
    import-machinery case, it gets no exemption at all -- see _child.py's own
    lookup_lines=False fix, which stops the HARNESS's traceback formatting from ever
    triggering linecache itself, so no legitimate caller needs one here), so this
    still sets file_unmodelled even on hosts where linecache's own content read
    happens to bypass the sandbox via a stdlib module that cached a real, unpatched
    builtins.open reference before this run started."""
    outside = tmp_path.parent / (tmp_path.name + "_secret.txt")
    outside.write_text("TOP-SECRET-CONTENT\n")
    p = program(tmp_path, "import linecache\n"
                          f"line = linecache.getline({str(outside)!r}, 1)\n"
                          "if 'TOP-SECRET' in line: raise IndexError('leaked: ' + line)\n")
    out = run_once(PYTHON, p, [], timeout=30, files=(p,))
    assert out.violated(("IndexError",)) is None, out
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=2, jobs=1, concolic=False)
    assert v.verdict != "false", v
    outside.unlink()


def test_linecache_updatecache_driven_directly_cannot_supply_a_false_witness(tmp_path):
    """MG's review of PR 115 (2026-09-29): the same channel as
    test_linecache_driven_directly_cannot_supply_a_false_witness, called one
    layer lower -- linecache.updatecache(path) directly, rather than through
    getline(). Same fix, same guarantee: no exemption for linecache's own
    os.stat() of an undeclared path, so this can never contribute a FALSE."""
    outside = tmp_path.parent / (tmp_path.name + "_secret2.txt")
    outside.write_text("TOP-SECRET-CONTENT\n")
    p = program(tmp_path, "import linecache\n"
                          f"lines = linecache.updatecache({str(outside)!r})\n"
                          "if lines and 'TOP-SECRET' in lines[0]: raise IndexError('leaked: ' + lines[0])\n")
    out = run_once(PYTHON, p, [], timeout=30, files=(p,))
    assert out.violated(("IndexError",)) is None, out
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=2, jobs=1, concolic=False)
    assert v.verdict != "false", v
    outside.unlink()


def test_pkgutil_get_data_on_a_sys_path_insert_package_cannot_read_an_undeclared_resource(tmp_path):
    """MG's review of PR 115 (2026-09-29): pkgutil.get_data(pkg, resource) reads a
    resource file NEXT TO a package's __init__.py -- the package itself imports
    cleanly (it's on sys.path, matching the intended fix), but the resource file is
    DATA, never something an import reads, so it must stay refused as undeclared
    exactly like any other file the program opens directly. Reusing the SAME
    import-machinery exemption for this would have handed a raw, unsandboxed file
    read to any package with a resource file next to it."""
    extlib = tmp_path / "extlib"
    (extlib / "pkz").mkdir(parents=True)
    (extlib / "pkz" / "__init__.py").write_text("")
    (extlib / "pkz" / "res.txt").write_text("TOP-SECRET-CONTENT")
    p = program(tmp_path, f"import sys\nsys.path.insert(0, {str(extlib)!r})\n"
                          "import pkgutil\n"
                          "data = pkgutil.get_data('pkz', 'res.txt')\n"
                          "if data == b'TOP-SECRET-CONTENT': raise IndexError('leaked: ' + repr(data))\n")
    out = run_once(PYTHON, p, [], timeout=30, files=(p,))
    assert out.violated(("IndexError",)) is None, out
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=2, jobs=1, concolic=False)
    assert v.verdict != "false", v


def test_spec_from_file_location_exec_module_driven_directly_cannot_read_an_undeclared_file(tmp_path):
    """MG's review of PR 115 (2026-09-29): the idiomatic "manual import" pattern
    (importlib.util.spec_from_file_location + module_from_spec +
    spec.loader.exec_module) called directly by the target on an arbitrary,
    undeclared path never passes through _find_and_load -- its own frame sits
    immediately above exec_module, not a real import's dispatcher -- so it must
    stay refused exactly like SourceFileLoader.get_data() called directly
    (test_a_loader_driven_directly_cannot_read_an_undeclared_file)."""
    outside = tmp_path.parent / (tmp_path.name + "_secret3.txt")
    outside.write_text("TOP-SECRET-CONTENT")
    p = program(tmp_path, "import importlib.util, importlib.machinery\n"
                          f"loader = importlib.machinery.SourceFileLoader('modx', {str(outside)!r})\n"
                          f"spec = importlib.util.spec_from_file_location('modx', {str(outside)!r}, loader=loader)\n"
                          "mod = importlib.util.module_from_spec(spec)\n"
                          "try:\n"
                          "    spec.loader.exec_module(mod)\n"
                          "    leaked = True\n"
                          "except SyntaxError:\n"
                          "    leaked = True  # the read succeeded; only compiling it as code failed\n"
                          "if leaked: raise IndexError('leaked')\n")
    out = run_once(PYTHON, p, [], timeout=30, files=(p,))
    assert out.status == "raised" and out.exc == "FileNotFoundError", out
    assert out.violated(("IndexError",)) is None, out
    v = verify(p, ("IndexError",), python=PYTHON, budget_seconds=60, max_runs=2, jobs=1, concolic=False)
    assert v.verdict != "false", v
    outside.unlink()


def test_concolic_argv_log_snapshots_before_target_mutation(tmp_path, monkeypatch):
    from tharkun.sv import _concolic_worker
    from tharkun.sv.sites import build_harness

    h = build_harness("import sys\nif __name__ == '__main__':\n"
                      "    if sys.argv == []:\n        sys.argv.append('changed')\n"
                      "        raise IndexError\n", ("IndexError",))
    harness = tmp_path / "harness.py"
    harness.write_text(h.source)
    monkeypatch.setitem(sys.modules, "_sv_verifier", _concolic_worker._stub())
    assert _concolic_worker._vector(str(harness), h.seed()) == [{"kind": "argv_full", "value": []}]


@pytest.mark.parametrize("value", [[], [""], ["not-the-program"], ["雪", "\0", "\ud800"]])
def test_full_argv_is_a_plain_list_and_replays_original_values(tmp_path, value):
    from tharkun.sv.nondet_harness import drive
    from tharkun.sv.prove import run_probe

    p = program(tmp_path, f"import sys\nif type(sys.argv) is list and sys.argv == {value!r}:\n"
                           "    sys.argv.append('mutated')\n    raise IndexError\n")
    vector = [{"kind": "argv_full", "value": value}]
    assert replay(p, {"vector": vector}, ("IndexError",), python=PYTHON)[0]
    for out in (run_once(PYTHON, p, vector, timeout=30), drive(p, vector)):
        assert out.violated(("IndexError",)) == "IndexError"
        assert out.draws[0]["kind"] == "argv_full" and out.draws[0]["value"] == value
    probe = run_probe(PYTHON, p, vector, [], [], timeout=30)
    assert probe["exc"] == "IndexError" and probe["draws"][0]["value"] == value


@pytest.mark.parametrize("vector", [
    [{"kind": "argv_full", "value": "not-a-list"}],
    [{"kind": "argv_full", "value": ["ok", 1]}],
    [{"kind": "argv_full", "value": None}],
    [{"kind": "argv_full"}], [None],
    [{"kind": "argv_full", "value": []}, {"kind": "argv", "value": []}],
    [{"kind": "argv_full", "value": []}, {"kind": "argv_full", "value": []}],
    [{"kind": "int", "value": 7}, {"kind": "argv_full", "value": []}],
])
def test_malformed_full_argv_is_rejected_before_any_target_statement(tmp_path, vector):
    from tharkun.sv.nondet_harness import drive
    from tharkun.sv.prove import run_probe

    p = program(tmp_path, "print('TARGET EXECUTED')\ntry: raise ValueError\nexcept BaseException: raise IndexError\n")
    child = run_once(PYTHON, p, vector, timeout=30)
    assert child.status == "invalid-draw" and child.exc is None
    driven = drive(p, vector, target_line=1)
    assert driven.status == "invalid-draw" and not driven.reached
    probe = run_probe(PYTHON, p, vector, [], [], timeout=30)
    assert probe["status"] == "invalid-draw"
    assert not replay(p, {"vector": vector}, ("IndexError",), python=PYTHON)[0]


def test_optional_boot_argv_does_not_consume_a_primitive_draw(tmp_path):
    p = program(tmp_path, "from _sv_verifier import nondet_int\nimport sys\n"
                           "if nondet_int() == 7 and sys.argv == [__file__]: raise IndexError\n")
    out = run_once(PYTHON, p, [{"kind": "int", "value": 7}], timeout=30)
    vector = [{k: d[k] for k in ("kind", "value")} for d in out.draws]
    assert vector == [{"kind": "argv_full", "value": [p]}, {"kind": "int", "value": 7}]
    assert replay(p, {"vector": vector}, ("IndexError",), python=PYTHON)[0]


@pytest.mark.parametrize("access", ["import sys as s\na = s.argv", "from sys import argv as a"])
def test_lifted_boot_is_first_even_before_first_argv_access_and_mutation(tmp_path, monkeypatch, access):
    from tharkun.sv import _concolic_worker
    from tharkun.sv.sites import build_harness

    source = ("from _sv_verifier import nondet_int\nif __name__ == '__main__':\n"
              "    n = nondet_int()\n" + "\n".join("    " + s for s in access.splitlines()) +
              "\n    if n == 7 and a == []:\n        a.append('mutated')\n        raise IndexError\n")
    h = build_harness(source, ("IndexError",))
    assert h.kinds == ["int", "argv_full"]
    path = tmp_path / "harness.py"; path.write_text(h.source)
    monkeypatch.setitem(sys.modules, "_sv_verifier", _concolic_worker._stub())
    vector = _concolic_worker._vector(str(path), dict(zip(h.params, (7, []))))
    assert vector == [{"kind": "argv_full", "value": []}, {"kind": "int", "value": 7}]
    p = program(tmp_path, source)
    assert replay(p, {"vector": vector}, ("IndexError",), python=PYTHON)[0]


def test_full_argv_literals_are_not_restricted_to_nondet_str_printability(tmp_path):
    from tharkun.sv.engine import Candidates, admissible, harvest_constants
    from tharkun.solve.environment import proof_blocker

    p = program(tmp_path, "import sys\nif sys.argv == ['雪'] or sys.argv == ['\\0']: raise IndexError\n")
    harvested = harvest_constants(p)
    assert ["雪"] in harvested["argv_full"] and ["\0"] in harvested["argv_full"]
    assert "雪" not in harvested["str"] and "\0" not in harvested["str"]
    assert admissible("argv_full", ["雪", "\0"]) and not admissible("argv_full", [1])
    candidates = Candidates(harvested, deepen=False)
    assert candidates.value("argv_full", 1) == [""]
    assert proof_blocker("from sys import argv\n") is not None


def test_full_argv_preserves_all_previously_harvested_cli_literal_examples(tmp_path):
    from tharkun.sv.engine import harvest_constants

    literals = [f"literal-{i}" for i in range(64)]
    p = program(tmp_path, "import sys\nvalues = " + repr(literals) + "\n")
    harvested = harvest_constants(p)
    assert all([p, *tail] in harvested["argv_full"] for tail in harvested["argv"])


@pytest.mark.parametrize("literals", [[], [f"雪{i}" for i in range(64)]])
def test_full_argv_preserves_old_cli_vectors_with_mixed_domain_literals(tmp_path, literals):
    from tharkun.sv.engine import harvest_constants

    p = program(tmp_path, "import sys\nvalues = " + repr(literals) + "\nif len(sys.argv) == 2 and sys.argv[1] == 'boom': raise IndexError\n")
    harvested = harvest_constants(p)
    assert [p, "boom"] in harvested["argv_full"]
    assert [p, "x", "x"] in harvested["argv_full"]
    assert all([p, *tail] in harvested["argv_full"] for tail in harvested["argv"])
    assert len(harvested["argv_full"]) <= 160

@pytest.mark.parametrize('options,version',
[('', '3.14'),
 ('options:\n  language_version: "3.14"\n', '3.14'),
 ("options:\n  language_version: '3.14' # comment\n", '3.14'),
 ("options:\n  language_version: '3.15'\n", '3.15'),
 ("options:\n  language_version: '3.13'\n", '3.13'),
 ("options:\n  language_version: '3.14.4'\n", '3.14.4'),
 ("options:\n  language_version: ''\n", ''),
 ("options:\n  language_version: ' 3.14 '\n", ' 3.14 '),
 ("options:\n  language_version: 'arbitrary'\n", 'arbitrary'),
 ("options:\n  language_version: 'null'\n", 'null'),
 ('options:\n  language_version: 3.14.4\n', '3.14.4'),
 ('options:\n  language_version: arbitrary\n', 'arbitrary')])
def test_task_version_preserves_plain_strings(tmp_path, options, version):
    program = tmp_path / "program.py"
    program.write_text("assert True\n")
    task = tmp_path / "task.yml"
    task.write_text("format_version: '2.0'\ninput_files: program.py\n" + options)
    env = TaskEnvironment.from_task(str(task))
    assert env.language_version == version and type(env.language_version) is str

@pytest.mark.parametrize('options',
['options:\n  language_version: 3.14\n',
 'options:\n  language_version: null\n',
 'options:\n  language_version: true\n',
 'options:\n  language_version: []\n',
 'options:\n  language_version: {}\n',
 'options:\n  language_version: &v "3.14"\n',
 'options:\n  language_version: "3\\.14"\n',
 'options:\n  language_version: "3.14"\n  language_version: "3.15"\n',
 'options:\n  language_version: "3.14"\noptions:\n  language_version: "3.15"\n',
 'options: {language_version: "3.14"}\n',
 'options: null\n',
 'options:\n  nested:\n    language_version: "3.15"\n',
 'language_version: "3.15"\n',
 'options:\n  <<: {language_version: "3.15"}\n',
 'defaults: &o\n  language_version: "3.15"\noptions: *o\n',
 'options:\n  language_version: |\n    3.14\n'])
def test_task_version_refuses_ambiguous_or_nonstring_declarations(tmp_path, options):
    task = tmp_path / "task.yml"
    task.write_text("format_version: '2.0'\ninput_files: program.py\n" + options)
    with pytest.raises(ValueError):
        TaskEnvironment.from_task(str(task))


def test_files_only_environment_constructor_keeps_default_version():
    assert TaskEnvironment(("p.py",)).language_version == "3.14"


@pytest.mark.parametrize("declaration", ['required_files: a_helper.py\n', 'input_files: []\nrequired_files: a_helper.py\n', 'input_files: null\nrequired_files: a_helper.py\n', 'input_files:\n  - z_main.py\n  - missing.py\nrequired_files: a_helper.py\n', "input_files: ['z_main.py', 'missing.py']\nrequired_files: a_helper.py\n", "input_files: ['z_*.py', 'missing*.py']\nrequired_files: a_helper.py\n", 'input_files: z_main.py\ninput_files: a_helper.py\n', 'input_files: &main z_main.py\nrequired_files: a_helper.py\n', 'source: &main z_main.py\ninput_files: *main\nrequired_files: a_helper.py\n', 'input_files: {file: z_main.py}\nrequired_files: a_helper.py\n', 'input_files: z_main.py\nrequired_files: [a_helper.py, missing.txt]\n', 'input_files: z_main.py\nrequired_files: a_helper.py\nrequired_files: missing.txt\n', 'input_files: z_main.py\nrequired_files: &helper a_helper.py\n'])
def test_file_projection_refuses_partial_or_ambiguous_inputs(tmp_path, declaration):
    from tharkun.solve.environment import TaskEnvironment, TaskMetadataError
    for name in ("z_main.py", "a_aux.py", "a_helper.py"):
        (tmp_path / name).write_text("pass\n")
    task = tmp_path / "task.yml"
    task.write_text("format_version: '2.0'\n" + declaration)
    with pytest.raises(TaskMetadataError):
        TaskEnvironment.from_task(str(task))


@pytest.mark.parametrize("declaration", ['input_files: z_main.py\nrequired_files: a_helper.py\n', '\'input_files\': z_main.py\n"required_files": a_helper.py\n', 'input_files:\n  - z_main.py\n  - a_aux.py\nrequired_files:\n  - a_helper.py\n', "input_files: ['z_main.py', 'a_aux.py']\nrequired_files: [a_helper.py]\n", "input_files: 'z_*.py'\nrequired_files: a_helper.py\n"])
def test_primary_file_order_never_sorts_required_helpers_first(tmp_path, declaration):
    from tharkun.solve.environment import TaskEnvironment, TaskMetadataError
    for name in ("z_main.py", "a_aux.py", "a_helper.py"):
        (tmp_path / name).write_text("pass\n")
    task = tmp_path / "task.yml"
    task.write_text("format_version: '2.0'\n" + declaration)
    environment = TaskEnvironment.from_task(str(task))
    files = environment.files
    assert files[0] == str(tmp_path / "z_main.py")
    assert files[-1] == str(tmp_path / "a_helper.py")
    if "a_aux.py" in declaration:
        assert files[1] == str(tmp_path / "a_aux.py")


@pytest.mark.parametrize("declaration", ['input_files: z_main.py\n- input_files: a_helper.py\nrequired_files: a_helper.py\n', '- input_files: a_helper.py\ninput_files: z_main.py\n', 'input_files: z_main.py\nrequired_files: a_helper.py\n- required_files: missing.py\n'])
def test_dangling_root_sequence_cannot_create_a_task_admission(tmp_path, declaration):
    from tharkun.solve.environment import TaskEnvironment, TaskMetadataError, _task_language_version
    for name in ("z_main.py", "a_helper.py"):
        (tmp_path / name).write_text("pass\n")
    task = tmp_path / "task.yml"
    task.write_text("format_version: '2.0'\n" + declaration)
    with pytest.raises(TaskMetadataError):
        TaskEnvironment.from_task(str(task))
    with pytest.raises(ValueError, match="root sequence|block task mapping"):
        _task_language_version(task.read_text().splitlines(keepends=True))


def test_legitimate_indentless_file_and_properties_sequences_stay_bound(tmp_path):
    from tharkun.solve.environment import TaskEnvironment
    for name in ("z_main.py", "a_helper.py"):
        (tmp_path / name).write_text("pass\n")
    task = tmp_path / "task.yml"
    task.write_text("format_version: '2.0'\ninput_files:\n- z_main.py\nrequired_files:\n- a_helper.py\n"
                    "properties:\n- property_file: property.prp\n  expected_verdict: false\n")
    environment = TaskEnvironment.from_task(str(task))
    files = environment.files
    assert list(files) == [str(tmp_path / "z_main.py"), str(tmp_path / "a_helper.py")]


@pytest.mark.parametrize("declaration", [
    "input_files: z_main.py\n  required_files: a_helper.py\n",
    "input_files: z_main.py\nrequired_files: a_helper.py\n  input_files: other.py\n",
    "input_files: [z_main.py]\n  required_files: a_helper.py\n",
])
def test_file_projection_refuses_indented_children_after_scalar(tmp_path, declaration):
    from tharkun.solve.environment import TaskMetadataError, _task_language_version
    for name in ("z_main.py", "a_helper.py", "other.py"):
        (tmp_path / name).write_text("pass\n")
    task = tmp_path / "task.yml"
    task.write_text("format_version: '2.0'\n" + declaration)
    with pytest.raises(TaskMetadataError) as refused:
        TaskEnvironment.from_task(str(task))
    assert refused.value.files == ()
    with pytest.raises(ValueError):
        _task_language_version(task.read_text().splitlines(keepends=True))


@pytest.mark.parametrize("selector", [
    "0xFF", ".nan", "0b101", "1_000", "1:20", "2026-10-11", "+.Inf", "-1_000",
])
@pytest.mark.parametrize("field", ["input_files", "required_files"])
def test_file_projection_refuses_plain_yaml_nonstring_names(tmp_path, selector, field):
    from tharkun.solve.environment import TaskMetadataError
    (tmp_path / "z_main.py").write_text("pass\n")
    (tmp_path / selector).write_text("pass\n")
    task = tmp_path / "task.yml"
    prefix = "input_files: z_main.py\n" if field == "required_files" else ""
    task.write_text("format_version: '2.0'\n" + prefix + f"{field}: {selector}\n")
    with pytest.raises(TaskMetadataError) as refused:
        TaskEnvironment.from_task(str(task))
    assert refused.value.files == ()


@pytest.mark.parametrize("selector", [
    "0xFF", ".nan", "0b101", "1_000", "1:20", "2026-10-11", "+.Inf", "-1_000",
])
@pytest.mark.parametrize("field", ["input_files", "required_files"])
def test_file_projection_keeps_quoted_yaml_nonstring_names(tmp_path, selector, field):
    (tmp_path / "z_main.py").write_text("pass\n")
    (tmp_path / selector).write_text("pass\n")
    task = tmp_path / "task.yml"
    prefix = "input_files: z_main.py\n" if field == "required_files" else ""
    task.write_text("format_version: '2.0'\n" + prefix + f"{field}: '{selector}'\n")
    expected = [str(tmp_path / "z_main.py")] if field == "required_files" else []
    assert list(TaskEnvironment.from_task(str(task)).files) == [*expected, str(tmp_path / selector)]


@pytest.mark.parametrize("selector", ["./0xFF", "1_000.py", "1:20.py", "2026-10-11.py", "3.14.4"])
def test_file_projection_keeps_ordinary_string_paths(tmp_path, selector):
    path = tmp_path / selector
    path.write_text("pass\n")
    task = tmp_path / "task.yml"
    task.write_text("format_version: '2.0'\n" + f"input_files: {selector}\n")
    assert TaskEnvironment.from_task(str(task)).files == (str(path),)
