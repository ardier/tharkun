"""[internal ref] / [internal ref]: the program runs in the child exactly as under ``python program.py``.

The child ran it through ``runpy.run_path`` inside ``main()``: five extra frames counted against the recursion limit (a
``RecursionError`` for ``f(994..998)`` where ``python`` reaches the ``assert False``: a WRONG TRUE), ``__package__ == ''``,
``__loader__`` None, ``__builtins__`` a dict, ``sys.path`` holding ``tharkun/sv`` (so ``import spec`` found OUR module) -- WRONG
FALSEs that the grader, sharing the model, confirmed.  Every probe is compared with the real ``python probe.py``.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tharkun.sv import _child, nondet_harness, verify
from tharkun.sv.engine import run_once
from tharkun.sv.prove import run_probe

RECURSION = textwrap.dedent("""
    def f(n):
        if n == 0:
            return 0
        return 1 + f(n - 1)
    f({n})
    assert False
""")

#: (name, source): a property violation (AssertionError) natively only when the model is the native one
ATTRIBUTES = {
    "m01 __package__ is None": "assert __package__ is None\n",
    "m02 __spec__ is None": "assert __spec__ is None\n",
    "m03 __loader__ is set": "assert __loader__ is not None\n",
    "m04 the loader is a SourceFileLoader": "import importlib.machinery as m\nassert type(__loader__) is m.SourceFileLoader\n",
    "m05 __builtins__ is the module": "assert type(__builtins__).__name__ == 'module'\n",
    "m06 __cached__ is None": "assert __cached__ is None\n",
    "m07 __name__ and __file__": "import os\nassert __name__ == '__main__' and os.path.isabs(__file__)\n",
    "m08 sys.modules['__main__'] is this module": "import sys\nassert sys.modules['__main__'].__dict__ is globals()\n",
    "m12 no tharkun/sv on sys.path": "import sys\nassert not any(p.endswith('/tharkun/sv') for p in sys.path)\n",
    "m13 import spec is a ModuleNotFoundError": "try:\n    import spec\nexcept ImportError:\n    pass\nelse:\n    assert False\n",
    "m14 import environment is a ModuleNotFoundError": "try:\n    import environment\nexcept ImportError:\n    pass\nelse:\n    assert False\n",
    "m15 sys.path[0] is the program's directory": "import os, sys\nassert sys.path[0] == os.path.dirname(os.path.abspath(__file__))\n",
    "m16 sys.argv is [the program]": "import sys\nassert sys.argv == [__file__]\n",
    "m17 the recursion limit reads 1000": "import sys\nassert sys.getrecursionlimit() == 1000\n",
    "m18 a program's own recursion limit is its own": "import sys\nsys.setrecursionlimit(300)\ndef f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\nf(295)\nassert sys.getrecursionlimit() == 300\nassert False\n",
}


def native(tmp_path, source: str) -> str:
    """How ``python prog.py`` ends: ``clean`` or the uncaught exception's class name."""
    path = tmp_path / "prog.py"
    path.write_text(source)
    run = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=60, cwd=tmp_path)
    if run.returncode == 0:
        return "clean"
    last = [ln for ln in run.stderr.strip().splitlines() if ln and not ln.startswith(" ")][-1]
    return last.split(":")[0].split(".")[-1]


def child(tmp_path, source: str):
    path = tmp_path / "prog.py"
    path.write_text(source)
    return run_once(sys.executable, str(path), [], timeout=60)


def ends_like(out) -> str:
    return "clean" if out.status in ("clean", "exit") else out.exc


@pytest.mark.parametrize("n", [994, 995, 996, 997, 998, 999, 1000])
def test_the_recursion_window_is_the_native_one(tmp_path, n):
    source = RECURSION.format(n=n)
    want = native(tmp_path, source)
    assert want == ("AssertionError" if n <= 998 else "RecursionError")
    out = child(tmp_path, source)
    assert ends_like(out) == want, (n, out.status, out.exc)


@pytest.mark.parametrize("n", [994, 996, 998])
def test_a_program_that_reaches_its_assert_natively_is_never_true(tmp_path, n):
    path = tmp_path / "prog.py"
    path.write_text(RECURSION.format(n=n))
    verdict = verify(str(path), ("AssertionError",), budget_seconds=15, jobs=1)
    assert verdict.verdict == "false", (n, verdict.verdict, verdict.reason)


@pytest.mark.parametrize("label", sorted(ATTRIBUTES))
def test_the_childs_main_matches_the_native_one(tmp_path, label):
    source = ATTRIBUTES[label]
    want = native(tmp_path, source)
    out = child(tmp_path, source)
    assert ends_like(out) == want, (label, "native:", want, "child:", out.status, out.exc, out.msg)


@pytest.mark.parametrize("label", sorted(ATTRIBUTES))
def test_the_stationary_probe_matches_the_native_one(tmp_path, label):
    source = ATTRIBUTES[label]
    want = native(tmp_path, source)
    path = tmp_path / "prog.py"
    path.write_text(source)
    res = run_probe(sys.executable, str(path), [], [], [], timeout=60)
    got = "clean" if res["status"] in ("clean", "exit") else res.get("exc")
    assert got == want, (label, "native:", want, "probe:", res)


@pytest.mark.parametrize("label", sorted(ATTRIBUTES))
def test_the_in_process_drive_matches_the_native_one_where_the_process_is_not_ours(tmp_path, label):
    # ``drive`` runs inside the verifier's own process (advisory exploration): only the module attributes can match
    if label.startswith(("m12", "m13", "m14", "m15", "m16", "m17", "m18")):
        pytest.skip("sys.path, argv and the interpreter's own limit belong to the verifier's process here")
    source = ATTRIBUTES[label]
    want = native(tmp_path, source)
    path = tmp_path / "prog.py"
    path.write_text(source)
    out = nondet_harness.drive(str(path), [])
    got = "clean" if out.status in ("clean", "exit") else out.exc
    assert got == want, (label, want, out)


def test_a_task_module_named_like_one_of_our_files_is_the_tasks_own(tmp_path):
    (tmp_path / "spec.py").write_text("VALUE = 41\n")
    (tmp_path / "environment.py").write_text("VALUE = 42\n")
    source = "import spec, environment\nassert (spec.VALUE, environment.VALUE) == (41, 42)\n"
    assert native(tmp_path, source) == "clean"
    assert ends_like(child(tmp_path, source)) == "clean"


def test_the_recursion_limit_is_restored_and_a_relaunch_is_not_compounded():
    import sys as _sys

    before = _sys.getrecursionlimit()
    code = compile("import sys\nseen = sys.getrecursionlimit()\n", "<t>", "exec")
    mod = _child.native_main_module("/tmp/x.py")
    _child.run_as_main(code, mod)
    assert mod.__dict__["seen"] == before and _sys.getrecursionlimit() == before
    _child.run_as_main(code, _child.native_main_module("/tmp/x.py"))
    assert _sys.getrecursionlimit() == before
    assert _sys.getrecursionlimit.__module__ != "tharkun" and callable(_sys.setrecursionlimit)


# --- what stays different by construction: the frames BELOW the module frame ------------------------------------------

STACK = {
    "m20 a sys._getframe walk": "import sys\ndef d():\n    n = 0\n    f = sys._getframe()\n    while f:\n        n += 1\n        f = f.f_back\n    return n\nassert d() == 2\n",
    "m21 traceback.extract_stack": "import traceback\nassert len(traceback.extract_stack()) == 1\n",
    "m22 inspect.stack": "import inspect\nassert len(inspect.stack()) == 1\n",
    "from-import of currentframe": "from inspect import currentframe\nassert currentframe().f_back is None\n",
}


@pytest.mark.parametrize("label", sorted(STACK))
def test_a_program_that_reads_the_stack_above_its_module_is_flagged_and_never_a_witness(tmp_path, label):
    source = STACK[label]
    assert native(tmp_path, source) == "clean"                  # natively the module frame is the bottom one
    for run in (child(tmp_path, source),):
        assert "stack_unmodelled" in run.flags
        assert run.violated(("AssertionError",)) is None
    path = tmp_path / "prog.py"
    path.write_text(source)
    res = run_probe(sys.executable, str(path), [], [], [], timeout=60)
    assert "stack_unmodelled" in res["flags"]
    out = nondet_harness.drive(str(path), [])
    assert "stack_unmodelled" in out.flags and out.violated(("AssertionError",)) is None
    verdict = verify(str(path), ("AssertionError",), budget_seconds=15, jobs=1)
    assert verdict.verdict != "false", (label, verdict.verdict, verdict.reason)


def test_a_program_that_does_not_read_the_stack_is_not_flagged(tmp_path):
    out = child(tmp_path, "def f(n):\n    return n if n < 2 else f(n - 1) + f(n - 2)\nassert f(10) == 55\n")
    assert out.flags == [] and out.status == "clean"


# --- review round (MG, #153): threads, and the spellings the first scan missed ----------------------------------------

THREAD_RECURSION = """
import threading

out = []

def f(n):
    try:
        return f(n + 1)
    except RecursionError:
        return n

def work():
    out.append(f(0))

t = threading.Thread(target=work)
t.start()
t.join()
assert out[0] == 995
"""

SPELLINGS = {
    "st03 getattr(inspect, 'stack')": "import inspect\nassert len(getattr(inspect, 'stack')()) == 1\n",
    "st04 string-built _getframe and f_back": "import sys\nf = getattr(sys, '_get' + 'frame')(0)\nn = 0\nwhile f:\n    n += 1\n    f = getattr(f, 'f_' + 'back')\nassert n == 1\n",
    "st05 vars(sys)['_getframe']": "import sys\nf = vars(sys)['_getframe'](0)\nassert f.__getattribute__('f_back') is None\n",
    "st06 exec of a frame walk": "exec('import sys\\nassert sys._getframe(0).f_back is None')\n",
    "st07 a computed getattr next to sys": "import sys\nname = '_get' + str(len('abc')) + 'x'\nassert getattr(sys, name, None) is None\n",
    "st08 a name joined at runtime": "import sys\nf = getattr(sys, ''.join(['_getf', 'rame']))(0)\nassert f.f_back is None\n",
    "st09 __import__ of a computed module": "m = __import__('in' + 'spect')\nassert len(m.stack()) == 1\n",
    "st10 a thread by a computed import": "t = __import__('thread' + 'ing')\nassert t.current_thread() is not None\n",
    "st11 from-import of stack": "from inspect import stack\nassert len(stack()) == 1\n",
    "st12 star import of inspect": "from inspect import *\nassert len(stack()) == 1\n",
    "st14 exec of code built at runtime": "exec(''.join(chr(c) for c in %r))\n" % [ord(c) for c in "import sys\nassert sys._getframe(0).f_back is None\n"],
    "st15 __import__ of a name read at runtime": "name = ''.join(chr(c) for c in [105, 110, 115, 112, 101, 99, 116])\nm = __import__(name)\nassert len(m.stack()) == 1\n",
    "st13 sys.__dict__": "import sys\nassert sys.__dict__['_getframe'](0).f_back is None\n",
}


def test_a_program_that_creates_threads_is_flagged_and_never_a_witness(tmp_path):
    assert native(tmp_path, THREAD_RECURSION) == "clean"            # natively the thread's depth is its own
    out = child(tmp_path, THREAD_RECURSION)
    assert "stack_unmodelled" in out.flags and out.violated(("AssertionError",)) is None
    path = tmp_path / "prog.py"
    path.write_text(THREAD_RECURSION)
    assert "stack_unmodelled" in run_probe(sys.executable, str(path), [], [], [], timeout=60)["flags"]
    verdict = verify(str(path), ("AssertionError",), budget_seconds=15, jobs=1)
    assert verdict.verdict != "false", (verdict.verdict, verdict.reason)


@pytest.mark.parametrize("module", ["threading", "_thread", "concurrent.futures", "multiprocessing.pool", "asyncio", "socketserver"])
def test_every_module_that_runs_code_on_another_stack_flags_the_program(module):
    from tharkun.solve.environment import references_stack

    assert references_stack(f"import {module}\n")
    assert references_stack(f"from {module} import x\n")


@pytest.mark.parametrize("label", sorted(SPELLINGS))
def test_every_spelling_of_a_stack_read_is_flagged_not_just_the_literal_name(tmp_path, label):
    from tharkun.solve.environment import references_stack

    source = SPELLINGS[label]
    assert references_stack(source), label
    out = child(tmp_path, source)
    assert "stack_unmodelled" in out.flags and out.violated(("AssertionError",)) is None, (label, out.flags, out.exc)


def test_a_program_that_only_uses_stack_like_names_for_its_own_things_is_not_flagged():
    from tharkun.solve.environment import references_stack

    for body in ("class Stack:\n    def __init__(self):\n        self.stack = []\n    def push(self, x):\n        self.stack.append(x)\n",
                 "stack = []\nstack.append(1)\nprint(stack.pop())\n",
                 "import re\npat = re.compile('a')\nx = getattr(pat, 'pattern')\n",
                 "import sys\nprint(sys.argv)\nv = getattr(object(), 'x', 0)\n",
                 "name = 'code'\nmsg = 'trace of the stack'\nprint(name, msg)\n",
                 "class A:\n    pass\nsetattr(A(), 'x', 1)\ngetattr(A(), 'y', None)\n"):
        assert not references_stack(body), body


# --- run-time thread detection is the authority (MG, second round of #153) ----------------------------------------------

DEPTH = """
out = []

def f(n):
    try:
        return f(n + 1)
    except RecursionError:
        return n

def work():
    out.append(f(0))
"""

#: (id, files written next to the program, the program's head that obtains and starts a thread: the static scan must NOT see it)
HIDDEN_THREADS = {
    "th09 an aliased import_module of a built name": ({}, "from importlib import import_module as im\nT = im(chr(116) + 'hreading')\n"),
    "th10 exec of a literal that imports threading": ({}, "exec('import threading as T')\n"),
    "th11 eval of a literal __import__": ({}, "T = eval(\"__import__('threading')\")\n"),
    "th17 a helper module starts the thread": ({"helper.py": "import threading as T\n"}, "import helper\nT = helper.T\n"),
    "th18 a helper module that starts it at import": (
        {"starter.py": "import threading\n\ndef go(work):\n    t = threading.Thread(target=work)\n    t.start()\n    t.join()\n"},
        "import starter\n\nclass T:\n    class Thread:\n        def __init__(self, target):\n            self.target = target\n"
        "        def start(self):\n            starter.go(self.target)\n        def join(self):\n            pass\n"),
}


def hidden_thread_program(tmp_path, files, head, expect):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return head + DEPTH + f"t = T.Thread(target=work)\nt.start()\nt.join()\nassert out[0] == {expect}\n"


def native_thread_depth(tmp_path, files, head):
    probe = hidden_thread_program(tmp_path, files, head, 0).replace("assert out[0] == 0", "print(out[0])")
    path = tmp_path / "probe_native.py"
    path.write_text(probe)
    run = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    return int(run.stdout.strip())


@pytest.mark.parametrize("label", sorted(HIDDEN_THREADS))
def test_a_thread_the_static_scan_cannot_see_is_flagged_when_it_starts(tmp_path, label):
    from tharkun.solve.environment import references_stack

    files, head = HIDDEN_THREADS[label]
    depth = native_thread_depth(tmp_path, files, head)
    source = hidden_thread_program(tmp_path, files, head, depth)
    assert native(tmp_path, source) == "clean"
    assert not references_stack(source), f"{label}: the scan sees it, this test is about what it cannot see"
    out = child(tmp_path, source)
    assert "stack_unmodelled" in out.flags and out.violated(("AssertionError",)) is None, (label, out.flags, out.exc)
    path = tmp_path / "prog.py"
    res = run_probe(sys.executable, str(path), [], [], [], timeout=60)
    assert "stack_unmodelled" in res["flags"], (label, res.get("flags"))
    drive = nondet_harness.drive(str(path), [])
    assert "stack_unmodelled" in drive.flags and drive.violated(("AssertionError",)) is None
    verdict = verify(str(path), ("AssertionError",), budget_seconds=15, jobs=1)
    assert verdict.verdict != "false", (label, verdict.verdict, verdict.reason)


def test_drive_records_threads_while_a_foreign_worker_is_watching(tmp_path, monkeypatch):
    import threading
    from tharkun.sv import _probe
    from tharkun.solve.environment import references_stack

    assert _probe.C._State is not _child._State
    monkeypatch.setattr(_probe.C._State, "flags", set())
    source = "exec('import threading as T')\nt = T.Thread(target=lambda: None)\nt.start()\nt.join()\nassert False\n"
    assert not references_stack(source)
    path = tmp_path / "foreign_watch.py"
    path.write_text(source)
    before = threading.Thread.start
    undo = _probe.C.watch_threads()
    foreign_start = threading.Thread.start
    try:
        out = nondet_harness.drive(str(path), [])
        assert threading.Thread.start is foreign_start
    finally:
        undo()
    assert threading.Thread.start is before
    assert "stack_unmodelled" in out.flags
    assert out.violated(("AssertionError",)) is None


def test_every_way_into_a_thread_is_watched(tmp_path):
    import threading

    for label, body in {
        "Thread.start": "import threading\nthreading.Thread(target=lambda: None).start()\n",
        "a Thread subclass": "import threading\nclass T(threading.Thread):\n    def run(self):\n        pass\nT().start()\n",
        "_thread.start_new_thread": "import _thread, time\n_thread.start_new_thread(lambda: None, ())\n",
        "a ThreadPoolExecutor": "from concurrent.futures import ThreadPoolExecutor\nwith ThreadPoolExecutor(1) as p:\n    p.submit(lambda: 1).result()\n",
        "a Timer": "import threading\nt = threading.Timer(0, lambda: None)\nt.start()\nt.join()\n",
    }.items():
        out = child(tmp_path, body)
        assert "stack_unmodelled" in out.flags, (label, out.flags)
    plain = child(tmp_path, "x = [i for i in range(10)]\nassert sum(x) == 45\n")
    assert "stack_unmodelled" not in plain.flags


def test_the_drive_restores_threading_after_every_run(tmp_path):
    import threading

    before = (threading.Thread.start, getattr(threading, "_start_joinable_thread", None))
    path = tmp_path / "prog.py"
    path.write_text("import threading\nthreading.Thread(target=lambda: None).start()\n")
    nondet_harness.drive(str(path), [])
    assert (threading.Thread.start, getattr(threading, "_start_joinable_thread", None)) == before
