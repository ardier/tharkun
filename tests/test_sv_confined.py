"""``solve/environment.py::confined`` -- the REVERSIBLE environment model and file-system confinement (#1132b).

``install_files`` + ``install_mutation_confinement`` patch ``builtins``/``io``/``os``/``posix`` for the rest of a child
or worker process.  ``confined`` is the same installer put back afterwards, so a process that must keep running
(the Eregion inverter, replaying the user's package thousands of times in-process) can confine exactly the code it
runs.  Pinned here: the restore is exact; nesting and threads never tear the snapshot; the scratch and the cwd are
managed; declared files are readable and shadowed; the SAME victim battery as ``test_sv_fs_confinement.py`` leaves
the whole outside tree untouched; the positive controls still work, before and after.
"""
from __future__ import annotations

import os
import shutil
import threading
import pathlib

import pytest

import posix

from tharkun.solve import environment
from tharkun.solve.environment import Confinement, confined

from tests.test_sv_fs_confinement import FAMILIES, outside, tree  # noqa: F401  (``outside`` is a fixture)


def attributes():
    return list(environment._patched_attributes())


def originals():
    return [(mod, name, getattr(mod, name)) for mod, name in attributes() if hasattr(mod, name)]


def assert_restored(before):
    for mod, name, value in before:
        assert getattr(mod, name) is value, f"{mod.__name__}.{name} was not restored"
    assert environment._CONFINED == {"depth": 0}


def attempt(*ops):
    for op in ops:
        try:
            op()
        except OSError:
            pass


# -- reversibility --------------------------------------------------------------

#: run in a child (the installer cannot be undone in-process): every module-level attribute of every loaded module,
#: before and after ``install_files``; prints the ones that changed and that ``_patched_attributes`` does not name
PROBE = """
import json, sys
sys.path.insert(0, {src!r})
from tharkun.solve import environment
import subprocess, _posixsubprocess, pty, multiprocessing.process   # the spawn surface is loaded, so it is visible

def snapshot():
    out = {{}}
    for name, mod in list(sys.modules.items()):
        try:
            items = list(vars(mod).items())
        except TypeError:
            continue
        for attr, value in items:
            out[(name, attr)] = value
    return out

def main():
    missing = object()
    before = snapshot()
    environment.install_files((), {scratch!r}, set())
    after = snapshot()
    loaded = {{name for name, _ in before}}
    changed = {{k for k, v in after.items() if k[0] in loaded and before.get(k, missing) is not v}}
    named = {{(m.__name__, n) for m, n in environment._patched_attributes()}}
    print(json.dumps(sorted(f"{{a}}.{{b}}" for a, b in changed - named)))

main()
"""


def test_install_files_rewrites_nothing_the_snapshot_does_not_name(tmp_path):
    """``confined`` restores ``_patched_attributes()`` and nothing else, so the list must cover EVERYTHING the
    installer rewrites.  A change to the installer that patches one more attribute (process-spawn confinement
    rewrites ``os.system``, ``subprocess.Popen`` ...) fails here until it is added to the list."""
    import subprocess
    import sys

    src = str(pathlib.Path(environment.__file__).resolve().parents[2])
    out = subprocess.run([sys.executable, "-c", PROBE.format(src=src, scratch=str(tmp_path))],
                         capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout



def test_the_attribute_set_covers_what_the_installer_rewrites():
    names = {(m.__name__, n) for m, n in attributes()}
    for want in (("builtins", "open"), ("io", "open"), ("_io", "open"), ("io", "FileIO"), ("_io", "FileIO"),
                 ("os", "stat"), ("os", "open"), ("os", "listdir"), ("os", "scandir"), ("os", "remove"),
                 ("posix", "remove"), ("os", "rename"), ("posix", "rename"), ("os", "symlink"), ("posix", "symlink"),
                 ("os", "truncate"), ("os", "mkdir"), ("posix", "mkdir")):
        assert want in names


def test_every_patched_attribute_is_restored_after_a_normal_exit():
    before = originals()
    with confined() as c:
        assert isinstance(c, Confinement)
        assert any(getattr(mod, name) is not value for mod, name, value in before), "the model is installed"
        assert builtins_open_is_patched()
    assert_restored(before)


def builtins_open_is_patched():
    import builtins
    return builtins.open is not originals_open


originals_open = __import__("builtins").open


def test_every_patched_attribute_is_restored_after_an_exception_and_it_propagates():
    before = originals()
    with pytest.raises(RuntimeError, match="boom"):
        with confined():
            raise RuntimeError("boom")
    assert_restored(before)


def test_a_failed_install_leaves_nothing_behind(monkeypatch, tmp_path):
    before = originals()
    cwd = os.getcwd()

    def broken(files, scratch, flags):
        os.stat = lambda *a, **k: None          # half-installed
        raise OSError("install failed")

    monkeypatch.setattr(environment, "install_files", broken)
    with pytest.raises(OSError, match="install failed"):
        with confined():
            pass                                  # pragma: no cover
    monkeypatch.undo()
    assert_restored(before)
    assert os.getcwd() == cwd


# -- nesting, threads ---------------------------------------------------------


def test_nesting_shares_the_outer_context_and_the_last_exit_restores():
    before = originals()
    with confined() as outer:
        with confined() as inner:
            assert inner is not None and inner.scratch == outer.scratch and inner.flags is outer.flags
            patched = originals()
        assert any(getattr(m, n) is not v for m, n, v in before), "an inner exit leaves it confined"
        assert [(m, n) for m, n, _ in originals()] == [(m, n) for m, n, _ in patched]
    assert_restored(before)


def test_an_exception_in_the_inner_block_does_not_unconfine_the_outer_one():
    before = originals()
    with confined():
        with pytest.raises(ValueError):
            with confined():
                raise ValueError("inner")
        import builtins
        assert builtins.open is not originals_open
    assert_restored(before)


def test_two_overlapping_threads_never_tear_the_snapshot():
    before = originals()
    entered_a, entered_b = threading.Event(), threading.Event()
    leave_a, leave_b = threading.Event(), threading.Event()
    seen: dict = {}

    def run(name, entered, leave):
        with confined() as c:
            seen[name] = c
            entered.set()
            assert leave.wait(10)

    a = threading.Thread(target=run, args=("a", entered_a, leave_a))
    b = threading.Thread(target=run, args=("b", entered_b, leave_b))
    a.start()
    assert entered_a.wait(10)
    b.start()
    assert entered_b.wait(10)
    assert seen["a"].scratch == seen["b"].scratch, "the second context shares the first's model"
    leave_a.set()
    a.join(10)
    import builtins
    assert builtins.open is not originals_open, "still patched while the second thread is inside"
    leave_b.set()
    b.join(10)
    assert_restored(before)


def test_a_thread_started_inside_the_window_can_enter_and_be_joined_without_deadlock():
    # the shape of ``replay_deep``: the calling thread is inside ``confined`` and waits for a helper thread that
    # enters it too
    before = originals()
    done = []

    def helper():
        with confined():
            done.append(True)

    with confined():
        t = threading.Thread(target=helper)
        t.start()
        t.join(10)
    assert done == [True]
    assert_restored(before)


# -- scratch, cwd, flags -------------------------------------------------------


def test_the_default_scratch_exists_inside_is_the_cwd_and_is_gone_afterwards():
    cwd = os.getcwd()
    with confined() as c:
        assert os.path.isdir(c.scratch) and os.getcwd() == os.path.realpath(c.scratch)
        scratch = c.scratch
    assert not os.path.exists(scratch)
    assert os.getcwd() == cwd


def test_a_caller_given_scratch_survives_and_flags_are_the_callers(tmp_path):
    scratch = tmp_path / "mine"
    flags: set = set()
    with confined(scratch=str(scratch), flags=flags) as c:
        assert c.scratch == os.path.realpath(scratch)
        open("note.txt", "w").write("x")
    assert scratch.is_dir(), "a scratch the caller passed is never removed"
    assert flags == set() or flags <= {"file_effect", "file_unmodelled"}


def test_a_nested_contexts_flags_argument_is_filled_when_it_exits(tmp_path):
    victim = tmp_path / "v.txt"
    victim.write_text("keep")
    inner_flags: set = set()
    with confined():
        with confined(flags=inner_flags):
            attempt(lambda: os.remove(victim))
    assert {"file_effect", "file_unmodelled"} <= inner_flags
    assert victim.read_text() == "keep"


def test_the_cwd_is_restored_even_when_the_block_raises_and_even_if_the_block_changed_it(tmp_path):
    cwd = os.getcwd()
    with pytest.raises(KeyError):
        with confined():
            os.chdir(tmp_path)
            raise KeyError("x")
    assert os.getcwd() == cwd


# -- declared files ------------------------------------------------------------


def test_a_declared_file_is_readable_a_write_goes_to_a_scratch_copy_and_the_real_bytes_stay(tmp_path):
    real = tmp_path / "data.txt"
    real.write_text("real bytes")
    other = tmp_path / "undeclared.txt"
    other.write_text("secret")
    with confined(files=(str(real),)) as c:
        assert open(real).read() == "real bytes"
        with open(real, "w") as fh:
            fh.write("changed")
        assert open(real).read() == "changed"
        with pytest.raises(FileNotFoundError):
            open(other).read()
        assert not (c.flags - {"file_effect"} - {"file_unmodelled"})
    assert real.read_text() == "real bytes"
    assert other.read_text() == "secret"


# -- the victim battery --------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_victim_battery_leaves_the_outside_tree_untouched(outside, family):
    before = tree(outside)
    env = {"os": os, "shutil": shutil, "pathlib": pathlib, "posix": posix, "OUT": str(outside), "attempt": attempt}
    with confined() as c:
        exec(compile(FAMILIES[family], f"<{family}>", "exec"), env)
        flags = set(c.flags)
    assert tree(outside) == before
    assert {"file_effect", "file_unmodelled"} <= flags


def test_posix_remove_is_confined_too(outside):
    before = tree(outside)
    with confined() as c:
        with pytest.raises(OSError):
            posix.remove(str(outside / "victim.txt"))
    assert tree(outside) == before


def test_positive_controls_scratch_relative_calls_work_and_outside_calls_work_again_afterwards(outside, tmp_path):
    with confined():
        os.mkdir("d")
        open("d/f.txt", "w").write("f")
        os.rename("d/f.txt", "d/g.txt")
        os.remove("d/g.txt")
        os.rmdir("d")
        assert not os.path.exists("d")
    victim = outside / "victim.txt"
    os.rename(victim, outside / "renamed.txt")
    os.remove(outside / "renamed.txt")
    os.mkdir(outside / "made")
    pathlib.Path(outside / "made").rmdir()
    assert not (outside / "renamed.txt").exists()


# -- process-global effects (MAIN's note) ---------------------------------------


def test_a_relative_write_by_the_caller_after_exit_goes_to_the_original_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with confined():
        pass
    open("after.txt", "w").write("after")
    assert (tmp_path / "after.txt").read_text() == "after"


def test_a_write_from_a_second_thread_during_the_window_is_confined_not_misplaced(tmp_path, monkeypatch, outside):
    monkeypatch.chdir(tmp_path)
    victim = outside / "victim.txt"
    result: dict = {}

    def other_thread():
        try:
            open("from_thread.txt", "w").write("x")            # relative: lands in the scratch, never in tmp_path
            result["relative"] = "written"
        except OSError as exc:
            result["relative"] = repr(exc)
        try:
            os.remove(victim)
        except OSError as exc:
            result["remove"] = exc.errno

    with confined():
        t = threading.Thread(target=other_thread)
        t.start()
        t.join(10)
    assert not (tmp_path / "from_thread.txt").exists(), "a relative write cannot land in the caller's directory"
    import errno
    assert result["remove"] == errno.EROFS and victim.exists()


def test_a_first_import_of_subprocess_inside_a_block_is_not_left_refusing(tmp_path):
    """``subprocess`` binds ``_posixsubprocess.fork_exec`` as ``_fork_exec`` when it is first imported.  A first
    import inside a confined block keeps the refusing function unless the exit puts the original back
    (``environment.py::_unpoison_subprocess``), and the host process would then never spawn again."""
    import subprocess
    import sys
    code = (
        "import sys\n"
        "assert 'subprocess' not in sys.modules, 'the child must not have imported subprocess yet'\n"
        "from tharkun.solve.environment import confined\n"
        "with confined():\n"
        "    import subprocess\n"
        "    try:\n"
        "        subprocess.run(['true'])\n"
        "        refused = False\n"
        "    except OSError:\n"
        "        refused = True\n"
        "assert refused, 'a spawn inside the block must be refused'\n"
        "assert subprocess.run(['true']).returncode == 0, 'a spawn after the block must work'\n"
        "print('OK')\n"
    )
    src = str(pathlib.Path(environment.__file__).resolve().parents[2])      # the directory holding the ``tharkun`` package
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path),
                         env={**os.environ, "PYTHONPATH": src})
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr + out.stdout


def test_sv_thread_refusal_restores_every_native_starter_on_exception():
    before = [(owner, name, getattr(owner, name)) for owner, name in environment.thread_starters()]
    flags = set()
    with pytest.raises(RuntimeError, match="leave"):
        with confined(flags=flags, threads=False):
            for owner, name, _original in before:
                with pytest.raises(OSError) as caught:
                    getattr(owner, name)()
                assert caught.value.errno == 1
            raise RuntimeError("leave")
    assert {"stack_unmodelled", "file_unmodelled"} <= flags
    assert all(getattr(owner, name) is original for owner, name, original in before)
    assert environment._CONFINED == {"depth": 0}


def test_non_lifo_host_threads_share_sv_thread_refusal_until_last_exit():
    before = [(owner, name, getattr(owner, name)) for owner, name in environment.thread_starters()]
    start_a, start_b = threading.Event(), threading.Event()
    entered_a, entered_b = threading.Event(), threading.Event()
    leave_a, leave_b, exited_a = threading.Event(), threading.Event(), threading.Event()
    checked_b = threading.Event()
    errors = []

    def a():
        try:
            assert start_a.wait(60)
            with confined(threads=False):
                entered_a.set()
                assert leave_a.wait(60)
        except BaseException as error:
            errors.append(error)
        finally:
            exited_a.set()

    def b():
        try:
            assert start_b.wait(60)
            with confined(threads=False):
                entered_b.set()
                assert exited_a.wait(60)
                for owner, name, _original in before:
                    with pytest.raises(OSError):
                        getattr(owner, name)()
                checked_b.set()
                assert leave_b.wait(60)
        except BaseException as error:
            errors.append(error)

    # Both host workers already exist before any target phase begins.
    first, second = threading.Thread(target=a), threading.Thread(target=b)
    first.start()
    second.start()
    try:
        start_a.set()
        assert entered_a.wait(60)
        start_b.set()
        assert entered_b.wait(60)
        leave_a.set()
        assert checked_b.wait(60)
    finally:
        start_a.set()
        start_b.set()
        leave_a.set()
        leave_b.set()
        first.join(60)
        second.join(60)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert all(getattr(owner, name) is original for owner, name, original in before)
    assert environment._CONFINED == {"depth": 0}


@pytest.mark.parametrize("source", [
    "class C:\n def __del__(self): pass\nC()\n",
    "def done(self): pass\nC = type('C', (), {'__' + 'del' + '__': done})\nC()\n",
    "factory = type\nC = factory('C', (), {})\n",
    "from builtins import type as factory\nC = factory('C', (), {})\n",
    "class C: pass\nname = '__' + 'del' + '__'\nsetattr(C, name, lambda self: None)\n",
    "import atexit\natexit.register(print, 'later')\n",
    "import weakref\n",
    "from weakref import ref\n",
    "from multiprocessing.util import Finalize\n",
    "import signal\n",
    "import logging\n",
    "import asyncio\n",
    "from _weakref import ref\n",
    "import gc\ngc.callbacks.append(lambda *args: None)\n",
    "import sys\nsys.settrace(lambda *args: None)\n",
    "from sys import setprofile as register\nregister(lambda *args: None)\n",
    "import sys\nsys.addaudithook(lambda *args: None)\n",
    "from threading import _register_atexit as register\nregister(lambda: None)\n",
    "maker = type(type(0))\nC = maker('C', (), {})\n",
    "maker = int.__class__\nC = maker('C', (), {})\n",
])
def test_deferred_hooks_are_refused_only_for_reversible_execution(source):
    from tharkun.solve.extern import native_import_reason

    assert native_import_reason(source) is None
    reason = native_import_reason(source, reversible=True)
    assert reason and "reversible execution" in reason


@pytest.mark.parametrize("source", [
    "class C: pass\nc = C()\nassert isinstance(c, C)\n",
    "class C: pass\nsetattr(C, 'value', 3)\n",
    "import _sv_verifier as sv\nx = sv.nondet_int()\n",
])
def test_reversible_preflight_keeps_ordinary_classes_and_defined_stub_access(source):
    from tharkun.solve.extern import native_import_reason

    assert native_import_reason(source, reversible=True) is None


@pytest.mark.parametrize("source", [
    "import sys\nassert type(sys.argv) is list\n",
    "assert list is not type(0)\n",
    "assert type([]) is list is not type(0)\n",
])
def test_reversible_type_identity_does_not_expose_a_class(source):
    from tharkun.solve.extern import native_import_reason

    assert native_import_reason(source, reversible=True) is None


@pytest.mark.parametrize("source", [
    "maker = type(type(0))\n",
    "maker = type(0)\n",
    "assert type(0).__name__ == 'int'\n",
    "assert type(0)() is not None\n",
    "assert type(0) == int\n",
    "assert (maker := type(0)) is int\n",
    "from builtins import type as maker\nassert maker(0) is int\n",
    "assert type(*[0]) is int\n",
    "assert type('C', (), {}) is not None\n",
])
def test_reversible_type_values_still_cannot_escape(source):
    from tharkun.solve.extern import native_import_reason

    assert native_import_reason(source, reversible=True)


def test_reversible_preflight_follows_inherited_local_finalizers(tmp_path):
    from tharkun.solve.extern import native_import_reason

    helper = tmp_path / "helper.py"
    helper.write_text("class Parent:\n def __del__(self): pass\n")
    source = "from helper import Parent\nclass Child(Parent): pass\nChild()\n"
    program = tmp_path / "program.py"
    program.write_text(source)
    assert native_import_reason(source, str(program)) is None
    assert "helper.py" in native_import_reason(source, str(program), reversible=True)


@pytest.mark.parametrize("source", [
    "import builtins\nmaker = builtins.__dict__['type']\n",
    "from builtins import exec as run\nrun('pass')\n",
    "import sys\nx = sys.modules['__main__']\n",
])
def test_reversible_preflight_preserves_existing_reflection_refusals(source):
    from tharkun.solve.extern import native_import_reason

    assert native_import_reason(source)
    assert native_import_reason(source, reversible=True)


@pytest.mark.parametrize("name", ["type", "exec"])
def test_literal_getattr_cannot_escape_reversible_preflight(name):
    from tharkun.solve.extern import native_import_reason

    source = f"import builtins\noperation = getattr(builtins, {name!r})\n"
    assert native_import_reason(source, reversible=True)


def test_thread_refusal_setup_failure_restores_environment(tmp_path, monkeypatch):
    before = originals()
    cwd = os.getcwd()

    def unavailable(flags):
        raise RuntimeError("starter setup failed")

    monkeypatch.setattr(environment, "refuse_threads", unavailable)
    with pytest.raises(RuntimeError, match="starter setup failed"):
        with confined(scratch=str(tmp_path / "scratch"), threads=False):
            pytest.fail("target must not run without its thread boundary")
    assert_restored(before)
    assert os.getcwd() == cwd


@pytest.mark.parametrize("persistent", [False, True])
def test_concolic_terminal_guard_runs_after_abnormal_target_phase(tmp_path, monkeypatch, persistent):
    import json
    from tharkun.sv import _concolic_worker

    program = tmp_path / "program.py"
    program.write_text("pass\n")
    harness = tmp_path / "harness.py"
    harness.write_text("pass\n")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"files": [str(program)], "harness": str(harness), "scratch": str(tmp_path)}))
    events = []

    def target_phase(req_path, req):
        events.append("target")
        (tmp_path / "trusted-output").write_text("saved")
        raise RuntimeError("target phase failed")

    def install(files, scratch, flags):
        assert (tmp_path / "trusted-output").read_text() == "saved"
        assert files == (str(program), str(harness))
        assert scratch == str(tmp_path)
        events.append("files")

    monkeypatch.setattr(_concolic_worker, "_run_request", target_phase)
    monkeypatch.setattr(environment, "install_files", install)
    monkeypatch.setattr(environment, "refuse_threads", lambda flags: events.append("threads"))
    with pytest.raises(RuntimeError, match="target phase failed"):
        _concolic_worker.main(str(request), persistent=persistent)
    assert events == (["target", "files", "threads"] if persistent else ["target"])


@pytest.mark.parametrize("module, member", [
    ("operator", "attrgetter"), ("_operator", "methodcaller"),
    ("abc", "ABCMeta"), ("_py_abc", "ABCMeta"), ("enum", "EnumType"),
    ("dataclasses", "make_dataclass"), ("typing", "NamedTuple"),
    ("types", "new_class"), ("_types", "FunctionType"), ("inspect", "getmembers"),
    ("pickle", "loads"), ("_pickle", "loads"), ("marshal", "loads"),
    ("shelve", "open"), ("runpy", "run_path"), ("code", "InteractiveInterpreter"),
    ("codeop", "compile_command"), ("pydoc", "locate"),
])
@pytest.mark.parametrize("style", ["module", "from_alias"])
def test_class_builders_and_deferred_code_modules_stay_out_of_reversible_execution(module, member, style):
    from tharkun.solve.extern import native_import_reason

    source = (f"import {module} as reflection\nfactory = reflection.{member}\n" if style == "module"
              else f"from {module} import {member} as factory\n")
    assert native_import_reason(source) is None
    reason = native_import_reason(source, reversible=True)
    assert reason and "reversible execution" in reason


def test_operator_class_builder_cannot_hide_computed_finalizer():
    from tharkun.solve.extern import native_import_reason

    source = (
        "import operator\n"
        "def finish(self):\n open('victim', 'w').write('changed')\n"
        "maker = operator.attrgetter('__class__')(int)\n"
        "C = maker('C', (), {'__' + 'del' + '__': finish})\n"
        "instance = C()\n"
    )
    assert native_import_reason(source) is None
    assert native_import_reason(source, reversible=True)


@pytest.mark.parametrize("source", [
    "import os\nos.register_at_fork(before=lambda: None)\n",
    "from os import register_at_fork as register\nregister(before=lambda: None)\n",
    "import os\nregister = getattr(os, 'register_at_fork')\nregister(before=lambda: None)\n",
])
def test_fork_callbacks_cannot_outlive_reversible_execution(source):
    from tharkun.solve.extern import native_import_reason

    assert native_import_reason(source) is None
    assert native_import_reason(source, reversible=True)
