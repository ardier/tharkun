"""Every file-system-mutating call is confined to the scratch directory, in the engine's child AND in the hybrid worker.

``solve/environment.py::install_mutation_confinement``: ``install_files`` shadowed what a program reads and writes
through ``open``, but ``os.remove``/``rename``/``mkdir``/``rmdir``/``chmod``/... went straight to the host, so a
replay -- or the Atheris worker, which runs the program for real millions of times -- could delete, rename or
re-permission a file outside the scratch.  One test per call family, with victims outside the scratch, plus the
positive control (the same calls on scratch-relative paths still work).
"""
from __future__ import annotations

import os
import stat
import sys
import textwrap

import pytest

from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.solve.environment import TaskEnvironment
from tharkun.sv.engine import run_once, verify


def tree(root):
    """Everything under ``root``: names, kinds, modes, sizes, mtimes, contents."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            body = open(path, "rb").read() if stat.S_ISREG(st.st_mode) else (
                os.readlink(path) if stat.S_ISLNK(st.st_mode) else None)
            out[os.path.relpath(path, root)] = (stat.S_IFMT(st.st_mode), stat.S_IMODE(st.st_mode), st.st_size,
                                                st.st_mtime_ns, body)
    return out


@pytest.fixture
def outside(tmp_path):
    root = tmp_path / "outside"
    (root / "dir" / "sub").mkdir(parents=True)
    (root / "empty").mkdir()
    (root / "victim.txt").write_text("victim")
    (root / "other.txt").write_text("other")
    (root / "dir" / "inner.txt").write_text("inner")
    (root / "dir" / "sub" / "deep.txt").write_text("deep")
    os.utime(root / "victim.txt", (1_000_000_000, 1_000_000_000))
    os.chmod(root / "victim.txt", 0o640)
    return root


def program(tmp_path, body, preamble=""):
    p = tmp_path / "prog" / "t.py"
    p.parent.mkdir(exist_ok=True)
    head = "import os, shutil, pathlib, posix\n" + preamble + textwrap.dedent("""
        def attempt(*ops):
            for op in ops:
                try:
                    op()
                except OSError:
                    pass
        if __name__ == "__main__":
    """)
    p.write_text(head + textwrap.indent(textwrap.dedent(body), "    ") + "\n")
    return str(p)


def run_child(path):
    return run_once(sys.executable, path, [], timeout=60, files=TaskEnvironment.from_task(path).files)


FAMILIES = {
    "remove_unlink": "attempt(lambda: os.remove(OUT + '/victim.txt'), lambda: os.unlink(OUT + '/other.txt'), "
                     "lambda: posix.remove(OUT + '/victim.txt'))",
    "rename_replace": "open('mine.txt', 'w').write('mine')\n"
                      "attempt(lambda: os.rename(OUT + '/victim.txt', OUT + '/moved.txt'), "
                      "lambda: os.replace(OUT + '/other.txt', OUT + '/moved2.txt'), "
                      "lambda: os.rename('mine.txt', OUT + '/victim.txt'), "
                      "lambda: os.replace('mine.txt', OUT + '/other.txt'), "
                      "lambda: os.rename(OUT + '/victim.txt', 'stolen.txt'), "
                      "lambda: os.renames(OUT + '/victim.txt', OUT + '/a/b/c.txt'))",
    "mkdir_rmdir": "attempt(lambda: os.mkdir(OUT + '/new'), lambda: os.makedirs(OUT + '/p/q/r'), "
                   "lambda: os.rmdir(OUT + '/empty'), lambda: os.removedirs(OUT + '/empty'), "
                   "lambda: os.makedirs(OUT + '/dir/sub/more', exist_ok=True))",
    "chmod_chown_utime_truncate": "attempt(lambda: os.chmod(OUT + '/victim.txt', 0o777), "
                                  "lambda: os.chown(OUT + '/victim.txt', os.getuid(), os.getgid()), "
                                  "lambda: os.lchown(OUT + '/victim.txt', os.getuid(), os.getgid()), "
                                  "lambda: os.utime(OUT + '/victim.txt', (1, 1)), "
                                  "lambda: os.truncate(OUT + '/victim.txt', 0), "
                                  "lambda: os.chmod(OUT + '/dir', 0o000), "
                                  "lambda: os.chmod(path=OUT + '/victim.txt', mode=0o600))",
    "link_symlink": "open('mine.txt', 'w').write('mine')\n"
                    "attempt(lambda: os.link(OUT + '/victim.txt', 'hard.txt'), "
                    "lambda: os.symlink(OUT + '/victim.txt', 'soft.txt'), "
                    "lambda: os.symlink('mine.txt', OUT + '/newlink'), "
                    "lambda: os.link('mine.txt', OUT + '/newhard'), "
                    "lambda: os.symlink('../outside/victim.txt', 'rel.txt'))\n"
                    "attempt(lambda: open('hard.txt', 'w').write('x'), lambda: open('soft.txt', 'w').write('x'), "
                    "lambda: open('rel.txt', 'w').write('x'))",
    "shutil": "os.mkdir('src'); open('src/f.txt', 'w').write('f')\n"
              "attempt(lambda: shutil.rmtree(OUT + '/dir'), lambda: shutil.move(OUT + '/victim.txt', 'got.txt'), "
              "lambda: shutil.move('src', OUT + '/moved_src'), lambda: shutil.copytree('src', OUT + '/copied'), "
              "lambda: shutil.copy('src/f.txt', OUT + '/copied_file'), "
              "lambda: shutil.copystat('src/f.txt', OUT + '/victim.txt'), "
              "lambda: shutil.chown(OUT + '/victim.txt', user=os.getuid()))",
    "pathlib": "P = pathlib.Path\n"
               "attempt(lambda: P(OUT + '/victim.txt').unlink(), lambda: P(OUT + '/other.txt').rename(OUT + '/zz'), "
               "lambda: P(OUT + '/other.txt').replace(OUT + '/zz'), lambda: P(OUT + '/mk').mkdir(), "
               "lambda: P(OUT + '/mk/a/b').mkdir(parents=True), lambda: P(OUT + '/empty').rmdir(), "
               "lambda: P(OUT + '/victim.txt').touch(), lambda: P(OUT + '/touched').touch(), "
               "lambda: P(OUT + '/victim.txt').chmod(0o777), lambda: P(OUT + '/ln').symlink_to('victim.txt'), "
               "lambda: P(OUT + '/hl').hardlink_to(OUT + '/victim.txt'), "
               "lambda: P(OUT + '/victim.txt').write_text('overwritten'), "
               "lambda: P(OUT + '/other.txt').write_bytes(b'overwritten'))",
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_replay_cannot_mutate_the_file_system_outside_the_scratch(tmp_path, outside, family):
    before = tree(outside)
    path = program(tmp_path, FAMILIES[family], preamble=f"OUT = {str(outside)!r}\n")
    out = run_child(path)
    assert out.status == "clean", out.msg
    assert tree(outside) == before
    assert {"file_effect", "file_unmodelled"} <= set(out.flags)        # recorded: the run is never a violation or a TRUE


def test_the_same_calls_on_scratch_relative_paths_still_work(tmp_path):
    path = program(tmp_path, """
        os.mkdir('d'); os.makedirs('a/b')
        open('d/f.txt', 'w').write('f')
        os.rename('d/f.txt', 'd/g.txt'); os.replace('d/g.txt', 'a/b/h.txt')
        os.utime('a/b/h.txt', (5, 5)); os.chmod('a/b/h.txt', 0o600); os.truncate('a/b/h.txt', 0)
        os.symlink('h.txt', 'a/b/link'); os.link('a/b/h.txt', 'a/b/hard')
        pathlib.Path('p').mkdir(); pathlib.Path('p/q').touch(); pathlib.Path('p/q').unlink(); pathlib.Path('p').rmdir()
        os.remove('a/b/link'); os.remove('a/b/hard'); os.remove('a/b/h.txt'); os.rmdir('a/b'); os.rmdir('a'); os.rmdir('d')
        assert not os.path.exists('a') and not os.path.exists('d') and not os.path.exists('p')
    """)
    out = run_child(path)
    assert out.status == "clean", out.msg


def test_a_link_out_of_the_scratch_is_refused_even_when_its_own_path_is_inside(tmp_path, outside):
    """A symbolic link inside the scratch pointing out would let a later write escape: it is never created."""
    path = program(tmp_path, "attempt(lambda: os.symlink(OUT + '/victim.txt', 'soft.txt'))\n"
                             "assert not os.path.lexists('soft.txt')", preamble=f"OUT = {str(outside)!r}\n")
    assert run_child(path).status == "clean"


def test_a_dir_fd_argument_is_refused(tmp_path, outside):
    path = program(tmp_path, "fd = os.open('.', os.O_RDONLY)\nattempt(lambda: os.unlink('x', dir_fd=fd))",
                   preamble=f"OUT = {str(outside)!r}\n")
    out = run_child(path)
    assert "file_unmodelled" in out.flags


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_the_hybrid_worker_cannot_mutate_the_file_system_outside_the_scratch_either(tmp_path, outside):
    before = tree(outside)
    body = "s = _sv_verifier.nondet_str()\nassert len(s) >= 0\n" + "\n".join(FAMILIES.values())
    path = program(tmp_path, body, preamble=f"import _sv_verifier\nOUT = {str(outside)!r}\n")
    # the families share names (src/mine.txt are created each time the program runs): tolerate re-creation
    text = open(path).read().replace("os.mkdir('src'); ", "os.makedirs('src', exist_ok=True); ")
    open(path, "w").write(text)
    verify(path, ("AssertionError",), jobs=1, budget_seconds=9, prove=False, concolic=False, stages=["hybrid"])
    assert tree(outside) == before
