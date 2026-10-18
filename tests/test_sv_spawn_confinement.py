import os, stat, textwrap, sys
import pytest
from tharkun.solve.environment import TaskEnvironment
from tharkun.sv.engine import run_once, verify
from tharkun.hybrid.fdp_codec import atheris_available


def tree(root):
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
    root.mkdir()
    (root / "victim.txt").write_text("victim")
    return root

def program(tmp_path, body, preamble=""):
    p = tmp_path / "prog" / "t.py"
    p.parent.mkdir(exist_ok=True)
    head = "import os, shutil, pathlib, posix, subprocess, multiprocessing, pty, concurrent.futures\n" + preamble + textwrap.dedent("""
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
    "system_popen": "attempt(lambda: os.system('rm ' + OUT + '/victim.txt'), lambda: os.popen('rm ' + OUT + '/victim.txt').read(), lambda: posix.system('rm ' + OUT + '/victim.txt'))",
    "spawn": "attempt(lambda: os.spawnl(os.P_WAIT, '/bin/rm', 'rm', OUT + '/victim.txt'), lambda: os.posix_spawn('/bin/rm', ['rm', OUT + '/victim.txt'], os.environ), lambda: posix.posix_spawn('/bin/rm', ['rm', OUT + '/victim.txt'], os.environ))",
    "exec": "attempt(lambda: os.execl('/bin/rm', 'rm', OUT + '/victim.txt'), lambda: os.execv('/bin/rm', ['rm', OUT + '/victim.txt']), lambda: posix.execv('/bin/rm', ['rm', OUT + '/victim.txt']))",
    "fork": "attempt(lambda: os.fork(), lambda: os.forkpty(), lambda: posix.fork())",
    "subprocess": "attempt(lambda: subprocess.run(['rm', OUT + '/victim.txt']), lambda: subprocess.Popen(['rm', OUT + '/victim.txt']).wait())",
    "multiprocessing": "def f(x): os.remove(OUT + '/victim.txt')\n"
                       "attempt(lambda: multiprocessing.Process(target=f, args=(1,)).start(), lambda: list(concurrent.futures.ProcessPoolExecutor().map(f, [1])))",
    "pty": "attempt(lambda: pty.spawn(['rm', OUT + '/victim.txt']))",
    "asyncio": "import asyncio\n"
               "async def run_async(): await asyncio.create_subprocess_shell('rm ' + OUT + '/victim.txt')\n"
               "attempt(lambda: asyncio.run(run_async()))",
}

@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_replay_cannot_spawn_processes(tmp_path, outside, family):
    before = tree(outside)
    path = program(tmp_path, FAMILIES[family], preamble=f"OUT = {str(outside)!r}\n")
    out = run_child(path)
    print(out.msg)
    assert out.status == "clean", out.msg
    assert tree(outside) == before
    assert {"file_effect", "file_unmodelled"} <= set(out.flags)

def test_the_positive_control_allows_harmless_work_in_scratch(tmp_path):
    path = program(tmp_path, """
        open('harmless.txt', 'w').write('ok')
        assert os.path.exists('harmless.txt')
    """)
    out = run_child(path)
    print(out.msg)
    assert out.status == "clean", out.msg

@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_the_hybrid_worker_cannot_spawn_processes_either(tmp_path, outside):
    before = tree(outside)
    body = "s = _sv_verifier.nondet_str()\nassert len(s) >= 0\n" + "\n".join(FAMILIES.values())
    path = program(tmp_path, body, preamble=f"import _sv_verifier\nOUT = {str(outside)!r}\n")
    verify(path, ("AssertionError",), jobs=1, budget_seconds=9, prove=False, concolic=False, stages=["hybrid"])
    assert tree(outside) == before


def test_popen_is_still_a_class_and_isinstance_works(tmp_path, outside):
    path = program(tmp_path, '''
        import subprocess
        class MyPopen(subprocess.Popen): pass
        assert issubclass(MyPopen, subprocess.Popen)
        
        try:
            p = subprocess.Popen(['ls'])
        except OSError:
            pass
        
        try:
            subprocess.Popen.__new__(subprocess.Popen)
        except OSError:
            pass
    ''', preamble=f"OUT = {str(outside)!r}\n")
    out = run_child(path)
    print(out.msg)
    assert out.status == "clean", out.msg

def test_restoring_spawn_confinement_works():
    import sys
    import tharkun.solve.environment as env
    
    # We must explicitly import the modules so they are in sys.modules
    import os
    import subprocess
    import _posixsubprocess
    try:
        import posix
    except ImportError:
        pass
    try:
        import _winapi
    except ImportError:
        pass
    
    snapshots = {}
    for owner_path, attr_name in env.SPAWN_CONFINEMENT_TARGETS:
        parts = owner_path.split(".")
        if parts[0] not in sys.modules:
            continue
        owner = sys.modules[parts[0]]
        for p in parts[1:]:
            owner = getattr(owner, p, None)
        if hasattr(owner, attr_name):
            snapshots[(owner_path, attr_name)] = getattr(owner, attr_name)
    
    flags = set()
    env.install_spawn_confinement(flags)
    
    for (owner_path, attr_name), original in snapshots.items():
        parts = owner_path.split(".")
        owner = sys.modules[parts[0]]
        for p in parts[1:]:
            owner = getattr(owner, p)
            
        patched = getattr(owner, attr_name)
        assert patched is not original, f"{owner_path}.{attr_name} was not patched!"
        # ensure the name is preserved
        assert patched.__name__ == original.__name__ or patched.__name__ == attr_name
        
        setattr(owner, attr_name, original)
        assert getattr(owner, attr_name) is original
