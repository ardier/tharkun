"""Fork isolation and the fresh-process verdict boundary ([internal ref])."""
import os
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from tharkun.sv import engine
from tharkun.sv.forkserver import ForkServer

pytestmark = pytest.mark.skipif(sys.platform != 'linux',
                                reason='Linux fork server')


@pytest.fixture
def server():
    s = ForkServer(sys.executable)
    yield s
    s.close()


def run(server, tmp_path, source, vector=(), **kwargs):
    p = tmp_path / 'target.py'
    p.write_text(textwrap.dedent(source))
    return server.run(sys.executable, str(p), list(vector), timeout=kwargs.pop('timeout', 30), **kwargs)


def test_fds_are_closed_including_server_ipc(server, tmp_path):
    with open(tmp_path / 'parent-file', 'w') as fh:
        os.set_inheritable(fh.fileno(), True)
        out = run(server, tmp_path, '''
            import os
            for name in os.listdir('/proc/self/fd'):
                fd = int(name)
                if fd <= 2:
                    continue
                try:
                    os.fstat(fd)
                except OSError:
                    continue
                # CPython may open a new /dev/urandom FD when reseeding after closure.
                assert os.readlink('/proc/self/fd/' + name) == '/dev/urandom'
            os.close(0)
            os.close(1)
            os.close(2)
        ''')
        assert out.status == 'clean', out
        fh.write('still open')
    assert run(server, tmp_path, 'pass').status == 'clean'


def test_fresh_cwd_temp_and_environment(server, tmp_path):
    marker = tmp_path / 'seen'
    source = f'''
        import os, tempfile
        cwd = os.getcwd()
        assert tempfile.gettempdir() == cwd == os.environ['TMPDIR']
        assert os.environ['TMP'] == os.environ['TEMP'] == cwd
        assert not os.path.exists('dirty')
        open('dirty', 'w').close()
        with open({str(marker)!r}, 'a') as fh:
            fh.write(cwd + '\\n')
        tempfile.tempdir = '/'
        os.environ['TMPDIR'] = '/'
        os.chdir('/')
    '''
    for _ in range(2):
        assert run(server, tmp_path, source).status == 'clean'
    dirs = marker.read_text().splitlines()
    assert len(set(dirs)) == 2 and all(not os.path.exists(d) for d in dirs)


def test_random_state_and_model_are_new_per_child(server, tmp_path):
    marker = tmp_path / 'states'
    source = f'''
        import random
        from _sv_verifier import nondet_int
        with open({str(marker)!r}, 'a') as fh:
            fh.write(repr(random.getstate()) + '\\n')
        assert random.random() == 0.5
        random.seed(42)
        assert random.random() == random.Random(42).random()
        random.random = lambda: -123
        nondet_int()
    '''
    vec = [{'kind': 'rfloat', 'value': '0x1.0000000000000p-1'}, {'kind': 'int', 'value': 7}]
    for _ in range(2):
        out = run(server, tmp_path, source, vec)
        assert out.status == 'clean' and len(out.draws) == 2, out
    assert len(set(marker.read_text().splitlines())) == 2


def test_hashseed_is_fixed_at_exec_and_other_seed_uses_fresh(server, tmp_path):
    marker = tmp_path / 'hashes'
    source = f'''
        import os
        with open({str(marker)!r}, 'a') as fh:
            fh.write(os.environ['PYTHONHASHSEED'] + ':' + str(hash('fork isolation')) + '\\n')
        os.environ['PYTHONHASHSEED'] = '12345'
    '''
    for seed in (0, 0, 1):
        assert run(server, tmp_path, source, hashseed=seed).status == 'clean'
    a, b, c = marker.read_text().splitlines()
    assert a == b and a.startswith('0:') and c.startswith('1:') and a[2:] != c[2:]


def test_signals_timers_and_masks_do_not_leak(server, tmp_path):
    source = '''
        import signal
        assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        assert not signal.pthread_sigmask(signal.SIG_BLOCK, [])
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL, 60)
        signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGUSR1])
    '''
    for _ in range(2):
        assert run(server, tmp_path, source).status == 'clean'


def test_exec_server_is_safe_with_parent_threads(server, tmp_path):
    ready, stop = threading.Event(), threading.Event()
    def work():
        ready.set()
        stop.wait()
    t = threading.Thread(target=work)
    t.start()
    ready.wait()
    try:
        assert run(server, tmp_path, 'import threading; assert threading.active_count() == 1').status == 'clean'
    finally:
        stop.set()
        t.join()


def test_server_refuses_fork_even_after_a_thread_joined(tmp_path, monkeypatch):
    # Load the worker in a disposable interpreter so its thread guard cannot affect pytest.
    import subprocess
    worker_dir = str(Path(engine.CHILD).parent)
    code = f'''
import sys
sys.path.insert(0, {worker_dir!r})
import _forkserver as s
assert s._single_threaded()
t = s.threading.Thread(target=lambda: None)
t.start()
t.join()
assert not s._single_threaded()
s.os.fork = lambda: (_ for _ in ()).throw(AssertionError('must not fork'))
assert s._execute({{}}, {{}})['status'] == 'crashed'
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


def test_child_threads_do_not_contaminate_server(server, tmp_path):
    assert run(server, tmp_path, '''
        import threading
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
    ''').status == 'clean'
    assert run(server, tmp_path, 'pass').status == 'clean'


def test_rlimits_memory_and_cpu_are_per_child(server, tmp_path):
    limit = 128 * 2**20
    out = run(server, tmp_path, f'''
        import resource
        assert resource.getrlimit(resource.RLIMIT_AS) == ({limit}, {limit})
        assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
        assert resource.getrlimit(resource.RLIMIT_CPU) == (3, 3)
        b = bytearray(512 * 2**20)
    ''', memory_bytes=limit, timeout=3)
    assert out.exc == 'MemoryError' and engine._resource_exhausted(out)
    assert run(server, tmp_path, 'assert len(bytearray(150 * 2**20)) > 0',
               memory_bytes=256 * 2**20).status == 'clean'


@pytest.mark.parametrize('source,status,timeout', [
    # a crash or an exit is reported when the child ends, so its timeout is only a ceiling (30 s): under load a 0.2 s timeout
    # turned a SIGSEGV into 'timeout'.  Only the child that never ends needs a short one: it times out whatever the load.
    ('import os, signal; os.kill(os.getpid(), signal.SIGSEGV)', 'crashed', 30),
    ('import os; os._exit(0)', 'crashed', 30),
    ('import os; os._exit(9)', 'crashed', 30),
    ('import sys; sys.exit(7)', 'exit', 30),
    ('while True: pass', 'timeout', 0.2),
])
def test_crashes_exits_and_timeout_are_not_verdicts(server, tmp_path, source, status, timeout):
    out = run(server, tmp_path, source, timeout=timeout)
    assert out.status == status and out.violated(('BaseException',)) is None
    assert run(server, tmp_path, 'pass').status == 'clean'


def test_timeout_kills_descendants(server, tmp_path):
    # Startup consumes the allowance; this test needs the target to spawn its descendant.
    assert run(server, tmp_path, 'pass', timeout=30).status == 'clean'
    pidfile = tmp_path / 'pid'
    out = run(server, tmp_path, f'''
        import subprocess, sys
        p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        open({str(pidfile)!r}, 'w').write(str(p.pid))
        while True: pass
    ''', timeout=0.4)
    assert out.status == 'timeout'
    pid = int(pidfile.read_text())
    path = Path(f'/proc/{pid}/stat')
    assert not path.exists() or path.read_text().split()[2] == 'Z'


def test_live_server_reports_child_cpu(server, tmp_path):
    assert run(server, tmp_path, 'sum(range(2000000))').status == 'clean'
    before = server.cpu
    assert run(server, tmp_path, 'while True: pass', timeout=0.2).status == 'timeout'
    assert server.cpu > before + 0.03


def test_fork_startup_is_deducted_from_the_child_allowance(server, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tharkun.sv import forkserver
    clock = [0.0]
    start, dumps = server._start, forkserver.json.dumps
    requests = []
    def delayed_start():
        start()
        clock[0] += 0.6
    def capture(value, *args, **kwargs):
        if isinstance(value, dict) and "timeout" in value:
            requests.append(value)
        return dumps(value, *args, **kwargs)
    monkeypatch.setattr(server, "_start", delayed_start)
    monkeypatch.setattr(forkserver, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(forkserver.json, "dumps", capture)
    # Check exact subtraction without giving the real child a subsecond deadline.
    assert run(server, tmp_path, "pass", timeout=20).status == "clean"
    assert len(requests) == 1 and requests[0]["timeout"] == pytest.approx(19.4)
    assert server.cpu > 0


def test_exhausted_startup_never_executes_the_target(server, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tharkun.sv import forkserver
    clock = [0.0]
    start = server._start
    def delayed_start():
        start()
        clock[0] = 2.0
    monkeypatch.setattr(server, "_start", delayed_start)
    monkeypatch.setattr(forkserver, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    marker = tmp_path / "executed"
    out = run(server, tmp_path, f"open({str(marker)!r}, 'w').close()", timeout=1)
    assert out.status == "timeout" and not marker.exists()
    assert server.proc is None


def test_a_failed_request_write_still_drains_started_children(server, tmp_path, monkeypatch):
    import time
    assert run(server, tmp_path, "pass").status == "clean"
    original = server.stream
    marker = tmp_path / "grandchild"
    class FailedFlush:
        def write(self, data):
            return original.write(data)
        def flush(self):
            original.flush()
            deadline = time.monotonic() + 30
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            assert marker.exists(), "the request must have started its target"
            raise OSError("failed after sending")
        def close(self):
            original.close()
    monkeypatch.setattr(server, "stream", FailedFlush())
    out = run(server, tmp_path, f"""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                 start_new_session=True)
        open({str(marker)!r}, 'w').write(str(child.pid))
        while True: pass
    """, timeout=0.5)
    assert out.status == "crashed" and server.proc is None
    pid = int(marker.read_text())
    path = Path(f"/proc/{pid}/stat")
    assert not path.exists() or path.read_text().split()[2] == "Z"


def test_true_supporting_runs_never_fork(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('a TRUE-supporting run tried to fork')
    monkeypatch.setattr(ForkServer, 'run', forbidden)
    for source in ('x = 1', '''
        from _sv_verifier import nondet_bool
        x = nondet_bool()
        assert x in (True, False)
    '''):
        p = tmp_path / 'true.py'
        p.write_text(textwrap.dedent(source))
        v = engine.verify(str(p), ('IndexError',), prove=False, concolic=False, budget_seconds=60)
        assert v.verdict == 'true'


def test_every_false_is_replayed_and_rejected_forks_cannot_report_false(tmp_path, monkeypatch):
    p = tmp_path / 'false.py'
    p.write_text('from _sv_verifier import nondet_int\nx=nondet_int()\nif x: raise IndexError()\n')
    original = engine.run_once
    calls = []
    def fresh(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, 'run_once', fresh)
    v = engine.verify(str(p), ('IndexError',), prove=False, concolic=False, jobs=1, budget_seconds=60, lazy_vectors=0)
    assert v.verdict == 'false' and calls[-1] == v.witness['vector'] and len(calls) == 3
    p.write_text('from _sv_verifier import nondet_int\nx=nondet_int()\n')
    monkeypatch.setattr(ForkServer, 'run', lambda *a, **k: engine.Outcome(
        'raised', 'IndexError', ['IndexError'], draws=[{'kind': 'int', 'value': 1}], flags=['forked']))
    v = engine.verify(str(p), ('IndexError',), prove=False, concolic=False, jobs=1, budget_seconds=60, lazy_vectors=0)
    assert v.verdict == 'unknown' and 'fresh replay' in v.reason


def test_child_crashes_and_memory_exhaustion_cannot_prove_true(tmp_path):
    p = tmp_path / 'crash.py'
    p.write_text('from _sv_verifier import nondet_int\nimport os\nif nondet_int(): os._exit(0)\n')
    v = engine.verify(str(p), ('IndexError',), prove=False, concolic=False,
                      budget_seconds=60, max_runs=8)
    assert v.verdict == 'unknown'


def test_proof_canaries_stay_fresh(tmp_path, monkeypatch):
    p = tmp_path / 'canary.py'
    p.write_text('from _sv_verifier import nondet_int\nx=nondet_int()\n')
    monkeypatch.setattr(engine, '_static_proof', lambda *a, **k: engine.Verdict('true'))
    monkeypatch.setattr(ForkServer, 'run', lambda *a, **k: pytest.fail('forked proof canary'))
    v = engine.verify(str(p), ('IndexError',), concolic=False, max_runs=5, budget_seconds=60)
    assert v.verdict == 'true'


@pytest.mark.parametrize('component', ['concolic:drive', 'hypothesis'])
def test_portfolio_draw_vectors_fork_then_replay(component, tmp_path, monkeypatch):
    from tharkun.sv import stages
    p = tmp_path / 'portfolio.py'
    p.write_text('from _sv_verifier import nondet_int\nif nondet_int() == 123: raise IndexError()\n')
    vec = [{'kind': 'int', 'value': 123}]
    stage = stages.Stage(component, component,
                         propose=lambda ctx, budget: [stages.Candidate(vec, component)], share=0.4)
    monkeypatch.setattr(stages, 'select', lambda *a, **k: [stage])
    calls = []
    original = ForkServer.run
    def explore(self, *args, **kwargs):
        calls.append(args[2])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(ForkServer, 'run', explore)
    v = engine.verify(str(p), ('IndexError',), prove=False, jobs=1, budget_seconds=60, lazy_vectors=0)
    assert calls == [vec] and v.verdict == 'false' and v.component == component
    assert v.runs == 4  # initial fresh, exploration, fresh confirmation, validator replay


def test_portfolio_draw_vectors_lazy_fresh_by_default(tmp_path, monkeypatch):
    from tharkun.sv import stages
    p = tmp_path / 'portfolio.py'
    p.write_text('from _sv_verifier import nondet_int\nif nondet_int() == 123: raise IndexError()\n')
    vec = [{'kind': 'int', 'value': 123}]
    stage = stages.Stage('hypothesis', 'hypothesis',
                         propose=lambda ctx, budget: [stages.Candidate(vec, 'hypothesis')], share=0.4)
    monkeypatch.setattr(stages, 'select', lambda *a, **k: [stage])
    monkeypatch.setattr(ForkServer, 'run', lambda *a, **k: pytest.fail('forked during lazy window'))
    v = engine.verify(str(p), ('IndexError',), prove=False, jobs=1, budget_seconds=60)
    assert v.verdict == 'false' and v.runs == 3


def test_fresh_discovery_still_needs_validator_replay(tmp_path, monkeypatch):
    p = tmp_path / 'fresh.py'
    p.write_text('raise IndexError()\n')
    original = engine.run_once
    calls = []
    def fresh(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, 'run_once', fresh)
    v = engine.verify(str(p), ('IndexError',), prove=False, concolic=False, budget_seconds=60)
    assert v.verdict == 'false' and calls == [[], []]


def test_unencodable_deep_vector_is_inconclusive(server, tmp_path):
    # Faster search reaches CPython's decimal-digit JSON limit within a task budget.
    vector = [{'kind': 'int', 'value': 10**5000}]
    out = run(server, tmp_path, 'pass', vector)
    assert out.status == 'crashed'
    p = tmp_path / 'target.py'
    out = engine.run_once(sys.executable, str(p), vector, timeout=30)
    assert out.status == 'crashed'
    assert run(server, tmp_path, 'pass').status == 'clean'


def test_site_startup_threads_run_only_after_fork(tmp_path):
    import venv
    env = tmp_path / 'venv'
    venv.EnvBuilder(with_pip=False).create(env)
    site_dir = env / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    marker = tmp_path / 'site-pids'
    (site_dir / 'sitecustomize.py').write_text(
        "import os, threading\n"
        "t = threading.Thread(target=lambda: None)\nt.start()\nt.join()\n"
        f"with open({str(marker)!r}, 'a') as f: f.write(str(os.getpid()) + '\\n')\n")
    server = ForkServer(str(env / 'bin/python'))
    try:
        for _ in range(2):
            out = run(server, tmp_path, f'''
                import os
                pids = open({str(marker)!r}).read().splitlines()
                assert str(os.getpid()) in pids
                assert str(os.getppid()) not in pids
            ''')
            assert out.status == 'clean', out
        assert len(marker.read_text().splitlines()) == 2
    finally:
        server.close()


def test_detached_grandchild_is_killed_and_reaped(server, tmp_path):
    marker = tmp_path / 'grandchild_pid'
    source = f'''
import subprocess, sys
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
open({str(marker)!r}, 'w').write(str(p.pid))
'''
    out = run(server, tmp_path, source)
    assert out.status == 'clean'
    gc_pid = int(marker.read_text())
    assert not Path(f'/proc/{gc_pid}').exists()


def test_detached_grandchild_killed_on_timeout(server, tmp_path):
    # Startup consumes the allowance; this test needs the target to spawn its grandchild.
    assert run(server, tmp_path, 'pass', timeout=30).status == 'clean'
    marker = tmp_path / 'grandchild_pid'
    source = f'''
import subprocess, sys
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
open({str(marker)!r}, 'w').write(str(p.pid))
while True: pass
'''
    out = run(server, tmp_path, source, timeout=5)
    assert out.status == 'timeout'
    gc_pid = int(marker.read_text())
    assert not Path(f'/proc/{gc_pid}').exists()


def test_atexit_handlers_run_in_fork_children(server, tmp_path):
    marker = tmp_path / 'atexit_ran'
    source = f'''
import atexit
atexit.register(lambda: open({str(marker)!r}, 'w').write('done'))
'''
    out = run(server, tmp_path, source)
    assert out.status == 'clean'
    assert marker.exists() and marker.read_text() == 'done'


def test_atexit_exception_cannot_influence_verdict(server, tmp_path):
    source = '''
import atexit
atexit.register(lambda: (_ for _ in ()).throw(IndexError('from atexit')))
'''
    out = run(server, tmp_path, source)
    assert out.status == 'clean'
    assert out.violated(('IndexError',)) is None


@pytest.mark.parametrize('lazy_seconds, fork_values', [
    (float('inf'), [3, 4]),
    (0, [0, 1, 2, 3, 4]),
], ids=['vector-window', 'zero-wall-window'])
def test_lazy_start_switches_to_fork_after_n_vectors(tmp_path, monkeypatch, lazy_seconds, fork_values):
    from tharkun.sv import stages
    p = tmp_path / 'multi.py'
    p.write_text('from _sv_verifier import nondet_int\nx = nondet_int()\n')
    cands = [stages.Candidate([{'kind': 'int', 'value': i}], 'hypothesis') for i in range(5)]
    stage = stages.Stage('hypothesis', 'hypothesis', propose=lambda ctx, budget: cands, share=0.5)
    monkeypatch.setattr(stages, 'select', lambda *a, **k: [stage])
    fork_calls = []
    original_fork = ForkServer.run
    def recorded_fork(self, *args, **kwargs):
        fork_calls.append(args[2])
        return original_fork(self, *args, **kwargs)
    monkeypatch.setattr(ForkServer, 'run', recorded_fork)

    # Isolate the vector window from the default two-second wall window; a zero
    # wall window switches immediately. Keep real child execution and deadlines.
    v = engine.verify(str(p), ('IndexError',), prove=False, jobs=1, max_runs=6,
                      budget_seconds=60, run_timeout=30, cpu_time_only=True,
                      lazy_vectors=3, lazy_seconds=lazy_seconds)
    assert fork_calls == [[{'kind': 'int', 'value': i}] for i in fork_values]
    assert v.verdict == 'unknown' and v.runs == 6
