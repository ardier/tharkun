"""Resource-truncated executions are neither completed leaves nor witnesses (.116)."""
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tharkun.sv import _child, engine, prove
from tharkun.sv.forkserver import ForkServer
from tharkun.sv.nondet_harness import DriveOutcome, drive


MODES = ('fresh', 'fork', 'drive', 'probe')


def program(tmp_path, source):
    path = tmp_path / 'target.py'
    path.write_text(textwrap.dedent(source).lstrip('\n'))
    return str(path)


def observe(mode, path, *, memory_bytes=None):
    if mode == 'drive':
        return drive(path, [])
    if mode == 'probe':
        row = prove.run_probe(sys.executable, path, [], [], [], timeout=30, memory_bytes=memory_bytes)
        return engine.Outcome(row['status'], row.get('exc'), row.get('mro', []), flags=row.get('flags', []))
    if mode == 'fork':
        server = ForkServer(sys.executable)
        try:
            out = server.run(sys.executable, path, [], timeout=30, memory_bytes=memory_bytes)
            assert 'forked' in out.flags, out
            return out
        finally:
            server.close()
    return engine.run_once(sys.executable, path, [], timeout=30, memory_bytes=memory_bytes)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('error', ['RecursionError', 'MemoryError'])
@pytest.mark.parametrize('tail', ['pass', 'raise AssertionError("after limit")'])
def test_caught_builtin_limits_remain_sticky(mode, error, tail, tmp_path):
    path = program(tmp_path, f'try:\n    raise {error}("limit")\nexcept {error}:\n    pass\n{tail}\n')
    out = observe(mode, path)
    assert out.status == ('clean' if tail == 'pass' else 'raised'), out
    assert 'resource_exhausted' in out.flags, out
    assert engine._resource_exhausted(out)
    assert out.violated(('AssertionError',)) is None


@pytest.mark.parametrize('mode', MODES)
def test_actual_caught_recursion_limit_is_not_a_witness(mode, tmp_path):
    path = program(tmp_path, 'def f():\n    f()\ntry:\n    f()\nexcept RecursionError:\n    pass\nassert False\n')
    out = observe(mode, path)
    assert out.exc == 'AssertionError', out
    assert 'resource_exhausted' in out.flags, out
    assert out.violated(('AssertionError',)) is None


@pytest.mark.parametrize('mode', ['fresh', 'fork', 'probe'])
def test_actual_caught_address_space_limit_is_not_a_witness(mode, tmp_path):
    path = program(tmp_path, 'try:\n    x = [0] * 10**12\nexcept MemoryError:\n    pass\nassert False\n')
    out = observe(mode, path, memory_bytes=256 * 2**20)
    assert out.exc == 'AssertionError' and 'resource_exhausted' in out.flags, out
    assert out.violated(('AssertionError',)) is None


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('error', ['RecursionError', 'MemoryError'])
def test_resource_subclasses_cannot_also_supply_a_property_witness(mode, error, tmp_path):
    path = program(tmp_path, f'class Limited({error}, AssertionError):\n    pass\nraise Limited()\n')
    out = observe(mode, path)
    assert error in out.mro and 'AssertionError' in out.mro, out
    assert 'resource_exhausted' in out.flags
    assert out.violated(('AssertionError',)) is None


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('name', ['RecursionError', 'MemoryError'])
def test_fake_resource_names_remain_real_property_exceptions(mode, name, tmp_path):
    path = program(tmp_path, f'class {name}(AssertionError):\n    pass\nraise {name}()\n')
    out = observe(mode, path)
    assert not engine._resource_exhausted(out), out
    assert out.violated(('AssertionError',)) == 'AssertionError'


@pytest.mark.parametrize('mode', ['fresh', 'drive'])
def test_observation_never_calls_an_exceptions_class_property(mode, tmp_path):
    path = program(tmp_path, '''
        class Ordinary(Exception):
            @property
            def __class__(self):
                raise AssertionError('observer invoked user code')
        try:
            raise Ordinary()
        except Ordinary:
            pass
    ''')
    out = observe(mode, path)
    assert out.status == 'clean' and not engine._resource_exhausted(out), out


@pytest.mark.parametrize('mode', ['fresh', 'drive'])
def test_observation_bypasses_metaclass_mro_descriptors(mode, tmp_path):
    out = observe(mode, program(tmp_path, '''
        class Meta(type):
            @property
            def __mro__(cls):
                raise AssertionError('observer invoked metaclass')
        class Ordinary(Exception, metaclass=Meta):
            pass
        try:
            raise Ordinary()
        except Ordinary:
            pass
    '''))
    assert out.status == 'clean' and not engine._resource_exhausted(out), out


@pytest.mark.parametrize('kind', [engine.Outcome, DriveOutcome])
@pytest.mark.parametrize('error', ['RecursionError', 'MemoryError'])
def test_terminal_fallback_uses_builtin_mro_not_display_name(kind, error):
    terminal = kind('raised', 'Derived', [error, 'AssertionError'])
    assert engine._resource_exhausted(terminal)
    assert terminal.violated(('AssertionError',)) is None
    fake = kind('raised', error, ['AssertionError', 'Exception', 'BaseException', 'object'])
    assert not engine._resource_exhausted(fake)
    assert fake.violated(('AssertionError',)) == 'AssertionError'


@pytest.mark.parametrize('source,classes', [
    ('def f(n):\n    assert n < 5000\n    f(n+1)\nf(0)\n', ('AssertionError',)),
    ('table = {i:i for i in range(3000)}\ndef f(n):\n    table[n]\n    f(n+1)\nf(0)\n', ('KeyError',)),
    ('def f():\n    f()\ntry:\n    f()\nexcept RecursionError:\n    pass\n', ('AssertionError',)),
    ('import _sv_verifier\nif _sv_verifier.nondet_bool():\n    raise RecursionError\n', ('AssertionError',)),
])
def test_truncated_deterministic_and_finite_tree_runs_are_unknown(tmp_path, source, classes):
    path = program(tmp_path, source)
    out = engine.verify(path, classes, prove=False, stages=[], concolic=False,
                        budget_seconds=60, run_timeout=30, max_runs=8, jobs=1)
    assert out.verdict == 'unknown', out


@pytest.mark.parametrize('row', [
    {'status': 'clean', 'flags': ['resource_exhausted']},
    {'status': 'raised', 'exc': 'AssertionError', 'mro': ['AssertionError'], 'flags': ['resource_exhausted']},
    {'status': 'raised', 'exc': 'Limited', 'mro': ['MemoryError', 'AssertionError']},
    {'status': 'raised', 'exc': 'Limited', 'mro': ['RecursionError', 'AssertionError']},
    {'status': 'clean', 'flags': ['resource_unobserved']},
])
def test_stationary_rejects_resource_runs_before_facts_or_replay(tmp_path, monkeypatch, row):
    from tharkun.knowledge import KnowledgeBase
    path = program(tmp_path, 'pass\n')
    row = dict(row, draws=[], instances=[{'loop': [1, 0], 'trip': 0}])
    monkeypatch.setattr(prove, 'run_probe', lambda *a, **k: row)
    monkeypatch.setattr(engine, 'run_once', lambda *a, **k: pytest.fail('resource probe must not be replayed'))
    knowledge = KnowledgeBase()
    result = prove._stationary(path, ('AssertionError',), SimpleNamespace(loops=[], draws=[], output_files=False),
                               python=sys.executable, remaining=lambda: 60, run_timeout=30,
                               memory_bytes=None, jobs=1, say=lambda s: None, knowledge=knowledge)
    assert result is None and list(knowledge) == []


@pytest.mark.parametrize('error', ['RecursionError', 'MemoryError'])
def test_independent_static_proof_survives_a_resource_cut_canary(tmp_path, monkeypatch, error):
    path = program(tmp_path, 'xs = [1]\nprint(xs[0])\n')
    monkeypatch.setattr(engine, 'run_once', lambda *a, **k: engine.Outcome('raised', error, [error]))
    out = engine.verify(path, ('IndexError',), budget_seconds=60, run_timeout=30, concolic=False, stages=[], jobs=1)
    assert out.verdict == 'true' and out.component.startswith('proof:'), out
    assert out.detail['canary']['cut'] is True


def test_missing_monitoring_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delattr(sys, 'monitoring')
    out = drive(program(tmp_path, 'assert False\n'), [])
    assert 'resource_unobserved' in out.flags and out.violated(('AssertionError',)) is None


def test_busy_monitoring_fails_closed_without_releasing_foreign_tools(tmp_path):
    m = sys.monitoring
    owned = []
    before = [m.get_tool(i) for i in range(6)]
    try:
        for i, owner in enumerate(before):
            if owner is None:
                m.use_tool_id(i, 'test-resource-busy')
                owned.append(i)
        during = [m.get_tool(i) for i in range(6)]
        out = drive(program(tmp_path, 'assert False\n'), [])
        assert 'resource_unobserved' in out.flags and out.violated(('AssertionError',)) is None
        assert [m.get_tool(i) for i in range(6)] == during
    finally:
        for i in owned:
            m.free_tool_id(i)
    assert [m.get_tool(i) for i in range(6)] == before


@pytest.mark.parametrize('name', ['register_callback', 'set_events'])
def test_partial_monitoring_setup_releases_its_slot(tmp_path, monkeypatch, name):
    m = sys.monitoring
    before = [m.get_tool(i) for i in range(6)]
    original = getattr(m, name)
    failed = []
    def fail_once(*args):
        if not failed:
            failed.append(True)
            raise RuntimeError('setup refused')
        return original(*args)
    monkeypatch.setattr(m, name, fail_once)
    out = drive(program(tmp_path, 'assert False\n'), [])
    assert failed and 'resource_unobserved' in out.flags
    assert out.violated(('AssertionError',)) is None
    assert [m.get_tool(i) for i in range(6)] == before


@pytest.mark.parametrize('change', ['mask', 'handled_callback', 'unwind_callback', 'owner'])
def test_observer_integrity_checks_mask_callbacks_and_owner(monkeypatch, change):
    m = sys.monitoring
    before = [m.get_tool(i) for i in range(6)]
    monkeypatch.setattr(_child._State, 'flags', set())
    restore = _child._watch_resources()
    tool = next(i for i in range(6) if m.get_tool(i) == 'tharkun.sv.resources')
    try:
        assert m.get_events(tool) == m.events.EXCEPTION_HANDLED | m.events.PY_UNWIND
        if change == 'mask':
            m.set_events(tool, 0)
        elif change.endswith('_callback'):
            event = m.events.EXCEPTION_HANDLED if change == 'handled_callback' else m.events.PY_UNWIND
            m.register_callback(tool, event, lambda *args: None)
        else:
            m.set_events(tool, 0)
            m.free_tool_id(tool)
            m.use_tool_id(tool, 'replacement-owner')
        restore()
        assert 'resource_unobserved' in _child._State.flags
        if change == 'owner':
            assert m.get_tool(tool) == 'replacement-owner'
        else:
            assert m.get_tool(tool) is None
    finally:
        if m.get_tool(tool) in ('tharkun.sv.resources', 'replacement-owner'):
            m.set_events(tool, 0)
            for event in (m.events.EXCEPTION_HANDLED, m.events.PY_UNWIND):
                m.register_callback(tool, event, None)
            m.free_tool_id(tool)
    assert [m.get_tool(i) for i in range(6)] == before


def test_monitor_and_line_tracer_are_preserved_and_flags_do_not_leak(tmp_path):
    m = sys.monitoring
    tool = next(i for i in range(6) if m.get_tool(i) is None)
    seen = []
    def on_line(code, line):
        if code.co_filename.endswith('target.py'):
            seen.append(line)
    def tracer(frame, event, arg):
        return tracer
    old_trace = sys.gettrace()
    m.use_tool_id(tool, 'foreign-resource-test')
    m.register_callback(tool, m.events.LINE, on_line)
    m.set_events(tool, m.events.LINE)
    methods = tuple(getattr(m, name) for name in ('set_events', 'register_callback', 'get_tool'))
    limit = sys.getrecursionlimit()
    try:
        sys.settrace(tracer)
        bad = drive(program(tmp_path, 'try:\n    raise MemoryError\nexcept MemoryError:\n    pass\nx = 1\n'), [], target_line=5)
        assert bad.reached and 'resource_exhausted' in bad.flags
        assert sys.gettrace() is tracer and sys.getrecursionlimit() == limit
        clean = drive(program(tmp_path, 'x = 1\n'), [], target_line=1)
        assert clean.reached and clean.flags == []
        assert sys.gettrace() is tracer
        assert seen and m.get_tool(tool) == 'foreign-resource-test'
        assert m.get_events(tool) == m.events.LINE
        assert tuple(getattr(m, name) for name in ('set_events', 'register_callback', 'get_tool')) == methods
    finally:
        sys.settrace(old_trace)
        m.set_events(tool, 0)
        m.register_callback(tool, m.events.LINE, None)
        m.free_tool_id(tool)


@pytest.mark.parametrize('mode', ['fresh', 'drive'])
def test_native_monitoring_function_identity_is_preserved(mode, tmp_path):
    out = observe(mode, program(tmp_path, "import sys\nassert type(sys.monitoring.set_events).__name__ != 'function'\n"))
    assert out.status == 'clean' and 'import_state_unmodelled' in out.flags, out


@pytest.mark.parametrize('action', [
    'm.set_events(t, 0); m.set_events(t, mask)',
    'm.clear_tool_id(t)',
    'm.free_tool_id(t)',
    'm.register_callback(t, m.events.EXCEPTION_HANDLED, None)',
    'm.get_tool(t)',
    'import operator; operator.methodcaller("set_events", t, 0)(m); m.set_events(t, mask)',
    'import functools; functools.partial(m.set_events, t)(0); m.set_events(t, mask)',
    'import functools; p=functools.partial.__new__(functools.partial, m.set_events, t, 0); p(); m.set_events(t, mask)',
])
def test_monitor_disable_restore_and_state_reads_cannot_supply_a_witness(tmp_path, action):
    source = ('import sys\nm=sys.monitoring\nt=4\n'
              'mask=m.events.EXCEPTION_HANDLED|m.events.PY_UNWIND\n'
              + action + '\ntry:\n    raise RecursionError\nexcept RecursionError:\n    pass\nassert False\n')
    out = engine.run_once(sys.executable, program(tmp_path, source), [], timeout=30)
    assert 'import_state_unmodelled' in out.flags and out.violated(('AssertionError',)) is None, out


def test_imported_helper_monitoring_access_is_also_unusable(tmp_path):
    (tmp_path / 'helper.py').write_text('import sys\nimport functools\nm=sys.monitoring\n'
                                     'def interfere():\n'
                                     '    p=functools.partial.__new__(functools.partial, m.set_events, 4, 0)\n'
                                     '    p()\n'
                                     '    try: raise RecursionError\n'
                                     '    except RecursionError: pass\n'
                                     '    m.set_events(4, m.events.EXCEPTION_HANDLED|m.events.PY_UNWIND)\n')
    path = program(tmp_path, 'import helper\nhelper.interfere()\nassert False\n')
    out = engine.run_once(sys.executable, path, [], timeout=30)
    assert 'import_state_unmodelled' in out.flags and out.violated(('AssertionError',)) is None, out


def test_foreign_callback_cannot_hide_a_caught_limit(tmp_path):
    out = engine.run_once(sys.executable, program(tmp_path, '''
        import sys
        m = sys.monitoring
        t = next(i for i in range(6) if m.get_tool(i) is None)
        def callback(code, offset, function, arg):
            try:
                raise MemoryError()
            except MemoryError:
                pass
        m.use_tool_id(t, 'target-callback')
        m.register_callback(t, m.events.CALL, callback)
        m.set_events(t, m.events.CALL)
        len([])
        assert False
    '''), [], timeout=30)
    assert 'import_state_unmodelled' in out.flags and out.violated(('AssertionError',)) is None, out
