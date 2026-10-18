"""Only unobservable output effects may be forgotten by a stationarity proof."""
import ast
from textwrap import dedent
import sys
import subprocess

import pytest

from tharkun.solve.drawflow import plan_draws
from tharkun.solve.output_effects import output_only
from tharkun.sv.prove import prove_dynamic


SAFE = [
    '''
    from _sv_verifier import nondet_int
    stream = open('/missing-parent/output.txt', 'w', encoding='utf-8')
    for unused in range(1 + abs(nondet_int())):
        print('fixed text', file=stream)
        stream.write('more')
    stream.close()
    assert 4 + 5 == 9
    ''',
    '''
    from _sv_verifier import nondet_int
    for unused in range(1 + abs(nondet_int())):
        with open('payload.bin', 'wb') as destination:
            destination.write(b'bytes')
    assert 3 * 7 == 21
    ''',
    '''
    from _sv_verifier import nondet_int
    destination = open('log.txt', 'a')
    for unused in range(1 + abs(nondet_int())):
        destination.write('append')
    destination.close()
    ''',
]


def prove_io(tmp_path, source, classes=('AssertionError',)):
    path = tmp_path / 'output.py'
    path.write_text(dedent(source))
    return prove_dynamic(str(path), classes, python=sys.executable, remaining=lambda: 15, jobs=1)


@pytest.mark.parametrize('source', SAFE)
def test_output_only_programs_and_renamings(tmp_path, source):
    source = dedent(source)
    for program in (source, source.replace('stream', 'sink').replace('destination', 'target')
                    .replace('unused', 'counter')):
        plan = plan_draws(program)
        assert plan.ok and plan.output_files, plan.reason
        verdict = prove_io(tmp_path, program)
        assert verdict is not None and verdict.verdict == 'true', verdict


@pytest.mark.parametrize('suffix', [
    "stream.read()", "stream.seek(0)", "value = stream.write('x')",
    "alias = stream", "print(stream)", "stream = open('other', 'w')",
    "with open('out.txt', 'r') as readback:\n    assert readback.read() == 'x'",
    "try:\n    stream.write('x')\nexcept OSError:\n    assert False",
    "def print(*args, **kwargs):\n    kwargs['file'].read()\nprint(file=stream)",
])
def test_readback_escape_or_observation_refuses(tmp_path, suffix):
    source = dedent(SAFE[0]) + '\n' + suffix
    for program in (source, source.replace('stream', 'sink').replace('unused', 'counter')):
        with pytest.raises(ValueError):
            output_only(ast.parse(program))
        assert prove_io(tmp_path, program) is None


def test_environment_exception_property_refuses(tmp_path):
    assert prove_io(tmp_path, SAFE[0], ('OSError',)) is None
    assert prove_io(tmp_path, SAFE[0], ('Exception',)) is None


def test_successful_output_does_not_hide_later_violation(tmp_path):
    # The literal missing parent normally stops before the assertion. The proof
    # must also consider successful writes, not certify from that failed open.
    source = dedent(SAFE[0]).replace('assert 4 + 5 == 9', 'assert False')
    result = prove_io(tmp_path, source)
    assert result is None or result.verdict != 'true'


@pytest.mark.parametrize('value', ['42', 'object()', "b'bytes'"])
def test_unsupported_write_value_never_certifies(tmp_path, value):
    source = dedent(SAFE[0]).replace("stream.write('more')", f'stream.write({value})')
    assert prove_io(tmp_path, source, ('TypeError',)) is None


def test_custom_path_protocol_never_certifies(tmp_path):
    source = '''
class Path:
    def __fspath__(self):
        return 1 / 0
''' + dedent(SAFE[0]).replace("'/missing-parent/output.txt'", 'Path()')
    assert prove_io(tmp_path, source, ('ZeroDivisionError',)) is None


@pytest.mark.parametrize('load', ['import string', 'from string import ascii_letters'])
def test_importing_written_output_never_certifies(tmp_path, load):
    (tmp_path / '_sv_verifier.py').write_text('def nondet_int(): return 0\n')
    source = f"""
from _sv_verifier import nondet_int
stream = open({str(tmp_path / 'string.py')!r}, 'w')
stream.write('raise AssertionError("import observed output")\\n')
stream.close()
{load}
for unused in range(1 + abs(nondet_int())):
    pass
"""
    for program in (source, source.replace('stream', 'sink').replace('unused', 'counter')):
        with pytest.raises(ValueError, match='import'):
            output_only(ast.parse(program))
        assert prove_io(tmp_path, program) is None
        native = subprocess.run([sys.executable, str(tmp_path / 'output.py')],
                                capture_output=True, text=True)
        assert native.returncode != 0 and 'AssertionError: import observed output' in native.stderr


def test_nested_import_after_output_refuses(tmp_path):
    source = dedent(SAFE[0]) + '\ndef load():\n    import string\nload()\n'
    assert not plan_draws(source).ok
    assert prove_io(tmp_path, source) is None


@pytest.mark.parametrize('membership', ['in', 'not in'])
def test_output_model_preserves_module_names(tmp_path, membership):
    (tmp_path / '_sv_verifier.py').write_text('def nondet_int(): return 0\n')
    source = f"""
from _sv_verifier import nondet_int
stream = open({str(tmp_path / 'out.txt')!r}, 'w')
for unused in range(1 + abs(nondet_int())):
    stream.write('x')
stream.close()
assert 'open' {membership} dir()
"""
    for program in (source, source.replace('stream', 'sink').replace('unused', 'counter')):
        result = prove_io(tmp_path, program)
        native = subprocess.run([sys.executable, str(tmp_path / 'output.py')],
                                capture_output=True, text=True)
        if membership == 'not in':
            assert native.returncode == 0
            assert result is not None and result.verdict == 'true'
        else:
            assert native.returncode != 0 and 'AssertionError' in native.stderr
            assert result is None or result.verdict != 'true'
