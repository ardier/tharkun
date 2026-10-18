"""Concrete provenance checks for the narrow callback-contract surface."""
import math
import random

import pytest

from tests.contracts.test_math_module import VALID
from tharkun.solve.contracts import closed_call_hooks
from tharkun.solve.contracts.reachability import _CLOSED


CALLS = {
    'abs': 'abs(-1)', 'divmod': 'divmod(5, 2)', 'float': 'float(1)',
    'int': 'int(1.0)', 'len': 'len([])', 'list': 'list(())', 'range': 'range(2)',
    'repr': 'repr([])', 'set': 'set([1])', 'math.log': 'math.log(1)',
    'math.sqrt': 'math.sqrt(1)', 'random.random': 'random.random()',
    'random.randrange': 'random.randrange(1)', 'random.seed': 'random.seed(1)',
    'list.append': '[].append(1)', 'list.clear': '[1].clear()',
    'list.copy': '[1].copy()', 'list.pop': '[1].pop()', 'list.remove': '[1].remove(1)',
    'dict.clear': '{1: 2}.clear()', 'dict.copy': '{1: 2}.copy()',
    'dict.pop': '{1: 2}.pop(1)', 'dict': 'dict([(1, 2)])',
    'dict.update': '{}.update([(1, 2)])',
    'set.add': 'set().add(1)', 'set.clear': '{1}.clear()', 'set.copy': '{1}.copy()',
    'set.pop': '{1}.pop()', 'set.remove': '{1}.remove(1)', 'set.update': 'set().update([1])',
    'frozenset.copy': 'frozenset([1]).copy()',
    'str.join': "''.join(['a'])", 'str.find': "'a'.find('a')",
    'bytes.join': "b''.join([b'a'])", 'bytes.find': "b'a'.find(b'a')",
    'bytearray.append': 'bytearray().append(1)', 'bytearray.clear': 'bytearray([1]).clear()',
    'bytearray.copy': 'bytearray([1]).copy()', 'bytearray.pop': 'bytearray([1]).pop()',
    'bytearray.remove': 'bytearray([1]).remove(1)',
    'bytearray.join': "bytearray().join([b'a'])", 'bytearray.find': "bytearray(b'a').find(b'a')",
}
# [internal ref] batch 5: every function of math (a valid call each) and every constant
CALLS.update({f'math.{n}': f"math.{n}({', '.join(repr(a) for a in args)})" for n, args in VALID.items()})
CALLS.update({f'math.{n}': f'math.{n}' for n in ('pi', 'e', 'tau', 'inf', 'nan')})


@pytest.mark.parametrize('name', CALLS)
def test_native_call_returns_builtin_value(name):
    assert closed_call_hooks(name) is not None
    value = eval(CALLS[name], {'math': math, 'random': random})
    assert type(value) in (int, float, bool, str, bytes, bytearray, list, tuple, dict, set,
                           frozenset, range, type(None))


def test_every_native_row_is_exercised():
    # The verifier primitive has its independent SV input contract; it is not a
    # CPython callable and is covered by the resolver's nondet source fixture.
    assert set(CALLS) == (_CLOSED - {'_sv_verifier.nondet_int'}) | {'dict', 'dict.update', 'random.seed'}


def test_containers_can_return_source_objects():
    class C:
        pass
    x = C()
    assert [x].pop() is x
    assert {0: x}.pop(0) is x
    assert {x}.pop() is x


def test_mapping_conversion_reads_keys():
    class Mapping:
        def keys(self):
            raise AssertionError('keys reached')
    for call in (dict, {}.update):
        with pytest.raises(AssertionError, match='keys reached'):
            call(Mapping())
    assert closed_call_hooks('dict') == closed_call_hooks('dict.update') == {'keys'}
