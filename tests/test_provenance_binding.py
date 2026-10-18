"""Python's binder is the oracle; provenance must cover every accepted binding."""
import ast
import inspect

from hypothesis import given, settings, strategies as st
import pytest

from tharkun.solve.determinism import determinism_census
from tharkun.solve.provenance import Provenance, UNKNOWN
from tharkun.solve.pyresolve import Resolver
from tests.test_address_provenance import renamed


BINDING_CONTROLS = [
    ("def f(**kw): return kw['x']\nvalue = f(**{'x': object()}, **{'y': 1})", False),
    ("def f(x=object(), /, **kw): return x\nvalue = f(x=1)", False),
    ("def f(x=object(), /, **kw): return x\nvalue = f(**{'x': 1})", False),
    ("def f(*items): return items[1]\nvalue = f(*(1,), *(object(),))", False),
    ("def f(x, /, **kw): return kw['x']\nvalue = f(1, x=object())", False),
    ("def f(x=1, /, **kw): return kw['x']\nvalue = f(x=object())", False),
    ("def f(x=1, /, **kw): return x\nvalue = f(x=2)", True),
    ("def f(**kw): return kw['x']\nvalue = f(**{'x': 3}, **{'y': 1})", True),
    ("def f(a, b): return 0\ndef values(): return (1, 2)\nf(*values())\nvalue = 0", True),
    ("def f(**kw): return 0\ndef values(): return {'item': 1}\nvalue = f(**values())", True),
]


@pytest.mark.parametrize('source,safe', BINDING_CONTROLS)
def test_binding_and_renaming(source, safe):
    source += '\nprint(str(value) * 2)'
    for program in (source, renamed(source)):
        result = determinism_census(program)
        assert result.ok is safe, (program, result.reason)


@pytest.mark.parametrize('source', [s for s, safe in BINDING_CONTROLS[:2]])
def test_review_repro_never_true(tmp_path, source):
    from tharkun.sv.engine import verify

    path = tmp_path / 'binding.py'
    path.write_text(source + "\n[][0] if str(value)[-9] == 'a' else None")
    result = verify(str(path), ('IndexError',), budget_seconds=2, jobs=1, prove=False)
    assert result.verdict != 'true', result


TOKENS = ('0', "'safe'", 'object()', '[object()]', '(1, object())')


@st.composite
def binding_cases(draw):
    posonly = draw(st.integers(0, 2))
    normal = draw(st.integers(0, 2))
    positional = [f'p{i}' for i in range(posonly + normal)]
    defaults = draw(st.integers(0, len(positional)))
    pieces = []
    for i, name in enumerate(positional):
        if i >= len(positional) - defaults:
            name += '=' + draw(st.sampled_from(TOKENS))
        pieces.append(name)
        if i + 1 == posonly:
            pieces.append('/')
    vararg = draw(st.booleans())
    keywords = [f'k{i}' for i in range(draw(st.integers(0, 2)))]
    if vararg:
        pieces.append('*rest')
    elif keywords:
        pieces.append('*')
    for name in keywords:
        if draw(st.booleans()):
            name += '=' + draw(st.sampled_from(TOKENS))
        pieces.append(name)
    if draw(st.booleans()):
        pieces.append('**extra')
    args = draw(st.lists(st.sampled_from(TOKENS), max_size=len(positional) + 2))
    kwargs = draw(st.dictionaries(st.sampled_from(positional + keywords + ['other']),
                                 st.sampled_from(TOKENS), max_size=5))
    call = []
    for arg in args:
        call.append('*(' + arg + ',)' if draw(st.booleans()) else arg)
    for key, value in kwargs.items():
        call.append('**{' + repr(key) + ': ' + value + '}' if draw(st.booleans())
                    else key + '=' + value)
    return ', '.join(pieces), args, kwargs, ', '.join(call)


@settings(max_examples=300, derandomize=True, deadline=None)
@given(binding_cases())
def test_binding_provenance_covers_signature_bind(case):
    signature, args, kwargs, call = case
    definition = f'def f({signature}): pass\n'
    namespace = {}
    exec(definition, namespace)
    try:
        bound = inspect.signature(namespace['f']).bind(
            *(eval(a) for a in args), **{k: eval(v) for k, v in kwargs.items()})
    except TypeError:
        return  # No function body runs for a rejected binding.
    bound.apply_defaults()
    tree = ast.parse(definition + f'f({call})')
    provenance = Provenance(Resolver(tree, ()))
    function = tree.body[0]

    def contains(values, actual):
        if UNKNOWN in values:
            return True
        if type(actual) in (int, str):
            return (type(actual).__name__,) in values
        if type(actual) in (tuple, list, dict):
            tag = 'dict' if isinstance(actual, dict) else 'seq'
            items = actual.values() if tag == 'dict' else actual
            return any(v[0] == tag and all(contains(provenance.get(('elem', v[1])), x)
                                          for x in items) for v in values)
        return False  # object() has no safe scalar representation.

    for name, actual in bound.arguments.items():
        values = provenance.get(('name', function, name))
        assert contains(values, actual), (definition, call, name, values, actual)


@pytest.mark.parametrize('call,protocol', [
    ('f(**Values())', "def keys(self):\n        global text\n        text = object()\n        return ['key']\n    def __getitem__(self, key):\n        return 1"),
    ('f(*Values())', "def __iter__(self):\n        global text\n        text = object()\n        return iter([1])"),
])
def test_expansion_effects_cannot_disappear_when_arguments_are_ignored(call, protocol):
    source = f"text = 1\nclass Values:\n    {protocol}\ndef f(*args, **kw): return 0\n{call}\nprint(text)"
    for program in (source, source.replace('Values', 'Entries').replace('text', 'message')
                    .replace('f(', 'consume(')):
        assert not determinism_census(program).ok


LITERAL_STAR_SAFE = [
    "def f(a, b): return '%s:%s' % (a, b)\nvalue = f(*[1, 'two'])",
    "def f(a, b): return '%s:%s' % (a, b)\ndef relay(items): return f(*items)\nrelay([1, 'two'])\nvalue = relay([3, 'four'])",
    "def f(a, b): return '%s:%s' % (a, b)\ndef relay(items): return f(*items)\nrelay((1, 'two'))\nvalue = relay([3, 'four'])",
    "def f(a): return a\nitems = [1]\nalias = items\nvalue = f(*alias)",
    "def f(a=object()): return a\nvalue = f(*[1])",
    "def f(a=1): return a\nvalue = f(*[])",
    "def f(a, /, b, *rest, flag=1, **kw): return (a, b, rest, flag, kw)\nvalue = f(*[1, 'two', 3], flag=4, extra=5)",
    "class Box:\n    def f(self, a, b): return '%s:%s' % (a, b)\nvalue = Box().f(*[1, 'two'])",
    "def f(a, b): return (a, b)\nvalue = f(*[[1, 2], (3, 4)])",
]

LITERAL_STAR_UNSAFE = [
    "def f(a): return a\nvalue = f(*[object()])",
    "def f(a): return '%s' % a\nvalue = f(*[object()])",
    "def f(a): return a\nx = 1\nvalue = f(*[x])",
    "def f(a): return a\nvalue = f(*list((1,)))",
    "def f(a): return a\ndef relay(items): return f(*items)\nrelay([1])\nvalue = relay(range(1))",
    "def f(a=object()): return a\ndef relay(items): return f(*items)\nrelay([1])\nvalue = relay([])",
    "def f(a): return a\nvalue = f(*[1, 2])",
    "def f(a): return a\nvalue = f(*[1], a=2)",
    "def f(a): return a\nitems = [1, 2]\nvalue = f(*items[:1])",
    "def f(a): return a\nitems = [1]\nvalue = f(*(items + []))",
    "def f(a): return a\nitems = [1]\nvalue = f(*(items * 1))",
    "def f(a): return a\nitems = [1]\nvalue = f(*items.copy())",
    "def f(a): return a\nitems = [1]\nalias = items\nalias[0] = 2\nvalue = f(*items)",
    "def f(*items): return items\nitems = [1]\nalias = items\nitems += [object()]\nvalue = f(*alias)",
    "def f(a): return a\nitems = [[1]]\nitems[0].append(2)\nvalue = f(*items)",
    "def f(a): return a\nitems = [{'x': 1}]\nitems[0]['x'] = 2\nvalue = f(*items)",
    "def f(a): return a\ndef mutate(items): items.reverse()\nitems = [1]\nmutate(items)\nvalue = f(*items)",
    "def f(a): return a\nitems = [1]\nmin([1], key=items.append)\nvalue = f(*items)",
]


@pytest.mark.parametrize('source,safe', [(s, True) for s in LITERAL_STAR_SAFE]
                         + [(s, False) for s in LITERAL_STAR_UNSAFE])
def test_literal_star_calls_keep_exact_binding_and_refuse_uncertain_carriers(source, safe):
    source += '\nprint(str(value) * 2)'
    for program in (source, renamed(source)):
        result = determinism_census(program)
        assert result.ok is safe, (program, result.reason)


@pytest.mark.parametrize('mutation', [
    'alias.append(2)', 'alias.extend([2])', 'alias.insert(0, 2)',
    'alias.reverse()', 'alias.sort()', 'alias.clear()', 'alias.pop()',
    'alias.remove(1)', 'alias[0] = 2', 'alias[:] = [2]', 'alias += [2]', 'alias *= 2',
    'min([1], key=alias.append)',
])
def test_same_type_mutation_invalidates_exact_star_slots(mutation):
    source = f'def consume(*items): return items\nvalues = [1]\nalias = values\n{mutation}\nvalue = consume(*values)\nprint(value)'
    tree = ast.parse(source)
    provenance = Provenance(Resolver(tree, ()))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == 'consume')
    assert provenance.arguments(call) is None
    assert not determinism_census(source).ok


def test_literal_star_binds_each_slot_without_unioning_positions():
    tree = ast.parse('def f(a, b): return a\ndef relay(items): return f(*items)\nrelay([1, "two"])\nrelay([3, "four"])')
    provenance = Provenance(Resolver(tree, ()))
    function = tree.body[0]
    assert provenance.get(('name', function, 'a')) == {('int',)}
    assert provenance.get(('name', function, 'b')) == {('str',)}


@pytest.mark.parametrize('call', ['f(*[1, 2])', 'f(*[1], a=2)'])
def test_exact_literal_star_keeps_python_arity_rejection(call):
    tree = ast.parse('def f(a): return a\n' + call)
    provenance = Provenance(Resolver(tree, ()))
    assert provenance.value(tree.body[-1].value) == set()
