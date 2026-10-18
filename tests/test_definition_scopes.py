"""Definition headers use the enclosing scope, even when the body is unused.

Comprehensions/genexps open their OWN scope (PEP 289), including when they sit inside a
definition header that itself resolves to the enclosing scope: a comprehension's ``for``
target must not leak into that enclosing scope (or into a sibling header), and conversely a
name the comprehension itself binds must not be mistaken for an outer/builtin name -- except
for the first ``for``'s iterable, which genuinely runs before the comprehension's own frame
exists and so sees the enclosing scope.
"""
import ast
import os
import subprocess
import symtable
import sys
from textwrap import indent

import pytest

from tharkun.solve.census import site_census
from tharkun.solve.determinism import determinism_census
from tharkun.solve.pyresolve import Resolver, Top
from tharkun.sv.engine import verify

BAD = '1 / (hash("x") != 3914910683811788151)'
DECORATOR = '@(lambda f: f if hash("x") != 3914910683811788151 else 1 / 0)\ndef f(hash):\n    pass\n'

#: MG's reproducer ([internal ref]): a comprehension target in ONE default (`unused`'s) must not
#: shadow the builtin `hash` used by a LATER, sibling default (`value`'s). Native CPython
#: (3.11-3.14, siphash13) makes `run()` equal this exact literal at PYTHONHASHSEED=2.
COMPREHENSION_LEAK = '''def run():
    def unused(x=[0 for hash in ()]):
        pass
    def value(x=hash("x")):
        return x
    return value()
'''
COMPREHENSION_LEAK_TARGET = 3914910683811788151


def renamed(source):
    tree = ast.parse(source)
    names = {'f': 'renamed', 'outer': 'wrapper', 'Box': 'Crate', 'x': 'value'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node.name = names.get(node.name, node.name)
        elif isinstance(node, ast.Name):
            node.id = names.get(node.id, node.id)
        elif isinstance(node, ast.arg):
            node.arg = names.get(node.arg, node.arg)
    return ast.unparse(tree)


def native(path, seed):
    return subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONHASHSEED=str(seed)))


@pytest.mark.parametrize('context', ['module', 'function', 'class'])
def test_decorator_hash_lookup_is_not_a_parameter(tmp_path, context):
    source = DECORATOR
    if context == 'function':
        source = 'def outer():\n' + indent(source, '    ') + 'outer()\n'
    elif context == 'class':
        source = 'class Box:\n' + indent(source, '    ')
    for program in (source, renamed(source)):
        path = tmp_path / 'decorator.py'
        path.write_text(program)
        assert native(path, 0).returncode == native(path, 1).returncode == 0
        assert 'ZeroDivisionError' in native(path, 2).stderr
        assert not determinism_census(program).ok
        assert not site_census(program, ['ZeroDivisionError']).no_site
        result = verify(str(path), ['ZeroDivisionError'], python=sys.executable,
                        budget_seconds=60, run_timeout=30, max_runs=4, jobs=1)
        assert result.verdict != 'true', result


@pytest.mark.parametrize('source', [
    DECORATOR,
    f'def f(x={BAD}, hash=0): pass',
    f'def f(*, x={BAD}, hash=0): pass',
    f'f = lambda x={BAD}, hash=0: None',
    f'def f(hash: {BAD}): pass',
    f'def f(hash) -> {BAD}: pass',
])
def test_header_names_resolve_outside_parameter_scope(source):
    tree = ast.parse(source)
    resolver = Resolver(tree, ())
    names = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == 'hash']
    assert names and all(resolver.is_builtin(n) for n in names)
    assert all(resolver.live(n) for n in names)


@pytest.mark.parametrize('source', [
    f'def f(x={BAD}, hash=0): pass',
    f'def f(*, x={BAD}, hash=0): pass',
    f'f = lambda x={BAD}, hash=0: None',
])
def test_unused_definition_defaults_still_execute(tmp_path, source):
    for program in (source, renamed(source)):
        path = tmp_path / 'defaults.py'
        path.write_text(program)
        assert 'ZeroDivisionError' in native(path, 2).stderr
        assert not site_census(program, ['ZeroDivisionError']).no_site
        assert not determinism_census(program).ok
        result = verify(str(path), ['ZeroDivisionError'], python=sys.executable,
                        budget_seconds=60, run_timeout=30, max_runs=4, jobs=1)
        assert result.verdict != 'true', result


def test_uncalled_outer_body_stays_dead():
    source = 'def outer():\n' + indent(f'def f(x={BAD}): pass\n', '    ')
    tree = ast.parse(source)
    resolver = Resolver(tree, ())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    assert not resolver.live(call)
    assert site_census(source, ['ZeroDivisionError']).no_site


@pytest.mark.skipif(sys.version_info < (3, 14), reason='PEP 649 needs Python 3.14')
@pytest.mark.parametrize('header', [f'def f(hash: {BAD}): pass\n', f'def f(hash) -> {BAD}: pass\n'])
def test_annotations_are_lazy_and_use_enclosing_names(tmp_path, header):
    path = tmp_path / 'annotations.py'
    path.write_text(header)
    assert native(path, 2).returncode == 0  # no annotation access yet
    path.write_text('import typing\n' + header + 'typing.get_type_hints(f)\n')
    assert 'ZeroDivisionError' in native(path, 2).stderr
    assert not determinism_census(header).ok  # conservative about later evaluation
    result = verify(str(path), ['ZeroDivisionError'], python=sys.executable,
                    budget_seconds=60, run_timeout=30, max_runs=4, jobs=1)
    assert result.verdict != 'true', result


@pytest.mark.parametrize('source', [
    DECORATOR.replace('def f(hash)', 'def hash(value)'),
    f'def f(x={BAD}): pass\ndef hash(value): return value\n',
])
def test_future_module_binding_does_not_hide_builtin(source):
    assert not determinism_census(source).ok
    assert not site_census(source, ['ZeroDivisionError']).no_site


@pytest.mark.parametrize('source', [
    '@(lambda f: f)\ndef f(hash=3): return hash + 1\nassert f() == 4\n',
    'def outer(hash):\n    def f(x=hash(3)): return x\n    return f()\nassert outer(lambda x: x + 2) == 5\n',
])
def test_safe_decorators_defaults_and_enclosing_shadowing(tmp_path, source):
    for program in (source, renamed(source)):
        assert determinism_census(program).ok
        path = tmp_path / 'safe.py'
        path.write_text(program)
        assert native(path, 2).returncode == 0


@pytest.mark.skipif(sys.version_info < (3, 12), reason='type parameters need Python 3.12')
def test_unmodelled_generic_annotation_scope_refuses():
    tree = ast.parse('def f[hash](x: hash): pass\nf(None)')
    resolver = Resolver(tree, ())
    with pytest.raises(Top, match='type-parameter'):
        resolver.check(tree.body[0])


# -- [internal ref]: comprehension scoping inside a definition header ---------------------------------

@pytest.mark.parametrize('seed,expect_target', [(0, False), (1, False), (2, True)])
def test_comprehension_leak_repro_native_hash_values(tmp_path, seed, expect_target):
    """Ground truth: native CPython, not the resolver. At seed 2, `run()` really does equal
    the literal the (fixed) verifier must never certify as unreachable."""
    path = tmp_path / 'repro.py'
    path.write_text(COMPREHENSION_LEAK + 'print(run())\n')
    result = native(path, seed)
    assert result.returncode == 0, result.stderr
    assert (result.stdout.strip() == str(COMPREHENSION_LEAK_TARGET)) == expect_target


def test_comprehension_target_does_not_shadow_sibling_default_builtin():
    """Direction 1 (the reported bug): `unused`'s comprehension binds `hash` to its OWN scope,
    not to `run`'s -- so `value`'s sibling default still sees the builtin `hash`."""
    tree = ast.parse(COMPREHENSION_LEAK)
    resolver = Resolver(tree, ())
    value_default_hash = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == 'hash' and isinstance(n.ctx, ast.Load))
    assert resolver.is_builtin(value_default_hash)
    resolver.check_all()  # must resolve cleanly: this is a real, certifiable builtin call


def test_comprehension_leak_keeps_determinism_and_verifier_honest(tmp_path):
    source = COMPREHENSION_LEAK + f'assert run() != {COMPREHENSION_LEAK_TARGET}\n'
    census = determinism_census(source)
    assert not census.ok, census  # a real hash() call must be seen, not swallowed
    assert not site_census(source, ['AssertionError']).no_site
    path = tmp_path / 'comprehension_leak.py'
    path.write_text(source)
    result = verify(str(path), ['AssertionError'], python=sys.executable,
                    budget_seconds=60, run_timeout=30, max_runs=8, jobs=1)
    assert result.verdict != 'true', result


def test_comprehension_builtin_as_own_target_resolves_both_ways():
    """Direction 2 (the converse): a read inside a comprehension that the comprehension itself
    binds must resolve to the comprehension's own local, not the outer builtin -- EXCEPT the
    first `for`'s iterable, which runs before the comprehension's frame exists and genuinely
    sees the builtin, even though the comprehension binds the very same name."""
    tree = ast.parse('xs = [range for range in range(3)]\n')
    resolver = Resolver(tree, ())
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == 'range'
             and isinstance(n.ctx, ast.Load)]
    elt_read, iter_read = loads
    assert not resolver.is_builtin(elt_read)   # comprehension-local: shadows the builtin
    assert resolver.is_builtin(iter_read)      # the iterable: escapes to the real builtin
    resolver.check_all()
    assert determinism_census('xs = [range for range in range(3)]\nassert xs == [0, 1, 2]\n').ok


def test_comprehension_scope_isolated_from_enclosing_function_reads_and_writes():
    """A comprehension that reuses an outer local's name as both its target AND its element
    keeps its own binding fully separate from that outer local -- in both directions."""
    source = 'def outer():\n    hash = (10, 20)\n    return [hash for hash in hash]\n'
    tree = ast.parse(source)
    resolver = Resolver(tree, ())
    listcomp = next(n for n in ast.walk(tree) if isinstance(n, ast.ListComp))
    outer_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    elt_read = listcomp.elt
    target_store = listcomp.generators[0].target
    iter_read = listcomp.generators[0].iter
    assert isinstance(elt_read.ctx, ast.Load) and isinstance(target_store.ctx, ast.Store) \
        and isinstance(iter_read.ctx, ast.Load) and elt_read.id == target_store.id == iter_read.id == 'hash'
    assert resolver.function_of(elt_read) is listcomp        # comprehension-local read
    assert resolver.function_of(target_store) is listcomp    # comprehension-local target
    assert resolver.function_of(iter_read) is outer_def       # the iterable: outer's local
    resolver.check_all()
    exec(compile(source + 'assert outer() == [10, 20]\n', '<test>', 'exec'), {})


def test_comprehension_scope_agrees_with_compiler_symtable():
    """Independent oracle: CPython's own `symtable` on MG's reproducer agrees with the fixed
    resolver -- `hash` is global in `run`'s own frame (where `value`'s default actually runs),
    never captured by the comprehension's target. Pre-3.12, the comprehension gets its own
    child function scope (PEP 289) in which `hash` IS local; PEP 709 (3.12+) inlines list/set/
    dict comprehensions into the enclosing frame's bytecode, so that child scope disappears
    from `symtable` even though the language-level scoping -- what `pyresolve.py` models from
    the AST -- is unchanged (confirmed natively above: `value()` still sees the builtin on
    3.14). Assert what holds on every supported version; check the pre-3.12 child only when
    `symtable` still reports one.
    """
    top = symtable.symtable(COMPREHENSION_LEAK, '<test>', 'exec')
    run_st = next(c for c in top.get_children() if c.get_name() == 'run')
    run_hash = run_st.lookup('hash')
    assert run_hash.is_global() and not run_hash.is_local()
    listcomp_children = [c for c in run_st.get_children() if c.get_name() == 'listcomp']
    if listcomp_children:  # < 3.12: comprehensions still get their own symtable scope
        comp_hash = listcomp_children[0].lookup('hash')
        assert comp_hash.is_local() and not comp_hash.is_global()

    tree = ast.parse(COMPREHENSION_LEAK)
    resolver = Resolver(tree, ())
    value_default_hash = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == 'hash' and isinstance(n.ctx, ast.Load))
    assert resolver.is_builtin(value_default_hash) == (not run_hash.is_local())


@pytest.mark.parametrize('source', [
    '[y for x in range(3) if (y := x * 2) > 0]',
    '[[y := x for x in range(3)] for _ in range(1)]',
])
def test_walrus_in_comprehension_escapes_to_enclosing_scope(source):
    """PEP 572: a `:=` target inside a comprehension binds in the nearest enclosing
    function/module scope, never the comprehension's own -- unlike an ordinary `for` target."""
    tree = ast.parse(source)
    resolver = Resolver(tree, ())
    y_store = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and n.id == 'y' and isinstance(n.ctx, ast.Store))
    assert resolver.function_of(y_store) is tree
    resolver.check_all()
