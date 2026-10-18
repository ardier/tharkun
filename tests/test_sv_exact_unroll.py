"""Range expansion refuses large cardinalities before materializing their values."""
import ast

import pytest

from tharkun.sv import exact_unroll


@pytest.fixture
def no_range_materialization(monkeypatch):
    class UnmaterializableRange:
        def __init__(self, *args):
            self.values = range(*args)

        def __len__(self):
            return len(self.values)

        def __iter__(self):
            pytest.fail("oversized range was materialized")

    monkeypatch.setattr(exact_unroll, "range", UnmaterializableRange, raising=False)


@pytest.mark.parametrize("bounds", [
    "65",
    "1152921504606846976",
    "1152921504606846976, -1152921504606846976, -1",
    "-9223372036854775808, 9223372036854775808",
])
@pytest.mark.parametrize("source", ["for i in range({bounds}):\n    pass\n", "xs = [i for i in range({bounds})]\n"])
def test_large_range_is_not_expanded_or_materialized(no_range_materialization, bounds, source):
    tree = ast.parse(source.format(bounds=bounds))
    original = ast.dump(tree)
    exact_unroll.unroll_for_loops(tree)
    assert ast.dump(tree) == original


@pytest.mark.parametrize("start,stop,step", [
    (0, 65, 1),
    (0, 1 << 60, 1),
    (1 << 60, -(1 << 60), -1),
    (-(1 << 63), 1 << 63, 1),
])
def test_large_counter_loop_is_not_expanded_or_materialized(no_range_materialization, start, stop, step):
    comparison = "<" if step > 0 else ">"
    tree = ast.parse(f"i = {start}\nwhile i {comparison} {stop}:\n    i += {step}\n")
    original = ast.dump(tree)
    exact_unroll.unroll_for_loops(tree)
    assert ast.dump(tree) == original


@pytest.mark.parametrize("args", [(0, 0, 1), (0, 64, 1), (64, 0, -1)])
def test_ranges_within_capacity_keep_all_values(args):
    assert exact_unroll._trips(*args) == list(range(*args))


@pytest.mark.parametrize("args", [(0, 0, 1), (0, 64, 1), (64, 0, -1)])
def test_comprehensions_within_capacity_keep_all_values(args):
    bounds = ", ".join(map(str, args))
    tree = ast.parse(f"xs = [i for i in range({bounds})]\n")
    exact_unroll.unroll_for_loops(tree)
    assert ast.literal_eval(tree.body[0].value) == list(range(*args))


@pytest.mark.parametrize("loop", [
    "for i in range(n):\n    pass\n",
    "i = 0\nwhile i < n:\n    i += 1\n",
    "xs = [i for i in range(n)]\n",
])
@pytest.mark.parametrize("other_scope", [False, True])
def test_parameter_constant_does_not_erase_an_unbound_load_in_another_scope(loop, other_scope):
    import textwrap

    source = "def f(n):\n    return n\n"
    if other_scope:
        source += "def g():\n" + textwrap.indent(loop, "    ") + "f(3)\ng()\n"
    else:
        source += "f(3)\n" + loop
    tree = ast.parse(source)
    original = ast.dump(tree)
    assert "n" not in exact_unroll.module_constants(tree)
    exact_unroll.unroll_for_loops(tree)
    assert ast.dump(tree) == original
    from tharkun.sv.exact_subset import audit
    assert audit(tree, set()) is not None
    with pytest.raises(NameError):
        exec(compile(tree, "<unbound-loop-limit>", "exec"), {})


@pytest.mark.parametrize("definition", [
    "@n\ndef f(n):\n    return n\n",
    "def f(n) -> n:\n    return n\n",
])
def test_parameter_constant_does_not_include_loads_outside_the_function_body(definition):
    tree = ast.parse(definition + "f(3)\n")
    assert "n" not in exact_unroll.module_constants(tree)


def test_parameter_constant_in_its_own_function_body_still_unrolls():
    source = "def f(n):\n    i = 0\n    while i < n:\n        i += 1\n    return i\nassert f(3) == 3\n"
    tree = ast.parse(source)
    assert exact_unroll.module_constants(tree) == {"n": 3}
    exact_unroll.unroll_for_loops(tree)
    assert not any(isinstance(n, ast.While) for n in ast.walk(tree))
    exec(compile(ast.fix_missing_locations(tree), "<bound-loop-limit>", "exec"), {})
