"""[internal ref]: ``solve/lowering`` -- exact comprehension lowering, by differential execution.

The lowering is trusted only as far as these tests can show the original and the lowered
program indistinguishable: the same result, the same exception (type and message), the same
ORDER of every side effect.  Each case runs both with ``exec`` and compares a full trace.
The first block is the five counterexamples that showed Eregion's own comprehension pass
(``eregion_forward_passes/passes/forward/lower_comprehensions.py``, an inversion-shaped
lowering) is not semantics-preserving; the rest pin the scope, order and laziness rules in
``comprehension.py``'s docstring, and a seeded random differential closes the gaps.
"""

from __future__ import annotations

import ast
import random
import textwrap

import pytest

from tharkun.solve.lowering import lower_comprehensions, origin_of, resolve


PRELUDE = '''
LOG = []

def eff(tag, value=None):
    LOG.append(tag)
    return value

def boom(tag, value, bad):
    LOG.append(tag)
    if value == bad:
        raise ValueError(f"boom {value}")
    return value

class Box:
    def __init__(self):
        self.items = [1, 2]
        self.name = "box"
    def add(self, extra):
        LOG.append("Box.add")
        return self.items + extra
'''


def _run(source: str, call: str):
    """Execute ``source`` and evaluate ``call``; return (result repr | exception, LOG)."""
    namespace: dict = {}
    exec(compile(PRELUDE + textwrap.dedent(source), "<prog>", "exec"), namespace)
    log_before = len(namespace["LOG"])
    try:
        result = ("ok", repr(eval(call, namespace)))
    except BaseException as exc:  # noqa: BLE001 -- the point is to compare what happens
        result = ("raised", type(exc).__name__, str(exc))
    return result, tuple(namespace["LOG"][log_before:])


def lowered_source(source: str):
    prelude_tree = ast.parse(PRELUDE + textwrap.dedent(source))
    lowered = lower_comprehensions(prelude_tree)
    return lowered, ast.unparse(lowered.tree)


def same(source: str, call: str, *, lowers: int | None = None):
    """The lowered program behaves exactly like the original for ``call``."""
    lowered, text = lowered_source(source)
    if lowers is not None:
        assert lowered.lowered == lowers, (lowered.lowered, lowered.refused, text)
    original = _run(source, call)
    after = _run_text(text, call, lowered.restore_names)
    assert after == original, f"\noriginal: {original}\nlowered : {after}\n{text}"
    return original


def _run_text(text: str, call: str, restore=lambda message: message):
    """Run the LOWERED text; ``restore`` maps fresh loop names back in exception messages
    (``cannot access local variable '_lw3_c'`` names the lowered variable)."""
    namespace: dict = {}
    exec(compile(text, "<lowered>", "exec"), namespace)
    log_before = len(namespace["LOG"])
    try:
        result = ("ok", repr(eval(call, namespace)))
    except BaseException as exc:  # noqa: BLE001
        result = ("raised", type(exc).__name__, restore(str(exc)))
    return result, tuple(namespace["LOG"][log_before:])


# ---------------------------------------------------------------------------
# the five ways eregion's inversion-shaped pass is not semantics-preserving
# (each program measured against ``lower_comprehensions`` of eregion-forward-passes on
# 2026-10-05: the original and that pass's output behave differently; here they must not)
# ---------------------------------------------------------------------------


def test_1_the_loop_variable_does_not_leak():
    # eregion's pass leaves ``i`` bound to the last element: (3, [2, 4, 6]) for (100, [2, 4, 6])
    result = same(
        """
        def main(xs):
            i = 100
            y = [i * 2 for i in xs]
            return i, y
        """,
        "main([1, 2, 3])",
        lowers=1,
    )
    assert result[0] == ("ok", "(100, [2, 4, 6])")


@pytest.mark.parametrize("name, source, call, expected", [
    # eregion's pass indexes ``xs[i]`` under ``len(xs)``: TypeError for an iterator, a generator
    # and a set, KeyError(0) for a dict, and for a dict keyed 0, 1, ... the VALUES for the keys
    ("iterator", "def main(n):\n    it = iter(range(n))\n    y = [v + 1 for v in it]\n    return y\n",
     "main(3)", "[1, 2, 3]"),
    ("generator", "def main(g):\n    y = [v for v in g]\n    return y\n",
     "main(x for x in [1, 2])", "[1, 2]"),
    ("set", "def main(s):\n    y = [v for v in s]\n    return y\n", "main({7})", "[7]"),
    ("dict", "def main(d):\n    y = [k for k in d]\n    return y\n", "main({'a': 1})", "['a']"),
    ("dict keyed 0 and 1", "def main(d):\n    y = [k for k in d]\n    return y\n",
     "main({0: 'x', 1: 'y'})", "[0, 1]"),
])
def test_2_an_iterable_that_is_not_a_sequence_is_iterated_not_indexed(name, source, call, expected):
    result = same(source, call, lowers=1)
    assert result[0] == ("ok", expected)


def test_3_the_first_iterable_is_evaluated_once_and_is_not_rebound_midway():
    # eregion's pass re-evaluates ``src()`` on every index step and in ``len(...)``
    result = same(
        """
        def src():
            eff("src")
            return [1, 2, 3]
        def main():
            y = [v for v in src()]
            return y
        """,
        "main()",
        lowers=1,
    )
    assert result[1] == ("src",)
    # ... and an assignment expression in the element changes what the index loop reads
    # ([1, 9] for [1, 2, 3]); here it is REFUSED (it binds in the enclosing scope), not rewritten
    result = same(
        """
        def main():
            xs = [1, 2, 3]
            y = [(xs := [9, 9]) and v for v in xs]
            return y
        """,
        "main()",
        lowers=0,
    )
    assert result[0] == ("ok", "[1, 2, 3]")


def test_4_a_nested_comprehension_that_reuses_a_name_keeps_both():
    # eregion's pass names both loops from the same variable: [[0], [0]] for [[0], [0, 2]]
    result = same(
        """
        def main(xs):
            y = [[x * 2 for x in range(x)] for x in xs]
            return y
        """,
        "main([1, 2])",
        lowers=2,
    )
    assert result[0] == ("ok", "[[0], [0, 2]]")


def test_5_any_all_and_next_over_a_generator_stay_lazy():
    # eregion's pass made these a list comprehension: every element is evaluated (e, e, e for e)
    # and next(list) is a TypeError
    source = """
        def main():
            return any(boom("e", v, 2) == 1 for v in [1, 2, 3])
    """
    lowered, text = lowered_source(source)
    assert lowered.lowered == 0 and "generator expression" in lowered.refused[0].reason
    assert _run(source, "main()") == (("ok", "True"), ("e",))
    assert _run_text(text, "main()") == _run(source, "main()")
    all_source = "def main():\n    return all(boom('e', v, 9) < 2 for v in [5, 1, 1])"
    lowered_all, all_text = lowered_source(all_source)
    assert lowered_all.lowered == 0
    assert _run_text(all_text, "main()") == _run(all_source, "main()") == (("ok", "False"), ("e",))
    next_source = "def main():\n    return next(v for v in [5, 6, 7])"
    lowered_next, next_text = lowered_source(next_source)
    assert lowered_next.lowered == 0
    assert _run_text(next_text, "main()") == (("ok", "5"), ())


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------


def test_a_comprehension_variable_shadows_an_outer_name_only_inside():
    same(
        """
        def main(xs):
            x = "outer"
            ys = [x for x in xs]
            zs = [x + "!" for _ in xs]
            return x, ys, zs
        """,
        "main([1, 2])",
        lowers=2,
    )


def test_a_nested_comprehension_that_rebinds_the_name_keeps_its_own():
    same(
        """
        def main(xs):
            return [[x * 2 for x in range(x)] for x in xs]
        """,
        "main([1, 2, 3])",
        lowers=2,
    )


def test_a_lambda_that_rebinds_the_name_is_not_renamed():
    same(
        """
        def main(xs):
            return [(lambda x: x + 1)(x) for x in xs]
        """,
        "main([1, 2])",
        lowers=1,
    )


def test_the_later_iterable_sees_the_earlier_variable():
    same(
        """
        def main():
            return [(a, b) for a in range(3) for b in range(a)]
        """,
        "main()",
        lowers=1,
    )


def test_a_tuple_target_and_a_starred_target():
    same("def main(ps):\n    return [a + b for a, b in ps]", "main([(1, 2), (3, 4)])", lowers=1)
    same("def main(ps):\n    return [(a, r) for a, *r in ps]", "main([[1, 2, 3], [4]])", lowers=1)


def test_the_loop_variable_is_a_closure_cell_shared_by_late_binding_lambdas():
    same(
        "def main():\n    fs = [lambda: x for x in range(3)]\n    return [f() for f in fs]",
        "main()",
        lowers=2,
    )


def test_a_comprehension_at_module_level_leaves_no_loop_variable_behind():
    lowered, text = lowered_source("squares = [v * v for v in range(4)]\nv = 'mine'\n")
    namespace: dict = {}
    exec(compile(text, "<m>", "exec"), namespace)
    assert namespace["squares"] == [0, 1, 4, 9] and namespace["v"] == "mine"
    assert lowered.lowered == 1


def test_a_comprehension_in_a_class_body_is_left_alone():
    source = """
        class C:
            n = 3
            ys = [v for v in range(2)]
        def main():
            return C.ys
    """
    lowered, text = lowered_source(source)
    assert lowered.lowered == 0 and lowered.refused and "class body" in lowered.refused[0].reason
    assert _run_text(text, "main()") == _run(source, "main()")


def test_a_method_inside_a_class_is_still_lowered():
    same(
        """
        class C:
            def m(self, xs):
                return [x + 1 for x in xs]
        def main():
            return C().m([1, 2])
        """,
        "main()",
        lowers=1,
    )


# ---------------------------------------------------------------------------
# order of evaluation around a hoisted comprehension
# ---------------------------------------------------------------------------


def test_earlier_operands_are_evaluated_before_the_loop_runs():
    result = same(
        """
        def main():
            return eff("left", 1) + sum([eff("elt", v) for v in [1, 2]])
        """,
        "main()",
        lowers=1,
    )
    assert result[1] == ("left", "elt", "elt")


def test_a_call_arguments_before_the_comprehension_are_evaluated_first():
    same(
        """
        def main():
            return max(eff("a", 0), *[]) if False else len([eff("x", 1)] + [eff("y", v) for v in [1, 2]])
        """,
        "main()",
        lowers=1,
    )


def test_a_bound_method_is_loaded_before_the_comprehension_runs():
    # the attribute is looked up BEFORE the arguments are evaluated; the comprehension
    # deletes it, so a late lookup would raise AttributeError
    source = """
        def main():
            b = Box()
            def kill(v):
                Box.add = lambda self, extra: ("replaced", extra)
                return v
            return b.add([kill(v) for v in [1]])
    """
    result = same(source, "main()", lowers=1)
    assert result[0][0] == "ok"


def test_the_callee_name_is_read_before_the_comprehension_rebinds_it():
    same(
        """
        def f(x):
            return ("f", x)
        def main():
            global f
            def swap(v):
                global f
                f = lambda x: ("g", x)
                return v
            return f([swap(v) for v in [1]])
        """,
        "main()",
        lowers=1,
    )


def test_two_comprehensions_in_one_expression_keep_their_order():
    result = same(
        """
        def main():
            return [eff("a", v) for v in [1, 2]] + [eff("b", v) for v in [3]]
        """,
        "main()",
        lowers=2,
    )
    assert result[1] == ("a", "a", "b")


def test_an_exception_in_an_earlier_operand_comes_before_the_comprehension():
    result = same(
        """
        def main():
            return boom("first", 1, 1) + len([eff("elt", v) for v in [1]])
        """,
        "main()",
        lowers=1,
    )
    assert result[0][0] == "raised" and result[1] == ("first",)


def test_a_dict_comprehension_evaluates_the_key_before_the_value():
    result = same(
        """
        def main():
            return {eff("k", v): eff("v", v * 2) for v in [1, 2]}
        """,
        "main()",
        lowers=1,
    )
    assert result[1] == ("k", "v", "k", "v")


def test_a_dict_comprehension_with_an_unhashable_key_raises_after_both_are_evaluated():
    same(
        """
        def main():
            return {eff("k", [v]): eff("v", v) for v in [1]}
        """,
        "main()",
        lowers=1,
    )


def test_conditions_are_evaluated_in_order_and_short_circuit_per_element():
    same(
        """
        def main():
            return [v for v in range(5) if eff("c1", v % 2 == 0) if eff("c2", v > 0)]
        """,
        "main()",
        lowers=1,
    )


def test_a_comprehension_on_the_right_of_and_is_evaluated_only_when_the_left_is_true():
    same(
        """
        def main(flag):
            return flag and [eff("elt", v) for v in [1, 2]]
        """,
        "main(False)",
        lowers=1,
    )
    same(
        """
        def main(flag):
            return flag and [eff("elt", v) for v in [1, 2]]
        """,
        "main(True)",
        lowers=1,
    )
    same(
        """
        def main(flag):
            return flag or [eff("elt", v) for v in [1, 2]]
        """,
        "main(0)",
        lowers=1,
    )
    same(
        """
        def main(a, b):
            return a and b or [eff("elt", v) for v in [1]]
        """,
        "main(1, 0)",
        lowers=1,
    )


def test_a_comprehension_in_one_branch_of_a_conditional_expression():
    for flag in (True, False):
        same(
            """
            def main(flag):
                return [eff("then", v) for v in [1]] if flag else [eff("else", v) for v in [2, 3]]
            """,
            f"main({flag})",
            lowers=2,
        )
    same(
        """
        def main(flag):
            return [eff("then", v) for v in [1]] if eff("test", flag) else 7
        """,
        "main(False)",
        lowers=1,
    )


def test_a_comprehension_in_a_condition_and_in_an_elif():
    same(
        """
        def main(x):
            if x > 5:
                return "big"
            elif [v for v in range(x) if eff("elif", v) == 1]:
                return "has one"
            return "none"
        """,
        "main(3)",
        lowers=1,
    )


def test_a_comprehension_as_the_iterable_of_a_for_loop():
    same(
        """
        def main():
            out = []
            for a in [eff("it", v) for v in [1, 2]]:
                out.append(eff("body", a))
            return out
        """,
        "main()",
        lowers=1,
    )


def test_a_comprehension_in_the_value_of_an_attribute_store_and_a_subscript_store():
    same(
        """
        def main():
            b = Box()
            d = {}
            b.items = [eff("a", v) for v in [1, 2]]
            d[eff("key", 0)] = [eff("val", v) for v in [3]]
            return b.items, d
        """,
        "main()",
        lowers=2,
    )


def test_nested_comprehensions_in_the_element_the_filter_and_the_iterable():
    same(
        """
        def main(rows):
            return [[c * 2 for c in r if c] for r in rows if [x for x in r if x > 1]]
        """,
        "main([[1, 2], [0, 1], [3, 0, 4]])",
        lowers=3,
    )
    same(
        """
        def main():
            return [a for a in [b + 1 for b in range(3)]]
        """,
        "main()",
        lowers=2,
    )
    same(
        """
        def main(n):
            return [(a, b) for a in range(n) for b in [a * 2 for _ in range(a)]]
        """,
        "main(3)",
        lowers=2,
    )


def test_an_unbound_name_in_an_earlier_operand_is_raised_before_the_loop():
    # NameError must come from the same place, before the comprehension's effects
    same(
        """
        def main():
            return undefined_name + len([eff("elt", v) for v in [1]])
        """,
        "main()",
        lowers=1,
    )


# ---------------------------------------------------------------------------
# generator expressions: lazy, so lowered only where the consumer drains them first
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("consumer", ["list", "tuple", "sorted"])
def test_a_generator_drained_by_a_builtin_is_lowered_exactly(consumer):
    same(
        f"""
        def main():
            return {consumer}(boom("e", v, 9) for v in [3, 1, 2])
        """,
        "main()",
        lowers=1,
    )
    # an exception part-way: the same exception, the same effects before it
    result = same(
        f"""
        def main():
            return {consumer}(boom("e", v, 2) for v in [3, 2, 1])
        """,
        "main()",
        lowers=1,
    )
    assert result[0][0] == "raised" and result[1] == ("e", "e")


def test_sorted_with_key_and_reverse_keywords():
    same(
        "def main():\n    return sorted((eff('e', v) for v in [3, 1, 2]), key=lambda x: -x, reverse=True)",
        "main()",
        lowers=1,
    )


def test_join_on_a_literal_drains_its_generator_first():
    same("def main():\n    return ', '.join(str(eff('e', v)) for v in [1, 2])", "main()", lowers=1)
    # a non-str element is only noticed after the whole sequence was built
    result = same("def main():\n    return ''.join(eff('e', v) for v in ['a', 1, 'c'])", "main()", lowers=1)
    assert result[0][0] == "raised" and result[1] == ("e", "e", "e")


@pytest.mark.parametrize(
    "consumer",
    ["sum", "min", "max", "any", "all", "next", "set", "frozenset", "dict", "enumerate", "zip", "map"],
)
def test_a_generator_given_to_a_lazy_consumer_is_refused(consumer):
    arg = "((v, v) for v in [1])" if consumer == "dict" else "(v for v in [1])"
    source = f"def main():\n    return {consumer}{arg}"
    lowered, text = lowered_source(source)
    assert lowered.lowered == 0 and lowered.refused
    assert "generator expression" in lowered.refused[0].reason


def test_a_generator_that_is_not_the_sole_argument_is_refused():
    lowered, _ = lowered_source("def main():\n    return sorted((v for v in [2, 1]), key=abs)\n")
    assert lowered.lowered == 1  # a keyword is fine: the iterable is still the only positional
    lowered, _ = lowered_source("def main():\n    return list((v for v in [1]), 2)\n")
    assert lowered.lowered == 0


def test_a_rebound_builtin_does_not_count_as_the_builtin():
    lowered, _ = lowered_source("def list(g):\n    return 1\ndef main():\n    return list(v for v in [1])\n")
    assert lowered.lowered == 0 and lowered.refused


def test_a_bare_generator_is_refused():
    lowered, _ = lowered_source("def main():\n    g = (v for v in [1, 2])\n    return next(g)\n")
    assert lowered.lowered == 0 and lowered.refused


# ---------------------------------------------------------------------------
# shapes whose order is not modelled are left exactly as they were
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, why",
    [
        ("return f(*[v for v in [1]])", "*args"),
        ("return f(**{k: 1 for k in 'a'})", "**kwargs"),
        ("return 1 < 2 < len([v for v in [1]])", "chained comparison"),
        ("return f'{[v for v in [1]]}'", "JoinedStr"),
        ("return (y := [v for v in [1]])", "NamedExpr"),
        ("return (lambda: [v for v in [1]])()", "lambda"),
        ("x = 0\n    x += len([v for v in [1]])\n    return x", "AugAssign"),
        ("x: list = [v for v in [1]]\n    return x", "AnnAssign"),
        ("n = 3\n    while len([v for v in range(n)]) > 5:\n        n -= 1\n    return n", "loop test"),
        ("with open(__file__) if [v for v in [1]] else None:\n        pass", "with"),
    ],
)
def test_unmodelled_shapes_are_refused_and_left_in_place(body, why):
    source = f"def f(*a, **k):\n    return 0\ndef main():\n    {body}\n"
    lowered, text = lowered_source(source)
    assert lowered.lowered == 0, (why, text)
    assert lowered.refused, why
    # the program text is semantically unchanged (the same tree minus nothing)
    assert ast.dump(ast.parse(text)) == ast.dump(ast.parse(PRELUDE + textwrap.dedent(source)))


def test_a_program_that_reaches_its_own_namespace_is_not_lowered():
    for reach in ("globals()", "locals()", "vars()", "dir()"):
        source = f"def main():\n    ns = {reach}\n    return [v for v in range(2)]\n"
        lowered, _ = lowered_source(source)
        assert lowered.lowered == 0 and "namespace" in lowered.refused[0].reason


def test_a_refusal_inside_an_unrelated_statement_leaves_the_others_lowered():
    lowered, _ = lowered_source(
        "def main():\n    a = [v for v in range(2)]\n    b = f(*[v for v in a])\n    return a, b\n"
        "def f(*a):\n    return a\n"
    )
    assert lowered.lowered == 1 and len(lowered.refused) == 1


# ---------------------------------------------------------------------------
# provenance: every node maps back to the original program
# ---------------------------------------------------------------------------

PROVENANCE_PROGRAM = """
def main(xs, flag):
    d = {k: [c for c in k if c] for k in xs if k}
    e = flag and [x * 2 for x in xs]
    return sum([a for a in range(3)]) + len(d) + len(e or [])
"""


def test_every_lowered_node_resolves_to_a_node_of_the_original_program():
    original = ast.parse(textwrap.dedent(PROVENANCE_PROGRAM))
    lowered = lower_comprehensions(original)
    assert lowered.lowered == 4
    checked = synthetic = 0
    for node in ast.walk(lowered.tree):
        if not hasattr(node, "lineno"):
            continue
        origin = origin_of(node)
        assert origin is not None, ast.dump(node)[:80]
        source_node = resolve(original, origin.path)
        assert (source_node.lineno, source_node.col_offset) == (node.lineno, node.col_offset), (
            origin.text(), ast.dump(node)[:80],
        )
        if origin.synthetic:
            synthetic += 1
        else:
            assert type(source_node) is type(node) or isinstance(node, ast.Name), (
                origin.text(), type(source_node), type(node),
            )
        checked += 1
    assert checked > 40 and synthetic > 10


def test_the_input_tree_is_not_modified():
    original = ast.parse(textwrap.dedent(PROVENANCE_PROGRAM))
    before = ast.dump(original)
    lower_comprehensions(original)
    assert ast.dump(original) == before


def test_fresh_loop_names_are_reported_and_restored_in_text():
    lowered = lower_comprehensions(ast.parse("def main(xs):\n    return [x for x in xs]\n"))
    (fresh, original), = lowered.renamed.items()
    assert original == "x"
    assert lowered.restore_names(f"({fresh}) == 0") == "(x) == 0"
    assert lowered.synthetic and not set(lowered.synthetic) & set(lowered.renamed)


# ---------------------------------------------------------------------------
# a seeded random differential: random expressions with effects and exceptions
# ---------------------------------------------------------------------------


class _Gen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.depth_budget = 0

    def atom(self, vars_):
        r = self.rng
        pick = r.random()
        if pick < 0.30:
            return str(r.randint(-2, 4))
        if pick < 0.55 and vars_:
            return r.choice(vars_)
        if pick < 0.70:
            return f"eff({r.choice('pqrs')!r}, {r.randint(0, 3)})"
        if pick < 0.80:
            return f"xs[{r.randint(-1, 3)}]"
        return "n"

    def expr(self, vars_, depth=0):
        r = self.rng
        if depth >= 3:
            return self.atom(vars_)
        pick = r.random()
        if pick < 0.20:
            return self.atom(vars_)
        if pick < 0.38:
            op = r.choice(["+", "-", "*"])
            return f"({self.expr(vars_, depth + 1)} {op} {self.expr(vars_, depth + 1)})"
        if pick < 0.62:
            return self.listcomp(vars_, depth)
        if pick < 0.70:
            return f"len({self.listcomp(vars_, depth)})"
        if pick < 0.76:
            return f"sum({self.listcomp(vars_, depth)})"
        if pick < 0.82:
            return f"({self.expr(vars_, depth + 1)} and {self.expr(vars_, depth + 1)})"
        if pick < 0.88:
            return f"({self.expr(vars_, depth + 1)} or {self.expr(vars_, depth + 1)})"
        if pick < 0.94:
            return (
                f"({self.expr(vars_, depth + 1)} if {self.expr(vars_, depth + 1)} "
                f"else {self.expr(vars_, depth + 1)})"
            )
        return f"max({self.expr(vars_, depth + 1)}, {self.expr(vars_, depth + 1)})"

    def listcomp(self, vars_, depth):
        r = self.rng
        # Comprehension variables never reuse a name the function also binds (a parameter or a
        # local): CPython >= 3.12 inlines comprehensions (PEP 709) and, in one exotic shape (a
        # comprehension that BINDS a name, then a later comprehension nested in the same outer
        # one that READS the same function local), raises UnboundLocalError where the language
        # reference -- and 3.10/3.11, and this lowering -- do not.  The lowering has MORE
        # behaviours there, never fewer, which is the safe direction for a proof; see
        # docs/LOWERING.md.  Names are reused among sibling and nested comprehensions on purpose.
        name = r.choice(["a", "b", "x"])
        inner = [*vars_, name]
        source = r.choice(["xs", "range(3)", "range(n)", "[eff('i', 1), 2]", f"[{self.atom(vars_)}, {self.atom(vars_)}]"])
        cond = f" if {self.expr(inner, depth + 2)}" if r.random() < 0.35 else ""
        more = ""
        if r.random() < 0.2:
            name2 = r.choice(["c", "a"])
            inner.append(name2)
            more = f" for {name2} in range({self.expr(inner, depth + 2)} % 3)"
        return f"[{self.expr(inner, depth + 1)} for {name} in {source}{more}{cond}]"

    def program(self):
        r = self.rng
        a = self.expr(["u", "n"], 0)
        b = self.expr(["u", "n"], 0)
        shape = r.random()
        if shape < 0.4:
            body = f"    u = {a}\n    return u, {b}"
        elif shape < 0.7:
            body = f"    u = 1\n    w = {a}\n    return u, w, {b}"
        else:
            body = f"    u = 2\n    if {a}:\n        return {b}\n    return -1"
        return f"def main(xs, n):\n{body}\n"


@pytest.mark.parametrize("seed", range(40))
def test_random_programs_behave_identically_before_and_after_lowering(seed):
    rng = random.Random(1107 + seed)
    gen = _Gen(rng)
    lowered_any = 0
    for _ in range(25):
        source = gen.program()
        try:
            lowered, text = lowered_source(source)
        except SyntaxError:
            pytest.fail(source)
        lowered_any += lowered.lowered
        for args in ("([1, 2, 3], 2)", "([], 0)", "([0, 5], 3)", "([2], -1)"):
            call = f"main{args}"
            original = _run(source, call)
            after = _run_text(text, call, lowered.restore_names)
            assert after == original, f"\n{source}\ncall {call}\noriginal: {original}\nlowered : {after}\n{text}"
    assert lowered_any > 0


# ---------------------------------------------------------------------------
# a program that can reach its own namespace is left alone -- precisely
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, source, expected", [
    # a loop variable that merely happens to be called ``dir`` is a local, not the builtin
    ("local named dir", "def f(xs):\n    for dir in xs:\n        pass\n    return [x for x in xs]\n", True),
    ("parameter named locals", "def f(locals):\n    return locals\ndef g(xs):\n    return [x for x in xs]\n", True),
    ("attribute named vars", "def g(o, xs):\n    return o.vars, [x for x in xs]\n", True),
    # the builtin itself, however it is reached, makes fresh names observable
    ("builtin dir()", "def f(xs):\n    print(dir())\n    return [x for x in xs]\n", False),
    ("locals() in another function", "def h():\n    return locals()\ndef g(xs):\n    return [x for x in xs]\n", False),
    ("eval", "def g(xs):\n    eval('1')\n    return [x for x in xs]\n", False),
    # a comprehension variable binds in the comprehension, not in the function around it
    ("comprehension variable is not a local", "def f(xs):\n    y = [dir for dir in xs]\n    return dir()\n", False),
    # a module-level binding does not exempt a load: a function can run before it executes
    ("module-level dir", "dir = 5\ndef f():\n    return dir\ndef g(xs):\n    return [x for x in xs]\n", False),
    ("global declaration", "def f():\n    global dir\n    dir = 1\ndef g(xs):\n    return [x for x in xs]\n", False),
    # frame and namespace attributes
    ("frame locals", "import sys\ndef f(xs):\n    sys._getframe().f_locals\n    return [x for x in xs]\n", False),
    ("function globals", "def f(xs):\n    f.__globals__\n    return [x for x in xs]\n", False),
])
def test_a_program_that_can_reach_its_namespace_is_left_alone_and_only_that_one(name, source, expected):
    lowered, _text = lowered_source(source)
    assert (lowered.lowered > 0) is expected, (name, lowered.refused)
    if not expected:
        assert all("namespace" in r.reason for r in lowered.refused), lowered.refused


def test_a_comprehension_in_a_function_with_a_local_named_like_a_builtin_is_lowered_and_exact():
    # hexiom shape: ``for dir in DIRS`` next to a comprehension
    same(
        """
        DIRS = [1, 2, 3]
        def main(n):
            total = 0
            for dir in DIRS:
                total += dir * n
            ys = [v + dir for v in range(3)]
            return total, ys
        """,
        "main(2)",
        lowers=1,
    )


def test_a_fresh_name_is_restored_only_where_the_comprehensions_own_variable_is_bound():
    source = "def main(xs):\n    ys = [v * 2 for v in xs if v > 0]\n    return ys\n"
    lowered, _text = lowered_source(source)
    (fresh,) = lowered.renamed
    scope = lowered.scopes[fresh]
    elt = scope + (("elt", None),)
    cond = scope + (("generators", 0), ("ifs", 0))
    first_iter = scope + (("generators", 0), ("iter", None))
    text = f"{fresh} >= 1"
    assert lowered.at_site(text, elt) == "v >= 1"
    assert lowered.at_site(text, cond) == "v >= 1"
    # the first iterable is evaluated in the enclosing scope, where `v` means something else
    assert lowered.at_site(text, first_iter) is None
    # a node that stands for the whole comprehension, and a node outside it
    assert lowered.at_site(text, scope) is None
    assert lowered.at_site(text, (("body", 0), ("body", 1))) is None
    assert lowered.at_site(text, None) is None
    # a text with no fresh name is untouched anywhere
    assert lowered.at_site("x >= 1", first_iter) == "x >= 1"
    # a synthetic name is never restored
    synthetic = next(iter(lowered.synthetic))
    assert lowered.at_site(f"len({synthetic}) > 0", elt) is None
