"""Narrow absint coverage for exact builtin ``defaultdict(list | None)``."""

from __future__ import annotations

import textwrap

from tharkun.solve.absint import prove


HEAD = "import collections\n"


def verdict(body: str, *classes: str) -> str:
    return prove(HEAD + textwrap.dedent(body), list(classes or ("AssertionError",)), filename="<defaultdict>").verdict


def test_list_factory_missing_key_inserts_and_returns_mutable_alias():
    result = prove(HEAD + textwrap.dedent("""
        d = collections.defaultdict(list)
        first = d[4]
        first.append(9)
        assert d[4][0] == 9
        assert len(d) == 1
    """), ["AssertionError"], filename="<defaultdict-list>")
    assert result.verdict == "TRUE", (result.verdict, result.reason, result.sites)


def test_defaultdict_none_factory_retains_keyerror_semantics():
    source = HEAD + textwrap.dedent("""
        d = collections.defaultdict()
        d[4]
    """)
    try:
        exec(compile(source, "<native-defaultdict-none>", "exec"), {})
    except KeyError:
        pass
    else:
        raise AssertionError("defaultdict() with no factory must raise KeyError on a miss")
    result = prove(source, ["KeyError"], filename="<defaultdict-none>")
    assert result.verdict != "TRUE", (result.verdict, result.reason, result.sites)
    assert any("KeyError" in exc and fact.startswith("UNPROVEN")
               for _line, _col, _op, exc, fact in result.sites), result.explain()


def test_accepted_factory_shapes_discharge_constructor_typeerror():
    for expression in ("collections.defaultdict()", "collections.defaultdict(None)",
                       "collections.defaultdict(list)"):
        result = prove(HEAD + f"d = {expression}\n", ["TypeError"], filename="<defaultdict-valid-call>")
        assert result.verdict == "TRUE", (expression, result.verdict, result.reason, result.sites)


def test_unknown_keys_and_open_summaries_do_not_claim_missing_aliases_equal():
    assert verdict("""
        d = collections.defaultdict(list)
        for k in (1, 2):
            d[k]
        assert d[1] is d[2]
    """) != "TRUE"


def test_constructor_alias_is_supported():
    result = prove("""from collections import defaultdict
d = defaultdict(list)
d[2].append(3)
assert d[2][0] == 3
""", ["AssertionError"], filename="<defaultdict-alias>")
    assert result.verdict == "TRUE", (result.verdict, result.reason, result.sites)


def test_module_alias_is_supported():
    result = prove("""import collections as c
d = c.defaultdict(list)
d[2].append(3)
assert d[2][0] == 3
""", ["AssertionError"], filename="<defaultdict-module-alias>")
    assert result.verdict == "TRUE", (result.verdict, result.reason, result.sites)


def test_plain_dict_write_keeps_existing_behavior():
    assert verdict("""
        d = {1: 2}
        d[1] = 3
        assert d[1] == 3
    """) == "TRUE"


def test_defaultdict_other_factories_and_subclasses_are_refused():
    programs = [
        "d = collections.defaultdict(int)\nassert False",
        "d = collections.defaultdict(lambda: [])\nassert False",
        "d = collections.defaultdict(list, {1: []})\nassert False",
        "d = collections.defaultdict(list, x=1)\nassert False",
    ]
    for program in programs:
        result = prove(HEAD + program + "\n", ["AssertionError", "TypeError"], filename="<defaultdict-refusal>")
        assert result.verdict != "TRUE", (program, result)

    subclass = """\
        import collections
        class D(collections.defaultdict):
            pass
        d = D(list)
        d[1]
    """
    result = prove(textwrap.dedent(subclass) + "assert False\n", ["AssertionError"], filename="<defaultdict-subclass>")
    assert result.verdict != "TRUE"


def test_default_factory_attribute_mutation_is_refused():
    source = HEAD + """d = collections.defaultdict(list)
d.default_factory = None
try:
    d[1]
except KeyError:
    pass
else:
    assert False
"""
    try:
        exec(compile(source, "<native-defaultdict-mutation>", "exec"), {})
    except AssertionError as exc:
        raise AssertionError("native mutation should make the missing-key KeyError catchable") from exc
    result = prove(source, ["AssertionError"], filename="<defaultdict-mutation-assertion>")
    assert result.verdict != "TRUE", (result.verdict, result.reason, result.sites)
    key_result = prove(source, ["KeyError"], filename="<defaultdict-mutation-keyerror>")
    assert key_result.verdict != "TRUE", (key_result.verdict, key_result.reason, key_result.sites)


def test_default_factory_mutation_refuses_a_union_receiver():
    source = """import collections
import random
d = collections.defaultdict(list)
target = d if random.random() < 0.5 else {}
target.default_factory = None
assert False
"""
    try:
        exec(compile(source, "<native-defaultdict-union-store>", "exec"), {})
    except (AttributeError, AssertionError):
        pass
    else:
        raise AssertionError("each native receiver branch must hit a property exception")
    result = prove(source, ["AttributeError", "AssertionError"], filename="<defaultdict-union-store>")
    assert result.verdict != "TRUE", (result.verdict, result.reason, result.sites)


def test_separate_missing_keys_are_not_claimed_to_share_one_list():
    source = HEAD + """d = collections.defaultdict(list)
for k in (1, 2):
    d[k]
assert d[1] is d[2]
"""
    ns = {}
    try:
        exec(compile(source, "<native-defaultdict-identity>", "exec"), ns)
    except AssertionError:
        pass
    else:
        raise AssertionError("distinct missing keys must get distinct lists")
    assert prove(source, ["AssertionError"], filename="<defaultdict-identity>").verdict != "TRUE"


def test_factory_metadata_survives_heap_join_and_mapping():
    from tharkun.solve.absint.aval import BOT, V, const_int
    from tharkun.solve.absint.heap import DefaultDictObj
    from tharkun.solve.absint.state import _map_obj, ojoin

    left = DefaultDictObj({"left": const_int(1)}, True, BOT, BOT, "list")
    right = DefaultDictObj({"right": const_int(2)}, True, BOT, BOT, "list")
    joined = ojoin(left, None, right, None)
    assert type(joined) is DefaultDictObj and joined.factory == "list"
    assert not joined.exact and not joined.items
    mapped = _map_obj(left, lambda _value: V(st=frozenset({("int", 3)})))
    assert type(mapped) is DefaultDictObj and mapped.factory == "list"
    assert mapped.items["left"].st == frozenset({("int", 3)})


def test_loop_reallocation_does_not_lose_an_old_factory_list():
    source = HEAD + """d = collections.defaultdict(list)
old_refs = []
for k in (1, 2):
    old = d[k]
    old.append(k)
    old_refs.append(old)
assert old_refs[0] is old_refs[1]
"""
    try:
        exec(compile(source, "<native-defaultdict-loop>", "exec"), {})
    except AssertionError:
        pass
    else:
        raise AssertionError("fresh lists for distinct keys must remain distinct through old references")
    result = prove(source, ["AssertionError"], filename="<defaultdict-loop>")
    assert result.verdict != "TRUE", (result.verdict, result.reason, result.sites)
    assert "allocation would invalidate an in-flight heap reference" in result.reason, result.explain()
