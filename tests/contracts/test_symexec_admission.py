"""Admission must not run user hooks; snapshot containers cannot model live views."""
import ast
import bisect
import types

import pytest

from tharkun.sv import _symexec as S


def _runner(env=None, globals_=None):
    return S._Runner(ast.parse("def __sv_harness():\n    pass\n"), globals_ or {}, env or {}, {}, [])


def _refused(executor, source):
    try:
        executor.ev(ast.parse(source, mode="eval").body)
    except S.Unsupported:
        return True
    return False


def _bound(base, calls):
    class Meta(type):
        def __eq__(cls, other):
            calls.append("type equality")
            return other is base

        def __hash__(cls):
            calls.append("type hash")
            return hash(base)

    class Bound(base, metaclass=Meta):
        def __int__(self):
            calls.append("conversion")
            return 42

        def __float__(self):
            calls.append("conversion")
            return 42.0

        def __getitem__(self, key):
            calls.append("getitem")
            return 42

        def __eq__(self, other):
            calls.append("element equality")
            return True

    return Bound


@pytest.mark.parametrize("base,qualname", [(int, "int"), (float, "float")])
def test_scalar_admission_refuses_subclasses_without_hooks(base, qualname):
    calls = []
    value = _bound(base, calls)(7)
    ex = S._Exec({}, {"bound": value}, {})
    assert _refused(ex, f"{qualname}(bound)")
    assert calls == []


@pytest.mark.parametrize("base", [tuple, list, dict])
def test_subscript_dispatch_refuses_subclasses_without_hooks(base):
    calls = []
    value = _bound(base, calls)({0: 7} if base is dict else [7])
    runner = _runner({"bound": value})
    assert _refused(runner, "bound[0]")
    assert calls == []


def test_comparison_admission_refuses_spoofed_scalars_without_hooks():
    calls = []
    value = _bound(int, calls)(7)
    runner = _runner({"bound": value})
    runner.run(ast.parse("xs = [bound]\nn = xs.count(7)\n").body)
    assert runner.skipped or runner.poisoned
    assert calls == []


@pytest.mark.parametrize("method", ["keys", "values", "items"])
def test_live_dict_views_are_refused_and_not_exact(method):
    source = f"d = {{1: 2}}\nview = d.{method}()\nd[1] = 3\nd[4] = 5\nx = list(view)\n"
    tree = ast.parse(source)
    runner = _runner()
    runner.run(tree.body)
    assert runner.skipped or runner.poisoned
    assert not S.run_exact(tree, {}, {}, tree.body, [])["exact"]


@pytest.mark.parametrize("shadow", ["global", "local", "caller"])
def test_fake_module_identity_is_refused_without_calling_it(shadow):
    calls = []
    fake = types.ModuleType("bisect")
    fake.bisect_left = lambda *args: calls.append("fake call") or 99
    runner = _runner(globals_={"alias": fake if shadow == "global" else bisect})
    if shadow == "local":
        runner.env["alias"] = fake
    if shadow == "caller":
        runner.frames = [{"alias": fake}]
    assert _refused(runner, "alias.bisect_left([1, 3, 5], 3)")
    assert calls == []


def test_rebound_stdlib_callable_is_refused(monkeypatch):
    calls = []
    monkeypatch.setattr(bisect, "bisect_left", lambda *args: calls.append("replacement") or 99)
    runner = _runner(globals_={"alias": bisect})
    assert _refused(runner, "alias.bisect_left([1, 3, 5], 3)")
    assert calls == []


def test_ordinary_scalars_containers_and_real_alias_still_work():
    assert S._Exec({}, {"bound": 7}, {}).ev(ast.parse("int(bound)", mode="eval").body) == 7
    runner = _runner({"bound": (7,)}, {"alias": bisect})
    assert runner.ev(ast.parse("bound[0]", mode="eval").body) == 7
    assert runner.ev(ast.parse("alias.bisect_left([1, 3, 5], 3)", mode="eval").body) == 1


def test_negative_control_equality_admission_is_caught(monkeypatch):
    monkeypatch.setattr(S, "_exact_type", lambda value, classes: type(value) in classes)
    with pytest.raises(AssertionError):
        test_scalar_admission_refuses_subclasses_without_hooks(int, "int")


def test_negative_control_module_name_dispatch_is_caught(monkeypatch):
    def unsafe(runner, f, node):
        return runner._through_rule(f"{runner.g[f.value.id].__name__}.{f.attr}", node)
    monkeypatch.setattr(S._Runner, "_contract_call", unsafe)
    with pytest.raises(AssertionError):
        test_fake_module_identity_is_refused_without_calling_it("global")


def test_negative_control_snapshot_view_is_caught(monkeypatch):
    native = S._Runner._native_method
    def unsafe(runner, qualname, node, receiver):
        if qualname == "dict.values":
            return list(receiver.values())
        return native(runner, qualname, node, receiver)
    monkeypatch.setattr(S._Runner, "_native_method", unsafe)
    with pytest.raises(AssertionError):
        test_live_dict_views_are_refused_and_not_exact("values")


@pytest.mark.parametrize("convert", ["list", "tuple"])
def test_immediate_keys_snapshot_stays_native_and_exact(convert):
    source = f"d = {{1: 2}}\nsnapshot = {convert}(d.keys())\nd[4] = 5\n"
    tree = ast.parse(source)
    runner = _runner()
    runner.run(tree.body)
    assert not (runner.skipped or runner.poisoned)
    assert list(runner.env["snapshot"]) == [1]
    assert S.run_exact(tree, {}, {}, tree.body, [])["exact"]


@pytest.mark.parametrize("rebind_type", [False, True])
def test_changed_module_class_is_refused_without_attribute_hooks(rebind_type, monkeypatch):
    calls = []
    class Noisy(types.ModuleType):
        def __getattribute__(self, name):
            calls.append(name)
            return super().__getattribute__(name)
    original = type(bisect)
    bisect.__class__ = Noisy
    if rebind_type:
        monkeypatch.setattr(types, "ModuleType", Noisy)
    try:
        runner = _runner(globals_={"alias": bisect})
        assert _refused(runner, "alias.bisect_left([1, 3, 5], 3)")
        assert calls == []
    finally:
        bisect.__class__ = original
