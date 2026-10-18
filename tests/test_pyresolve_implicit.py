"""Implicit protocol calls must not disappear from either shared census."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tharkun.solve.census import site_census
from tharkun.solve.determinism import determinism_census
from tharkun.solve.pyresolve import Resolver
from tharkun.sv import verify


CHECK = 'assert hash("x") != 3914910683811788151'
KEYS = (Path(__file__).parent / "fixtures/census_implicit_keys.py").read_text()


def hook(name, args="self", after="return 0", before="", call="C()"):
    return f"class C:\n{before}    def {name}({args}):\n        {CHECK}\n        {after}\n{call}\n"


PROTOCOLS = {
    "mapping_kwargs": KEYS,
    "mapping_display": KEYS.replace("dict(**C())", "{**C()}"),
    "mapping_constructor": KEYS.replace("dict(**C())", "dict(C())"),
    "mapping_update": KEYS.replace("dict(**C())", "{}.update(C())"),
    "conditional_method": KEYS.replace("    def keys", "    if True:\n        def keys").replace(
        '        assert hash', '            assert hash').replace('        return []', '            return []'),
    "star_unpack": hook("__iter__", after="return iter(())", call="[*C()]"),
    "iteration": hook("__next__", before="    def __iter__(self):\n        return self\n", call="list(C())"),
    "enter": hook("__enter__", before="    def __exit__(self, *args):\n        pass\n", call="with C():\n    pass"),
    "exit": hook("__exit__", "self, *args", before="    def __enter__(self):\n        return self\n", call="with C():\n    pass"),
    "operator": hook("__add__", "self, other", call="C() + 1"),
    "comparison": hook("__eq__", "self, other", call="C() == 1"),
    "index": hook("__index__", call="[0][C()]"),
    "int": hook("__int__", call="int(C())"),
    "bool": hook("__bool__", after="return False", call="bool(C())"),
    "len": hook("__len__", call="len(C())"),
    "format": hook("__format__", "self, spec", after="return ''", call="f'{C()}'"),
    "str": hook("__str__", after="return ''", call="str(C())"),
    "repr": hook("__repr__", after="return ''", call="repr(C())"),
    "getattr": hook("__getattr__", "self, name", call="C().x"),
    "getattribute": hook("__getattribute__", "self, name", call="C().x"),
    "descriptor": hook("__get__", "self, obj, owner", call="class D:\n    x = C()\nD().x"),
    "decorator": hook("__call__", "self, f", after="return f", call="@C()\ndef f():\n    pass"),
    "metaclass": hook("__new__", "cls, *args", call="class D(metaclass=C):\n    pass").replace("class C:", "class C(type):"),
    "init_subclass": hook("__init_subclass__", "cls", call="class D(C):\n    pass"),
}


@pytest.mark.parametrize("name", PROTOCOLS)
def test_implicit_hook_is_live_and_never_certified(name):
    source = PROTOCOLS[name]
    result = subprocess.run([sys.executable, "-c", source], env=dict(os.environ, PYTHONHASHSEED="2"),
                            capture_output=True, text=True, timeout=5)
    assert result.returncode != 0 and "AssertionError" in result.stderr, result.stderr
    assert not site_census(source, ["AssertionError"]).no_site
    assert not determinism_census(source).ok


def test_nondunder_method_names_need_no_protocol_allowlist():
    for header in ["def future_protocol(self):", "async def future_protocol(self):"]:
        tree = ast.parse(f"class C:\n    {header}\n        {CHECK}\n")
        res = Resolver(tree, ())
        assert all(res.live(n) for n in ast.walk(tree))
    assert site_census("def unused():\n    assert False\n", ["AssertionError"]).no_site


def test_mapping_repro_cannot_return_true(tmp_path):
    path = tmp_path / "implicit_keys.py"
    path.write_text(KEYS)
    result = verify(str(path), ["AssertionError"], budget_seconds=2, jobs=1, concolic=False)
    assert result.verdict != "true", result
