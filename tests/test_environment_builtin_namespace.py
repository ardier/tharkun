"""A changed builtin namespace cannot certify a static proof ([internal ref])."""

import ast

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.absint.interp import Analyzer, FuncCtx, Unsupported
from tharkun.solve.absint.state import Frame, State
from tharkun.solve.absint.aval import NONE
from tharkun.solve.environment import proof_blocker


@pytest.mark.parametrize("source", [
    "__builtins__ = {}\ndef f():\n    raise ValueError()\ntry:\n    f()\nexcept NameError:\n    assert False\n",
    "__builtins__ = 1\ndef f():\n    raise ValueError()\ntry:\n    f()\nexcept TypeError:\n    assert False\n",
    "__builtins__ = {'ValueError': KeyError}\ndef f():\n    raise ValueError()\ntry:\n    f()\nexcept KeyError:\n    assert False\n",
    "b = {'ValueError': ValueError}\n__builtins__ = b\ndef f():\n    raise ValueError()\nb['ValueError'] = KeyError\ntry:\n    f()\nexcept KeyError:\n    assert False\n",
])
def test_changed_or_captured_builtin_mapping_preserves_native_negative(source):
    with pytest.raises(AssertionError):
        exec(source, {})
    result = prove(source, ("AssertionError",))
    assert result.verdict == "UNKNOWN" and not result.complete
    assert "__builtins__ binding" in result.reason


@pytest.mark.parametrize("source", [
    "__builtins__ = {}",
    "__builtins__: object = {}",
    "__builtins__ += {}",
    "(__builtins__ := {})",
    "del __builtins__",
    "if False:\n    __builtins__ = {}",
    "def __builtins__():\n    pass",
    "class __builtins__:\n    pass",
    "def f(__builtins__):\n    pass",
    "def f(*__builtins__):\n    pass",
    "def f(**__builtins__):\n    pass",
    "def f(*, __builtins__):\n    pass",
    "import sys as __builtins__",
    "from sys import version as __builtins__",
    "for __builtins__ in []:\n    pass",
    "with open('x') as __builtins__:\n    pass",
    "try:\n    pass\nexcept Exception as __builtins__:\n    pass",
    "match 1:\n    case __builtins__:\n        pass",
    "def f():\n    global __builtins__\n    __builtins__ = {}",
])
def test_every_binding_form_is_refused_before_slicing_or_static_lookup(source):
    assert proof_blocker(source) == "__builtins__ binding: unmodelled builtin namespace"
    result = prove(source + "\nassert 1 == 1\n", ("AssertionError",))
    assert result.verdict == "UNKNOWN" and not result.complete


@pytest.mark.parametrize("name", ["__builtins__", "external.py\x00__builtins__"])
def test_direct_argument_or_external_global_write_cannot_bypass_store_guard(name):
    analyzer = Analyzer("pass")
    state = State()
    state.frames.append(Frame("g"))
    with pytest.raises(Unsupported, match="binding the builtin namespace"):
        analyzer.write_var(state, 0, name, NONE)
    assert name not in state.frame.env


def test_delete_refuses_while_default_runtime_namespace_and_plain_builtins_remain():
    analyzer = Analyzer("pass")
    analyzer.run()
    state = State()
    state.frames.append(Frame("g"))
    state.frame.env["__builtins__"] = NONE
    context = FuncCtx(ast.parse("pass"), set(), set(), is_module=True)
    with pytest.raises(Unsupported, match="deleting the builtin namespace"):
        analyzer.delete_name("__builtins__", state, context)
    assert state.frame.env["__builtins__"] == NONE
    assert proof_blocker("__builtins__\nassert 1 == 1") is None
    result = prove("x = int(3)\nassert x == 3", ("AssertionError",))
    assert result.verdict == "TRUE", result.explain()


def test_engine_static_consumer_cannot_certify_replaced_namespace(tmp_path):
    from tharkun.sv.engine import _static_proof

    program = tmp_path / "program.py"
    program.write_text("__builtins__ = {}\ndef f():\n    raise ValueError()\ntry:\n    f()\nexcept NameError:\n    assert False\n")
    assert _static_proof(str(program), ("AssertionError",), lambda: 60, lambda _: None) is None
