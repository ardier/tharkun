"""Fixed native controls for MG #147 follow-ups, carried in [internal ref]."""
import ast
import math
import random

import pytest

from tharkun.solve.contracts.rules import get_rule
from tharkun.sv._symexec import _Exec, Unsupported


def test_math_function_attribute_refuses_without_visitor_crash():
    assert callable(math.sqrt)
    with pytest.raises(Unsupported, match="Attribute"):
        _Exec({"math": math}, {}, {}).ev(ast.parse("math.sqrt", mode="eval").body)


def test_math_constant_call_refuses_native_typeerror():
    with pytest.raises(TypeError, match="not callable"):
        math.pi(1)
    with pytest.raises(Unsupported, match="TypeError"):
        _Exec({"math": math}, {}, {}).ev(ast.parse("math.pi(1)", mode="eval").body)


def test_math_constant_attribute_still_returns_value():
    assert _Exec({"math": math}, {}, {}).ev(ast.parse("math.pi", mode="eval").body) == math.pi


# Literal wrong-arity samples; no generated native program executor.
TWINS = [
    ("open", "f = open()\n", open, ()),
    ("random.choice", "import random\nx = random.choice()\n", random.choice, ()),
    ("random.randint", "import random\nx = random.randint()\n", random.randint, ()),
    ("random.random", "import random\nx = random.random(0)\n", random.random, (0,)),
    ("random.randrange", "import random\nx = random.randrange()\n", random.randrange, ()),
    ("random.sample", "import random\nx = random.sample()\n", random.sample, ()),
    ("random.shuffle", "import random\nrandom.shuffle()\n", random.shuffle, ()),
    ("random.uniform", "import random\nx = random.uniform()\n", random.uniform, ()),
]


@pytest.mark.parametrize("qualname,source,native,args", TWINS)
def test_restored_arity_twins_match_native(qualname, source, native, args):
    with pytest.raises(TypeError):
        native(*args)
    assert get_rule(qualname).twin(qualname) == source
