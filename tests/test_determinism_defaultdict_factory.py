"""Narrow deterministic support for constant defaultdict factories."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tharkun.solve.determinism import determinism_census
from tharkun.sv import verify


SAFE_FLOAT = """\
from collections import defaultdict
d = defaultdict(lambda: float('inf'))
d[0, 1] = 3
assert d[0, 1] <= d[1, 0]
assert d[1, 0] == float('inf')
"""

SAFE_SIGNED_LITERAL = """\
from collections import defaultdict
d = defaultdict(lambda: -1.0)
d[1, 2] = -3
assert d[1, 2] < d[2, 1]
"""

SAFE_UPDATE_AND_RETURN = """\
from collections import defaultdict
def shortest(n, edges):
    dist = defaultdict(lambda: float('inf'))
    dist.update({(i, i): 0 for i in range(n)})
    dist.update(edges)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    return dist
actual = shortest(3, {(0, 1): 3, (1, 2): 4})
expected = {(0, 0): 0, (0, 1): 3, (0, 2): 7,
            (1, 0): float('inf'), (1, 1): 0, (1, 2): 4,
            (2, 0): float('inf'), (2, 1): float('inf'), (2, 2): 0}
expected_alias = expected
assert actual == expected_alias
assert actual[0, 2] == 7
"""


def test_constant_float_factory_proves_local_tuple_key_reads_and_writes():
    assert determinism_census(SAFE_FLOAT).ok


def test_constant_signed_literal_factory_proves_local_tuple_key_reads_and_writes():
    assert determinism_census(SAFE_SIGNED_LITERAL).ok


@pytest.mark.parametrize("factory", ["0", "1.5", "-1", "+1.5", "float('2.5')"])
def test_zero_argument_literal_factories_are_deterministic(factory):
    source = f"from collections import defaultdict\nd = defaultdict(lambda: {factory})\nx = d[1, 2]\n"
    assert determinism_census(source).ok, determinism_census(source).reason


def test_dictcomp_update_returned_mapping_equality_and_subscript_are_supported():
    result = determinism_census(SAFE_UPDATE_AND_RETURN)
    assert result.ok, result.reason


def test_generic_dictcomp_preserves_safe_string_key_lookup_certificate():
    source = """\
edges = {('a', 'b'): 1}
dist = {v: 0 for u, v in edges}
for (u, v), weight in edges.items():
    value = dist[v]
    assert value == 0
"""
    result = determinism_census(source)
    assert result.ok, result.reason


def test_every_helper_parameter_call_site_must_pass_a_dict_display():
    source = """\
from collections import defaultdict
def build(edges):
    d = defaultdict(lambda: 0)
    d.update(edges)
    return d
first = build({(0, 1): 2})
second = build({(1, 2): 3})
assert first == {(0, 1): 2}
assert second[1, 2] == 3
"""
    assert determinism_census(source).ok


def test_signed_zero_mapping_equality_is_deterministic():
    source = """\
from collections import defaultdict
def build():
    d = defaultdict(lambda: -0.0)
    d[0, 0] = -0.0
    return d
actual = build()
assert actual == {(0, 0): 0.0}
"""
    assert determinism_census(source).ok


def test_float_update_and_signed_zero_equality_are_deterministic():
    source = """\
from collections import defaultdict
def build():
    d = defaultdict(lambda: 1.5)
    d.update({(0, 0): -0.0, (0, 1): 2.5})
    return d
actual = build()
assert actual == {(0, 0): 0.0, (0, 1): 2.5}
"""
    assert determinism_census(source).ok


@pytest.mark.parametrize("source", [
        "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nprint(d[1, 2])\n",
        "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nprint(str(d[1, 2]))\n",
        "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nprint(format(d[1, 2], '.1f'))\n",
        "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nprint(f'{d[1, 2]}')\n",
        "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nprint(repr(d))\n",
        "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nprint(repr([d]))\n",
        "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nprint(str(d))\n",
        "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nprint(format(d))\n",
        "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nprint(f'{d!r}')\n",
        "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nraise Exception(d)\n",
])
def test_text_conversion_of_mapping_or_factory_values_is_refused(source):
    assert not determinism_census(source).ok, source


def test_factory_missing_read_inserts_once_per_key_natively():
    source = """\
from collections import defaultdict
calls = []
def factory():
    calls.append(1)
    return float('inf')
d = defaultdict(factory)
first = d[1, 2]
again = d[1, 2]
other = d[2, 3]
assert first == again == other == float('inf')
assert len(d) == 2 and len(calls) == 2
"""
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONSTARTUP"}}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run([sys.executable, "-B", "-c", source], text=True,
                               capture_output=True, timeout=30, env=env)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("source", [
    "from collections import defaultdict\nd = defaultdict(lambda: float('inf'), {(1, 2): 0})\nassert False\n",
    "from collections import defaultdict\nd = defaultdict(lambda x: float('inf'))\nassert False\n",
    "from collections import defaultdict\nd = defaultdict(lambda x=0: 1)\nassert False\n",
    "from collections import defaultdict\nd = defaultdict(lambda *, x=0: 1)\nassert False\n",
    "from collections import defaultdict\nd = defaultdict(lambda *args: 1)\nassert False\n",
    "from collections import defaultdict\nd = defaultdict(lambda **kwargs: 1)\nassert False\n",
    "from collections import defaultdict\ndef make():\n    value = 1\n    return defaultdict(lambda: value)\nd = make()\nassert False\n",
    "from collections import defaultdict\nfloat = int\nd = defaultdict(lambda: float('inf'))\nassert False\n",
    "from collections import defaultdict\ndef f(float):\n    d = defaultdict(lambda: float('inf'))\n    return d[1, 2]\nf(0)\n",
    "from collections import defaultdict\nmake = defaultdict\nd = make(lambda: 0)\nassert False\n",
    "from collections import defaultdict\nfactory = lambda: 0\nd = defaultdict(factory)\nassert False\n",
    "from collections import defaultdict\ndefaultdict = dict\nd = defaultdict(lambda: 0)\nassert False\n",
    "from collections import defaultdict\ndef build(defaultdict):\n    d = defaultdict(lambda: 0)\n    return d[0, 1]\nbuild(dict)\n",
    "import collections as c\nc.defaultdict = dict\nd = c.defaultdict(lambda: 0)\nassert False\n",
    "from collections import defaultdict\ncalls = []\nd = defaultdict(lambda: (calls.append(1), float('inf'))[1])\nassert False\n",
    "from collections import defaultdict\nd = defaultdict(lambda: float('nan'))\nassert d[1, 2] < 0\n",
    "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nd[True, 2] = 1\nassert False\n",
    "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nd[1, (2, 3)] = 1\nassert False\n",
    "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nd[1, 2] = True\nassert False\n",
    "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nd[object(), 2] = 1\nassert False\n",
    "from collections import defaultdict\nd = defaultdict(lambda: object())\nx = d[1, 2]\n",
    "from collections import defaultdict\ndef build():\n    d = defaultdict(lambda: 0)\n    d[0, 0] = 0\n    return d\nfor key in build():\n    print(key)\n",
    "from collections import defaultdict\ndef build():\n    d = defaultdict(lambda: 0)\n    return d\nreturned = build()\ndef observe():\n    assert returned == {(0, 0): 0}\nobserve()\n",
    "from collections import defaultdict\ndef build():\n    d = defaultdict(lambda: 0)\n    return d\nbox = {}\nbox[0] = build()\n",
    "from collections import Counter, defaultdict\nc = Counter()\nc[1] += 1\nd = defaultdict(lambda: 0)\nassert c[1] == 1 and d[0, 0] == 0\n",
    "from collections import defaultdict, deque\nq = deque()\nq.append(1)\nassert q.popleft() == 1\nd = defaultdict(lambda: 0)\nassert d[0, 0] == 0\n",
    "from collections import defaultdict\nd = defaultdict(lambda: id(object()))\nd[0, 1]\n",
    "from collections import defaultdict\nd = defaultdict(lambda: 0)\nd.update({(x, x): 1 for x in {1, 2, 3}})\n",
    "from collections import defaultdict\ndef build(edges):\n    d = defaultdict(lambda: 0)\n    d.update(edges)\n    return d\nunknown = dict()\nresult = build(unknown)\n",
    "from collections import defaultdict\ndef build(edges):\n    d = defaultdict(lambda: 0)\n    d.update(edges)\n    return d\ngood = build({(0, 1): 1})\nbad = build(dict())\n",
    "from collections import defaultdict\ngraph = {(1, 2): 1}\nalias = graph\nalias[True] = 2\nd = defaultdict(lambda: 0)\nd.update(graph)\n",
    "from collections import defaultdict\ndef build():\n    d = defaultdict(lambda: 0)\n    d[1, 2] = 1\n    return d\nexpected = {(1, 2): 1}\nalias = expected\nalias[(2, 3)] = float('nan')\nassert build() == expected\n",
    "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nd[0, 0] = float('inf')\nassert d[0, 0] - d[1, 1] == 0\n",
    "from collections import defaultdict\nd = defaultdict(lambda: float('inf'))\nd[0, 0] = float('inf')\nassert d[0, 0] + -float('inf') < 0\n",
])
def test_factory_and_escape_boundaries_never_become_certificates(source):
    assert not determinism_census(source).ok, determinism_census(source).reason


def test_public_verifier_accepts_the_narrow_local_factory_case(tmp_path):
    path = tmp_path / "safe_defaultdict.py"
    path.write_text(SAFE_FLOAT, encoding="utf-8")
    result = verify(str(path), ("AssertionError",), python=sys.executable,
                    budget_seconds=60, run_timeout=30, max_runs=1, jobs=1,
                    prove=False, concolic=False, canary_seconds=0,
                    early_stop_seconds=0, witness_dir=tmp_path / "witness")
    assert result.verdict == "true", (result.component, result.reason)


def test_public_verifier_accepts_update_return_equality_and_subscript(tmp_path):
    path = tmp_path / "safe_defaultdict_full_slice.py"
    path.write_text(SAFE_UPDATE_AND_RETURN, encoding="utf-8")
    result = verify(str(path), ("AssertionError",), python=sys.executable,
                    budget_seconds=60, run_timeout=30, max_runs=1, jobs=1,
                    prove=False, concolic=False, canary_seconds=0,
                    early_stop_seconds=0, witness_dir=tmp_path / "witness")
    assert result.verdict == "true", (result.component, result.reason)


def test_public_verifier_does_not_promote_factory_side_effects(tmp_path):
    source = """\
from collections import defaultdict
calls = []
d = defaultdict(lambda: (calls.append(1), float('inf'))[1])
assert d[1, 2] == float('inf')
"""
    path = tmp_path / "effectful_defaultdict.py"
    path.write_text(source, encoding="utf-8")
    result = verify(str(path), ("AssertionError",), python=sys.executable,
                    budget_seconds=60, run_timeout=30, max_runs=1, jobs=1,
                    prove=False, concolic=False, canary_seconds=0,
                    early_stop_seconds=0, witness_dir=tmp_path / "witness")
    assert result.verdict != "true", (result.component, result.reason)
