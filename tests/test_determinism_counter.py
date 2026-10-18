"""The Counter certificate is restricted to the LCS-safe operation subset."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from tharkun.solve.determinism import determinism_census
from tharkun.sv import verify


def _counter_source(key: str, *, constructor: str = "Counter()") -> str:
    return (
        "from collections import Counter\n"
        f"counts = {constructor}\n"
        f"assert counts[{key}] == 0\n"
        "if counts:\n"
        "    assert False, 'missing lookup inserted a key'\n"
        f"counts[{key}] += 2\n"
        "selected = 2 if counts else 0\n"
        "if counts:\n"
        "    maximum = max(counts.values())\n"
        "else:\n"
        "    maximum = 0\n"
        "assert maximum == selected == 2\n"
    )


def _native(tmp_path, source: str, *, seed: int = 1):
    path = tmp_path / "counter_case.py"
    path.write_text(source, encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONSTARTUP"}}
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-B", str(path)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30, check=False,
    )


@pytest.mark.parametrize("predecessor", ("dp[i - 1, j]", "dp[i - 1, j - 1]"))
def test_lcs_shaped_counter_programs_are_certified_without_benchmark_files(predecessor):
    source = textwrap.dedent(f"""\
        from collections import Counter
        def lcs_length(s, t):
            dp = Counter()
            for i in range(len(s)):
                for j in range(len(t)):
                    if s[i] == t[j]:
                        dp[i, j] = {predecessor} + 1
            return max(dp.values()) if dp else 0
        assert lcs_length('witch', 'sandwich') == 2
        assert lcs_length('fun', '') == 0
    """)
    result = determinism_census(source)
    assert result.ok, result.reason


def test_counter_missing_read_and_exact_count_updates_are_natively_sound(tmp_path):
    for key in ("7", "'key'", "(1, 2)"):
        source = _counter_source(key)
        result = determinism_census(source)
        assert result.ok, result.reason
        native = _native(tmp_path, source)
        assert native.returncode == 0, native.stderr


def test_counter_constructor_aliases_keep_the_supported_subset():
    for source in (
        _counter_source("(2, 3)", constructor="Bag()")
        .replace("from collections import Counter", "from collections import Counter as Bag"),
        _counter_source("(2, 3)", constructor="module.Counter()")
        .replace("from collections import Counter", "import collections as module"),
    ):
        result = determinism_census(source)
        assert result.ok, result.reason


def test_plain_counter_alias_preserves_known_heap_identity():
    source = """\
        from collections import Counter
        c = Counter()
        alias = c
        assert alias[(2, 3)] == 0
        alias[(2, 3)] += 1
        assert c[(2, 3)] == 1
        assert max(c.values()) == 1
    """
    result = determinism_census(textwrap.dedent(source))
    assert result.ok, result.reason


def test_direct_max_values_shape_is_narrow():
    accepted = _counter_source("(1, 2)")
    assert determinism_census(accepted).ok

    for source in (
        "from collections import Counter\nc = Counter()\nmax(c.values(), default=0)\n",
        "from collections import Counter\nc = Counter()\nmax(c.values(), key=abs)\n",
        "from collections import Counter\nc = Counter()\nmax(c)\n",
        "from collections import Counter\nc = Counter()\nv = c.values()\nprint(v)\n",
    ):
        assert not determinism_census(source).ok


def _seed_dependent_counter_source(operation: str) -> str:
    value = "1" if operation == "counts.most_common()" else "index"
    return (
        "from collections import Counter\n"
        "keys = {'amber', 'blue', 'copper', 'denim', 'emerald', 'fuchsia', 'gray', 'hazel'}\n"
        "counts = Counter()\n"
        "for index, key in enumerate(keys, 1):\n"
        f"    counts[key] = {value}\n"
        f"print({operation})\n"
    )


def test_hash_ordered_counter_observations_remain_uncertified():
    for operation in (
        "tuple(counts)",
        "tuple(counts.keys())",
        "list(counts)",
        "counts.most_common()",
        "list(counts.values())",
    ):
        source = _seed_dependent_counter_source(operation)
        result = determinism_census(source)
        assert not result.ok, result.reason


def test_hash_order_control_cannot_prove_assertion_safety(tmp_path):
    source = _seed_dependent_counter_source("list(counts)") + (
        "assert next(iter(counts)) in {'amber', 'blue', 'copper', 'denim'}\n"
    )
    path = tmp_path / "counter_hash_order.py"
    path.write_text(source, encoding="utf-8")
    verdict = verify(
        str(path), ("AssertionError",), python=sys.executable,
        budget_seconds=60, run_timeout=30, max_runs=1, jobs=1,
        prove=False, concolic=False, canary_seconds=0,
    )
    assert verdict.verdict != "true", verdict


def test_nonempty_constructor_shapes_and_counter_escapes_are_refused():
    sources = (
        "from collections import Counter\nc = Counter(['a', 'b'])\n",
        "from collections import Counter\nc = Counter({'a': 1})\n",
        "from collections import Counter\nc = Counter()\nupdate = c.update\nupdate({1: 2})\n",
        "from collections import Counter\nc = Counter()\nget = c.get\nget(1)\n",
        "from collections import Counter\ndef inspect(value): return value[1]\nc = Counter()\ninspect(c)\n",
        "from collections import Counter\nc = Counter()\nc.update({1: 2})\n",
        "from collections import Counter\nCounter = dict\nc = Counter()\n",
        "from collections import Counter\ndef build(Counter): return Counter()\nbuild(dict)\n",
        "import collections as module\nmodule = type('M', (), {'Counter': dict})\nc = module.Counter()\n",
        "from collections import Counter\nmake = Counter\nc = make()\n",
        "from collections import Counter\nc = Counter()\nmax = lambda value: 0\nmax(c.values())\n",
    )
    for source in sources:
        assert not determinism_census(source).ok


def test_counter_certificate_refuses_nearby_key_and_count_types():
    sources = (
        "from collections import Counter\nc = Counter()\nc[1] = False\n",
        "from collections import Counter\nc = Counter()\nc[1] = 1.5\n",
        "from collections import Counter\nc = Counter()\nc[1] = 'one'\n",
        "from collections import Counter\nc = Counter()\nc[1] = None\n",
        "from collections import Counter\nc = Counter()\nc[1] = 1\nc[1] += 1.5\n",
        "from collections import Counter\nc = Counter()\nc[True] = 1\n",
        "from collections import Counter\nc = Counter()\nc[1.5] = 1\n",
        "from collections import Counter\nc = Counter()\nc[(1, True)] = 1\n",
        "from collections import Counter\nc = Counter()\nc[(1, (2,))] = 1\n",
        "from collections import Counter\nc = Counter()\nalias = c\nalias = {}\nalias[1] = 1\n",
        "from collections import Counter\n"
        "def make():\n    c = Counter()\n    return c\n"
        "counts = make()\nprint(counts[1])\n",
    )
    for source in sources:
        assert not determinism_census(source).ok


def test_custom_key_and_count_callbacks_raise_natively_and_are_refused(tmp_path):
    key_source = textwrap.dedent("""\
        from collections import Counter
        class Key:
            def __hash__(self): return 7
            def __eq__(self, other): raise AssertionError('key equality callback')
        counts = Counter()
        counts[Key()] = 1
        counts[Key()] = 2
    """)
    count_source = textwrap.dedent("""\
        from collections import Counter
        class Count:
            def __add__(self, other): raise AssertionError('count addition callback')
        counts = Counter()
        counts[1] = Count()
        counts[1] += 1
    """)
    for source in (key_source, count_source):
        assert not determinism_census(source).ok
        native = _native(tmp_path, source)
        assert native.returncode != 0
        assert "AssertionError:" in native.stderr
