"""Heap operations are deterministic only for address-free, comparable values."""

import ast
import heapq
import os
import subprocess
import sys

import pytest

from tharkun.solve.determinism import determinism_census
from tharkun.solve.pyresolve import Resolver, Top, _HEAPQ_STAR_EXPORTS
from tharkun.sv import verify


def _resolver(source, *, filename="<program>", declared_files=(), allowed=("heapq",)):
    return Resolver(ast.parse(source, filename=filename), allowed, filename=filename,
                    _declared_files=declared_files)


def test_heapq_star_import_resolves_exact_cpython314_exports():
    expected = (
        "heappush", "heappop", "heapify", "heapreplace", "heappushpop",
        "heappush_max", "heappop_max", "heapify_max", "heapreplace_max",
        "heappushpop_max", "nlargest", "nsmallest", "merge",
    )
    assert _HEAPQ_STAR_EXPORTS == expected
    assert tuple(heapq.__all__) == expected

    resolver = _resolver("from heapq import *\nheapify([])\n")
    assert resolver.imports.names == {name: ("heapq", name) for name in expected}
    call = next(n for n in ast.walk(resolver.tree) if isinstance(n, ast.Call))
    assert resolver.call_name(call) == "heapq.heapify"


@pytest.mark.parametrize("source", [
    "from math import *\n",
    "from heapq import *\nheapify = 1\n",
    "def heapify(x): return x\nfrom heapq import *\n",
    "from heapq import *\ndef f(heapify): return heapify\n",
])
def test_heapq_star_import_keeps_other_star_and_rebinding_refusals(source):
    with pytest.raises(Top):
        _resolver(source)


def test_heapq_star_import_refuses_declared_sibling_shadow(tmp_path):
    main = tmp_path / "main.py"
    sibling = tmp_path / "heapq.py"
    main.write_text("from heapq import *\n", encoding="utf-8")
    sibling.write_text("def heapify(xs): return xs\n", encoding="utf-8")

    with pytest.raises(Top, match="stdlib shadow 'heapq'"):
        _resolver(main.read_text(encoding="utf-8"), filename=str(main),
                  declared_files=(str(main), str(sibling)))


@pytest.mark.parametrize("source", [
    "import heapq as h\nh.heapify([3, 1, 2])\n",
    "from heapq import *\nh = [3, 1, 2]\nheapify(h)\nheappush(h, 0)\nx = heappop(h)\ny = heappushpop(h, 4)\nz = heapreplace(h, 5)\na = nsmallest(2, [3, 1, 2])\nb = nlargest(2, [3, 1, 2])\n",
    "import heapq\nh = [(1, 'b'), (2, 'a')]\nheapq.heapify(h)\nx = heapq.heappop(h)\n",
    "import heapq\nh = ['c', 'a', 'b']\nheapq.heapify(h)\nx = heapq.heappop(h)\n",
    "import heapq\nh = [3, 1, 2]\nheapq.heapify(h)\na = str(heapq.heappop(h))\nb = str(heapq.heappushpop(h, 4))\nc = str(heapq.heapreplace(h, 5))\nd = str(heapq.nsmallest(2, [3, 1, 2]))\nprint(a, b, c, d)\n",
])
def test_heapq_accepts_closed_builtin_comparison_values(source):
    assert determinism_census(source).ok


@pytest.mark.parametrize("source", [
    "import heapq\nh = [1, 'x']\nheapq.heapify(h)\n",
    "import heapq\nclass Node:\n    def __lt__(self, other): return id(self) < id(other)\nh = [Node(), Node()]\nheapq.heapify(h)\n",
    "import heapq\nclass Node:\n    def __lt__(self, other): raise AssertionError('user comparison')\nh = [Node(), Node()]\nheapq.heapify(h)\n",
    "import heapq\nclass Node: pass\nh = [(1, Node()), (1, Node())]\nheapq.heapify(h)\n",
    "import heapq\nh = [(*[1, 2], 'a'), (1, 'a')]\nheapq.heapify(h)\n",
    "import heapq\nh = []\nheapq.heapify(h)\nheapq.heappop(h)\n",
    "import heapq\nh = [1, 2]\nheapq.heapify(h)\nh.append(object())\nheapq.heappop(h)\n",
    "import heapq\nh = [1, 2]\nheapq.heapify(h)\nheapq.nsmallest(1, h, key=abs)\n",
    "import heapq\nh = [1, 2]\nheapq.heapify(h)\nlist(heapq.merge(h))\n",
    "import heapq\nh = [1, 2]\nheapq.heapify(h)\npush = heapq.heappush\npush(h, 3)\n",
    "import heapq\nclass Item:\n    def __repr__(self): return hex(id(self))\nh = [Item()]\nheapq.heapify(h)\nprint(repr(heapq.heappop(h)))\n",
])
def test_heapq_refuses_user_hooks_mixed_or_escaped_values(source):
    assert not determinism_census(source).ok


def test_mixed_heap_inputs_really_raise_typeerror_natively():
    with pytest.raises(TypeError):
        heapq.heapify([1, "x"])


def test_heapq_user_comparison_really_executes_natively():
    class Node:
        def __lt__(self, other):
            raise AssertionError("user comparison")

    with pytest.raises(AssertionError, match="user comparison"):
        heapq.heapify([Node(), Node()])


def test_heapq_address_based_comparison_really_executes_natively():
    class Node:
        def __lt__(self, other):
            return id(self) < id(other)

    values = [Node(), Node()]
    heapq.heapify(values)
    assert len(values) == 2


def _native_and_verify(tmp_path, source, exception, *, sibling=None):
    main = tmp_path / "main.py"
    main.write_text(source, encoding="utf-8")
    files = [str(main)]
    task = None
    if sibling is not None:
        module = tmp_path / "heapq.py"
        module.write_text(sibling, encoding="utf-8")
        files.append(str(module))
        task = tmp_path / "task.yml"
        task.write_text("format_version: '2.0'\ninput_files: main.py\nrequired_files:\n  - heapq.py\n"
                        "options:\n  language: Python\n", encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONSTARTUP"}}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    native = subprocess.run([sys.executable, "-B", str(main)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    result = verify(str(main), (exception,), python=sys.executable,
                    task=str(task) if task is not None else None,
                    budget_seconds=60, run_timeout=30, max_runs=1, jobs=1,
                    prove=False, concolic=False, canary_seconds=1)
    return native, result


@pytest.mark.parametrize(("source", "exception", "native_exc", "sibling"), [
    ("import heapq\nheapq.heapify([1, 'x'])\n", "TypeError", "TypeError", None),
    ("from heapq import heapify\ndef heapify(xs):\n    raise AssertionError('local shadow')\nheapify([])\n",
     "AssertionError", "AssertionError", None),
    ("from heapq import *\nheapify([])\n", "AssertionError", "AssertionError",
     "raise AssertionError('sibling shadow')\n"),
    ("import heapq\nclass Node:\n    def __lt__(self, other): raise AssertionError('user comparison')\n"
     "heapq.heapify([Node(), Node()])\n",
     "AssertionError", "AssertionError", None),
    ("import heapq\nclass Node:\n    def __lt__(self, other): return id(self) < id(other)\n"
     "h = [Node(), Node()]\nheapq.heapify(h)\n",
     "AssertionError", None, None),
])
def test_heapq_adversarial_native_and_verifier_controls(tmp_path, source, exception, native_exc, sibling):
    native, result = _native_and_verify(tmp_path, source, exception, sibling=sibling)
    if native_exc is None:
        assert native.returncode == 0, native.stderr
        assert result.verdict == "unknown", result
    else:
        assert native.returncode != 0 and f"{native_exc}:" in native.stderr, native.stderr
        assert result.verdict in {"false", "unknown"}, result
