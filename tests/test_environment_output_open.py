"""[internal ref] / [internal ref]: the shared environment model classifies a write-only ``open()`` as OUTPUT.

``tharkun.solve.environment.open_effect`` is the one classification; ``proof_blocker`` (the gate
absint and the engine's static proofs consult) lets an output open through and still blocks every
read.  What an output open can raise (``OUTPUT_OPEN_RAISES``) is part of the model, and no analysis
consulting it may drop it: ``absint`` has no model of ``open`` and so still gives no proof for a
program that opens a file.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from tharkun.solve.environment import (
    OUTPUT_MODES,
    OUTPUT_OPEN_RAISES,
    READ_MODES,
    open_effect,
    proof_blocker,
)


def _effect(expr: str) -> str:
    return open_effect(ast.parse(expr, mode="eval").body)


# ---------------------------------------------------------------------------
# the classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["w", "wb", "wt", "bw", "tw", "a", "ab", "at", "ba", "ta", "x", "xb", "xt", "bx", "tx"])
def test_a_constant_write_only_mode_is_output(mode):
    assert _effect(f"open(p, {mode!r})") == "output"
    assert _effect(f"open(p, mode={mode!r})") == "output"
    assert _effect(f"open(file=p, mode={mode!r})") == "output"


@pytest.mark.parametrize("mode", ["w+", "wb+", "w+b", "a+", "ab+", "x+", "xb+", "r+", "rb+", "+w"])
def test_an_update_mode_reads_and_is_not_output(mode):
    assert _effect(f"open(p, {mode!r})") == "unknown"


@pytest.mark.parametrize("mode", ["r", "rb", "rt", "br", "tr"])
def test_a_read_mode_is_read(mode):
    assert _effect(f"open(p, {mode!r})") == "read"


def test_no_mode_is_the_default_read():
    assert _effect("open(p)") == "read"
    assert _effect("open(p, encoding='utf-8')") == "read"


@pytest.mark.parametrize("expr", [
    "open(p, mode)",                  # not a constant
    "open(p, 'w' + 'b')",
    "open(p, m or 'w')",
    "open(p, 'ww')",                  # invalid: ValueError
    "open(p, 'wa')",
    "open(p, 'wbt')",
    "open(p, 'U')",
    "open(p, '')",
    "open(p, b'w')",                  # a bytes mode is a TypeError
    "open(p, 5)",
    "open(1, 'w')",                   # a file descriptor
    "open(fd=1)",
    "open()",
    "open(p, 'w', opener=f)",         # runs a callback
    "open(p, 'w', closefd=False)",
    "open(*args)",
    "open(p, *rest)",
    "open(**kw)",
    "open(p, 'w', mode='r')",         # mode twice: TypeError
    "open(p, 'w', 1, 'utf-8', None, None, 7)",   # more arguments than open takes
])
def test_everything_else_is_unknown(expr):
    assert _effect(expr) == "unknown", expr


@pytest.mark.parametrize("expr", [
    "open(p, 'w', 1)",
    "open(p, 'w', encoding='utf-8')",
    "open(p, 'w', encoding='latin1', newline='')",
    "open(p, 'wb', buffering=0)",
    "open(p, 'w', errors='replace')",
    "open(p, 'w', 1, 'utf-8', 'strict', '\\n')",
])
def test_the_plain_keyword_and_positional_arguments_are_output(expr):
    assert _effect(expr) == "output", expr


def test_the_mode_sets_are_what_python_accepts():
    for mode in OUTPUT_MODES:
        assert len(set(mode)) == len(mode) and set(mode) <= set("waxbt") and sum(c in "wax" for c in mode) == 1
    for mode in READ_MODES:
        assert set(mode) <= set("rbt") and "r" in mode
    assert not OUTPUT_MODES & READ_MODES
    assert all("+" not in m for m in OUTPUT_MODES | READ_MODES)


def test_what_an_output_open_can_raise_stays_part_of_the_model():
    # the checks a proof that accepts an output open must model or stay open at
    assert set(OUTPUT_OPEN_RAISES) == {"OSError", "TypeError", "ValueError"}
    assert "TypeError" in ("TypeError", "AttributeError")  # the dynamic-typing property class
    with pytest.raises(TypeError):
        open(None, "w")
    with pytest.raises(ValueError):
        open("a\x00b", "w")


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def _blocker(src: str):
    return proof_blocker(textwrap.dedent(src))


@pytest.mark.parametrize("src", [
    "f = open(p, 'wb')\nf.write(b'x')\nf.close()\n",
    "with open(p, 'w', encoding='latin1', newline='') as fp:\n    fp.write('x')\n",
    "with open(p, 'ab') as fp:\n    fp.write(b'x')\n",
    "with open(p, 'xb') as fp:\n    fp.write(b'x')\n",
    "def write_ppm(im, filename):\n    with open(filename, 'wb') as fp:\n        fp.write(im)\n",
])
def test_a_write_only_open_is_not_an_input_effect(src):
    assert _blocker(src) is None


@pytest.mark.parametrize("src, what", [
    ("x = open(p).read()\n", "unknown file read"),
    ("with open(p, 'rb') as fp:\n    d = fp.read()\n", "unknown file read"),
    ("with open(p, 'r+') as fp:\n    d = fp.read()\n", "unknown file read"),
    ("with open(p, 'w+') as fp:\n    d = fp.read()\n", "unknown file read"),
    ("with open(p, 'a+') as fp:\n    d = fp.read()\n", "unknown file read"),
    ("with open(p, mode) as fp:\n    pass\n", "unknown file read"),
    ("with open(1, 'w') as fp:\n    pass\n", "unknown file read"),
    ("with open(p, 'w', opener=f) as fp:\n    pass\n", "unknown file read"),
    ("open('data.txt', 'r')\n", "unknown file read"),
])
def test_a_read_or_an_unclassifiable_open_still_blocks(src, what):
    blocker = _blocker(src)
    assert blocker is not None and what in blocker


def test_a_program_that_writes_a_file_and_reads_it_back_is_still_blocked_at_the_read():
    blocker = _blocker("""
        with open(p, 'w') as fp:
            fp.write('0')
        divisor = int(open(p).read())
        r = 1 // divisor
    """)
    assert blocker is not None and "file read at line 4" in blocker


@pytest.mark.parametrize("src", [
    "def open(p, m):\n    return 1\nopen(p, 'w')\n",
    "from io import open\nopen(p, 'w')\n",
    "from codecs import open as open\nopen(p, 'w')\n",
    "import gzip as open\nopen(p, 'w')\n",
    "open = lambda p, m: 1\nopen(p, 'w')\n",
    "def f(open):\n    open(p, 'w')\n",
    "for open in [print]:\n    open(p, 'w')\n",
    "with g() as open:\n    open(p, 'w')\n",
    "try:\n    pass\nexcept Exception as open:\n    pass\nopen(p, 'w')\n",
    "global open\nopen(p, 'w')\n",
    "class open:\n    pass\nopen(p, 'w')\n",
])
def test_a_rebound_open_is_not_the_builtin(src):
    assert _blocker(src) is not None


@pytest.mark.parametrize("src", [
    "import io\nio.open(p, 'w')\n",
    "import builtins\nbuiltins.open(p, 'w')\n",
    "from pathlib import Path\nPath(p).open('w')\n",
    "import os\nos.open(p, os.O_WRONLY)\n",
    "import tarfile\ntarfile.open(p, 'w')\n",
])
def test_only_the_builtin_name_open_is_classified_output(src):
    assert _blocker(src) is not None


def test_an_output_open_next_to_a_read_blocks_at_the_read_whichever_comes_first():
    assert "file read" in _blocker("a = open(p).read()\nopen(q, 'w').write('x')\n")
    assert "file read" in _blocker("open(q, 'w').write('x')\na = open(p).read()\n")


# ---------------------------------------------------------------------------
# what consults it
# ---------------------------------------------------------------------------


def test_absint_gives_no_proof_for_a_program_that_opens_a_file():
    from tharkun.solve.absint import prove

    result = prove(textwrap.dedent("""
        def main(p):
            with open(p, 'wb') as fp:
                fp.write(b'x')
            return 1
        main('out')
    """), ["ZeroDivisionError"], filename="<program>")
    # the gate lets the write through; absint has no model of open or with and aborts there
    assert result.verdict == "UNKNOWN" and "environment" not in result.reason
    assert not result.complete


def test_absint_is_stopped_at_the_gate_by_a_read_and_by_a_write_then_read_back():
    from tharkun.solve.absint import prove

    for src in (
        "def main(p):\n    return open(p).read()\nmain('x')\n",
        "def main(p):\n    with open(p, 'w') as fp:\n        fp.write('0')\n    return 1 // int(open(p).read())\nmain('x')\n",
    ):
        result = prove(src, ["ZeroDivisionError"], filename="<program>")
        assert result.verdict == "UNKNOWN" and result.reason.startswith("environment: unknown file read")


def test_the_determinism_census_uses_the_same_classification():
    from tharkun.solve.determinism import determinism_census

    ok = determinism_census("def main(p):\n    with open(p, 'xb') as fp:\n        fp.write(b'x')\nmain('o')\n")
    assert ok.ok, ok.reason
    for mode in ("r", "r+", "w+", "a+", "wa", "ww"):
        census = determinism_census(f"def main(p):\n    with open(p, {mode!r}) as fp:\n        fp.write('x')\nmain('o')\n")
        assert not census.ok, mode


def test_the_stationary_probes_output_modes_are_a_subset_of_the_shared_ones():
    from tharkun.solve.output_effects import MODES

    assert MODES <= OUTPUT_MODES
