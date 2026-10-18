"""[internal ref]: absint models the write side of a file opened by a write-only ``open()``.

The contracts (``contracts/rules_fileio.py``) state every raise claim; the model (``Analyzer.c_open``,
``outfile_method``, ``s_With``) upgrades a class to proven only for what it knows of the abstract
arguments, and refuses (``Unsupported``) everything else: a read, an update mode, a descriptor,
``buffering``, ``tell``/``seek``/attributes, a handle used as a value, another context manager.
What the environment does (``OSError``) and a closed file or an encoding error (``ValueError``) are
not property classes: a path that raises them ends.
"""

from __future__ import annotations

import ast
import os
import random
import textwrap

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.absint.interp import Analyzer, Unsupported


def _prove(src: str, classes=("ZeroDivisionError",)):
    return prove(textwrap.dedent(src), list(classes), filename="<program>")


def _unsupported(src: str) -> str:
    """The reason the analyzer refuses ``src`` (it runs to the end otherwise: the test fails)."""
    an = Analyzer(textwrap.dedent(src), "<program>")
    with pytest.raises(Unsupported) as info:
        an.run()
    return str(info.value)


# ---------------------------------------------------------------------------
# what is modelled
# ---------------------------------------------------------------------------


def test_a_text_file_written_in_a_with_block_is_proved():
    for classes in (["ZeroDivisionError"], ["TypeError"], ["IndexError", "KeyError"], ["AttributeError"]):
        result = _prove("""
            def main(p):
                with open(p, 'w', encoding='utf-8', newline='') as fp:
                    fp.write('P6\\n')
                    fp.write('x' + 'y')
                return 1 // 1
            main('out.ppm')
        """, classes)
        assert result.verdict == "TRUE", (classes, result.reason)
        assert result.complete


def test_the_bare_open_close_pair_and_every_method_of_the_family():
    result = _prove("""
        def main(p):
            fp = open(p, 'a')
            fp.write('x')
            fp.writelines(['a', 'b'])
            fp.flush()
            fp.close()
            return 1 // 1
        main('out')
    """, ["ZeroDivisionError", "TypeError", "AttributeError", "IndexError", "KeyError"])
    assert result.verdict == "TRUE", result.reason


def test_the_mode_spellings_are_all_output():
    for mode in ("w", "a", "x", "wt", "at", "xt", "tw"):
        result = _prove(f"""
            def main(p):
                with open(p, {mode!r}) as fp:
                    fp.write('x')
                return 1 // 1
            main('out')
        """, ["ZeroDivisionError", "TypeError"])
        assert result.verdict == "TRUE", (mode, result.reason)


def test_a_str_that_is_unknown_is_a_valid_path_and_a_valid_text():
    result = _prove("""
        import _sv_verifier
        def main(name):
            with open(name, 'w') as fp:
                fp.write(name)
            return 1 // 1
        main(_sv_verifier.nondet_str())
    """, ["ZeroDivisionError", "TypeError"])
    assert result.verdict == "TRUE", result.reason


def test_a_binary_file_takes_an_array_and_writelines_takes_a_list_of_arrays():
    result = _prove("""
        import array
        def main(p):
            buf = array.array('B', [0, 1, 2])
            with open(p, 'wb') as fp:
                fp.write(buf)
                fp.writelines([buf, buf])
            return 1 // 1
        main('out.bin')
    """, ["ZeroDivisionError", "TypeError", "IndexError"])
    assert result.verdict == "TRUE", result.reason


def test_two_handles_nested_handles_and_a_handle_passed_to_a_function():
    result = _prove("""
        def put(f, s):
            f.write(s)
        def main(p, q):
            with open(p, 'w') as a, open(q, 'wb') as b:
                put(a, 'x')
                with open(p, 'a') as c:
                    c.write('y')
            return 1 // 1
        main('a', 'b')
    """, ["ZeroDivisionError", "TypeError"])
    assert result.verdict == "TRUE", result.reason


def test_a_with_block_in_a_loop_and_with_return_break_continue():
    result = _prove("""
        def first(p, n):
            for i in range(n):
                with open(p, 'a') as fp:
                    fp.write('x')
                    if i > 3:
                        break
                    if i == 2:
                        continue
            while n > 0:
                with open(p, 'w') as fp:
                    fp.write('y')
                    n = n - 1
                    if n == 1:
                        return n
            return 0
        first('out', 5)
    """, ["ZeroDivisionError", "TypeError"])
    assert result.verdict == "TRUE", result.reason


def test_an_unrelated_error_after_the_file_is_still_found():
    result = _prove("""
        def main(p, d):
            with open(p, 'w') as fp:
                fp.write('x')
            return 1 // d
        main('out', 0)
    """)
    assert result.verdict == "UNKNOWN"
    divide = [r for r in result.records if r["exc"] == "ZeroDivisionError" and r["kind"] == "divide"]
    assert divide and divide[0]["status"] == "may_raise"


# ---------------------------------------------------------------------------
# the TypeErrors the contracts name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("write", ["fp.write(1)", "fp.write(None)", "fp.write(1.5)", "fp.write(['a'])", "fp.write(('a',))"])
def test_a_wrong_typed_write_to_a_text_file_is_an_unproven_typeerror(write):
    result = _prove(f"""
        def main(p):
            with open(p, 'w') as fp:
                {write}
            return 0
        main('out')
    """, ["TypeError"])
    assert result.verdict == "UNKNOWN"
    assert "TextIOWrapper.write:TypeError" in result.reason


def test_a_write_that_may_be_a_str_or_an_int_is_unproven():
    result = _prove("""
        import _sv_verifier
        def main(p, flag):
            x = 'a' if flag else 1
            with open(p, 'w') as fp:
                fp.write(x)
            return 0
        main('out', _sv_verifier.nondet_bool())
    """, ["TypeError"])
    assert result.verdict == "UNKNOWN"


def test_a_str_to_a_binary_file_and_an_int_element_in_writelines_are_unproven_typeerrors():
    for write in ("fp.write('abc')", "fp.write(1)", "fp.writelines(['a'])"):
        result = _prove(f"""
            def main(p):
                with open(p, 'wb') as fp:
                    {write}
                return 0
            main('out')
        """, ["TypeError"])
        assert result.verdict == "UNKNOWN", write
    result = _prove("""
        def main(p):
            with open(p, 'w') as fp:
                fp.writelines(['a', 1])
            return 0
        main('out')
    """, ["TypeError"])
    assert result.verdict == "UNKNOWN"


def test_a_non_str_path_is_an_unproven_typeerror_and_an_int_path_is_refused():
    result = _prove("""
        def main(p):
            with open(p, 'w') as fp:
                fp.write('x')
            return 0
        main(None)
    """, ["TypeError"])
    assert result.verdict == "UNKNOWN"
    assert "open:TypeError" in result.reason
    assert "descriptor" in _unsupported("""
        def main(p):
            with open(p, 'w') as fp:
                fp.write('x')
        main(1)
    """)


# ---------------------------------------------------------------------------
# refused, never approximated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("src, why", [
    ("def main(p):\n    f = open(p)\n    return 0\nmain('x')\n", "not a write-only constant mode"),
    ("def main(p):\n    f = open(p, 'r+')\n    return 0\nmain('x')\n", "not a write-only constant mode"),
    ("def main(p):\n    f = open(p, 'w+')\n    return 0\nmain('x')\n", "not a write-only constant mode"),
    ("def main(p, m):\n    f = open(p, m)\n    return 0\nmain('x', 'w')\n", "not a write-only constant mode"),
    ("def main(p):\n    f = open(p, 'w', 0)\n    return 0\nmain('x')\n", "buffering"),
    ("def main(p):\n    f = open(p, 'w', buffering=1)\n    return 0\nmain('x')\n", "buffering"),
    ("def main(p):\n    f = open(p, 'w', closefd=True)\n    return 0\nmain('x')\n", "not a write-only constant mode"),
    ("def main(p):\n    f = open(p, 'wb', encoding='utf-8')\n    return 0\nmain('x')\n", "binary mode with a text option"),
    ("def main(p, e):\n    f = open(p, 'w', encoding=e)\n    return 0\nmain('x', 5)\n", "of an unknown type"),
])
def test_every_other_shape_of_open_is_unsupported(src, why):
    assert why in _unsupported(src)


@pytest.mark.parametrize("use", [
    "x = fp.tell()", "fp.seek(0)", "x = fp.read()", "x = fp.name", "x = fp.closed", "x = fp.mode",
    "x = fp.buffer", "fp.truncate()", "x = fp.fileno()", "print('x', file=fp)", "fp.write('a', 'b')",
    "fp.write(s='a')", "fp.flush(1)", "fp.close(1)", "fp.writelines()",
])
def test_a_handle_used_in_any_other_way_is_unsupported(use):
    _unsupported(f"""
        def main(p):
            fp = open(p, 'w')
            {use}
        main('out')
    """)


@pytest.mark.parametrize("ctx", ["io.StringIO()", "threading.Lock()", "open_cm()"])
def test_another_context_manager_is_unsupported(ctx):
    _unsupported(f"""
        import io
        import threading
        def open_cm():
            return 1
        def main():
            with {ctx} as f:
                pass
        main()
    """)


def test_a_with_whose_target_is_not_a_name_is_unsupported():
    assert "plain name" in _unsupported("""
        def main(p):
            box = [0]
            with open(p, 'w') as box[0]:
                pass
        main('out')
    """)


def test_a_read_is_stopped_at_the_environment_gate_and_a_write_then_read_back_too():
    for src in (
        "def main(p):\n    return open(p).read()\nmain('x')\n",
        "def main(p):\n    with open(p, 'w') as fp:\n        fp.write('0')\n    return 1 // int(open(p).read())\nmain('x')\n",
    ):
        result = _prove(src)
        assert result.verdict == "UNKNOWN" and result.reason.startswith("environment: unknown file read")


# ---------------------------------------------------------------------------
# the differential: no TRUE for a program that raises, over random output programs
# ---------------------------------------------------------------------------

TEXT_VALUES = ["'x'", "s", "s + 'y'", "a", "None", "1.5", "[s]", "'p' * 2", "str(a)"]
ARRAY_VALUES = ["buf", "s", "a", "None", "[1]"]


def _program(rng: random.Random):
    binary = rng.random() < 0.3
    mode = rng.choice(["wb", "ab", "xb"] if binary else ["w", "a", "x"])
    value = rng.choice(ARRAY_VALUES if binary else TEXT_VALUES)
    method = rng.choice(["write", "write", "writelines", "flush", "close"])
    if method == "write":
        stmt = f"fp.write({value})"
    elif method == "writelines":
        stmt = f"fp.writelines([{value}, {rng.choice(ARRAY_VALUES if binary else TEXT_VALUES)}])"
    else:
        stmt = f"fp.{method}()"
    tail = rng.choice(["1 // d", "[1, 2][d]", "{0: 1}[d]", "1 // 1"])
    return (
        "import array\n"
        "def main(path, s, a, d):\n"
        "    buf = array.array('B', [1, 2])\n"
        f"    with open(path, {mode!r}) as fp:\n"
        f"        {stmt}\n"
        f"    return {tail}\n"
    )


def _raises(src: str, args, tmp_path) -> str | None:
    cwd = os.getcwd()
    os.chdir(tmp_path)
    for name in os.listdir(tmp_path):
        os.remove(name)
    ns: dict = {}
    try:
        exec(compile(src, "<native>", "exec"), ns)
        ns["main"](*args)
    except BaseException as exc:  # noqa: BLE001
        return type(exc).__name__
    finally:
        os.chdir(cwd)
    return None


@pytest.mark.parametrize("seed", range(20))
def test_a_program_absint_proves_safe_never_raises_that_class(seed, tmp_path):
    rng = random.Random(1082_10 + seed)
    proved = 0
    for _ in range(12):
        src = _program(rng)
        args = (f"out{rng.randint(0, 9)}", rng.choice(["", "a", "go"]), rng.randint(-2, 3), rng.choice([0, 1, 2]))
        call = src + f"main{args!r}\n"
        for cls in ("TypeError", "ZeroDivisionError", "IndexError", "KeyError", "AttributeError"):
            result = prove(call, [cls], filename="<fuzz>", time_budget=20.0)
            if result.verdict != "TRUE":
                continue
            proved += 1
            assert _raises(src, args, tmp_path) != cls, f"WRONG TRUE for {cls}:\n{call}"
    assert proved > 0
