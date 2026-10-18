"""[internal ref]: the write side of a file opened by a write-only ``open()`` in ``tharkun.sv._symexec``.

The executor runs the program past ``with open(path, "w") as fp: fp.write(...)`` and the rest of the
family (``contracts/rules_fileio.py``, read through ``adapters/concolic.py``), ending a path where the
contract says the call raises (a wrong-typed argument: a ``TypeError``), and refusing every other
shape (a read, an update mode, a descriptor, ``buffering``, a handle used as a value).  What is
written cannot come back to the program, so nothing is tracked.  Differential tests run the same
programs natively over a box of inputs.
"""

from __future__ import annotations

import itertools
import os
import random
import sys
import textwrap

import pytest

z3 = pytest.importorskip("z3")

from tharkun.sv import _symexec  # noqa: E402
from tharkun.sv.sites import build_harness  # noqa: E402


def _solve(src: str, classes=("KeyError",)):
    h = build_harness(textwrap.dedent(src), classes, None)
    assert h is not None, src
    mod = {"__name__": "sv_harness_test"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    got = _symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites],
                                 h.params, h.kinds, {})
    return {s.get("kind", s["exc"]): got[s["line"]] for s in h.sites}, h


HEAD = "import _sv_verifier\nif __name__ == \"__main__\":\n"


def _prog(draws: str, body: str) -> str:
    return HEAD + textwrap.indent(textwrap.dedent(draws), "    ") + textwrap.indent(textwrap.dedent(body), "    ")


# ---------------------------------------------------------------------------
# the shapes
# ---------------------------------------------------------------------------


def test_the_program_runs_past_a_text_write_and_the_site_after_it_is_solved():
    got, _ = _solve(_prog("a = _sv_verifier.nondet_int()\n", """
        with open("out.txt", "w") as fp:
            fp.write("header")
        if a == 41:
            raise KeyError("after the file")
    """))
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["a"] == 41


def test_a_bare_open_close_pair_and_flush_and_writelines():
    got, _ = _solve(_prog("a = _sv_verifier.nondet_int()\n", """
        fp = open("out.txt", "w")
        fp.write("x")
        fp.writelines(["a", "b"])
        fp.flush()
        fp.close()
        if a == 7:
            raise KeyError("after")
    """))
    assert got["raise"][0] == "sat" and got["raise"][1]["a"] == 7


def test_a_string_draw_is_a_valid_text_write_and_a_valid_path():
    got, _ = _solve(_prog("s = _sv_verifier.nondet_str()\n", """
        with open(s, "w") as fp:
            fp.write(s)
        if s == "go":
            raise KeyError("after")
    """))
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["s"] == "go"


def test_a_wrong_typed_write_is_a_typeerror_which_ends_the_path():
    # the int draw is written to a text file: TypeError, so the raise after it is never reached
    got, _ = _solve(_prog("a = _sv_verifier.nondet_int()\n", """
        with open("out.txt", "w") as fp:
            fp.write(a)
        raise KeyError("only reachable if fp.write(a) did not raise")
    """))
    assert got["raise"][0] in ("unsat", "unreached")


def test_a_typeerror_in_one_branch_leaves_the_other_branch_reachable():
    got, _ = _solve(_prog("a = _sv_verifier.nondet_int()\ns = _sv_verifier.nondet_str()\n", """
        with open("out.txt", "w") as fp:
            if a > 0:
                fp.write(s)
            else:
                fp.write(a)
        if a > 5:
            raise KeyError("only the str branch can get here")
    """))
    outcome, vector = got["raise"]
    assert outcome == "sat" and vector["a"] > 5


def test_a_binary_handle_takes_bytes_and_a_str_is_a_typeerror():
    ok, _ = _solve(_prog("a = _sv_verifier.nondet_int()\n", """
        with open("out.bin", "wb") as fp:
            fp.write(b"abc")
        if a == 3:
            raise KeyError("after")
    """))
    assert ok["raise"][0] == "sat"
    bad, _ = _solve(_prog("s = _sv_verifier.nondet_str()\n", """
        with open("out.bin", "wb") as fp:
            fp.write(s)
        raise KeyError("unreachable: a str to a binary file is a TypeError")
    """))
    assert bad["raise"][0] in ("unsat", "unreached")


def test_writelines_checks_every_element():
    got, _ = _solve(_prog("a = _sv_verifier.nondet_int()\n", """
        with open("out.txt", "w") as fp:
            fp.writelines(["ok", a])
        raise KeyError("unreachable: an int element is a TypeError")
    """))
    assert got["raise"][0] in ("unsat", "unreached")


def test_two_handles_in_one_with_statement():
    got, _ = _solve(_prog("a = _sv_verifier.nondet_int()\n", """
        with open("a.txt", "w") as f, open("b.txt", "a") as g:
            f.write("x")
            g.write("y")
        if a == 2:
            raise KeyError("after")
    """))
    assert got["raise"][0] == "sat"


# ---------------------------------------------------------------------------
# refused, never approximated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opener", [
    "open('in.txt')",                    # a read
    "open('in.txt', 'r')",
    "open('f.txt', 'w+')",               # an update mode
    "open('f.txt', 'a+')",
    "open(a, 'w')",                      # an int path: a file descriptor
    "open('f.txt', 'w', 0)",             # buffering
    "open('f.txt', 'w', buffering=1)",
    "open('f.txt', 'w', closefd=True)",
    "open('f.txt', mode_name)",          # a mode that is not a constant
    "open('f.txt', 'wb', encoding='utf-8')",
])
def test_every_other_shape_of_open_is_unsupported(opener):
    # the statement is skipped and the name its body assigns is poisoned: a site that reads it is
    # unsupported, never solved as if the file had been modelled
    got, _ = _solve(_prog("a = _sv_verifier.nondet_int()\nmode_name = 'w'\n", f"""
        with {opener} as fp:
            y = a
        if y == 1:
            raise KeyError("after")
    """))
    assert got["raise"][0] == "unsupported", (opener, got)


@pytest.mark.parametrize("use", [
    "bool(fp)",
    "fp.name",
    "fp.closed",
    "fp == fp",
    "fp.tell()",
    "fp.seek(0)",
    "fp.read()",
    "print('x', file=fp)",
    "fp[0]",
    "len(fp)",
])
def test_a_handle_used_in_any_other_way_is_unsupported(use):
    got, _ = _solve(_prog("a = _sv_verifier.nondet_int()\n", f"""
        with open("out.txt", "w") as fp:
            pass
        x = {use}
        if x == 1:
            raise KeyError("after")
    """))
    assert got["raise"][0] == "unsupported", (use, got)


def test_a_handle_in_a_condition_is_unsupported_and_is_never_taken_as_truthy():
    got, _ = _solve(_prog("a = _sv_verifier.nondet_int()\n", """
        fp = open("out.txt", "w")
        if fp:
            y = a
        else:
            y = 0
        if y == 1:
            raise KeyError("after")
    """))
    assert got["raise"][0] == "unsupported"


def test_a_return_or_break_inside_the_with_body_is_unsupported():
    got, _ = _solve("""
        import _sv_verifier

        def write_it(a):
            with open("out.txt", "w") as fp:
                fp.write("x")
                return a
            return 0

        if __name__ == "__main__":
            a = _sv_verifier.nondet_int()
            r = write_it(a)
            if r == 4:
                raise KeyError("after")
    """)
    assert got["raise"][0] == "unsupported"


def test_another_context_manager_is_unsupported_as_before():
    got, _ = _solve(_prog("a = _sv_verifier.nondet_int()\n", """
        import threading
        with threading.Lock():
            y = a
        if y == 1:
            raise KeyError("after")
    """))
    assert got["raise"][0] in ("unsupported", "unreached")


def test_a_rebound_open_is_not_the_builtin():
    got, _ = _solve("""
        import _sv_verifier

        def open(path, mode):
            return [0]

        if __name__ == "__main__":
            a = _sv_verifier.nondet_int()
            with open("out.txt", "w") as fp:
                y = a
            if y == 3:
                raise KeyError("after")
    """)
    assert got["raise"][0] == "unsupported"


# ---------------------------------------------------------------------------
# the differential: random programs against native Python
# ---------------------------------------------------------------------------

# only constructs the executor models: a statement it cannot model is skipped (a search aid's
# approximation: a later site is still solved), which would make "the model fires natively" false
TEXT_ARGS = ["\"x\"", "s", "s + \"!\"", "a", "\"y\" * 2", "t", "None", "b\"z\""]
BIN_ARGS = ["b\"x\"", "b\"yz\"", "s", "a", "None", "\"x\""]
SITES = ["a == 3", "a > 1 and s == \"go\"", "s == \"x!\"", "a + 1 == 0", "t == \"ab\""]
INTS = range(-2, 5)
STRS = ["", "a", "ab", "go", "x!", "xyz"]


def _program(rng: random.Random):
    binary = rng.random() < 0.35
    mode = rng.choice(["wb", "ab", "xb"] if binary else ["w", "a", "x"])
    arg = rng.choice(BIN_ARGS if binary else TEXT_ARGS)
    method = rng.choice(["write", "write", "writelines", "flush"])
    if method == "write":
        stmt = f"fp.write({arg})"
    elif method == "writelines":
        stmt = f"fp.writelines([{arg}, {rng.choice(BIN_ARGS if binary else TEXT_ARGS)}])"
    else:
        stmt = "fp.flush()"
    cond = rng.choice(["a > 0", "s == \"go\"", "True"])
    site = rng.choice(SITES)
    return (
        f"if {cond}:\n"
        f"    with open(\"out_{mode}\", \"{mode}\") as fp:\n"
        f"        {stmt}\n"
        f"else:\n"
        f"    fp = open(\"alt_{mode}\", \"{mode}\")\n"
        f"    fp.close()\n"
        f"if {site}:\n"
        f"    raise KeyError('site')\n"
    )


def _native_fires(body: str, a: int, s: str, tmp) -> bool:
    cwd = os.getcwd()
    os.chdir(tmp)
    for name in os.listdir(tmp):
        os.remove(name)
    try:
        ns = {"a": a, "s": s, "t": s}
        exec(compile(textwrap.dedent(body), "<native>", "exec"), ns)
    except KeyError:
        return True
    except (TypeError, ValueError, OSError):
        return False
    finally:
        os.chdir(cwd)
    return False


@pytest.mark.parametrize("seed", range(25))
def test_random_output_programs_agree_with_native_python(seed, tmp_path):
    rng = random.Random(1082_00 + seed)
    checked = 0
    for _ in range(6):
        body = _program(rng)
        sym = _prog("a = _sv_verifier.nondet_int()\ns = _sv_verifier.nondet_str()\nt = s\n", body)
        got, _h = _solve(sym)
        outcome, payload = got["raise"]
        fires = {(a, s): _native_fires(body, a, s, tmp_path) for a, s in itertools.product(INTS, STRS)}
        if outcome == "sat":
            assert _native_fires(body, payload["a"], payload["s"], tmp_path), f"model does not fire:\n{body}\n{payload}"
            checked += 1
        elif outcome in ("unsat", "unreached"):
            assert not any(fires.values()), f"claimed {outcome} but {[v for v, f in fires.items() if f][:3]} fire:\n{body}"
            checked += 1
        elif outcome == "unknown":
            continue
        else:
            assert outcome == "unsupported", (outcome, body)
    assert checked >= 1
