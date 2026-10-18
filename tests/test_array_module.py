"""[internal ref] ([internal ref]): the ``array`` module in absint (docs/ABSINT.md Slice 3).

Before this, ``array`` was in ``determinism.py::DET_MODULES``/``census.py::MODULES`` (the
determinism census and the static site census already allowed it) but ``interp.py::MODULES``
had no entry for it at all -- any program that so much as wrote ``import array`` and touched
the result could never get a single check proven through the abstract interpreter, and
MG's PR 89 lesson (``set_bottom`` on an unrecognized shape produced 9 wrong TRUEs) means the
scope here is narrow ON PURPOSE: construction (from a list, from another array of the SAME
typecode, or empty), indexing/slicing/``len``/iteration (all reused verbatim from list's own
heap machinery -- ``heap.py::ArrayObj`` subclasses ``ListObj``), ``append``/``extend``, and the
one binop real SV-Benchmarks code needs (``array('B', [0]) * n``, scimark's own construction
idiom). Everything else (``insert``/``pop``/``remove``/``sort``/``clear``, slice assignment,
in-place repetition, mixed-typecode operations, the platform-dependent ``'l'``/``'L'``
typecodes, ``'u'``) refuses -- ``Unsupported`` -- rather than guess.

Typecode semantics: an integer typecode's stored element is the constant fact of the
non-raising path narrowed to that typecode's exact range (real ``OverflowError`` on a
value provably or possibly out of range); ``'f'`` rounds a known-exact float to the nearest
representable ``struct.pack('<f', ...)`` value (CPython's own conversion) -- and, measured
here, never RAISES ``OverflowError`` for doing so (it SATURATES an out-of-range magnitude to
``+-inf`` instead, unlike ``struct.pack`` itself); ``'d'`` is a C double, i.e. a Python float
exactly, so it neither rounds nor overflows.
"""

from __future__ import annotations

from tharkun.solve.absint import prove
from tharkun.solve.census import site_census
from tharkun.solve.determinism import determinism_census

DL = ("KeyError", "IndexError")
DT = ("TypeError", "AttributeError")
TE = ("TypeError",)
OF = ("OverflowError",)
AS = ("AssertionError",)


def _prove(src, classes):
    return prove(src, classes, filename="t.py", time_budget=10)


def _prove_storage_assertion(src, classes=AS):
    result = _prove(src, classes)
    assert result.verdict == "TRUE", result.reason
    assert result.complete and any("AssertionError" in site[3] for site in result.sites)
    unsupported = _prove(src, OF)
    assert unsupported.verdict == "UNKNOWN"
    assert unsupported.reason == "unsupported property classes: OverflowError"
    return result


# ---------------------------------------------------------------- construction

def test_construct_from_a_list_and_read_it_back():
    # every element of a list/array is one JOINED summary (list's own, pre-existing
    # precision limit, unrelated to array): an index of a multi-element literal proves
    # its BOUNDS (the join of 1, 2, 3), not one exact per-index value.
    src = 'import array\na = array.array("i", [1, 2, 3])\nassert a[0] >= 1 and a[0] <= 3\n'
    r = _prove(src, AS)
    assert r.verdict == "TRUE", r.reason
    src_single = 'import array\na = array.array("i", [7])\nassert a[0] == 7\n'
    r_single = _prove(src_single, AS)
    assert r_single.verdict == "TRUE", r_single.reason


def test_construct_empty_with_only_a_typecode():
    src = 'import array\na = array.array("i")\nassert len(a) == 0\n'
    r = _prove(src, AS)
    assert r.verdict == "TRUE", r.reason


def test_from_array_import_array_works_the_same_as_the_module_attribute():
    src = 'from array import array\na = array("i", [1, 2])\nassert len(a) == 2\n'
    r = _prove(src, AS)
    assert r.verdict == "TRUE", r.reason


def test_an_unknown_or_computed_typecode_is_refused_not_guessed():
    src = 'import array\nimport _sv_verifier\ntc = "i" if _sv_verifier.nondet_bool() else "d"\na = array.array(tc, [1])\n'
    r = _prove(src, TE)
    assert r.verdict != "TRUE"


def test_platform_dependent_and_unicode_typecodes_are_refused():
    # 'l'/'L' ("long"): itemsize is genuinely platform-dependent in CPython's own array
    # module (4 bytes on Windows' LLP64 long, 8 on 64-bit Linux/macOS) -- refused rather
    # than assume one platform. 'u': no str/bytes element model exists to back it.
    for tc in ("l", "L", "u"):
        r = _prove(f'import array\na = array.array("{tc}", [1])\n', TE)
        assert r.verdict != "TRUE", tc


def test_zero_arg_array_call_is_refused_arity():
    r = _prove('import array\na = array.array()\n', TE)
    assert r.verdict != "TRUE"


def test_typecode_mismatch_between_two_arrays_is_refused():
    src = 'import array\na = array.array("i", [1, 2])\nb = array.array("d", a)\n'
    r = _prove(src, TE)
    assert r.verdict != "TRUE"


# ---------------------------------------------------------------- indexing / slicing / len

def test_in_range_index_is_exact():
    # a SINGLE-element array's elem summary is exact; a multi-element one is bounded (see
    # test_construct_from_a_list_and_read_it_back).
    src = 'import array\na = array.array("i", [42])\nassert a[0] == 42\nassert a[-1] == 42\n'
    r = _prove(src, AS)
    assert r.verdict == "TRUE", r.reason


def test_out_of_range_index_is_never_true():
    src = (
        'import _sv_verifier\n'
        'import array\n'
        'a = array.array("i", [1, 2, 3])\n'
        'i = _sv_verifier.nondet_int()\n'
        'if i < 0 or i > 10:\n'
        '    raise SystemExit\n'
        'x = a[i]\n'
    )
    r = _prove(src, DL)
    assert r.verdict == "UNKNOWN", r.reason


def test_len_is_exact():
    src = 'import array\na = array.array("i", [1, 2, 3])\nassert len(a) == 3\n'
    r = _prove(src, AS)
    assert r.verdict == "TRUE", r.reason


def test_slicing_preserves_typecode_and_length():
    src = (
        'import array\n'
        'a = array.array("i", [1, 2, 3, 4, 5])\n'
        'b = a[1:3]\n'
        'assert len(b) == 2\n'
        'b.append(9)\n'  # still a real (typed) array, not degraded to a plain list
        'assert len(b) == 3\n'
    )
    r = _prove(src, AS)
    assert r.verdict == "TRUE", r.reason


def test_symbolic_length_construction_and_indexing_compose():
    # W49's raytrace Canvas pattern ([internal ref]): a length linked to a SYMBOLIC
    # expression (width * height * 3), not a literal -- must carry a usable length fact
    # the same way list's own `[X] * n` segmentation does.
    src = (
        'import _sv_verifier\n'
        'import array\n'
        'width = _sv_verifier.nondet_int()\n'
        'height = _sv_verifier.nondet_int()\n'
        'if width < 0 or height < 0:\n'
        '    raise SystemExit\n'
        'buf = array.array("B", [0] * (width * height * 3))\n'
        'i = _sv_verifier.nondet_int()\n'
        'if i < 0 or i >= len(buf):\n'
        '    raise SystemExit\n'
        'r = _sv_verifier.nondet_int()\n'
        'buf[i] = max(0, min(255, r))\n'
        'assert 0 <= buf[i] <= 255\n'
    )
    _prove_storage_assertion(src, DL + AS)


def test_scimark_construct_then_repeat_then_index_composes():
    # scimark's own idiom: `array('d', [0]) * (w * h)`, then indexed by a checked bound.
    src = (
        'import _sv_verifier\n'
        'from array import array\n'
        'w = _sv_verifier.nondet_int()\n'
        'h = _sv_verifier.nondet_int()\n'
        'if w < 0 or h < 0:\n'
        '    raise SystemExit\n'
        'data = array("d", [0]) * (w * h)\n'
        'i = _sv_verifier.nondet_int()\n'
        'if i < 0 or i >= len(data):\n'
        '    raise SystemExit\n'
        'x = data[i]\n'
    )
    r = _prove(src, DL)
    assert r.verdict == "TRUE", r.reason


# ---------------------------------------------------------------- append / extend / iteration

def test_append_extends_the_length_fact():
    src = 'import array\na = array.array("i", [1, 2])\na.append(3)\nassert len(a) == 3\n'
    r = _prove(src, AS)
    assert r.verdict == "TRUE", r.reason


def test_extend_from_a_list_or_another_array():
    src = (
        'import array\n'
        'a = array.array("i", [1, 2])\n'
        'a.extend([3, 4])\n'
        'b = array.array("i", [5])\n'
        'a.extend(b)\n'
        'assert len(a) == 5\n'
    )
    r = _prove(src, AS)
    assert r.verdict == "TRUE", r.reason


def test_extend_with_the_wrong_element_type_is_never_true():
    src = 'import array\na = array.array("i", [1, 2])\na.extend(["x", "y"])\n'
    r = _prove(src, TE)
    assert r.verdict == "UNKNOWN", r.reason


def test_iteration_over_an_array_does_not_abort():
    src = (
        'import array\n'
        'a = array.array("i", [1, 2, 3])\n'
        'total = 0\n'
        'for x in a:\n'
        '    total = total + x\n'
        'assert isinstance(total, int)\n'
    )
    r = _prove(src, TE)
    assert r.verdict == "TRUE", r.reason


def test_list_of_an_array_materializes_a_plain_list():
    src = 'import array\na = array.array("i", [1, 2, 3])\nb = list(a)\nassert len(b) == 3\n'
    r = _prove(src, AS)
    assert r.verdict == "TRUE", r.reason


# ---------------------------------------------------------------- refused mutators/operators

def test_unsupported_methods_and_operators_refuse_rather_than_guess():
    for src in (
        'import array\na = array.array("i", [1, 2])\na.insert(0, 5)\n',
        'import array\na = array.array("i", [1, 2])\na.pop()\n',
        'import array\na = array.array("i", [1, 2])\na.remove(1)\n',
        'import array\na = array.array("i", [1, 2])\na.sort()\n',
        'import array\na = array.array("i", [1, 2])\na.clear()\n',
        'import array\na = array.array("i", [1, 2])\na *= 3\n',
        'import array\na = array.array("i", [1, 2, 3])\na[0:2] = [9, 9]\n',
    ):
        r = _prove(src, TE)
        assert r.verdict != "TRUE", src


# ---------------------------------------------------------------- typecode semantics: ints

def test_overflow_on_typecode_b_is_never_true_when_unconstrained():
    src = 'import _sv_verifier\nimport array\nn = _sv_verifier.nondet_int()\na = array.array("b", [n])\n'
    r = _prove(src, OF)
    assert r.verdict == "UNKNOWN", r.reason


def test_bounded_typecode_b_storage_invariant_is_proven():
    src = (
        'import _sv_verifier\n'
        'import array\n'
        'n = _sv_verifier.nondet_int()\n'
        'if n < -128 or n > 127:\n'
        '    raise SystemExit\n'
        'a = array.array("b", [n])\n'
        'assert -128 <= a[0] <= 127\n'
    )
    _prove_storage_assertion(src)


def test_overflow_boundary_is_exact_for_every_signed_and_unsigned_width():
    # one in-range and one just-out-of-range literal per int typecode: a non-raising
    # store's element is the MEET of what it was known to be and the typecode's own
    # range, so an in-range literal round-trips exactly.
    cases = [
        ("b", 127, 128), ("B", 255, 256),
        ("h", 32767, 32768), ("H", 65535, 65536),
        ("i", 2147483647, 2147483648), ("I", 4294967295, 4294967296),
        ("q", 2**63 - 1, 2**63), ("Q", 2**64 - 1, 2**64),
    ]
    for tc, ok_val, bad_val in cases:
        ok_src = f'import array\na = array.array("{tc}", [{ok_val}])\nassert a[0] == {ok_val}\n'
        r_ok = _prove(ok_src, AS)
        assert r_ok.verdict == "TRUE", (tc, r_ok.reason)
        bad_src = f'import array\na = array.array("{tc}", [{bad_val}])\n'
        r_bad = _prove(bad_src, OF)
        assert r_bad.verdict == "UNKNOWN", (tc, r_bad.reason)


# ---------------------------------------------------------------- typecode semantics: floats

def test_f_saturation_is_native_and_proven_as_a_storage_assertion():
    # CPython's array module SATURATES an out-of-range 'f' magnitude to +-inf rather than
    # raising (measured: `array.array('f', [1e300])[0] == inf`) -- unlike `struct.pack`,
    # which DOES raise for the same input. This must never be modelled as a check the
    # program can fail.
    import array
    assert array.array("f", [1e300])[0] > 1e300
    src = 'import array\na = array.array("f", [1e300])\nassert a[0] > 1e300\n'
    _prove_storage_assertion(src)
    assert _prove(src.replace("> 1e300", "== 1e300"), AS).verdict == "UNKNOWN"


def test_f_rounding_is_never_wrongly_proven_exact():
    # array('f', [0.1])[0] is 0.10000000149011612, never 0.1 exactly. An assertion that ASSUMES no
    # rounding happened fails at run time, so it must never be PROVEN; the assertion that the value
    # differs from the double 0.1 holds at run time, and is proven now that float literals are exact
    # ([internal ref]; before it the literal was the generic FLOAT and neither form could be decided).
    never = 'import array\na = array.array("f", [0.1])\nassert a[0] == 0.1\n'
    r = _prove(never, AS)
    assert r.verdict != "TRUE", r.reason
    rounded = 'import array\na = array.array("f", [0.1])\nassert a[0] != 0.1\n'
    assert _prove(rounded, AS).verdict == "TRUE"
    # a C double is a Python float exactly: no rounding, so the equality IS provable
    exact = 'import array\na = array.array("d", [0.1])\nassert a[0] == 0.1\n'
    assert _prove(exact, AS).verdict == "TRUE"


def test_d_typecode_is_a_real_double_no_refusal():
    src = 'import array\na = array.array("d", [1.5, 2.5])\nassert len(a) == 2\n'
    r = _prove(src, AS)
    assert r.verdict == "TRUE", r.reason


def test_int_into_a_float_typecode_is_accepted_float_into_an_int_typecode_is_not():
    ok = 'import array\na = array.array("d", [1])\nassert len(a) == 1\n'
    r_ok = _prove(ok, AS)
    assert r_ok.verdict == "TRUE", r_ok.reason
    bad = 'import array\na = array.array("i", [1.5])\n'
    r_bad = _prove(bad, TE)
    assert r_bad.verdict == "UNKNOWN", r_bad.reason


# ---------------------------------------------------------------- census / determinism

def test_array_programs_are_still_deterministic():
    src = 'import array\na = array.array("i", [1, 2, 3])\na.append(4)\nx = a[0] + len(a)\n'
    assert determinism_census(src).ok


def test_array_programs_are_resolved_by_the_static_site_census():
    # the census must recognize array indexing as a real IndexError SITE (no_site=False),
    # not fail to resolve the program and vacuously claim there is none.
    src = 'import array\na = array.array("i", [1, 2, 3])\nx = a[0]\n'
    res = site_census(src, ("IndexError",))
    assert res.no_site is False, res.reason
