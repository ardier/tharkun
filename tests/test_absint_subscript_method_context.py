"""Explicit builtin ``__getitem__``/``__setitem__`` calls are not subscript contexts."""

import pytest

from tharkun.solve.absint import prove


DIRECT_CALLS = (
    "d = {1: 2}\nx = d.__getitem__(1)",
    "xs = [2]\nx = xs.__getitem__(0)",
    "t = (2,)\nx = t.__getitem__(0)",
    "d = defaultdict(list)\nx = d.__getitem__(1)",
    "d = {}\nd.__setitem__(1, 2)",
    "xs = [0]\nxs.__setitem__(0, 2)",
    "d = defaultdict(list)\nd.__setitem__(1, 2)",
)


@pytest.mark.parametrize("source", DIRECT_CALLS)
def test_explicit_builtin_subscript_method_is_a_named_refusal(source):
    result = prove("from collections import defaultdict\n" + source + "\n", ["AssertionError"])
    assert result.verdict == "UNKNOWN", (source, result.verdict, result.reason)
    assert "builtin subscript rule requires a SubscriptCtx" in result.reason


def test_direct_getitem_and_setitem_calls_keep_native_success_and_error_semantics():
    native_cases = (
        ("d = {1: 2}\nassert d.__getitem__(1) == 2", None),
        ("d = {}\nd.__getitem__(1)", KeyError),
        ("xs = [2]\nassert xs.__getitem__(0) == 2", None),
        ("xs = [2]\nxs.__getitem__(4)", IndexError),
        ("t = (2,)\nassert t.__getitem__(0) == 2", None),
        ("t = (2,)\nt.__getitem__(4)", IndexError),
        ("d = defaultdict(list)\nassert d.__getitem__(1) == [] and 1 in d", None),
        ("d = {}\nd.__setitem__(1, 2)\nassert d[1] == 2", None),
        ("xs = [0]\nxs.__setitem__(0, 2)\nassert xs[0] == 2", None),
        ("xs = [0]\nxs.__setitem__(4, 2)", IndexError),
        ("d = defaultdict(list)\nd.__setitem__(1, 2)\nassert d[1] == 2", None),
    )
    for source, expected in native_cases:
        try:
            exec("from collections import defaultdict\n" + source, {})
        except BaseException as exc:  # noqa: BLE001 - exact CPython exception is the control
            assert type(exc) is expected, (source, type(exc), expected)
        else:
            assert expected is None, (source, expected)


@pytest.mark.parametrize(
    "source",
    (
        "d = {1: 2}\nassert d[1] == 2",
        "xs = [2]\nassert xs[0] == 2",
        "t = (2,)\nassert t[0] == 2",
        "d = defaultdict(list)\nd[1].append(9)\nassert d[1][0] == 9",
    ),
)
def test_bracket_subscripts_keep_the_existing_proof_route(source):
    result = prove("from collections import defaultdict\n" + source + "\n", ["AssertionError"])
    assert result.verdict == "TRUE", (source, result.verdict, result.reason, result.sites)
