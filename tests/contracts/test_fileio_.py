"""Tests backing every claim in ``tharkun.solve.contracts.rules_fileio`` ([internal ref]): the write side of a file
opened by ``open(..., "w" | "a" | "x")``, on CPython 3.14.4."""

from __future__ import annotations

import array
import itertools

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get
from tharkun.solve.contracts.rules_fileio import OUTPUT_HANDLE_TYPES

OTHER_THAN_TYPEERROR = sorted(set(PROPERTY_CLASSES) - {"TypeError"})
NON_PROPERTY = (ValueError, OSError, LookupError)


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "out")


def _text(path):
    return open(path, "w", encoding="utf-8")


def _binary(path):
    return open(path, "wb")


HANDLES = {"io.TextIOWrapper": _text, "io.BufferedWriter": _binary}
GOOD = {"io.TextIOWrapper": "abc", "io.BufferedWriter": b"abc"}
BAD = {
    "io.TextIOWrapper": [b"abc", 1, None, 1.5, ["a"], ("a",), bytearray(b"a"), object()],
    "io.BufferedWriter": ["abc", 1, None, 1.5, ["a"], (1,), {1: 2}, object()],
}


def _no_property_class_but(fn, allowed):
    """``fn`` raises nothing, or only exceptions in ``allowed``; any other property class fails."""
    try:
        fn()
    except allowed:
        return
    except tuple(PROPERTY_CLASSES.values()) as exc:  # noqa: PERF203
        pytest.fail(f"unexpected property exception {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# the handle kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode, kind", [
    ("w", "io.TextIOWrapper"), ("a", "io.TextIOWrapper"), ("x", "io.TextIOWrapper"),
    ("wt", "io.TextIOWrapper"), ("tw", "io.TextIOWrapper"), ("at", "io.TextIOWrapper"), ("xt", "io.TextIOWrapper"),
    ("wb", "io.BufferedWriter"), ("ab", "io.BufferedWriter"), ("xb", "io.BufferedWriter"),
    ("bw", "io.BufferedWriter"), ("ba", "io.BufferedWriter"), ("bx", "io.BufferedWriter"),
])
def test_open_output_returns_the_handle_its_mode_names(tmp_path, mode, kind):
    contract = get("open.output")
    assert contract.tested and contract.returns.tested
    with open(str(tmp_path / f"f-{mode}"), mode) as fh:
        assert f"{type(fh).__module__.lstrip('_')}.{type(fh).__name__}" in {"io.TextIOWrapper", "io.BufferedWriter"}
        assert type(fh).__name__ == kind.split(".")[1]
    assert OUTPUT_HANDLE_TYPES[kind == "io.BufferedWriter"] == kind


def test_the_contracts_exist_for_both_kinds_and_every_method():
    for kind in ("io.TextIOWrapper", "io.BufferedWriter"):
        for method in ("write", "writelines", "flush", "close", "__enter__", "__exit__"):
            contract = get(f"{kind}.{method}")
            assert contract is not None and contract.tested, (kind, method)
            assert contract.may_raise_other  # ValueError (closed), OSError (the environment)
    assert get("io.TextIOWrapper.tell") is None and get("io.TextIOWrapper.seek") is None


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(HANDLES))
class TestWrite:
    def contract(self, kind):
        return get(f"{kind}.write")

    def test_the_good_argument_returns_its_length(self, kind, path):
        with HANDLES[kind](path) as fh:
            assert fh.write(GOOD[kind]) == 3

    def test_a_wrong_type_is_a_typeerror(self, kind, path):
        with HANDLES[kind](path) as fh:
            for bad in BAD[kind]:
                with pytest.raises(TypeError):
                    fh.write(bad)

    @pytest.mark.parametrize("call", ["fh.write()", "fh.write('a', 'b')", "fh.write(s='a')"])
    def test_a_wrong_arity_or_a_keyword_is_a_typeerror(self, kind, path, call):
        with HANDLES[kind](path) as fh:
            with pytest.raises(TypeError):
                eval(call.replace("'a'", repr(GOOD[kind])), {"fh": fh})

    def test_the_argument_is_checked_before_the_file(self, kind, path):
        fh = HANDLES[kind](path)
        fh.close()
        with pytest.raises(TypeError):
            fh.write(BAD[kind][1])
        with pytest.raises(ValueError):
            fh.write(GOOD[kind])

    def test_the_other_six_classes_never_escape_for_any_plain_argument(self, kind, path):
        with HANDLES[kind](path) as fh:
            for arg in itertools.chain([GOOD[kind]], BAD[kind], [bytearray(b"x"), memoryview(b"x"), array.array("d", [1.0])]):
                _no_property_class_but(lambda arg=arg: fh.write(arg), (TypeError, ValueError, OSError))

    def test_claims_are_tested_and_have_both_open_world_flags_where_a_callback_can_run(self, kind, path):
        contract = self.contract(kind)
        assert contract.tested and contract.may_raise_other
        assert contract.raise_claim("TypeError").condition not in ("never", "unknown")
        if kind == "io.TextIOWrapper":
            assert not contract.effects.may_execute_callback
            assert all(contract.raise_claim(e).condition == "never" for e in OTHER_THAN_TYPEERROR)
        else:
            assert contract.effects.may_execute_callback
            assert all("propagates uncaught" in contract.raise_claim(e).condition for e in OTHER_THAN_TYPEERROR)


def test_a_str_subclass_is_accepted_by_a_text_file_and_none_of_its_methods_runs(path):
    calls = []

    class Spy(str):
        def __getattribute__(self, name):
            calls.append(name)
            return super().__getattribute__(name)

        def __str__(self):
            calls.append("__str__")
            raise AssertionError("must not run")

        def __len__(self):
            calls.append("__len__")
            raise ZeroDivisionError("must not run")

        def encode(self, *a, **k):
            calls.append("encode")
            raise KeyError("must not run")

        def find(self, *a, **k):
            calls.append("find")
            raise IndexError("must not run")

        def __iter__(self):
            calls.append("__iter__")
            raise FloatingPointError("must not run")

        def __getitem__(self, i):
            calls.append("__getitem__")
            raise AttributeError("must not run")

        def __eq__(self, other):
            calls.append("__eq__")
            raise TypeError("must not run")

        __hash__ = str.__hash__

    with _text(path) as fh:
        assert fh.write(Spy("abc")) == 3
        fh.writelines([Spy("de")])
    assert [c for c in calls if c not in ("__class__",)] == [], calls
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == "abcde"


@pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
def test_a_binary_write_propagates_what_its_buffer_provider_raises(path, exc_name):
    provider = raiser("__buffer__", exc_name)()
    with _binary(path) as fh:
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            fh.write(provider)
    assert "propagates uncaught" in get("io.BufferedWriter.write").raise_claim(exc_name).condition or exc_name == "TypeError"


def test_a_binary_write_accepts_every_builtin_buffer(path):
    with _binary(path) as fh:
        assert fh.write(b"ab") == 2
        assert fh.write(bytearray(b"ab")) == 2
        assert fh.write(memoryview(b"ab")) == 2
        assert fh.write(array.array("B", [1, 2])) == 2
        assert fh.write(array.array("d", [1.0])) == 8


def test_a_text_write_can_fail_to_encode_with_a_valueerror_which_is_not_a_property_class(path):
    with _text(path) as fh:
        with pytest.raises(UnicodeEncodeError) as info:
            fh.write("\ud800")
    assert isinstance(info.value, ValueError)


def test_a_write_to_a_closed_file_is_a_valueerror(path):
    for kind, opener in HANDLES.items():
        fh = opener(path)
        fh.close()
        with pytest.raises(ValueError):
            fh.write(GOOD[kind])


# ---------------------------------------------------------------------------
# writelines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(HANDLES))
class TestWritelines:
    def test_a_list_of_the_right_elements_returns_none(self, kind, path):
        with HANDLES[kind](path) as fh:
            assert fh.writelines([GOOD[kind], GOOD[kind]]) is None
            assert fh.writelines((GOOD[kind],)) is None
            assert fh.writelines(x for x in [GOOD[kind]]) is None

    def test_a_non_iterable_is_a_typeerror_on_an_open_file(self, kind, path):
        with HANDLES[kind](path) as fh:
            with pytest.raises(TypeError):
                fh.writelines(1)
            with pytest.raises(TypeError):
                fh.writelines(None)

    def test_a_wrong_element_is_a_typeerror_and_the_earlier_ones_stay_written(self, kind, path):
        with HANDLES[kind](path) as fh:
            with pytest.raises(TypeError):
                fh.writelines([GOOD[kind], BAD[kind][1], GOOD[kind]])
        with open(path, "rb") as fh:
            assert fh.read() == b"abc"

    def test_the_file_is_checked_first(self, kind, path):
        fh = HANDLES[kind](path)
        fh.close()
        with pytest.raises(ValueError):
            fh.writelines(1)
        with pytest.raises(ValueError):
            fh.writelines([GOOD[kind]])

    @pytest.mark.parametrize("call", ["fh.writelines()", "fh.writelines([], [])", "fh.writelines(lines=[])"])
    def test_a_wrong_arity_or_a_keyword_is_a_typeerror(self, kind, path, call):
        with HANDLES[kind](path) as fh:
            with pytest.raises(TypeError):
                eval(call, {"fh": fh})

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_what_the_iterable_raises_propagates(self, kind, path, exc_name):
        iterable = raiser("__iter__", exc_name)()
        with HANDLES[kind](path) as fh:
            with pytest.raises(PROPERTY_CLASSES[exc_name]):
                fh.writelines(iterable)

    def test_the_claims(self, kind, path):
        contract = get(f"{kind}.writelines")
        assert contract.tested and contract.may_raise_other and contract.effects.may_execute_callback
        assert contract.raise_claim("TypeError").condition not in ("never", "unknown")
        assert all("propagates uncaught" in contract.raise_claim(e).condition for e in OTHER_THAN_TYPEERROR)


def test_a_str_argument_is_iterated_by_character_for_text_and_is_a_typeerror_for_binary(path):
    with _text(path) as fh:
        fh.writelines("abc")
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == "abc"
    with _binary(path) as fh:
        with pytest.raises(TypeError):
            fh.writelines("abc")


# ---------------------------------------------------------------------------
# flush, close, the context manager
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(HANDLES))
class TestFlushCloseAndTheContextManager:
    def test_flush_returns_none_and_takes_no_argument(self, kind, path):
        with HANDLES[kind](path) as fh:
            assert fh.flush() is None
            with pytest.raises(TypeError):
                fh.flush(1)

    def test_close_returns_none_is_idempotent_and_takes_no_argument(self, kind, path):
        fh = HANDLES[kind](path)
        with pytest.raises(TypeError):
            fh.close(1)
        assert fh.close() is None and fh.closed
        assert fh.close() is None

    def test_after_close_every_other_method_is_a_valueerror(self, kind, path):
        fh = HANDLES[kind](path)
        fh.close()
        for call in (lambda: fh.flush(), lambda: fh.write(GOOD[kind]), lambda: fh.writelines([GOOD[kind]]), lambda: fh.__enter__()):
            with pytest.raises(ValueError):
                call()

    def test_enter_returns_the_receiver_and_exit_closes_and_returns_none(self, kind, path):
        fh = HANDLES[kind](path)
        assert fh.__enter__() is fh
        assert fh.__exit__(None, None, None) is None and fh.closed

    def test_exit_never_suppresses_the_bodys_exception(self, kind, path):
        fh = HANDLES[kind](path)
        assert fh.__exit__(ValueError, ValueError("x"), None) is None and fh.closed
        with pytest.raises(ZeroDivisionError):
            with HANDLES[kind](path) as inner:
                inner.write(GOOD[kind])
                1 // 0
        assert inner.closed

    def test_exit_on_a_closed_file_is_fine(self, kind, path):
        fh = HANDLES[kind](path)
        fh.close()
        assert fh.__exit__(None, None, None) is None

    def test_a_return_inside_the_with_still_closes(self, kind, path):
        def body():
            with HANDLES[kind](path) as fh:
                fh.write(GOOD[kind])
                return fh

        assert body().closed

    def test_the_claims(self, kind, path):
        for method in ("flush", "close", "__enter__", "__exit__"):
            contract = get(f"{kind}.{method}")
            assert contract.tested and not contract.effects.may_execute_callback
            assert all(contract.raise_claim(e).condition == "never" for e in OTHER_THAN_TYPEERROR)
        for method in ("flush", "close", "__enter__"):
            assert get(f"{kind}.{method}").raise_claim("TypeError").condition not in ("never", "unknown")
        assert get(f"{kind}.__exit__").raise_claim("TypeError").condition == "never"


# ---------------------------------------------------------------------------
# open.output
# ---------------------------------------------------------------------------


class TestOpenOutput:
    contract = get("open.output")

    def test_the_metadata(self):
        assert self.contract.tested and self.contract.may_raise_other
        assert self.contract.effects.may_execute_callback  # __fspath__
        assert self.contract.raise_claim("TypeError").condition not in ("never", "unknown")
        assert all("propagates uncaught" in self.contract.raise_claim(e).condition for e in OTHER_THAN_TYPEERROR)

    @pytest.mark.parametrize("call", [
        "open(None, 'w')",
        "open(1.5, 'w')",
        "open([], 'w')",
        "open(P, 'w', encoding=1)",
        "open(P, 'w', errors=1)",
        "open(P, 'w', newline=1)",
        "open(P, 'w', buffering='x')",
        "open(P, 'w', -1, None, None, None, True, None, 7)",
        "open(P, 'w', mode='w')",
        "open(P, 'w', nope=1)",
    ])
    def test_the_typeerror_conditions(self, path, call):
        with pytest.raises(TypeError):
            eval(call, {"P": path})

    def test_the_path_may_be_str_bytes_or_pathlike(self, path, tmp_path):
        import pathlib

        open(path, "w").close()
        open(path.encode(), "wb").close()
        open(pathlib.Path(path), "w").close()

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_what_the_paths_fspath_raises_propagates(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            open(raiser("__fspath__", exc_name)(), "w")

    def test_str_path_and_plain_arguments_never_raise_a_property_class(self, tmp_path):
        base = str(tmp_path)
        shapes = [
            lambda: open(base + "/new", "w"),
            lambda: open(base + "/new2", "wb"),
            lambda: open(base, "w"),                                  # a directory: IsADirectoryError
            lambda: open(base + "/absent/x", "w"),                    # FileNotFoundError
            lambda: open("", "w"),
            lambda: open(base + "/n\x00", "w"),                       # ValueError
            lambda: open(base + "/e", "w", encoding="nope"),          # LookupError
            lambda: open(base + "/e2", "wb", encoding="utf-8"),       # ValueError
            lambda: open(base + "/e3", "w", newline="x"),             # ValueError
            lambda: open(base + "/e4", "w", buffering=0),             # ValueError
        ]
        for shape in shapes:
            _no_property_class_but(shape, NON_PROPERTY)

    def test_the_non_property_failures(self, tmp_path):
        base = str(tmp_path)
        with pytest.raises(OSError):
            open(base, "w")
        with pytest.raises(OSError):
            open(base + "/absent/x", "w")
        (tmp_path / "exists").write_text("x")
        with pytest.raises(FileExistsError):
            open(base + "/exists", "x")
        with pytest.raises(ValueError):
            open(base + "/n\x00", "w")
        with pytest.raises(ValueError):
            open(base + "/b", "wb", encoding="utf-8")
        with pytest.raises(LookupError):
            open(base + "/c", "w", encoding="nope")

    def test_the_shape_excludes_a_descriptor_update_modes_and_unbuffered_io(self, tmp_path):
        base = str(tmp_path)
        with pytest.raises(ValueError):
            open(base + "/u", "w", buffering=0)                        # text: not this shape
        assert type(open(base + "/u2", "wb", buffering=0)).__name__ == "FileIO"
        with open(base + "/rw", "w+") as fh:                           # reads: not this shape
            assert fh.read() == ""


def test_every_contract_of_this_batch_is_consumed_by_an_adapter():
    """Mechanical form of "no dead entries": ``open.output`` is the contract ``open`` maps to for the
    absint model and the symbolic executor, and every method contract is one an engine can read through
    ``output_handle_contract``."""
    from tharkun.solve.absint.interp import Analyzer
    from tharkun.solve.contracts import rules_fileio as fileio_
    from tharkun.solve.contracts.adapters import absint as absint_adapter
    from tharkun.solve.contracts.adapters import common
    from tharkun.solve.contracts.adapters import concolic as concolic_adapter

    assert common.BUILTIN_SHAPE_CONTRACTS == {"open": "open.output"}
    assert hasattr(Analyzer, "c_open")
    assert absint_adapter.contract_for_builtin("open") is get("open.output")
    assert concolic_adapter.output_open_contract() is get("open.output")
    reachable = {"open.output"}
    for binary in (False, True):
        for method in common.OUTPUT_HANDLE_METHOD_ARITY:
            contract = common.output_handle_contract(binary, method)
            assert contract is not None
            assert absint_adapter.output_handle_contract(binary, method) is contract
            assert concolic_adapter.output_handle_contract(binary, method) is contract
            reachable.add(contract.qualname)
    assert set(fileio_.QUALNAMES) == reachable


def test_the_union_open_contract_stays_untested_and_is_not_what_the_models_read():
    union = get("open")
    assert union is not None and not union.tested  # the type it returns depends on the mode
    from tharkun.solve.contracts.adapters import absint as absint_adapter

    assert absint_adapter.contract_for_builtin("open").qualname == "open.output"


def test_the_shape_table_covers_exactly_the_documented_arities():
    from tharkun.solve.contracts.adapters.common import output_handle_call_shape

    assert output_handle_call_shape("write", 1, False)
    assert not output_handle_call_shape("write", 0, False)
    assert not output_handle_call_shape("write", 2, False)
    assert not output_handle_call_shape("write", 1, True)
    assert output_handle_call_shape("flush", 0, False) and not output_handle_call_shape("flush", 1, False)
    assert output_handle_call_shape("__exit__", 3, False) and not output_handle_call_shape("__exit__", 0, False)
    assert not output_handle_call_shape("tell", 0, False) and not output_handle_call_shape("seek", 1, False)


def test_a_withdrawn_contract_withdraws_the_models(monkeypatch):
    from tharkun.solve.contracts import registry
    from tharkun.solve.contracts.adapters import common

    monkeypatch.delitem(registry.REGISTRY, "io.TextIOWrapper.write")
    assert common.output_handle_contract(False, "write") is None
    assert common.output_handle_contract(True, "write") is not None
