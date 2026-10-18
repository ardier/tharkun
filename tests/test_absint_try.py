"""Tests for exact try/except/else/finally semantics in absint ([internal ref]).

Tests each construct against both absint and native CPython 3.14.4 (values AND exceptions),
plus a negative control per construct.
"""

from __future__ import annotations

import textwrap
import os
import subprocess
import sys
import types
import pytest

from tharkun.solve.absint import prove

DL = ("KeyError", "IndexError")
DT = ("TypeError", "AttributeError")
AR = ("ZeroDivisionError", "FloatingPointError")
AS = ("AssertionError",)


def _src(s: str) -> str:
    return textwrap.dedent(s).strip()


def test_module_lambda_defaults_keywords_and_alias_match_native():
    source = _src("""
        saved = 1
        f = lambda x=saved, /, y=2, *, z=3: x + y + z
        saved = 9
        alias = f
        assert alias(z=4) == 7
    """)
    ok, exc, _ = run_native(source)
    assert ok and not exc
    result = prove(source, ("AssertionError",))
    assert result.verdict == "TRUE", result.explain()


def test_module_lambda_body_exception_is_caught_like_native():
    source = _src("""
        f = lambda: [][0]
        try:
            f()
        except IndexError:
            pass
    """)
    ok, exc, _ = run_native(source)
    assert ok and not exc
    result = prove(source, ("IndexError", "KeyError"))
    assert result.verdict == "TRUE", result.explain()


def test_uncaught_lambda_index_error_matches_native_and_is_not_proved_safe():
    source = _src("""
        f = lambda: [][0]
        f()
    """)
    ok, exc, _ = run_native(source)
    assert not ok and exc == "IndexError"
    result = prove(source, ("IndexError", "KeyError"))
    assert result.verdict != "TRUE", result.explain()


def test_lambda_positional_only_call_error_matches_native():
    source = _src("""
        f = lambda x, /: x
        f(x=1)
    """)
    ok, exc, _ = run_native(source)
    assert not ok and exc == "TypeError"
    result = prove(source, ("TypeError",))
    assert result.verdict != "TRUE", result.explain()


@pytest.mark.parametrize("source", [
    """
    def make(k):
        return lambda x: x + k
    f = make(1)
    assert f(2) == 3
    """,
    """
    class C:
        f = lambda x: x + 1
    assert C.f(2) == 3
    """,
])
def test_nested_or_class_lambda_remains_native_valid_but_unsupported(source):
    source = _src(source)
    ok, exc, _ = run_native(source)
    assert ok and not exc
    result = prove(source, ("AssertionError", "TypeError"))
    assert result.verdict == "UNKNOWN" and result.reason


def run_native(src: str):
    """Run natively on CPython 3.14.4; return (completed_normally, exc_class_name, namespace)."""
    ns = {}
    try:
        exec(src, ns)
        return True, "", ns
    except Exception as e:
        return False, type(e).__name__, ns


# ==============================================================================
# 1. Bare except
# ==============================================================================

def test_bare_except_catches_all_and_matches_native():
    src = _src("""
        x = 0
        try:
            raise KeyError()
        except:
            x = 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_bare_except_negative_control():
    # Negative control: assert x != 1
    src = _src("""
        x = 0
        try:
            raise KeyError()
        except:
            x = 1
        assert x != 1
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "AssertionError"
    r = prove(src, DL + AS)
    assert r.verdict == "UNKNOWN"


# ==============================================================================
# 2. Handler matching by class identity
# ==============================================================================

def test_except_class_identity_matches_and_matches_native():
    src = _src("""
        x = 0
        try:
            raise KeyError()
        except KeyError:
            x = 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_except_class_identity_negative_control():
    # Negative control: KeyError raised but only IndexError caught
    src = _src("""
        try:
            raise KeyError()
        except IndexError:
            pass
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "KeyError"
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert any(c[2] == "raise" and "KeyError" in c[3] for c in r.sites)


# ==============================================================================
# 3. Handler matching by subclass / MRO
# ==============================================================================

def test_except_subclass_matching_builtin_and_matches_native():
    src = _src("""
        x = 0
        try:
            raise KeyError()
        except LookupError:
            x = 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_except_subclass_matching_user_class_and_matches_native():
    src = _src("""
        class MyKeyError(KeyError):
            pass

        x = 0
        try:
            raise MyKeyError()
        except KeyError:
            x = 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_except_subclass_negative_control():
    # Negative control: custom exception inherits from Exception, not KeyError
    src = _src("""
        class OtherError(Exception):
            pass

        try:
            raise OtherError()
        except KeyError:
            pass
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "OtherError"
    # Even if OtherError is not in DL, it is unhandled
    r = prove(src, DL)
    # The uncaught OtherError is non-property, but check records safely; let's test with KeyError
    src_fail = _src("""
        class MyKey(KeyError):
            pass
        try:
            raise MyKey()
        except IndexError:
            pass
    """)
    r_fail = prove(src_fail, DL)
    assert r_fail.verdict == "UNKNOWN"


# ==============================================================================
# 4. Handler matching by tuple of classes
# ==============================================================================

def test_except_tuple_of_classes_matches_and_matches_native():
    src = _src("""
        x = 0
        try:
            raise KeyError()
        except (TypeError, KeyError):
            x = 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_except_tuple_negative_control():
    # Negative control: neither class in tuple matches
    src = _src("""
        try:
            raise KeyError()
        except (TypeError, ValueError):
            pass
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "KeyError"
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"


# ==============================================================================
# divmod's integer transfer and single ZeroDivisionError route ([internal ref])
# ==============================================================================

@pytest.mark.parametrize(("a", "b", "q", "r"), [
    (7, 3, 2, 1), (-7, 3, -3, 2), (7, -3, -3, -2), (-7, -3, 2, -1), (0, 3, 0, 0),
])
def test_divmod_integer_signs_and_tuple_slots_match_native(a, b, q, r):
    src = _src(f"""
        pair = divmod({a}, {b})
        q, r = pair
        assert q == {q}
        assert r == {r}
        assert pair[0] == {q}
        assert pair[1] == {r}
    """)
    ok, exc, _ = run_native(src)
    assert ok, exc
    result = prove(src, AS)
    assert result.verdict == "TRUE", result.explain()


def test_divmod_zero_divisor_routes_one_matching_property_exception():
    src = _src("""
        try:
            divmod(7, 0)
        except ZeroDivisionError:
            caught = True
        assert caught
    """)
    ok, exc, _ = run_native(src)
    assert ok, exc
    result = prove(src, AR + AS)
    assert result.verdict == "TRUE", result.explain()
    assert not [record for record in result.records if record["kind"] == "divmod:ZeroDivisionError"], result.records


def test_divmod_zero_divisor_routes_arithmetic_error_superclass():
    src = _src("""
        try:
            divmod(7, 0)
        except ArithmeticError:
            caught = True
        assert caught
    """)
    ok, exc, _ = run_native(src)
    assert ok, exc
    result = prove(src, AR + AS)
    assert result.verdict == "TRUE", result.explain()


def test_divmod_zero_divisor_does_not_match_key_error():
    src = _src("""
        try:
            divmod(7, 0)
        except KeyError:
            pass
    """)
    ok, exc, _ = run_native(src)
    assert not ok and exc == "ZeroDivisionError"
    result = prove(src, AR)
    assert result.verdict == "UNKNOWN"
    routed = [site for site in result.sites if site[2] == "divmod:ZeroDivisionError"]
    assert len(routed) == 1, result.sites


def test_divmod_maybe_zero_has_one_route_and_arithmetic_property_is_not_true():
    src = _src("""
        import _sv_verifier
        divisor = _sv_verifier.nondet_int()
        try:
            q, r = divmod(7, divisor)
        except ZeroDivisionError:
            pass
        else:
            assert q * divisor + r == 7
    """)
    result = prove(src, AR + AS)
    assert result.verdict != "TRUE", result.explain()
    assert not [site for site in result.sites if site[2] == "divmod:ZeroDivisionError"], result.sites


def test_divmod_divisor_bounded_positive_does_not_route_zero_division():
    src = _src("""
        import _sv_verifier
        divisor = abs(_sv_verifier.nondet_int()) + 1
        try:
            divmod(7, divisor)
        except ZeroDivisionError:
            assert False
        else:
            completed = True
        assert completed
    """)
    result = prove(src, AR + AS)
    assert result.verdict == "TRUE", result.explain()
    records = [record for record in result.records if record["kind"] == "divmod:ZeroDivisionError"]
    assert len(records) == 1 and records[0]["status"] == "safe", result.records


def test_divmod_maybe_zero_unhandled_has_one_call_site_record_and_native_zero_raises(monkeypatch):
    src = _src("""
        import _sv_verifier
        divisor = _sv_verifier.nondet_int()
        divmod(7, divisor)
    """)
    result = prove(src, AR)
    assert result.verdict == "UNKNOWN", result.explain()
    assert result.complete
    matching = [record for record in result.records if record["kind"] == "divmod:ZeroDivisionError"]
    assert len(matching) == 1, result.records
    assert matching[0]["status"] == "may_raise"

    fake_verifier = types.ModuleType("_sv_verifier")
    fake_verifier.nondet_int = lambda: 0
    monkeypatch.setitem(sys.modules, "_sv_verifier", fake_verifier)
    ok, exc, _ = run_native(src)
    assert not ok and exc == "ZeroDivisionError"


def test_divmod_definite_zero_outside_try_has_one_call_site_record():
    src = _src("divmod(7, 0)")
    result = prove(src, AR)
    assert result.verdict == "UNKNOWN"
    matching = [record for record in result.records if record["kind"] == "divmod:ZeroDivisionError"]
    assert len(matching) == 1, result.records
    assert matching[0]["status"] == "may_raise"


@pytest.mark.parametrize("draw", [False, True])
def test_one_sided_module_binding_fails_closed_and_matches_native(draw, monkeypatch):
    src = _src("""
        import _sv_verifier
        if _sv_verifier.nondet_bool():
            x = 1
        try:
            x
        except NameError:
            raise TypeError
    """)
    fake_verifier = types.ModuleType("_sv_verifier")
    fake_verifier.nondet_bool = lambda: draw
    monkeypatch.setitem(sys.modules, "_sv_verifier", fake_verifier)
    ok, exc, _ = run_native(src)
    assert (ok, exc) == ((True, "") if draw else (False, "TypeError"))
    result = prove(src, DT)
    assert result.verdict == "UNKNOWN", result.explain()


def test_one_sided_local_binding_fails_closed_and_must_bound_control_stays_provable():
    src = _src("""
        import _sv_verifier
        def f():
            if _sv_verifier.nondet_bool():
                x = 1
            try:
                x
            except UnboundLocalError:
                raise TypeError
        f()
    """)
    result = prove(src, DT)
    assert result.verdict == "UNKNOWN", result.explain()

    must_bound = _src("""
        import _sv_verifier
        if _sv_verifier.nondet_bool():
            x = 1
        else:
            x = 1
        assert x == 1
    """)
    assert prove(must_bound, AS).verdict == "TRUE"


def test_one_sided_unused_binding_and_later_write_do_not_poison_join():
    unused = _src("""
        import _sv_verifier
        if _sv_verifier.nondet_bool():
            x = 1
        assert True
    """)
    assert prove(unused, AS).verdict == "TRUE"

    rebound = _src("""
        import _sv_verifier
        if _sv_verifier.nondet_bool():
            x = 1
        x = 3
        assert x == 3
    """)
    assert prove(rebound, AS).verdict == "TRUE"


def test_divmod_alias_and_bool_operands_remain_outside_narrow_admission():
    alias_src = _src("""
        import _sv_verifier
        alias = divmod
        divisor = _sv_verifier.nondet_int()
        try:
            alias(7, divisor)
        except ZeroDivisionError:
            pass
    """)
    alias_result = prove(alias_src, AR)
    assert alias_result.verdict == "UNKNOWN"
    bool_src = _src("""
        try:
            divmod(7, True)
        except TypeError:
            pass
    """)
    bool_result = prove(bool_src, DT)
    assert bool_result.verdict == "UNKNOWN"


def test_shadowed_divmod_keeps_user_function_analysis():
    src = _src("""
        def divmod(a, b):
            return (a, b)
        q, r = divmod(7, 0)
        assert q == 7 and r == 0
    """)
    ok, exc, _ = run_native(src)
    assert ok, exc
    result = prove(src, AS)
    assert result.verdict == "TRUE", result.explain()


def test_divmod_without_registered_rule_remains_unknown(monkeypatch):
    from tharkun.solve.contracts import rules

    original = rules.get_rule
    monkeypatch.setattr(rules, "get_rule", lambda name: None if name == "divmod" else original(name))
    result = prove("divmod(7, 3)", AS)
    assert result.verdict == "UNKNOWN"
    assert "divmod has no rule" in result.reason


def test_divmod_without_tested_contract_remains_unknown(monkeypatch):
    from tharkun.solve.contracts.adapters import absint as contracts

    original = contracts.contract_for_builtin
    calls = 0

    def missing_after_builtin_resolution(name):
        nonlocal calls
        if name == "divmod":
            calls += 1
            return original(name) if calls == 1 else None
        return original(name)

    monkeypatch.setattr(contracts, "contract_for_builtin", missing_after_builtin_resolution)
    result = prove("divmod(7, 3)", AS)
    assert result.verdict == "UNKNOWN"
    assert "divmod() has no tested contract" in result.reason


# ==============================================================================
# 5. Variable binding in except clause (`as name`) and deletion
# ==============================================================================

def test_except_as_name_binds_and_deletes_after_handler():
    src = _src("""
        x = 0
        try:
            raise KeyError()
        except KeyError as err:
            x = 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    # Native CPython deletes err
    assert "err" not in ns
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_except_as_name_overwrites_and_deletes_prior_name():
    # Native CPython deletes the name even if it was bound before!
    src = _src("""
        err = 10
        try:
            raise KeyError()
        except KeyError as err:
            pass
    """)
    ok, exc, ns = run_native(src)
    assert ok
    assert "err" not in ns


def test_except_as_name_negative_control():
    src = _src("""
        x = 0
        try:
            raise KeyError()
        except KeyError as err:
            x = 1
        assert x != 1
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "AssertionError"
    r = prove(src, DL + AS)
    assert r.verdict == "UNKNOWN"


# ==============================================================================
# 6. Else clause
# ==============================================================================

def test_else_executes_when_no_exception_and_matches_native():
    src = _src("""
        x = 0
        try:
            x = 1
        except KeyError:
            x = 10
        else:
            x = 2
        assert x == 2
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 2
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_else_skipped_when_exception_raised_and_matches_native():
    src = _src("""
        x = 0
        try:
            raise KeyError()
        except KeyError:
            x = 10
        else:
            x = 2
        assert x == 10
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 10
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_else_negative_control():
    # Negative control: assert x == 2 when else was skipped
    src = _src("""
        x = 0
        try:
            raise KeyError()
        except KeyError:
            x = 10
        else:
            x = 2
        assert x == 2
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "AssertionError"
    r = prove(src, DL + AS)
    assert r.verdict == "UNKNOWN"


# ==============================================================================
# 7. Finally on normal and caught exits
# ==============================================================================

def test_finally_on_normal_exit_and_matches_native():
    src = _src("""
        x = 0
        try:
            x = 1
        finally:
            x = x + 1
        assert x == 2
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 2
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


def test_finally_on_caught_handler_exit_and_matches_native():
    src = _src("""
        x = 0
        try:
            raise KeyError()
        except KeyError:
            x = 10
        finally:
            x = x + 1
        assert x == 11
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 11
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_finally_normal_negative_control():
    src = _src("""
        x = 0
        try:
            x = 1
        finally:
            x = x + 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "AssertionError"
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN"


# ==============================================================================
# 8. Finally on return, break, continue
# ==============================================================================

def test_finally_runs_before_return_and_matches_native():
    src = _src("""
        depth = 0
        def helper():
            global depth
            try:
                depth = depth + 1
                return 42
            finally:
                depth = depth - 1

        res = helper()
        assert res == 42
        assert depth == 0
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["res"] == 42 and ns["depth"] == 0
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


def test_finally_runs_on_break_and_matches_native():
    src = _src("""
        x = 0
        while True:
            try:
                break
            finally:
                x = x + 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


def test_finally_runs_on_continue_and_matches_native():
    src = _src("""
        x = 0
        done = False
        while not done:
            try:
                done = True
                continue
            finally:
                x = x + 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


def test_finally_return_negative_control():
    src = _src("""
        depth = 0
        def helper():
            global depth
            try:
                depth = depth + 1
                return 42
            finally:
                depth = depth - 1

        res = helper()
        assert depth != 0
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "AssertionError"
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN"


# ==============================================================================
# 9. Finally overriding return and swallowing exception
# ==============================================================================

def test_finally_overrides_earlier_return_and_matches_native():
    src = _src("""
        def f():
            try:
                return 1
            finally:
                return 2

        assert f() == 2
    """)
    ok, exc, ns = run_native(src)
    assert ok
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


def test_finally_swallows_exception_with_return_and_matches_native():
    src = _src("""
        def f():
            try:
                raise KeyError()
            finally:
                return 0

        assert f() == 0
    """)
    ok, exc, ns = run_native(src)
    assert ok
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_finally_override_negative_control():
    src = _src("""
        def f():
            try:
                return 1
            finally:
                return 2

        assert f() == 1
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "AssertionError"
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN"


# ==============================================================================
# 10. Bare raise (re-raise)
# ==============================================================================

def test_bare_raise_reraises_active_exception_and_matches_native():
    src = _src("""
        x = 0
        try:
            try:
                raise KeyError()
            except KeyError:
                raise
        except KeyError:
            x = 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_bare_raise_negative_control():
    # Negative control: outer except does not catch re-raised KeyError
    src = _src("""
        try:
            try:
                raise KeyError()
            except KeyError:
                raise
        except IndexError:
            pass
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "KeyError"
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"


# ==============================================================================
# 11. Nested try statements
# ==============================================================================

def test_nested_try_inner_unhandled_caught_by_outer_and_matches_native():
    src = _src("""
        x = 0
        try:
            try:
                raise KeyError()
            except ValueError:
                pass
        except KeyError:
            x = 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_nested_try_negative_control():
    src = _src("""
        x = 0
        try:
            try:
                raise KeyError()
            except ValueError:
                pass
        except KeyError:
            x = 1
        assert x != 1
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "AssertionError"
    r = prove(src, DL + AS)
    assert r.verdict == "UNKNOWN"


# ==============================================================================
# 12. Callee exception propagation (call_func)
# ==============================================================================

def test_callee_exception_caught_by_caller_and_matches_native():
    src = _src("""
        def callee():
            raise KeyError("missing")

        x = 0
        try:
            callee()
        except KeyError:
            x = 1
        assert x == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["x"] == 1
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_callee_exception_negative_control():
    src = _src("""
        def callee():
            raise KeyError("missing")

        try:
            callee()
        except IndexError:
            pass
    """)
    ok, exc, ns = run_native(src)
    assert not ok and exc == "KeyError"
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"


# ==============================================================================
# 13. Target mini tasks
# ==============================================================================

def test_ta_caught_keyerror_callee_true():
    src = _src("""
        import _sv_verifier

        def do_patch(has_changes: int) -> int:
            if not has_changes:
                raise KeyError("feature_changes")
            return 200

        def main():
            has_changes = _sv_verifier.nondet_int()
            if not (has_changes == 0):
                return
            try:
                do_patch(has_changes)
            except KeyError:
                pass

        main()
    """)
    r = prove(src, DL)
    assert r.verdict == "TRUE", r.explain()


def test_tracer_counter_try_finally_shape():
    src = _src("""
        class Tracer:
            def __init__(self):
                self.depth = 0

            def trace(self, n):
                try:
                    self.depth = self.depth + 1
                    return 1
                finally:
                    self.depth = self.depth - 1

        def main():
            t = Tracer()
            t.trace(5)
            assert t.depth == 0

        main()
    """)
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


# ==============================================================================
# 14. Named refusals
# ==============================================================================

def test_refusal_try_star():
    src = _src("""
        try:
            pass
        except* ExceptionGroup:
            pass
    """)
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert "except* / ExceptionGroup is not modelled" in r.reason


def test_refusal_exception_group():
    src = _src("""
        raise ExceptionGroup("eg", [ValueError()])
    """)
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert "except* / ExceptionGroup is not modelled" in r.reason


def test_refusal_exception_chaining():
    src = _src("""
        raise KeyError() from None
    """)
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert "exception chaining effects on state are not modelled" in r.reason


def test_refusal_bare_raise_outside_handler():
    src = _src("""
        raise
    """)
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert "bare raise outside an except handler" in r.reason


# ==============================================================================
# 15. Soundness review (Claude Fable, lane TRY): the rules of the rewrite, each with a native control and a
#     negative control.  ``prove`` never answers FALSE, so a wrong verdict is a TRUE the native run refutes.
# ==============================================================================

def _native(src: str):
    """(completed normally, exception class name) of the program on this CPython."""
    ok, exc, _ns = run_native(src)
    return ok, exc


def test_implicit_exception_of_an_operation_is_caught_and_matches_native():
    # a check an operation records (here the KeyError of a dict subscript) is an exception in flight
    src = _src("""
        d = {}
        x = 0
        try:
            d[1]
        except KeyError:
            x = 1
        assert x == 1
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_implicit_exception_negative_control():
    src = _src("""
        d = {}
        try:
            d[1]
        except IndexError:
            pass
    """)
    assert _native(src) == (False, "KeyError")
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert any(c[2] == "key" and "KeyError" in c[3] for c in r.sites), r.explain()   # recorded at ITS site


def test_a_possible_raise_keeps_both_paths_and_matches_native():
    src = _src("""
        import _sv_verifier
        d = {}
        if _sv_verifier.nondet_int() > 0:
            d[1] = 0
        try:
            y = d[1]
        except KeyError:
            y = -1
        assert y >= -1
    """)
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()
    # negative control: the handler path (y == -1) is real, so y == 0 is not provable
    r = prove(src.replace("assert y >= -1", "assert y == 0"), DL + AS)
    assert r.verdict == "UNKNOWN"


def test_callee_side_effects_before_the_raise_reach_the_handler_and_matches_native():
    # the handler sees the state AT the raise (never the state before the statement)
    src = _src("""
        g = 0
        def f():
            global g
            g = 1
            raise KeyError()
        try:
            f()
        except KeyError:
            pass
        assert g == 1
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()
    r = prove(src.replace("assert g == 1", "assert g == 0"), DL + AS)   # negative control
    assert r.verdict == "UNKNOWN"


def test_callee_possible_raise_caught_by_caller_and_matches_native():
    src = _src("""
        def get(d, k):
            return d[k]
        d = {}
        try:
            v = get(d, 1)
        except KeyError:
            v = 0
        assert v == 0
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()
    r = prove(src.replace("except KeyError", "except IndexError"), DL + AS)   # negative control
    assert r.verdict == "UNKNOWN"


def test_break_in_a_loop_inside_try_finally_does_not_run_finally_and_matches_native():
    src = _src("""
        x = 0
        try:
            while True:
                break
            x = 1
        finally:
            x = x + 1
        assert x == 2
    """)
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


def test_break_in_a_loop_inside_try_keeps_the_continuation_negative_control():
    # the continuation after the loop is real: a KeyError there must be found (a break routed to the
    # try's finally would lose it and prove TRUE)
    src = _src("""
        try:
            while True:
                break
            {}[0]
        finally:
            pass
    """)
    assert _native(src) == (False, "KeyError")
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert any("KeyError" in c[3] for c in r.sites), r.explain()


def test_continue_in_a_loop_inside_try_finally_and_matches_native():
    src = _src("""
        n = 0
        x = 0
        try:
            while n < 2:
                n = n + 1
                continue
            x = 1
        finally:
            x = x + 1
        assert x == 2
    """)
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


def test_a_raise_inside_a_loop_reaches_a_try_around_the_loop_and_matches_native():
    src = _src("""
        x = 0
        steps = range(5)
        try:
            for i in steps:
                x = i
                if i == 3:
                    raise KeyError()
        except KeyError:
            y = x
            assert 0 <= y <= 4
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()
    r = prove(src.replace("assert 0 <= y <= 4", "assert y == 5"), DL + AS)   # negative control
    assert r.verdict == "UNKNOWN"


def test_finally_runs_for_an_unhandled_exception_then_it_propagates_and_matches_native():
    src = _src("""
        x = 0
        try:
            try:
                raise KeyError()
            finally:
                x = 1
        except KeyError:
            pass
        assert x == 1
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()
    r = prove(src.replace("assert x == 1", "assert x == 0"), DL + AS)   # negative control
    assert r.verdict == "UNKNOWN"


def test_a_raise_in_a_handler_or_in_else_goes_to_the_outer_try_and_matches_native():
    src = _src("""
        x = 0
        try:
            try:
                raise KeyError()
            except KeyError:
                raise IndexError()
            except IndexError:
                x = 5
        except IndexError:
            x = 1
        try:
            try:
                pass
            except KeyError:
                x = 5
            else:
                raise KeyError()
        except KeyError:
            x = x + 1
        assert x == 2
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_except_as_name_is_deleted_on_every_exit_and_matches_native():
    # after the handler the name is unbound: reading it is a NameError, which ends the program BEFORE the
    # KeyError site, so data-lookup holds (natively: NameError)
    src = _src("""
        try:
            raise KeyError()
        except KeyError as e:
            pass
        e
        {}[0]
    """)
    assert _native(src) == (False, "NameError")
    r = prove(src, DL)
    assert r.verdict == "TRUE", r.explain()
    src_break = _src("""
        for i in range(1):
            try:
                raise KeyError()
            except KeyError as e:
                break
        e
        {}[0]
    """)
    assert _native(src_break) == (False, "NameError")
    r = prove(src_break, DL)
    assert r.verdict == "TRUE", r.explain()


def test_except_as_name_deletion_negative_control():
    # the same program with the name still bound reaches the KeyError site
    src = _src("""
        try:
            raise KeyError()
        except KeyError as e:
            pass
        e = 1
        e
        {}[0]
    """)
    assert _native(src) == (False, "KeyError")
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"


def test_user_exception_classes_match_by_identity_not_by_name():
    src = _src("""
        class E(KeyError):
            pass
        E1 = E
        class E(ValueError):
            pass
        try:
            raise E1()
        except E:
            pass
    """)
    assert _native(src) == (False, "E")      # the first E, a KeyError, escapes
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert any(c[2] == "raise" and "KeyError" in c[3] for c in r.sites), r.explain()


def test_a_raised_class_is_instantiated_and_matches_native():
    src = _src("""
        g = 0
        class E(KeyError):
            def __init__(self):
                global g
                g = 1
        try:
            raise E
        except E:
            pass
        assert g == 1
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()
    r = prove(src.replace("assert g == 1", "assert g == 0"), DL + AS)   # negative control
    assert r.verdict == "UNKNOWN"


def test_bare_raise_in_a_called_function_reraises_the_handled_exception_and_matches_native():
    src = _src("""
        def f():
            raise
        x = 0
        try:
            try:
                raise KeyError()
            except KeyError:
                f()
        except KeyError:
            x = 1
        assert x == 1
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_an_exception_leaving_a_class_body_is_not_dropped():
    src = _src("""
        class A:
            raise KeyError()
    """)
    assert _native(src) == (False, "KeyError")
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert any(c[2] == "raise" and "KeyError" in c[3] for c in r.sites), r.explain()
    src_caught = _src("""
        x = 0
        try:
            class A:
                raise KeyError()
        except KeyError:
            x = 1
        assert x == 1
    """)
    assert _native(src_caught) == (True, "")
    r = prove(src_caught, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_tuple_clause_matches_two_exceptions_of_one_try_and_matches_native():
    src = _src("""
        import _sv_verifier
        c = _sv_verifier.nondet_int()
        x = 0
        try:
            if c > 0:
                raise KeyError()
            raise IndexError()
        except (KeyError, IndexError):
            x = 1
        assert x == 1
    """)
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


# ---- named refusals: what is NOT modelled is UNKNOWN by name, never a guess

def test_refusal_non_property_raise_inside_try_would_otherwise_be_a_wrong_true():
    # natively: int("x") raises ValueError, the handler sets x, the KeyError site is reached
    src = _src("""
        s = "x"
        x = 0
        try:
            n = int(s)
        except ValueError:
            x = 1
        if x == 1:
            {}[0]
    """)
    assert _native(src) == (False, "KeyError")
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert "static callable raises" in r.reason, r.reason


@pytest.mark.parametrize("body, reason", [
    ('try:\n    "abc".index("b")\nexcept ValueError:\n    pass', "contract rule"),
    ('try:\n    undefined_name\nexcept NameError:\n    pass', "NameError inside a try"),
    ('try:\n    x = float("1.5")\nexcept ValueError:\n    pass', "static callable raises"),
    ('try:\n    x = chr(70)\nexcept ValueError:\n    pass', "static callable raises"),
    ('try:\n    r = range(0, 1, 1)\nexcept ValueError:\n    pass', "static callable raises"),
    ('def f():\n    try:\n        pass\n    finally:\n        raise\nf()', "bare raise inside a finally"),
    ('import _sv_verifier\nT = KeyError if _sv_verifier.nondet_int() > 0 else ValueError\ntry:\n    raise KeyError()\n'
     'except T:\n    pass', "not a single value"),
    ('class A(ValueError):\n    pass\nclass B(A, KeyError):\n    pass\ntry:\n    raise B()\nexcept KeyError:\n    pass',
     "multiple inheritance"),
    ('try:\n    while True:\n        pass\nexcept KeyError:\n    pass', "ends every path"),
])
def test_named_refusals_inside_try(body, reason):
    r = prove(_src(body), DL)
    assert r.verdict == "UNKNOWN"
    assert reason in r.reason, r.reason


def test_refusal_harness_draw_inside_try():
    # the harness refuses an invalid draw with a ValueError the program could swallow: not modelled, so the
    # scheduler's own replay keeps the last word (tests/test_sv_draw_vectors.py, the swallowing attacks)
    src = _src("""
        import _sv_verifier
        try:
            x = _sv_verifier.nondet_int()
        except ValueError:
            x = 0
    """)
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN"
    assert "static callable raises" in r.reason, r.reason


def test_f_string_inside_try_refuses_unrouted_conversion_raises():
    src = _src("""
        x = 0
        try:
            s = f"value {x}"
        except KeyError:
            x = 1
        assert x == 0
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "contract rule" in r.reason

# ---- Length hints catch exception identity, including user subclasses ([internal ref]).

@pytest.mark.parametrize("consumer, supported", [
    ("xs = list(L())", True),
    ("xs = sorted(L())", True),
    ("xs = []\nxs.extend(L())", True),
    ("xs = []\nxs += L()", False),
])
@pytest.mark.parametrize("definition, raised", [
    ("", "TypeError"),
    ("class E(TypeError):\n    pass", "E"),
    ("class E(TypeError):\n    pass\nclass F(E):\n    pass", "F"),
])
def test_length_hint_typeerror_subclasses_keep_caller_continuation(consumer, supported, definition, raised):
    src = (
        definition + "\n"
        "g = 0\n"
        "class L:\n"
        "    def __iter__(self):\n"
        "        return iter([1])\n"
        "    def __len__(self):\n"
        "        global g\n"
        "        g = 1\n"
        f"        raise {raised}()\n"
        + consumer + "\nassert g == 1"
    )
    ok, exc, ns = run_native(src)
    assert ok and exc == "" and ns["xs"] == [1] and ns["g"] == 1
    r = prove(src, AS + DT)
    assert r.verdict == ("TRUE" if supported else "UNKNOWN"), r.explain()
    if not supported:
        assert "in-place builtin sequence" in r.reason
    # The old PR178 lost E(TypeError)'s continuation and incorrectly certified this assertion.
    bad = src + "\nassert False"
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS).verdict == "UNKNOWN"


@pytest.mark.parametrize("definition, raised", [
    ("class E(ValueError):\n    pass", "E"),
    ("class TypeError(Exception):\n    pass", "TypeError"),
])
def test_length_hint_other_exception_identities_propagate(definition, raised):
    src = (
        definition + "\n"
        "class L:\n"
        "    def __iter__(self):\n"
        "        return iter([1])\n"
        "    def __len__(self):\n"
        f"        raise {raised}()\n"
        "xs = []\nxs.extend(L())\nassert False"
    )
    assert _native(src) == (False, raised)
    # Both are non-property exceptions; they end execution before the failing assertion.
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()
    caught = (
        src.rsplit("xs = []", 1)[0]
        + "caught = 0\ntry:\n    xs = []\n    xs.extend(L())\n"
        f"except {raised}:\n    caught = 1\n"
        "assert caught == 1"
    )
    assert _native(caught) == (True, "")
    r = prove(caught, AS)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "builtin method raises" in r.reason
    bad = caught.replace("assert caught == 1", "assert caught == 0")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS).verdict == "UNKNOWN"

@pytest.mark.parametrize("bases", ["TypeError, ValueError", "TypeError, TypeError"])
def test_multiple_exception_bases_are_refused_before_identity_is_guessed(bases):
    src = (
        f"class E({bases}):\n    pass\n"
        "class L:\n"
        "    def __iter__(self):\n"
        "        return iter([1])\n"
        "    def __len__(self):\n"
        "        raise E()\n"
        "xs = []\nxs.extend(L())\nassert False"
    )
    ok, exc, _ = run_native(src)
    assert not ok
    assert exc == ("AssertionError" if bases == "TypeError, ValueError" else "TypeError")
    r = prove(src, AS + DT)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "multiple inheritance" in r.reason


def test_a_caught_subscript_uses_the_original_raising_state_without_duplicate_contract_exit():
    src = _src("""
        g = 0
        def f():
            global g
            g = 1
            return {}[0]
        try:
            f()
        except KeyError:
            pass
        assert g == 1
    """)
    assert _native(src) == (True, "")
    r = prove(src, AS + DL)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("assert g == 1", "assert g == 0")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS + DL).verdict == "UNKNOWN"


@pytest.mark.parametrize("setup, body, caught, reason", [
    ("", "[1].index(2)", "ValueError", "AssertionError"),
    ("", "(1,).index(2)", "ValueError", "AssertionError"),
    ("", "[1].remove(2)", "ValueError", "AssertionError"),
    ("f = [1].index", "f(2)", "ValueError", "AssertionError"),
    ("def f():\n    [1].index(2)", "f()", "ValueError", "AssertionError"),
    ("s = {1}", "s.remove(2)", "KeyError", "builtin method raises"),
    ("import random", "random.choice([])", "IndexError", "static callable raises"),
    ("", "range(1, 2, False)", "ValueError", "static callable raises"),
    ("f = int\ns = 'x'", "f(s)", "ValueError", "static callable raises"),
    ("import io\nf = io.StringIO()", "f.seek(-1)", "ValueError", "builtin method raises"),
    ("", "x, y = (1,)", "ValueError", "unpacking"),
    ("xs = [1, 2]", "xs[::2] = []", "ValueError", "slice operation"),
    ("xs = [1]", "xs[::0]", "ValueError", "slice operation"),
    ("", "1 << -1", "ValueError", "negative count"),
    ("", "1 >> -1", "ValueError", "negative count"),
    ("", "10.0 ** 1000", "OverflowError", "float arithmetic"),
    ("d = {1: 1}", "for k in d:\n    d[2] = 2", "RuntimeError", "iteration"),
    ("s = {1}", "for k in s:\n    s.add(2)", "RuntimeError", "iteration"),
    ("", "raise UnicodeDecodeError()", "TypeError", "constructor"),
    ("class E(UnicodeDecodeError):\n    pass", "raise E()", "TypeError", "constructor"),
    ("class L:\n    def __len__(self):\n        return 100000000000000000000000", "if L():\n    pass", "OverflowError", "Py_ssize_t"),
])
def test_callable_and_implicit_raises_cannot_hide_a_failing_handler(setup, body, caught, reason):
    src = setup + "\ntry:\n" + textwrap.indent(body, "    ") + f"\nexcept {caught}:\n    assert False"
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN", r.explain()
    assert reason in r.reason, r.reason


@pytest.mark.parametrize("setup, body", [
    ("", "ValueError(x=1)"),
    ("class E(ValueError):\n    pass", "E(x=1)"),
    ("class C:\n    pass", "C(1)"),
    ("e = ValueError()", "e()"),
])
def test_constructor_typeerrors_keep_the_actual_handler_path(setup, body):
    src = setup + "\ntry:\n    " + body + "\nexcept TypeError:\n    assert False"
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS + DT)
    assert r.verdict == "UNKNOWN", r.explain()


def test_analyzed_user_method_can_raise_and_caller_continuation_is_not_lost():
    src = _src("""
        class C:
            def f(self):
                raise ValueError()
        c = C()
        n = 0
        try:
            c.f()
        except ValueError:
            n = 1
        assert n == 1
    """)
    assert _native(src) == (True, "")
    assert prove(src, AS + DT).verdict == "TRUE"
    bad = src.replace("assert n == 1", "assert n == 0")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS + DT).verdict == "UNKNOWN"


@pytest.mark.parametrize("name", [
    "BaseException", "Exception", "TypeError", "ValueError", "RuntimeError",
    "LookupError", "IndexError", "KeyError", "AssertionError", "ArithmeticError",
    "ZeroDivisionError", "FloatingPointError", "OverflowError",
])
def test_supported_builtin_exception_constructor_and_keyword_boundary(name):
    src = f"n = 0\ntry:\n    raise {name}(1, 'x')\nexcept {name}:\n    n = 1\nassert n == 1"
    assert _native(src) == (True, "")
    assert prove(src, AS + DT).verdict == "TRUE"
    bad = f"try:\n    {name}(x=1)\nexcept TypeError:\n    assert False"
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS + DT).verdict == "UNKNOWN"


@pytest.mark.parametrize("setup, call", [
    ("xs = []", "xs.index(1)"),
    ("xs = []", "xs.remove(1)"),
    ("xs = ()", "xs.index(1)"),
    ("xs = []\nlookup = xs.index", "lookup(1)"),
    ("xs = []\nlookup = xs.remove", "lookup(1)"),
    ("xs = [1]\nxs.clear()", "xs.index(1)"),
])
def test_empty_sequence_search_routes_valueerror_and_preserves_the_heap(setup, call):
    src = (setup + "\nalias = xs\ncaught = False\ntry:\n    " + call
           + "\nexcept ValueError:\n    caught = True\nassert caught and len(alias) == 0")
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("assert caught and len(alias) == 0", "assert not caught")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS).verdict == "UNKNOWN"
    bad_handler = src.replace("caught = True", "assert False")
    assert _native(bad_handler) == (False, "AssertionError")
    assert prove(bad_handler, AS).verdict == "UNKNOWN"


@pytest.mark.parametrize("setup, call", [
    ("xs = []\nalias = xs", "xs.pop()"),
    ("xs = []\nalias = xs\npop = xs.pop", "pop()"),
])
def test_empty_list_pop_routes_indexerror_and_preserves_the_heap(setup, call):
    src = (setup + "\ncaught = False\nfinalized = False\ntry:\n    " + call
           + "\nexcept IndexError:\n    caught = True\nfinally:\n    finalized = True"
           + "\nassert caught and finalized and len(alias) == 0")
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()
    bad_handler = src.replace("caught = True", "assert False")
    assert _native(bad_handler) == (False, "AssertionError")
    assert prove(bad_handler, AS).verdict == "UNKNOWN"
    bad_postcondition = src.replace("assert caught and finalized", "assert not caught and finalized")
    assert _native(bad_postcondition) == (False, "AssertionError")
    assert prove(bad_postcondition, AS).verdict == "UNKNOWN"


def test_empty_bound_list_pop_keeps_its_original_receiver():
    src = _src("""
        xs = []
        alias = xs
        pop = xs.pop
        xs = [7]
        caught = False
        try:
            pop()
        except IndexError:
            caught = True
        assert caught and len(alias) == 0 and len(xs) == 1 and xs[0] == 7
    """)
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


def test_nonempty_list_pop_remains_outside_the_empty_raise_route():
    src = _src("""
        xs = [7]
        try:
            value = xs.pop()
        except IndexError:
            assert False
        assert value == 7 and len(xs) == 0
    """)
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN", r.explain()


@pytest.mark.parametrize("call, caught", [
    ("[].pop(0)", "IndexError"),
    ("[].pop(index=0)", "TypeError"),
])
def test_empty_list_pop_with_arguments_remains_outside_the_noarg_route(call, caught):
    src = f"try:\n    {call}\nexcept {caught}:\n    assert False"
    assert _native(src) == (False, "AssertionError")
    assert prove(src, AS).verdict == "UNKNOWN"


@pytest.mark.parametrize("handler", ["LookupError", "Exception", "", "(KeyError, IndexError)"])
def test_empty_list_pop_matches_property_class_superclass_tuple_and_bare_handlers(handler):
    clause = f"except {handler}:" if handler else "except:"
    src = "caught = False\ntry:\n    [].pop()\n" + clause + "\n    caught = True\nassert caught"
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_nonmatching_keyerror_does_not_hide_uncaught_empty_pop_property_site():
    src = "try:\n    [].pop()\nexcept KeyError:\n    assert False"
    assert _native(src) == (False, "IndexError")
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN", r.explain()
    assert any(line == 2 and op == "pop" and exc == "IndexError"
               for line, _col, op, exc, _fact in r.sites), r.explain()


def test_uncaught_empty_list_pop_keeps_the_data_lookup_site():
    src = "xs = []\nxs.pop()"
    assert _native(src) == (False, "IndexError")
    r = prove(src, DL)
    assert r.verdict == "UNKNOWN", r.explain()
    assert any(line == 2 and op == "pop" and exc == "IndexError"
               for line, _col, op, exc, _fact in r.sites), r.explain()


def test_empty_list_pop_else_finally_and_reraise_match_native():
    src = _src("""
        handled = False
        took_else = False
        finalized = False
        try:
            [].pop()
        except IndexError:
            handled = True
        else:
            took_else = True
        finally:
            finalized = True
        assert handled and not took_else and finalized
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()

    reraised = "try:\n    [].pop()\nexcept IndexError:\n    raise"
    assert _native(reraised) == (False, "IndexError")
    assert prove(reraised, DL).verdict == "UNKNOWN"


def test_empty_list_pop_nonmatching_inner_handler_reaches_outer_handler():
    src = _src("""
        inner = False
        outer = False
        try:
            try:
                [].pop()
            except KeyError:
                inner = True
        except IndexError:
            outer = True
        assert outer and not inner
    """)
    assert _native(src) == (True, "")
    r = prove(src, DL + AS)
    assert r.verdict == "TRUE", r.explain()


def test_list_pop_empty_only_on_one_path_remains_refused():
    src = _src("""
        import _sv_verifier
        xs = []
        if _sv_verifier.nondet_int() > 0:
            xs.append(7)
        try:
            xs.pop()
        except IndexError:
            pass
    """)
    import sys
    import types

    old = sys.modules.get("_sv_verifier")
    draws = types.ModuleType("_sv_verifier")
    try:
        for value in (0, 1):
            draws.nondet_int = lambda value=value: value
            sys.modules["_sv_verifier"] = draws
            assert _native(src) == (True, "")
    finally:
        if old is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old
    r = prove(src, DL + AS)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "list.pop" in r.reason and "raises are not routed" in r.reason


def test_empty_search_callee_and_finally_keep_the_actual_exception_state():
    src = _src("""
        seen = 0
        def item():
            global seen
            seen = 7
            return 1
        def lookup(xs):
            return xs.index(item())
        xs = []
        caught = False
        finalized = False
        try:
            lookup(xs)
        except ValueError:
            caught = True
        finally:
            finalized = True
        assert seen == 7 and caught and finalized and len(xs) == 0
    """)
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("seen == 7", "seen == 0")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS).verdict == "UNKNOWN"


def test_empty_search_uses_the_builtin_exception_identity_after_name_shadowing():
    src = _src("""
        E = ValueError
        ValueError = KeyError
        caught = 0
        try:
            try:
                [].index(1)
            except ValueError:
                assert False
        except E:
            caught = 1
        assert caught == 1
    """)
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("assert caught == 1", "assert caught == 0")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS).verdict == "UNKNOWN"


def test_bound_empty_search_uses_the_receiver_not_a_rebound_variable():
    src = _src("""
        xs = []
        lookup = xs.index
        xs = [1]
        caught = False
        try:
            lookup(1)
        except ValueError:
            caught = True
        assert caught and len(xs) == 1
    """)
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


@pytest.mark.parametrize("setup, call, caught", [
    ("xs = [1]", "xs.index(2)", "ValueError"),
    ("xs = [1]", "xs.remove(2)", "ValueError"),
    ("xs = (1,)", "xs.index(2)", "ValueError"),
    ("xs = []\nlookup = xs.index\nxs.append(1)", "lookup(2)", "ValueError"),
    ("xs = []", "xs.index()", "TypeError"),
    ("xs = []", "xs.remove(1, 2)", "TypeError"),
    ("xs = []", "xs.index(value=1)", "TypeError"),
    ("xs = []", "xs.index(1, 'bad')", "TypeError"),
    ("xs = ()", "xs.index(1, None)", "TypeError"),
    ("class Bound:\n    def __index__(self):\n        raise KeyError()\nxs = []", "xs.index(1, Bound())", "KeyError"),
    ("class L(list):\n    def index(self, x):\n        raise KeyError()\nxs = L()", "xs.index(1)", "KeyError"),
])
def test_sequence_search_does_not_hide_failing_handlers_or_admit_unmodeled_forms(setup, call, caught):
    src = setup + "\ntry:\n    " + call + "\nexcept " + caught + ":\n    assert False"
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN", r.explain()


@pytest.mark.parametrize("setup, call, length", [
    ("xs = [1]", "xs.index(2)", 1),
    ("xs = (1,)", "xs.index(2)", 1),
    ("xs = [1]", "xs.remove(2)", 1),
    ("xs = [1, 2]", "xs.index(3)", 2),
    ("xs = (1, 2)", "xs.index(0)", 2),
    ("xs = [1, 2]", "xs.remove(3)", 2),
    ("xs = [False, True]", "xs.index(2)", 2),
    ("xs = [1, 2]", "xs.index(False)", 2),
    ("xs = ['a', 'b']", "xs.index('z')", 2),
    ("xs = ('a', 'b')", "xs.index('z')", 2),
    ("xs = ['a', 'b']", "xs.remove('z')", 2),
    ("xs = [None]", "xs.remove(1)", 1),
    ("xs = [1, 2]", "xs.index(3, -100, 100)", 2),
    ("xs = (1, 2)", "xs.index(3, False, True)", 2),
    ("xs = [1]\nlookup = xs.index", "lookup(2)", 1),
    ("xs = [1]\nlookup = xs.remove", "lookup(2)", 1),
    ("xs = []\nlookup = xs.index\nxs.append(1)", "lookup(2)", 1),
    ("xs = [1]\nlookup = xs.index\nxs = [2]", "lookup(2)", 1),
    ("xs = [1]", "xs.index(1, 1)", 1),
    ("xs = (1,)", "xs.index(1, 0, 0)", 1),
    ("xs = [1, 2, 3]", "xs.index(2, 1, 1)", 3),
    ("xs = (1, 2, 3)", "xs.index(2, 2, 1)", 3),
    ("xs = [1, 2]", "xs.index(1, 3)", 2),
    ("xs = [1, 2, 3]", "xs.index(2, -1, -2)", 3),
    ("xs = (1, 2, 3)", "xs.index(2, -1, -1)", 3),
    ("xs = [1, 2]", "xs.index(1, -3, -10)", 2),
    ("xs = (1, 2)", "xs.index(1, 0, -2)", 2),
    ("xs = [1, 2]", "xs.index(1, True, True)", 2),
    ("xs = (1, 2)", "xs.index(True, True, False)", 2),
    ("xs = [1]", "xs.index(1, 1267650600228229401496703205376)", 1),
    ("xs = (1,)", "xs.index(1, -1267650600228229401496703205376, -1267650600228229401496703205375)", 1),
    ("xs = ['a', 'b']", "xs.index('a', 1, 0)", 2),
    ("xs = [None]", "xs.index(None, 0, 0)", 1),
    ("xs = [1]\nlookup = xs.index", "lookup(1, 1)", 1),
    ("xs = [1]\nlookup = xs.index\nxs = [1, 1]", "lookup(1, 1)", 2),
])
def test_definitely_absent_sequence_search_routes_valueerror_without_mutation(setup, call, length):
    src = (setup + "\nalias = xs\ncaught = False\ntry:\n    " + call
           + "\nexcept ValueError:\n    caught = True\nelse:\n    assert False\nassert caught and len(alias) == " + str(length))
    assert _native(src) == (True, "")
    r = prove(src, AS + DT)
    assert r.verdict == "TRUE", r.explain()
    for bad in (src.replace("caught = True", "assert False"),
                src.replace("assert caught and", "assert not caught and")):
        assert _native(bad) == (False, "AssertionError")
        r = prove(bad, AS + DT)
        assert r.verdict == "UNKNOWN" and r.complete, r.explain()
        assert "AssertionError" in r.reason


@pytest.mark.parametrize("method", ["index", "remove"])
def test_absent_search_callee_routes_post_argument_state_through_finally(method):
    src = _src("""
        g = 0
        xs = [1]
        def item():
            global g
            g = 7
            return 2
        def lookup():
            xs.METHOD(item())
            g = 100
        caught = False
        finalized = False
        try:
            lookup()
        except ValueError:
            caught = True
        finally:
            finalized = True
        assert g == 7 and caught and finalized and len(xs) == 1 and xs[0] == 1
    """).replace("METHOD", method)
    assert _native(src) == (True, "")
    r = prove(src, AS + DT + DL)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("g == 7", "g == 0")
    assert _native(bad) == (False, "AssertionError")
    r = prove(bad, AS + DT + DL)
    assert r.verdict == "UNKNOWN" and r.complete, r.explain()


@pytest.mark.parametrize("setup, call", [
    ("xs = [1]", "xs.index(1)"),
    ("xs = (1,)", "xs.index(True)"),
    ("xs = [1]", "xs.remove(True)"),
    ("xs = [1]", "xs.index(1.0)"),
    ("xs = [1.0]", "xs.remove(1)"),
    ("xs = [1, 3]", "xs.index(2)"),  # joined interval includes 2
    ("xs = ['a', 'b']", "xs.index('a')"),
    ("v = float('nan')\nxs = [v]", "xs.index(v)"),
    ("xs = [1]\nlookup = xs.index\nxs.append(2)", "lookup(2)"),
    ("xs = [1]", "xs.index(1, 0, 1)"),
    ("xs = (1, 1)", "xs.index(1, 1, 2)"),
    ("xs = [1, 2]", "xs.index(1, 1)"),  # nonempty slice exclusion is still refused
    ("xs = [1]", "xs.index(1, -1267650600228229401496703205376, 1267650600228229401496703205376)"),
    ("xs = [1]", "xs.index(1, False, True)"),
    ("xs = [1]", "xs.index(1, 0.0, 0)"),
    ("xs = (1,)", "xs.index(1, 0, None)"),
    ("xs = [1.0]", "xs.index(1, 0, 0)"),
    ("xs = [1]", "xs.index(1.0, 0, 0)"),
    ("xs = [[1]]", "xs.index([1], 0, 0)"),
    ("xs = [1]", "xs.index(2, 'bad')"),
    ("xs = (1,)", "xs.index(2, None)"),
    ("xs = [1]", "xs.index(2, 0.0)"),
    ("xs = [1]", "xs.remove(2, 3)"),
    ("xs = [1]", "xs.index(value=2)"),
])
def test_uncertain_search_and_unsupported_call_forms_remain_named_refusals(setup, call):
    src = setup + "\ntry:\n    " + call + "\nexcept Exception:\n    pass\nassert True"
    assert _native(src) == (True, "")
    r = prove(src, AS + DT)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()
    assert "unsupported:" in r.reason


def test_empty_index_slice_preserves_argument_order_effects_and_finally():
    src = _src("""
        g = 0
        xs = [1]
        def item():
            global g
            g = 1
            return 1
        def start():
            global g, xs
            g = g * 10 + 2
            xs = [2]
            return 0
        def stop():
            global g
            g = g * 10 + 3
            return 0
        caught = False
        finalized = False
        try:
            xs.index(item(), start(), stop())
        except ValueError:
            caught = True
        else:
            assert False
        finally:
            finalized = True
        assert g == 123 and caught and finalized and xs[0] == 2
    """)
    assert _native(src) == (True, "")
    r = prove(src, AS + DT + DL)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("g == 123", "g == 12")
    assert _native(bad) == (False, "AssertionError")
    r = prove(bad, AS + DT + DL)
    assert r.verdict == "UNKNOWN" and r.complete, r.explain()


@pytest.mark.parametrize("raising_arg", ["item", "start"])
def test_empty_index_slice_does_not_override_an_earlier_argument_exception(raising_arg):
    src = _src("""
        g = 0
        def item():
            global g
            g = 1
            ITEM
        def start():
            global g
            g = 2
            START
        def stop():
            global g
            g = 3
            return 0
        caught = False
        try:
            [1].index(item(), start(), stop())
        except KeyError:
            caught = True
        except ValueError:
            assert False
        assert caught and g == EXPECTED
    """).replace("ITEM", "raise KeyError()" if raising_arg == "item" else "return 1")
    src = src.replace("START", "raise KeyError()" if raising_arg == "start" else "return 0")
    src = src.replace("EXPECTED", "1" if raising_arg == "item" else "2")
    assert _native(src) == (True, "")
    r = prove(src, AS + DL)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("assert caught and", "assert not caught and")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS + DL).verdict == "UNKNOWN"


@pytest.mark.parametrize("setup, call", [
    ("xs = [1]", "xs.index(1, 0, 1)"),
    ("xs = (1, 1)", "xs.index(1, 1, 2)"),
    ("xs = [1]", "xs.index(1, -2, 2)"),
    ("xs = [1]", "xs.index(1, -1267650600228229401496703205376, 1267650600228229401496703205376)"),
    ("xs = [1]", "xs.index(1, False, True)"),
    ("xs = [1, 1]\ndef stop():\n    global xs\n    xs = [1]\n    return 2", "xs.index(1, 1, stop())"),
    ("xs = [1]\nstart = 0\ndef stop():\n    global start\n    start = 1\n    return 1", "xs.index(1, start, stop())"),
])
def test_nonempty_index_slice_cannot_hide_a_failing_else_or_replace_saved_operands(setup, call):
    src = setup + "\ntry:\n    " + call + "\nexcept ValueError:\n    pass\nelse:\n    assert False"
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()
    assert "builtin method raises" in r.reason


def test_absent_search_rechecks_the_receiver_after_argument_mutation():
    src = _src("""
        xs = [1]
        def item():
            xs[0] = 2
            return 2
        try:
            xs.index(item())
        except ValueError:
            pass
        assert False
    """)
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()
    assert "builtin method raises" in r.reason


def test_empty_search_still_refuses_user_comparisons_and_builtin_namespace_changes():
    src = _src("""
        class Item:
            def __eq__(self, other):
                raise KeyError()
        try:
            [].index(Item())
        except ValueError:
            assert False
    """)
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "user comparison" in r.reason
    rebound = "__builtins__ = {}\n" + src
    r = prove(rebound, AS)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "builtin namespace" in r.reason


@pytest.mark.parametrize("call", ["[].index(1)", "[].remove(1)", "().index(1)"])
def test_uncaught_empty_search_still_ends_execution_before_a_failing_assertion(call):
    src = call + "\nassert False"
    assert _native(src) == (False, "ValueError")
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()


@pytest.mark.parametrize("call", [
    "chr(-1)", "chr(0x110000)", "chr(1267650600228229401496703205376)", "chr(-1267650600228229401496703205376)",
    "range(0, 3, 0)", "range(-(1 << 100), 1 << 100, 0)",
    "int('nope')", "float('nope')", "int('bad')", "float('bad')",
    "int('')", "float('')", "int(' + ')", "float(' - ')",
    "int('²')", "float('²')",
])
def test_definitely_invalid_primitive_builtins_route_valueerror_and_match_native(call):
    src = "caught = False\ntry:\n    " + call + "\nexcept ValueError:\n    caught = True\nassert caught"
    assert _native(src) == (True, "")
    r = prove(src, AS + DT)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("caught = True", "assert False")
    assert _native(bad) == (False, "AssertionError")
    r = prove(bad, AS + DT)
    assert r.verdict == "UNKNOWN", r.explain()
    bad = src.replace("assert caught", "assert not caught")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS + DT).verdict == "UNKNOWN"


@pytest.mark.parametrize("callee, arguments", [("chr", "arg()"), ("range", "0, 3, arg()")])
def test_primitive_valueerror_alias_and_callee_keep_argument_effects_and_finally(callee, arguments):
    src = _src("""
        g = 0
        def arg():
            global g
            g = g + 1
            return VALUE
        def f():
            alias = CALLEE
            alias(ARGUMENTS)
            g = 100
        try:
            f()
        except ValueError:
            g = g + 2
        finally:
            g = g + 4
        assert g == 7
    """).replace("VALUE", "-1" if callee == "chr" else "0").replace("CALLEE", callee).replace("ARGUMENTS", arguments)
    assert _native(src) == (True, "")
    r = prove(src, AS + DT)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("assert g == 7", "assert g == 6")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS + DT).verdict == "UNKNOWN"


@pytest.mark.parametrize("call", ["chr(-1)", "range(0, 3, 0)", "int('nope')", "float('nope')"])
def test_primitive_valueerror_uses_builtin_exception_identity(call):
    src = ("E = ValueError\nValueError = KeyError\ncaught = False\ntry:\n    " + call
           + "\nexcept E:\n    caught = True\nassert caught")
    assert _native(src) == (True, "")
    r = prove(src, AS + DL)
    assert r.verdict == "TRUE", r.explain()


@pytest.mark.parametrize("call", [
    "chr(70)", "chr(True)", "chr(1.0)", "chr()", "chr(i=-1)",
    "chr(1 << 100)", "chr(-(1 << 100))",
    "range(0, 3, 1)", "range(0, 3, False)", "range(0, 3, 0.0)",
    "range(0.0, 3, 0)", "range(stop=3)", "range(0, 3)",
    "int('12')", "float('1.5')", "int('١')", "float('١.٥')",
    "int('1x')", "float('1x')", "int('nan')", "float('nan')", "float(' -NaN ')",
    "float('inf')", "float(' +INF ')", "float('Infinity')", "float(' -iNfInItY ')",
    "int()", "float()",
    "int('nope', 10)", "int('nope', base=10)", "int(x='nope')", "float(x='nope')",
    "float('nope', 1)", "int('nope', **{})", "float('nope', **{})",
])
def test_primitive_valueerror_does_not_admit_other_call_shapes(call):
    src = "try:\n    " + call + "\nexcept Exception:\n    assert False"
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "static callable raises" in r.reason, r.reason


@pytest.mark.parametrize("call, reason", [
    ("int(b'nope')", "constant of type bytes"),
    ("float(b'nope')", "constant of type bytes"),
    ("int('no' + 'pe')", "builtin sequence/set arithmetic"),
    ("float('no' + 'pe')", "builtin sequence/set arithmetic"),
])
def test_invalid_numeric_literal_refuses_unsupported_argument_evaluation(call, reason):
    src = "try:\n    " + call + "\nexcept Exception:\n    assert False"
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN", r.explain()
    assert reason in r.reason, r.reason


@pytest.mark.parametrize("call", ["chr(Item())", "range(0, 3, Item())", "int(Item())", "float(Item())"])
def test_primitive_valueerror_does_not_skip_custom_index_protocol(call):
    src = _src("""
        class Item:
            def __index__(self):
                raise KeyError()
        try:
            CALL
        except KeyError:
            assert False
    """).replace("CALL", call)
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN", r.explain()


@pytest.mark.parametrize("name, args", [("chr", "-1"), ("range", "0, 3, 0"), ("int", "'nope'"), ("float", "'nope'")])
def test_primitive_valueerror_does_not_override_user_callable_or_namespace(name, args):
    src = ("def " + name + "(*args):\n    raise KeyError()\ntry:\n    " + name + "(" + args
           + ")\nexcept KeyError:\n    assert False")
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN", r.explain()
    rebound = "__builtins__ = {}\n" + src
    r = prove(rebound, AS)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "builtin namespace" in r.reason, r.reason


@pytest.mark.parametrize("name, args", [("chr", "-1"), ("range", "0, 3, 0"), ("int", "'nope'"), ("float", "'nope'")])
def test_primitive_valueerror_requires_the_existing_tested_rule(monkeypatch, name, args):
    from tharkun.solve.contracts import rules
    original = rules.get_rule
    monkeypatch.setattr(rules, "get_rule", lambda n: None if n == name else original(n))
    src = "try:\n    " + name + "(" + args + ")\nexcept ValueError:\n    pass\nassert True"
    assert _native(src) == (True, "")
    r = prove(src, AS + DT)
    assert r.verdict == "UNKNOWN", r.explain()
    assert name + " has no rule" in r.reason, r.reason


@pytest.mark.parametrize("name", ["int", "float"])
def test_invalid_numeric_literal_alias_preserves_callee_state_and_finally(name):
    src = _src("""
        g = 0
        def f():
            global g
            g = 1
            alias = CAST
            alias('nope')
            g = 100
        try:
            f()
        except ValueError:
            g = g + 2
        finally:
            g = g + 4
        assert g == 7
    """).replace("CAST", name)
    assert _native(src) == (True, "")
    r = prove(src, AS + DT)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("assert g == 7", "assert g == 6")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS + DT).verdict == "UNKNOWN"


@pytest.mark.parametrize("name", ["int", "float"])
def test_invalid_numeric_literal_keeps_builtin_callable_identity(name):
    src = ("alias = " + name + "\ndef " + name + "(value):\n    raise KeyError()\n"
           "caught = False\ntry:\n    alias('nope')\nexcept ValueError:\n    caught = True\nassert caught")
    assert _native(src) == (True, "")
    r = prove(src, AS + DL)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("alias('nope')", name + "('nope')").replace("except ValueError:", "except KeyError:")
    bad = bad.replace("caught = True", "assert False")
    assert _native(bad) == (False, "AssertionError")
    assert prove(bad, AS + DL).verdict == "UNKNOWN"


@pytest.mark.parametrize("name", ["int", "float"])
def test_uncaught_invalid_numeric_literal_ends_before_failing_assertion(name):
    src = name + "('nope')\nassert False"
    assert _native(src) == (False, "ValueError")
    r = prove(src, AS + DT)
    assert r.verdict == "TRUE", r.explain()


@pytest.mark.parametrize("method", ["index", "rindex"])
def test_absent_string_search_routes_valueerror_under_try(method):
    src = _src("""
        caught = False
        try:
            'abc'.METHOD('z')
        except ValueError:
            caught = True
        else:
            assert False
        assert caught
    """).replace("METHOD", method)
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "TRUE", r.explain()
    bad = src.replace("caught = True", "assert False")
    assert _native(bad) == (False, "AssertionError")
    r = prove(bad, AS)
    assert r.verdict == "UNKNOWN" and r.complete, r.explain()
    assert "AssertionError" in r.reason


@pytest.mark.parametrize("method", ["index", "rindex"])
def test_present_string_search_else_cannot_be_proved_unreachable(method):
    src = _src("""
        try:
            'abc'.METHOD('b')
        except ValueError:
            pass
        else:
            assert False
    """).replace("METHOD", method)
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()
    assert "contract rule" in r.reason


@pytest.mark.parametrize("call", [
    "'abc'.index('z', 1)", "'abc'.rindex('z', 0, 2)",
    "'abc'.index('z', start=0)", "'abc'.rindex('z', 0, stop=2)",
    "'abc'.index()", "'abc'.rindex('z', 0, 1, 2)",
    "'abc'.index(b'z')", "'abc'.rindex(1)",
    "'abc'.index('')", "''.rindex('')",
])
def test_string_search_keeps_bounds_aliases_and_other_shapes_refused(call):
    src = "try:\n    " + call + "\nexcept Exception:\n    pass\nassert True"
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()
    assert "unsupported:" in r.reason


def test_string_search_bound_method_alias_remains_refused():
    src = "s = 'abc'\nlookup = s.index\ntry:\n    lookup('z')\nexcept ValueError:\n    pass"
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()


@pytest.mark.parametrize("haystack, needle", [
    ("z" * 65, "z"),
    ("z" * 65, "z" * 65),
])
def test_string_search_wildcard_receiver_or_needle_remains_refused(haystack, needle):
    src = ("try:\n    " + repr(haystack) + ".index(" + repr(needle) + ")\n"
           "except ValueError:\n    pass\nelse:\n    assert False")
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()


def test_string_search_absence_requires_every_finite_pair_to_be_absent():
    from tharkun.solve.absint.aval import V
    from tharkun.solve.absint.interp import Analyzer

    analyzer = Analyzer("pass\n")
    finite_absence = [V(s=frozenset({"abc", "def"})), V(s=frozenset({"z", "x"}))]
    one_matching_pair = [V(s=frozenset({"abc", "def"})), V(s=frozenset({"z", "a"}))]
    all_absent = _src("""
        for haystack in ("abc", "def"):
            for needle in ("z", "x"):
                try:
                    haystack.index(needle)
                except ValueError:
                    pass
                else:
                    assert False
    """)
    one_match = _src("""
        for haystack in ("abc", "def"):
            for needle in ("z", "a"):
                try:
                    haystack.index(needle)
                except ValueError:
                    pass
                else:
                    assert False
    """)
    assert _native(all_absent) == (True, "")
    assert _native(one_match) == (False, "AssertionError")
    assert analyzer._primitive_valueerror_call(None, "str.index", finite_absence, {}, None)
    assert not analyzer._primitive_valueerror_call(None, "str.rindex", one_matching_pair, {}, None)


def test_string_search_requires_the_existing_rule(monkeypatch):
    from tharkun.solve.contracts import rules
    original = rules.get_rule
    monkeypatch.setattr(rules, "get_rule", lambda name: None if name == "str.index" else original(name))
    src = "try:\n    'abc'.index('z')\nexcept ValueError:\n    pass\nassert True"
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()
    assert not r.complete
    assert "str.index has no rule" in r.reason


def test_string_search_requires_a_tested_contract(monkeypatch):
    from tharkun.solve.contracts.adapters import absint as contracts
    monkeypatch.setattr(contracts, "contract_for_method", lambda owner, name: None)
    src = "try:\n    'abc'.rindex('z')\nexcept ValueError:\n    pass\nassert True"
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN", r.explain()
    assert not r.complete
    assert "str.rindex has no tested contract" in r.reason


@pytest.mark.parametrize("expression", [
    "'abc'.find('z')", "'abc'.rfind('z')", "'abc'.count('z')", "'z' in 'abc'",
])
def test_nonraising_string_search_operations_are_not_routed_as_valueerror(expression):
    src = "try:\n    EXPRESSION\nexcept ValueError:\n    assert False\nassert True".replace("EXPRESSION", expression)
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN", r.explain()


# ==============================================================================
# 16. Non-raising builtin print calls in try blocks ([internal ref])
# ==============================================================================

def test_print_finite_primitives_and_literal_options_are_nonraising():
    src = _src("""
        alias = print
        try:
            alias(1, True, 'ok', None, 2.5, sep=':', end='!')
        except Exception:
            assert False
        else:
            assert True
    """)
    assert _native(src) == (True, "")
    r = prove(src, AS)
    assert r.verdict == "TRUE" and r.complete, r.explain()
    failing_handler = _src("""
        try:
            print(1, 'ok', sep=':')
        except Exception:
            pass
        else:
            assert False
    """)
    assert _native(failing_handler) == (False, "AssertionError")
    r = prove(failing_handler, AS)
    assert r.verdict == "UNKNOWN" and r.complete, r.explain()
    assert "AssertionError" in r.reason


@pytest.mark.parametrize("call", [
    "print('x' * 65)", "print('x', sep='s' * 65)", "print('x', end='e' * 65)",
    "print('\\ud800')", "print('x', sep='\\ud800')", "print('x', end='\\ud800')",
    "print('é')", "print('x', sep='é')", "print('x', end='é')",
    "print(_sv_verifier.nondet_str())", "print('x', sep=_sv_verifier.nondet_str())",
    "print('x', end=_sv_verifier.nondet_str())", "print(10 ** 5000)",
    "print(*('x',))", "print('x', **{})", "print('x', file=None)", "print('x', flush=False)",
])
def test_print_unsafe_or_unknown_shapes_remain_named_refusals(call):
    src = _src("""
        import _sv_verifier
        try:
            CALL
        except Exception:
            assert False
        else:
            assert True
    """).replace("CALL", call)
    # Some refusals include a native caught exception (notably the huge int); others succeed.
    native_src = src.replace("import _sv_verifier", "").replace("_sv_verifier.nondet_str()", "'safe'")
    ok, exc = _native(native_src)
    assert (ok and not exc) or (not ok and exc == "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()


def test_print_requires_the_existing_rule_and_tested_contract(monkeypatch):
    from tharkun.solve.contracts import rules
    from tharkun.solve.contracts.adapters import absint as contracts

    src = "try:\n    print('x')\nexcept Exception:\n    assert False\nelse:\n    assert True"
    assert _native(src) == (True, "")
    original_rule = rules.get_rule
    original_contract = contracts.contract_for_builtin
    monkeypatch.setattr(rules, "get_rule", lambda name: None if name == "print" else original_rule(name))
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete and "print has no rule" in r.reason, r.explain()
    monkeypatch.setattr(rules, "get_rule", original_rule)
    monkeypatch.setattr(contracts, "contract_for_builtin",
                        lambda name: None if name == "print" else original_contract(name))
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete and "print has no tested contract" in r.reason, r.explain()


@pytest.mark.parametrize("src, native_result", [
    (_src("""
        import sys, io
        old = sys.stdout
        try:
            sys.stdout = io.StringIO()
            print('x')
        except Exception:
            sys.stdout = old
            assert False
        else:
            sys.stdout = old
            assert True
    """), (True, "")),
    (_src("""
        import contextlib, io
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                print('x')
        except Exception:
            assert False
        else:
            assert True
    """), (True, "")),
    (_src("""
        import sys, io
        old = sys.stdout
        stream = io.StringIO()
        sys.stdout = stream
        stream.close()
        try:
            print('x')
        except Exception:
            sys.stdout = old
            assert False
        else:
            sys.stdout = old
            assert True
    """), (False, "AssertionError")),
])
def test_print_stdout_mutation_or_closed_stream_is_not_admitted(src, native_result):
    assert _native(src) == native_result
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()


def test_print_user_formatting_method_that_raises_is_not_admitted():
    src = _src("""
        class Bad:
            def __str__(self):
                raise RuntimeError('formatting')
        try:
            print(Bad())
        except Exception:
            assert False
        else:
            assert True
    """)
    assert _native(src) == (False, "AssertionError")
    r = prove(src, AS)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()


@pytest.mark.parametrize("src, encoding", [
    (_src("""
        try:
            print('é')
        except ValueError:
            1 // 0
    """), "ascii:strict"),
    (_src(r"""
        try:
            print('\ud800')
        except ValueError:
            1 // 0
    """), "utf-8:strict"),
])
def test_print_nonascii_can_raise_under_inherited_stdout_encoding(src, encoding):
    child = (f"source = {src!r}\n"
             "try:\n    exec(source, {})\nexcept Exception as error:\n    print(type(error).__name__)\n")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = encoding
    native = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True,
                            text=True, check=True)
    assert native.stdout.strip() == "ZeroDivisionError"
    r = prove(src, AR)
    assert r.verdict == "UNKNOWN" and not r.complete, r.explain()


# ==============================================================================
# round's exact built-in scalar transfer
# ==============================================================================

@pytest.mark.parametrize(("value", "expected"), [
    ("0.5", 0), ("1.5", 2), ("2.5", 2), ("-0.5", 0), ("-1.5", -2), ("True", 1),
])
def test_round_builtin_scalar_matches_native_and_assertions(value, expected):
    src = _src(f"result = round({value})\nassert result == {expected}")
    native_src = _src(f"result = round({value})\nassert type(result) is int and result == {expected}")
    ok, exc, ns = run_native(native_src)
    assert ok and type(ns["result"]) is int and ns["result"] == expected, (ok, exc, ns)
    result = prove(src, ["AssertionError"])
    assert result.verdict == "TRUE", result.explain()


def test_round_finite_float_inside_try_is_modelled():
    src = _src("""
        try:
            result = round(2.5)
        except ValueError:
            result = -1
        assert result == 2
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["result"] == 2, (ok, exc, ns)
    result = prove(src, ["AssertionError"])
    assert result.verdict == "TRUE", result.explain()


@pytest.mark.parametrize(("literal", "exception"), [
    ("math.nan", "ValueError"), ("math.inf", "OverflowError"), ("-math.inf", "OverflowError"),
])
def test_round_nonfinite_caught_input_remains_refused(literal, exception):
    src = _src(f"""
        import math
        value = {literal}
        caught = False
        try:
            round(value)
        except {exception}:
            caught = True
        assert caught
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["caught"], (ok, exc, ns)
    result = prove(src, ["AssertionError"])
    assert result.verdict == "UNKNOWN", result.explain()
    assert "round() of a float that may raise" in result.explain()


@pytest.mark.parametrize(("literal", "exception"), [
    ("math.nan", "ValueError"),
    ("math.inf", "OverflowError"),
    ("-math.inf", "OverflowError"),
])
def test_round_nonfinite_handler_assertion_is_not_proved(literal, exception):
    src = _src(f"""
        import math
        value = {literal}
        try:
            round(value)
        except {exception}:
            assert False
    """)
    ok, exc, _ = run_native(src)
    assert not ok and exc == "AssertionError"
    result = prove(src, ["AssertionError"])
    assert result.verdict == "UNKNOWN", result.explain()
    assert "round() of a float that may raise" in result.explain()


def test_round_infinite_outside_try_ends_the_path():
    src = _src("""
        import math
        round(math.inf)
        assert False
    """)
    ok, exc, _ = run_native(src)
    assert not ok and exc == "OverflowError"
    result = prove(src, ["AssertionError"])
    assert result.verdict == "TRUE", result.explain()


@pytest.mark.parametrize("call", [
    "round(1.25, 0)", "round(1.25, None)", "round(number=1.5)", "round(number=1.5, ndigits=0)",
])
def test_round_valid_keyword_and_ndigits_forms_are_not_mis_modelled(call):
    src = _src(f"result = {call}\nassert result == result")
    ok, exc, _ = run_native(src)
    assert ok, (exc, call)
    result = prove(src, ["AssertionError"])
    assert result.verdict == "UNKNOWN", result.explain()


def test_round_custom_dunder_remains_refused():
    src = _src("""
        class Custom:
            def __round__(self):
                return 1
        result = round(Custom())
        assert result == 1
    """)
    ok, exc, ns = run_native(src)
    assert ok and ns["result"] == 1, (ok, exc, ns)
    result = prove(src, ["AssertionError"])
    assert result.verdict == "UNKNOWN", result.explain()

# Unknown imports can raise ImportError/ModuleNotFoundError before binding.  The
# abstract interpreter may assume success only when its existing module or
# Resolver tables prove that success; otherwise a surrounding handler is not
# allowed to disappear from the proof.
@pytest.mark.parametrize(("statement", "handler", "reason"), [
    ("import _absint_p112_missing_xyz", "ImportError", "ImportError inside a try"),
    ("import _absint_p112_missing_xyz", "ModuleNotFoundError", "ImportError inside a try"),
    ("from math import _absint_p112_missing_name_xyz", "ImportError", "math import origin is not resolved"),
    ("from math import _absint_p112_missing_name_xyz", "Exception", "math import origin is not resolved"),
    ("import _absint_p112_missing_xyz", "", "ImportError inside a try"),  # bare except
])
def test_unknown_imports_with_catching_handlers_remain_unknown(statement, handler, reason):
    except_clause = "except:" if not handler else f"except {handler}:"
    src = _src(f"""
        caught = False
        try:
            {statement}
        {except_clause}
            caught = True
        assert not caught
    """)
    ok, exc, _ = run_native(src)
    assert not ok and exc == "AssertionError"
    result = prove(src, AS)
    assert result.verdict == "UNKNOWN", result.explain()
    assert reason in result.explain()


def test_unknown_import_in_function_catches_module_not_found_and_refuses():
    src = _src("""
        def load():
            try:
                import _absint_p112_missing_function_xyz
            except ModuleNotFoundError:
                return "caught"
            return "loaded"
        assert load() == "loaded"
    """)
    ok, exc, _ = run_native(src)
    assert not ok and exc == "AssertionError"
    result = prove(src, AS)
    assert result.verdict == "UNKNOWN", result.explain()
    assert "ImportError inside a try" in result.explain()


def test_modeled_stdlib_and_contract_imports_inside_try_remain_usable():
    for stmt in [
        "import math",
        "from math import sqrt",
        "import heapq",
        "from heapq import heappop",
        "import operator",
        "from operator import add",
    ]:
        src = _src(f"""
            caught = False
            try:
                {stmt}
            except ImportError:
                caught = True
            assert not caught
        """)
        ok, exc, _ = run_native(src)
        assert ok, exc
        result = prove(src, AS)
        assert result.verdict == "TRUE", result.explain()


def test_operator_native_contract_imports_use_resolved_metadata():
    from tharkun.solve.absint.interp import Analyzer
    from tharkun.solve.pyresolve import module_attrs

    native_src = _src("""
        import operator as ops
        from operator import add as plus
        assert ops.add(2, 3) == plus(2, 3) == 5
    """)
    ok, exc, _ = run_native(native_src)
    assert ok, exc

    # Keep import admission separate from operator-call precision: this
    # abstraction recognizes the module/export metadata, not call results.
    src = _src("""
        import operator as ops
        from operator import add as plus
        x = 1
        y = x + 1
        assert y == 2
    """)
    analyzer = Analyzer(src)
    resolver = analyzer._governing_resolver(None)
    assert resolver is not None
    assert "operator" in resolver.allowed
    assert "add" in (module_attrs("operator") or ())
    assert analyzer._import_known_to_succeed("operator", None, analyzer.tree.body[0])
    assert analyzer._import_known_to_succeed(
        "operator", None, analyzer.tree.body[1], name="add", alias="plus")
    result = prove(src, AS)
    assert result.verdict == "TRUE", result.explain()


def test_missing_operator_export_inside_try_is_not_assumed_success():
    src = _src("""
        caught = False
        try:
            from operator import _absint_missing_operator_export_xyz
        except ImportError:
            caught = True
        assert not caught
    """)
    ok, exc, _ = run_native(src)
    assert not ok and exc == "AssertionError"
    result = prove(src, AS)
    assert result.verdict == "UNKNOWN", result.explain()
    assert "operator import origin is not resolved" in result.explain()


def test_operator_import_root_shadow_is_still_rejected(tmp_path):
    (tmp_path / "operator.py").write_text("raise TypeError('shadow import failed')\n")
    program = tmp_path / "main.py"
    src = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        caught = False
        try:
            import operator
        except TypeError:
            caught = True
        assert not caught
    """)
    program.write_text(src)
    proc = subprocess.run([sys.executable, "-S", "-c", src], capture_output=True, text=True)
    assert proc.returncode != 0 and "AssertionError" in proc.stderr, proc.stderr
    result = prove(src, AS, filename=str(program))
    assert result.verdict == "UNKNOWN", result.explain()
    assert "environment: stdlib shadow 'operator'" in result.explain()


@pytest.mark.parametrize("module", ["list", "protocol"])
def test_absint_contract_module_allowlist_excludes_pseudo_module_families(module):
    from tharkun.solve.contracts.rules import native_module_names

    names = native_module_names()
    assert "operator" in names
    assert module not in names
    src = _src(f"""
        caught = False
        try:
            import {module}
        except ImportError:
            caught = True
        assert not caught
    """)
    ok, exc, _ = run_native(src)
    assert not ok and exc == "AssertionError"
    result = prove(src, AS)
    assert result.verdict == "UNKNOWN", result.explain()
    assert f"{module} import origin is not resolved" in result.explain()


def test_resolver_proven_task_module_imports_remain_usable(tmp_path):
    import sys
    from tharkun.solve.absint.interp import Analyzer

    module = f"_absint_p112_helper_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module}.py").write_text("VALUE = 7\n")
    program = tmp_path / "main.py"
    src = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import {module} as helper
        from {module} import VALUE
        assert helper.VALUE == VALUE == 7
    """)
    program.write_text(src)
    analyzer = Analyzer(src, filename=str(program))
    resolver = analyzer._governing_resolver(None)
    assert resolver is not None
    assert module in resolver.imports.modules.values()
    assert resolver.imports.names["VALUE"] == (f"$extern:{tmp_path / (module + '.py')}", "VALUE")
    import_node, from_node = analyzer.tree.body[2:4]
    assert analyzer._import_known_to_succeed(module, None, import_node)
    assert analyzer._import_known_to_succeed(
        module, None, from_node, name="VALUE", alias="VALUE")
    assert not analyzer._import_known_to_succeed(
        module, None, from_node, name="MISSING", alias="MISSING")
    old_path = sys.path[:]
    try:
        ok, exc, _ = run_native(src)
        assert ok, exc
        result = prove(src, AS, filename=str(program))
        assert result.verdict == "TRUE", result.explain()
    finally:
        sys.path[:] = old_path
        sys.modules.pop(module, None)


def test_shadowed_modeled_stdlib_import_inside_try_is_not_assumed_success(tmp_path):
    import sys

    (tmp_path / "heapq.py").write_text("raise TypeError('shadow import failed')\n")
    program = tmp_path / "main.py"
    src = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        caught = False
        try:
            import heapq
        except TypeError:
            caught = True
        assert not caught
    """)
    program.write_text(src)
    proc = subprocess.run(
        [sys.executable, "-S", "-c", src], capture_output=True, text=True)
    assert proc.returncode != 0 and "AssertionError" in proc.stderr, proc.stderr
    result = prove(src, AS, filename=str(program))
    assert result.verdict == "UNKNOWN", result.explain()
    assert "environment: stdlib shadow 'heapq'" in result.explain()


def test_shadow_origin_guard_runs_before_known_module_fastpath(tmp_path, monkeypatch):
    from tharkun.solve import extern
    from tharkun.solve.absint.interp import Analyzer, Unsupported

    (tmp_path / "heapq.py").write_text("raise TypeError('shadow import failed')\n")
    program = tmp_path / "main.py"
    src = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        try:
            import heapq
        except TypeError:
            pass
    """)
    program.write_text(src)
    # Isolate the helper from prove()'s earlier, whole-program import preflight;
    # Resolver itself still parses this real source and supplies its root snapshot.
    monkeypatch.setattr(extern, "native_import_reason", lambda *args, **kwargs: None)
    analyzer = Analyzer(src, filename=str(program))
    resolver = analyzer._governing_resolver(None)
    assert resolver is not None
    import_node = analyzer.tree.body[2].body[0]
    with pytest.raises(Unsupported, match="task root shadows stdlib module 'heapq'"):
        analyzer._import_known_to_succeed("heapq", None, import_node)


def test_shadowed_modeled_stdlib_import_outside_try_refuses(tmp_path):
    import sys

    (tmp_path / "heapq.py").write_text("raise TypeError('shadow import failed')\n")
    program = tmp_path / "main.py"
    src = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import heapq
        assert True
    """)
    program.write_text(src)
    proc = subprocess.run(
        [sys.executable, "-S", "-c", src], capture_output=True, text=True)
    assert proc.returncode != 0 and "TypeError: shadow import failed" in proc.stderr, proc.stderr
    result = prove(src, ("TypeError",), filename=str(program))
    assert result.verdict == "UNKNOWN", result.explain()
    assert "environment: stdlib shadow 'heapq'" in result.explain()


def test_unknown_import_outside_try_keeps_nonproperty_failure_irrelevant():
    src = _src("""
        import _absint_p112_missing_outside_try_xyz
        value = 7
        assert value == 7
    """)
    ok, exc, _ = run_native(src)
    assert not ok and exc == "ModuleNotFoundError"
    result = prove(src, ("TypeError",))
    assert result.verdict == "TRUE", result.explain()


def test_known_draw_caught_importerror_no_longer_proves_typeerror_absent(monkeypatch):
    import sys
    import types

    draws = iter((2577420508, 2577420585))
    verifier = types.ModuleType("_sv_verifier")
    verifier.nondet_int = lambda: next(draws)
    monkeypatch.setitem(sys.modules, "_sv_verifier", verifier)
    src = _src("""
        import _sv_verifier
        try:
            import _absint_p112_missing_draw_xyz
            value = 1
        except ImportError:
            value = "s"
        n = _sv_verifier.nondet_int()
        m = _sv_verifier.nondet_int()
        if n * 2654435761 % 4294967291 == 987654 and m == n + 77:
            result = value + 1
    """)
    ok, exc, _ = run_native(src)
    assert not ok and exc == "TypeError"
    result = prove(src, DT)
    assert result.verdict == "UNKNOWN", result.explain()
    assert not result.complete, result.explain()
    assert "ImportError inside a try" in result.explain()
