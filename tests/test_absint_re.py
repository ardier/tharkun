"""Shared regex rules: finite CPython folds and conservative unknown-string results.

Negative controls retain UNKNOWN, with a named refusal for target callbacks.
Successful matching is never inferred for an unknown string; later sites stay live.
"""
from __future__ import annotations

import textwrap

import pytest

from tharkun.solve.absint import prove

TE = ("TypeError",)
TA = ("TypeError", "AttributeError")
AS = ("AssertionError",)
DL = ("KeyError", "IndexError")


def _src(body: str) -> str:
    return textwrap.dedent(body)


# ------------------------------------------------------------------ PROVABLE
PROVABLE = {
    # contains_html's real shape (string_utils/validation.py): search-result
    # truthiness only, never a group.
    "search_truthiness_is_provable": (TA, """
        import re, _sv_verifier
        HTML_RE = re.compile(r"<[^>]+>")
        def contains_html(s):
            return HTML_RE.search(s) is not None
        s = _sv_verifier.nondet_str()
        print(contains_html(s))
    """),
    # words_count's real shape: findall()'s LENGTH only, never an element.
    "findall_length_is_provable": (TA, """
        import re, _sv_verifier
        WORDS_RE = re.compile(r"[a-zA-Z0-9_]+")
        s = _sv_verifier.nondet_str()
        n = len(WORDS_RE.findall(s))
        print(n)
    """),
    # is_number's shape (module-level form, not a compiled Pattern): match()
    # truthiness via the module-level function, not a Pattern method.
    "module_level_match_truthiness_is_provable": (TA, """
        import re, _sv_verifier
        s = _sv_verifier.nondet_str()
        print(re.match(r"[0-9]+", s) is not None)
    """),
    # fullmatch, search, and findall's None/length-only shape all provable
    # the same way through the module-level forms too.
    "module_level_fullmatch_truthiness_is_provable": (TA, """
        import re, _sv_verifier
        s = _sv_verifier.nondet_str()
        print(re.fullmatch(r"[0-9]+", s) is not None)
    """),
    # sub()'s result is unconditionally str -- usable in any str-shaped
    # context, e.g. concatenation.
    "sub_result_is_str": (TA, """
        import re, _sv_verifier
        s = _sv_verifier.nondet_str()
        out = re.sub(r"a", "b", s)
        print(out + "!")
    """),
    # split()'s result is a list -- len()/truthiness provable without
    # touching an element (the interleaved-group shape is unknown).
    "split_length_is_provable": (TA, """
        import re, _sv_verifier
        s = _sv_verifier.nondet_str()
        n = len(re.split(r",", s))
        print(n)
    """),
    # Pattern methods (not just module-level forms) behave the same way.
    "pattern_search_truthiness_is_provable": (TA, """
        import re, _sv_verifier
        p = re.compile(r"x")
        s = _sv_verifier.nondet_str()
        print(p.search(s) is not None)
    """),
    # keyword arguments sub/split legitimately take (count=/maxsplit=) do
    # not block the proof.
    "sub_with_count_keyword_is_provable": (TA, """
        import re, _sv_verifier
        p = re.compile(r"x")
        s = _sv_verifier.nondet_str()
        out = p.sub("y", s, count=1)
        print(out + "!")
    """),
}


@pytest.mark.parametrize("name", sorted(PROVABLE))
def test_provable(name):
    classes, body = PROVABLE[name]
    r = prove(_src(body), classes)
    assert r.verdict == "TRUE", r.explain()


# --------------------------------------------------------- NEVER_TRUE / negative controls
NEVER_TRUE = {
    # #1102b negative control (MAIN, owner-specified): a user object passed
    # where a str is expected must not give TRUE -- it is a real, reachable
    # TypeError.
    "user_object_where_str_expected_module_level": (TE, """
        import re
        class Weird:
            pass
        re.search(r"x", Weird())
    """),
    "user_object_where_str_expected_pattern_method": (TE, """
        import re
        class Weird:
            pass
        p = re.compile(r"x")
        p.search(Weird())
    """),
    "user_object_where_str_expected_compile_pattern": (TE, """
        import re
        class Weird:
            pass
        re.compile(Weird())
    """),
    "non_str_repl_to_sub": (TE, """
        import re
        p = re.compile(r"x")
        p.sub(5, "hello")
    """),
}


@pytest.mark.parametrize("name", sorted(NEVER_TRUE))
def test_never_true(name):
    classes, body = NEVER_TRUE[name]
    r = prove(_src(body), classes)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "unproven check" in r.reason or "callback or unknown regex argument kind" in r.reason, r.reason


# #1102b negative control (MAIN, owner-specified): a pattern built from a
# nondet string must not let a search/match/fullmatch PROVE success --
# standalone (not NEVER_TRUE: nothing is reachably violated in a program
# that never runs; this pins that the ANALYSIS stays honestly unproven, the
# same "gate, not arithmetic" shape as [internal ref]'s shift/floordiv negative
# controls).
def test_pattern_from_nondet_string_never_proves_a_match():
    r = prove(_src("""
        import re, _sv_verifier
        pat = _sv_verifier.nondet_str()
        p = re.compile(pat)
        s = _sv_verifier.nondet_str()
        assert p.search(s) is not None
    """), AS)
    assert r.verdict == "UNKNOWN", r.explain()


# A match must never be assumed to fail either -- the SAME honesty in the
# other direction (a real search() call CAN succeed; claiming it can't would
# be just as wrong a narrowing, only visible on the opposite assertion).
def test_search_result_never_proves_always_none_either():
    r = prove(_src("""
        import re, _sv_verifier
        p = re.compile(r"[0-9]+")
        s = _sv_verifier.nondet_str()
        assert p.search(s) is None
    """), AS)
    assert r.verdict == "UNKNOWN", r.explain()


# Two DIFFERENT compiled patterns must never be treated as the same object
# (a wrong "always equal" narrowing would be a real bug: they compile
# different text).
def test_two_different_patterns_are_not_proven_identical():
    r = prove(_src("""
        import re
        p = re.compile(r"a")
        q = re.compile(r"b")
        assert p is q
    """), AS)
    assert r.verdict == "UNKNOWN", r.explain()


# ------------------------------------------------------------------ UNSUPPORTED
# Constructs this coarse pass deliberately does not model -- must abort to
# UNKNOWN (never guess), same convention as test_absint.py's own UNSUPPORTED
# dict (fixed DL tail, only the abort matters).
UNSUPPORTED = {
    # Match's own methods/attributes are not modelled at all (that is the
    # whole point of the sound-coarse boundary: control flow around
    # match SUCCESS is provable; extracting a group is not, yet).
    "match_group_call": """
        import re, _sv_verifier
        p = re.compile(r"x")
        s = _sv_verifier.nondet_str()
        m = p.search(s)
        if m is not None:
            print(m.group())
        [][0]
    """,
    "match_start_call": """
        import re, _sv_verifier
        p = re.compile(r"x")
        s = _sv_verifier.nondet_str()
        m = p.match(s)
        if m is not None:
            print(m.start())
        [][0]
    """,
    # Findall/split elements are TOP -- using one (not just len()/
    # truthiness of the list) aborts.
    "findall_element_used": """
        import re, _sv_verifier
        p = re.compile(r"[a-z]+")
        s = _sv_verifier.nondet_str()
        xs = p.findall(s)
        if xs:
            print(xs[0].upper())
        [][0]
    """,
    "split_element_used": """
        import re, _sv_verifier
        s = _sv_verifier.nondet_str()
        xs = re.split(r",", s)
        print(xs[0].upper())
        [][0]
    """,
}


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_unsupported(name):
    r = prove(_src(UNSUPPORTED[name]), DL)
    assert r.verdict == "UNKNOWN" and r.reason


# ------------------------------------------------------ MG's PR 89 review (9 wrong TRUEs)
# MG's adversarial review ([host]:[local path] 89edge/) found the
# arity/keyword checks above rejecting a VALID call shape (positional flags/pos/endpos,
# the flags=/count=/maxsplit=/pos=/endpos= keywords, a callable repl) responded with
# `st.set_bottom()` -- which claims the call UNCONDITIONALLY raises, silently making
# every later site in the same block unreachable, a wrong TRUE. Fixed: every one of
# these now raises Unsupported instead (verdict UNKNOWN, never TRUE, regardless of
# WHY -- an unproven reachable check and "this exact shape isn't modelled yet" are
# both sound, so the assertion here is just "never TRUE", not a specific reason).
MG_ATTACKS = {
    "module_search_with_positional_flags": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        m = re.search('a', s, re.IGNORECASE)
        v = xs[0]
    """,
    "module_match_with_positional_flags": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        m = re.match('a', s, re.IGNORECASE)
        v = xs[0]
    """,
    "module_findall_with_positional_flags": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        r = re.findall('a', s, re.IGNORECASE)
        v = xs[0]
    """,
    "module_findall_with_flags_keyword": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        r = re.findall('a', s, flags=re.IGNORECASE)
        v = xs[0]
    """,
    "module_search_with_flags_keyword": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        m = re.search('a', s, flags=re.IGNORECASE)
        v = xs[0]
    """,
    "compile_with_flags_keyword": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        p = re.compile('a', flags=re.IGNORECASE)
        v = xs[0]
    """,
    "pattern_match_with_positional_pos": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        p = re.compile('a')
        m = p.match(s, 0)
        v = xs[0]
    """,
    "pattern_search_with_positional_pos_endpos": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        p = re.compile('a')
        m = p.search(s, 1, 3)
        v = xs[0]
    """,
    "pattern_findall_with_positional_pos": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        p = re.compile('a')
        r = p.findall(s, 0)
        v = xs[0]
    """,
    "module_split_with_positional_maxsplit": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        r = re.split('a', s, 1)
        v = xs[0]
    """,
    "module_sub_with_positional_count": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        r = re.sub('a', 'b', s, 1)
        v = xs[0]
    """,
    "module_sub_with_callable_repl": """
        import re, _sv_verifier
        xs = []
        s = _sv_verifier.nondet_str()
        r = re.sub('a', lambda m: 'b', s)
        v = xs[0]
    """,
}


@pytest.mark.parametrize("name", sorted(MG_ATTACKS))
def test_mg_attacks_never_prove_true(name):
    r = prove(_src(MG_ATTACKS[name]), DL)
    assert r.verdict != "TRUE", r.explain()


# The verdict-level repro MG posted directly ([host]:[local path]
# re_flags_hidden.py / re_flags_forced.py): a positional-flags re.search() call
# earlier in the program used to hide a LATER, unrelated IndexError entirely.
def test_positional_flags_does_not_hide_a_later_unrelated_site():
    r = prove(_src("""
        import re, _sv_verifier
        xs = []
        h = _sv_verifier.nondet_int()
        m = re.search("a", "abc", re.IGNORECASE)
        if h == 875841331:
            v = xs[0]
    """), DL)
    assert r.verdict != "TRUE", r.explain()
