"""[internal ref]: `x += <set>` (and the sibling set-to-sequence sinks) must never certify.

Found by MG (helper B) while reviewing [internal ref]: the determinism certificate's rule 3
(set-order tracking, ``determinism.py``'s ``_Census.check_sets``) re-typed a LIST
target as a tracked "set name" after ``acc += s`` (an ``ast.AugAssign``), because
the fixpoint that decides which names may hold a set treated any augmented
assignment the same as a real rebinding. ``x += <set>`` never changes ``x``'s type
(``list.__iadd__`` mutates and returns the same list; ``__add__``/``__or__``/etc.
return a value of the target's own type) -- only an earlier real binding tells us
whether a name is a set. Treating ``acc`` as set-tracked made ``acc != witness``
(list != list, order-sensitive) get checked as an order-blind set comparison
instead: a wrong TRUE on a program that raises AssertionError under
PYTHONHASHSEED=3 (and other seeds) but not others.

Fixed in two places (`_Census.check_sets` / `_Census.check_set_use`):
  - an ``AugAssign`` node never adds its target to ``set_names``/``set_attrs``;
  - an ``AugAssign``'s own value/target occurrence is only exempt from the
    order-sensitive-use check when the target is ALREADY a tracked set (a
    genuine ``s |= other_set``), never merely because it received one.
"""
from __future__ import annotations

from textwrap import dedent

import pytest

from tharkun.solve.determinism import determinism_census

# The exact repro (MG helper B / [internal ref]): natively raises AssertionError
# under PYTHONHASHSEED=3 (1 of 10 seeds tried), never under the others.
WRONG_TRUE_REPRO = """
    s = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}
    acc = []
    acc += s
    witness = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]
    assert acc != witness
    print("survived")
"""


def test_list_iadd_from_a_set_is_never_true():
    result = determinism_census(dedent(WRONG_TRUE_REPRO))
    assert result.ok is False, result.reason


def test_a_genuine_set_to_set_iadd_stays_safe():
    """Safe mirror: `acc |= s` where `acc` really is a set (both established by a
    real assignment, `set()`) is order-blind -- must NOT be caught by the fix
    above, which only removes AugAssign's power to falsely CREATE that tracking,
    not a real one already established by `acc = set()`."""
    source = """
        s = {1, 2, 3}
        acc = set()
        acc |= s
        print(len(acc))
    """
    result = determinism_census(dedent(source))
    assert result.ok is True, result.reason


# MG's sink map (rev/1081/setorder.py): 18 ways a set's hash-iteration-order can
# reach a sequence. Every one of these already used an order-SENSITIVE read
# downstream (`acc[0] != "banana"` or equivalent), so every one must refuse --
# this is the existing, deliberate design for the other 17 sinks (each flags at
# the point a set's contents flow into a non-set container, not only if a later
# read happens to be order-sensitive); `list_iadd` (first entry) is the one this
# fix newly covers, the other 17 were already correctly refused before [internal ref].
SET_ORDER_SINKS = {
    "list_iadd": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = []\nacc += s\nassert acc[0] != 'banana'\n"
    ),
    "list_extend": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = []\nacc.extend(s)\nassert acc[0] != 'banana'\n"
    ),
    "slice_assign": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = ['x']\nacc[:] = s\nassert acc[0] != 'banana'\n"
    ),
    "slice_insert": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = ['x']\nacc[0:0] = s\nassert acc[0] != 'banana'\n"
    ),
    "star_list": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = [*s]\nassert acc[0] != 'banana'\n"
    ),
    "star_tuple": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = (*s,)\nassert acc[0] != 'banana'\n"
    ),
    "list_ctor": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = list(s)\nassert acc[0] != 'banana'\n"
    ),
    "tuple_iadd": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = ()\nacc += tuple(s)\nassert acc[0] != 'banana'\n"
    ),
    "dict_fromkeys": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = list(dict.fromkeys(s))\nassert acc[0] != 'banana'\n"
    ),
    "for_append": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = []\nfor x in s:\n    acc.append(x)\nassert acc[0] != 'banana'\n"
    ),
    "listcomp": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = [x for x in s]\nassert acc[0] != 'banana'\n"
    ),
    "deque_extend": (
        "import collections\n"
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = collections.deque()\nacc.extend(s)\nassert acc[0] != 'banana'\n"
    ),
    "bytearray_iadd": (
        "s = {1, 2, 300 % 256, 17, 99, 250, 7}\n"
        "acc = bytearray()\nacc += bytes(s)\nassert acc[0] != 99\n"
    ),
    "str_join": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = [','.join(s)]\nassert not acc[0].startswith('banana')\n"
    ),
    "iadd_via_func": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "def add(a, b):\n    a += b\n    return a\n"
        "acc = add([], s)\nassert acc[0] != 'banana'\n"
    ),
    "zip_set": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = [a for a, b in zip(s, range(7))]\nassert acc[0] != 'banana'\n"
    ),
    "enumerate_set": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        "acc = [x for i, x in enumerate(s)]\nassert acc[0] != 'banana'\n"
    ),
}


@pytest.mark.parametrize("name", sorted(SET_ORDER_SINKS))
def test_set_to_sequence_sinks_are_never_true(name):
    result = determinism_census(SET_ORDER_SINKS[name])
    assert result.ok is False, (name, result.reason)


# -- [internal ref] part 2 (MG, rev/1093/mix1093.py + list_then_set_later.py) --------------
# Part 1's fix stopped `+=` from CREATING set-tracking, but set_names/set_attrs was
# still flow- AND scope-insensitive: a name counted as tracked if ANY of its
# bindings anywhere in the program was a set-source, so a list at the order-
# sensitive use was still wrongly exempted whenever the SAME name was also bound
# to a set somewhere else -- a later real assignment, a dead branch, or a `global`
# rebind inside a function that never runs before the use. Fixed by requiring
# EVERY binding of a name (across all scopes, all forms) to agree it is a set
# before the name is tracked at all; a single non-set binding untracks it
# everywhere, including at uses that come before that binding in the source.
_S = 's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
_W = 'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'

MIX_1093_NEVER_TRUE = {
    "list_then_set_later": (
        _S + _W + "acc = []\nacc += s\nassert acc != w\nacc = set()\n"
    ),
    "set_in_dead_branch": (
        _S + _W + "acc = []\nif len(s) > 100:\n    acc = set()\nacc += s\nassert acc != w\n"
    ),
    "tuple_then_set": (
        _S + _W + "acc = ()\nacc += tuple(s)\nassert acc != tuple(w)\nacc = frozenset()\n"
    ),
    "global_rebind": (
        _S + _W + "acc = []\ndef reset():\n    global acc\n    acc = set()\n"
        "acc += s\nassert acc != w\nreset()\n"
    ),
    # standalone file, byte-identical to rev/1093/list_then_set_later.py
    "list_then_set_later_standalone_file": (
        _S + _W + "acc = []\nacc += s\nassert acc != w\nacc = set()\n"
    ),
}


@pytest.mark.parametrize("name", sorted(MIX_1093_NEVER_TRUE))
def test_a_name_bound_to_a_set_anywhere_does_not_track_it_everywhere(name):
    result = determinism_census(MIX_1093_NEVER_TRUE[name])
    assert result.ok is False, (name, result.reason)


def test_a_genuinely_all_set_alias_still_stays_safe():
    """Safe mirror for part 2: EVERY binding of `acc` is a set-source (the
    fixpoint's alias case), so it must still be tracked and stay certified."""
    source = _S + "acc = set()\nacc |= s\nassert acc == set({'apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape'})\n"
    result = determinism_census(source)
    assert result.ok is True, result.reason


def test_a_set_stored_through_a_subscript_target_is_refused_not_ignored():
    """Regression for a fix-1 bug found while building part 2: a target this
    tracker does not bind a name for (a subscript) must still refuse when a
    set is written through it, not silently treat it as irrelevant."""
    source = _S + "acc = ['x']\nacc[:] = s\nassert acc[0] != 'banana'\n"
    result = determinism_census(source)
    assert result.ok is False, result.reason


def test_a_non_set_value_through_a_subscript_target_is_unaffected():
    """The same fix must not make an ordinary, unrelated subscript write (no
    set anywhere near it) refuse -- this regressed once while fixing the case
    above (tests/test_address_provenance.py's Grid.replace fixture)."""
    source = """
        class Box:
            def __init__(self):
                self.items = [[1, 2], [3, 4]]
            def replace(self, value):
                self.items[0] = [value]
        b = Box()
        b.replace(7)
        print(str(b.items[0][0]) * 2)
    """
    result = determinism_census(dedent(source))
    assert result.ok is True, result.reason


# -- [internal ref] part 3 (MG, rev/1093/p2_1093.py + audit) --------------------------------
# Part 2's greatest fixpoint only sees Assign/AnnAssign/NamedExpr targets. Two more
# real binding forms slipped past it entirely, each a wrong TRUE (component=
# deterministic) on 60556ad and on sv_2027:
#   (a) a function/lambda PARAMETER is not a binding the fixpoint records at all --
#       a same-named parameter can hold a list on every call while the tracker,
#       having seen only the module-level `acc = set()`, keeps `acc` tracked;
#   (b) a class-body `items = []` was folded into set_names (the bare name) but
#       NOT set_attrs (what `c.items` actually reads), so an unrelated
#       `other.items = set()` elsewhere wrongly certified `c.items` as a set.
# The audit for OTHER missed binding forms (import-as, global/nonlocal, del,
# match-case captures, starred targets, setattr()/__dict__) found two more live
# holes, confirmed a wrong TRUE on 60556ad by diffing against the SAME source with
# only the part-3 fix reverted, and confirmed each repro raises AssertionError
# natively under PYTHONHASHSEED=3 (verified separately, not re-checked on every
# CI run here -- these tests check the CENSUS verdict, matching this file's
# existing convention for the sink map above):
#   (c) a `match` capture (`case x:`) binds a name to the WHOLE subject value with
#       no Assign node for the fixpoint to see -- if that same bare name is also
#       assigned a set elsewhere, the capture's actual (possibly list) value could
#       be compared as if it were a set;
#   (d) `setattr(obj, "literal", value)` sets an attribute with no Attribute node
#       for the target-walk to see -- a non-set `value` here must still veto that
#       attribute name exactly as a `.attr = value` write would.
_PARAM_S = 's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
_PARAM_W = 'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'

# Verbatim from [host]:[local path] (MG's part-3 repros).
PARAM_AND_ATTR_NEVER_TRUE = {
    "param_shadows_set": (
        _PARAM_S + "acc = set()\ndef f(acc):\n    acc += s\n    return acc != w\nassert f([])\n"
    ),
    "kwonly_param": (
        _PARAM_S + "acc = set()\ndef f(*, acc):\n    acc += s\n    return acc != w\n"
        "assert f(acc=[])\n"
    ),
    "default_param": (
        _PARAM_S + "acc = set()\ndef f(acc=[]):\n    acc += s\n    return acc != w\nassert f()\n"
    ),
    "vararg_name": (
        _PARAM_S + "acc = set()\ndef f(*acc):\n    x = list(acc)\n    return x\n"
        "acc2 = []\nacc2 += s\nassert acc2 != w\n"
    ),
    "classbody_attr": (
        _PARAM_S + "class C:\n    items = []\nc = C()\nother = C()\nother.items = set()\n"
        "c.items += s\nassert c.items != w\n"
    ),
    "classbody_attr2": (
        _PARAM_S + "class C:\n    items = []\n    def add(self):\n        self.items += s\n"
        "c = C()\nd = C()\nd.items = set()\nc.add()\nassert c.items != w\n"
    ),
}
PARAM_AND_ATTR_NEVER_TRUE = {k: _PARAM_W + v for k, v in PARAM_AND_ATTR_NEVER_TRUE.items()}


@pytest.mark.parametrize("name", sorted(PARAM_AND_ATTR_NEVER_TRUE))
def test_a_parameter_or_classbody_attribute_is_never_wrongly_tracked(name):
    result = determinism_census(PARAM_AND_ATTR_NEVER_TRUE[name])
    assert result.ok is False, (name, result.reason)


def test_p2_1093_safe_mirror_still_certifies():
    """MG's `safe_mirror`: EVERY binding of `acc` is a genuine set, no parameter or
    class body involved -- must still certify TRUE."""
    source = _PARAM_S + "acc = set()\nacc |= s\nassert acc == set(w)\n"
    result = determinism_census(_PARAM_W + source)
    assert result.ok is True, result.reason


# -- [internal ref] part 3 audit: forms beyond MG's repros -----------------------------------
def test_match_case_whole_value_capture_is_never_wrongly_tracked():
    """`case x:` (no sub-pattern) binds `x` to the ENTIRE subject with no Assign
    node -- confirmed a wrong TRUE on 60556ad by reverting only this fix (the
    program's only other binding of `x` is `x = set()`, a real Assign, so the old
    fixpoint tracked it); natively raises AssertionError under PYTHONHASHSEED=3."""
    source = """
        s = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}
        w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]
        acc = []
        match acc:
            case x:
                pass
        x += s
        assert x != w
        x = set()
    """
    result = determinism_census(dedent(source))
    assert result.ok is False, result.reason


def test_setattr_with_a_literal_name_is_never_wrongly_tracked():
    """`setattr(obj, "items", value)` sets an attribute with no Attribute node for
    the target-walk to see -- confirmed a wrong TRUE on 60556ad by reverting only
    this fix (`c.items = set()` was the only VISIBLE binding of `items`, so the old
    fixpoint tracked it, silently missing the `setattr(..., "items", [])` binding);
    natively raises AssertionError under PYTHONHASHSEED=3."""
    source = """
        s = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}
        w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]
        class C:
            pass
        def mk():
            c = C()
            c.items = set()
            return c
        c = mk()
        setattr(c, "items", [])
        c.items += s
        acc = c.items
        assert acc != w
    """
    result = determinism_census(dedent(source))
    assert result.ok is False, result.reason


def test_classbody_set_attribute_with_no_conflicting_binding_stays_safe():
    """Safe mirror for (b): a class-body attribute that IS always a set (no
    conflicting list binding anywhere) must still certify -- the class-body-attr
    fix folds the class-body binding into set_attrs, it must not blanket-veto it."""
    source = """
        s = {"apple", "banana", "cherry"}
        class C:
            items = set()
        c = C()
        c.items |= s
        print(len(c.items))
    """
    result = determinism_census(dedent(source))
    assert result.ok is True, result.reason


def test_parameter_veto_does_not_leak_to_an_unrelated_tracked_name():
    """Safe mirror for (a): a parameter name that never collides with a genuinely
    all-set name elsewhere must not affect that name's own tracking."""
    source = """
        s = {"apple", "banana", "cherry"}
        def f(x):
            return len(x)
        acc = set()
        acc |= s
        f(3)
        print(len(acc))
    """
    result = determinism_census(dedent(source))
    assert result.ok is True, result.reason


def test_match_case_veto_does_not_leak_to_an_unrelated_tracked_name():
    """Safe mirror for (c): a match subject/capture unrelated to any tracked set
    name must not affect that name's own tracking."""
    source = """
        s = {"apple", "banana", "cherry"}
        acc = set()
        acc |= s
        n = 1
        match n:
            case 1:
                result = len(acc)
            case _:
                result = 0
        y = result + 1
    """
    result = determinism_census(dedent(source))
    assert result.ok is True, result.reason


# -- [internal ref] part 3 continued (MG, MG-blocks-PR-80): properties/__getattr__/hooks ----
# MG blocked PR 80 at bfc08e8 with three more verdict-level wrong TRUEs, one caused
# BY part 3's own class-body-attr registration, two pre-existing and already live on
# sv_2027:
#   - property_alias: a class-body Name `class D: items = set()` (folded into
#     set_attrs by part 3) was the ONLY visible binding of the attribute name
#     `items` -- an unrelated class C's `@property def items(self): return
#     self.store` (a LIST) hands out a value with no Assign node to see.
#   - property_store: the same shape, but the visible binding is a genuine
#     `d.items = set()` instance-attribute write (pre-existing since part 1,
#     not part 3's fault) -- a property elsewhere still returns a list.
#   - getattr_store: `d.items = set()` is again the only visible binding, but
#     the read (`C().items`) actually goes through `C.__getattr__`, which can
#     return anything for ANY attribute name, not just `items`.
# Fixed in `_Census.check_sets`: every class-body FunctionDef/AsyncFunctionDef/
# ClassDef (a method, including a @property -- the decorator does not change
# that it is still a FunctionDef named `items` in the class body -- or a nested
# class) vetoes ("attr", name); separately, any class defining __getattr__,
# __getattribute__, __get__ (read side) or __setattr__, __set__ (write side,
# audited the same way per MG's instruction) vetoes ALL attribute tracking
# globally, since none of these can be reasoned about per-name or per-class by
# this tracker. Verbatim from [host]:[local path]
PROPERTY_AND_HOOK_NEVER_TRUE = {
    "property_alias": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    items = set()\n"
        "class C:\n    def __init__(self):\n        self.store = []\n"
        "    @property\n    def items(self):\n        return self.store\n"
        "c = C()\nx = c.items\nx += s\nassert x != w\n"
    ),
    "property_store": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "class C:\n    def __init__(self):\n        self.store = []\n"
        "    @property\n    def items(self):\n        return self.store\n"
        "c = C()\nx = c.items\nx += s\nassert x != w\n"
    ),
    "getattr_store": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "class C:\n    def __getattr__(self, n):\n        return []\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
}


@pytest.mark.parametrize("name", sorted(PROPERTY_AND_HOOK_NEVER_TRUE))
def test_a_property_or_attr_hook_is_never_wrongly_tracked(name):
    result = determinism_census(PROPERTY_AND_HOOK_NEVER_TRUE[name])
    assert result.ok is False, (name, result.reason)


def test_unrelated_classbody_method_does_not_leak_to_an_unrelated_tracked_name():
    """Safe mirror: a class-body method whose name never collides with any
    tracked name/attribute must not affect that name's own tracking."""
    source = """
        s = {"apple", "banana", "cherry"}
        class C:
            def foo(self):
                return []
        acc = set()
        acc |= s
        c = C()
        y = c.foo()
        n = len(acc)
    """
    result = determinism_census(dedent(source))
    assert result.ok is True, result.reason


# -- [internal ref] part 4 (MG): a hook can be INSTALLED, not only DEFINED ------------------
# Part 3 fixed a class defining a hook by `def __getattr__(...)`. MG found a
# live wrong TRUE (component=deterministic) on sv_2027 4411f43: `class C:
# __getattr__ = some_func` (or `= lambda self, n: ...`) makes `some_func` the
# class's __getattr__ exactly as a `def` would, but it is an Assign, not a
# FunctionDef, so the part-3 def-only scan never saw it. Fixed by also
# recognizing a class-body Assign/AnnAssign/AugAssign (and tuple/list-unpacking
# target) whose target names a hook. `setattr(SomeClass, "__getattr__", f)` is
# also covered, though every construction tried already fails to resolve at
# the pyresolve layer regardless (setattr with a non-DUNDER_OK dunder literal
# name is refused as a "reflective attribute" before this analysis ever runs)
# -- defense-in-depth per MG's audit instruction, not exercised by any test
# here since it is provably unreachable today.
GETATTR_ASSIGN_NEVER_TRUE = {
    # Verbatim from [host]:[local path]
    "getattr_assign": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "class C:\n    __getattr__ = _ga\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    "getattr_lambda": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "class C:\n    __getattr__ = lambda self, n: []\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    # Audit-driven: a tuple-unpacking target that names a hook. Confirmed a
    # genuinely isolated wrong TRUE on 4411f43 (reverting only this fix
    # reproduces it; natively raises AssertionError under PYTHONHASHSEED=3).
    "getattr_tuple_unpack": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "class C:\n    __getattr__, _junk = _ga, None\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    # Audit-driven: an augmented-assignment target that names a hook (a
    # callable-list subclass, so `+=` mutates in place without crashing and
    # the hook stays callable). Natively raises AssertionError under
    # PYTHONHASHSEED=3. This construction is ALSO independently refused by
    # the address-provenance layer on the pre-fix code (an unrelated,
    # orthogonal block on this unusual a binding shape) -- kept as coverage
    # for the AugAssign-target detection itself, not as proof it is the only
    # thing standing between this program and a wrong TRUE.
    "getattr_augassign": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "class _Hook(list):\n    def __call__(self, n):\n        return []\n"
        "class C:\n    __getattr__ = _Hook()\n    __getattr__ += []\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
}


@pytest.mark.parametrize("name", sorted(GETATTR_ASSIGN_NEVER_TRUE))
def test_a_hook_installed_by_assignment_is_never_wrongly_tracked(name):
    result = determinism_census(GETATTR_ASSIGN_NEVER_TRUE[name])
    assert result.ok is False, (name, result.reason)


def test_unrelated_classbody_tuple_and_augassign_do_not_leak():
    """Safe mirror: a class-body tuple-unpacking target and an augmented
    assignment whose names never collide with a hook name or a tracked
    name/attribute must not affect that name's own tracking."""
    source = """
        s = {"apple", "banana", "cherry"}
        class C:
            foo, bar = 1, 2
            baz = 3
            baz += 0
        acc = set()
        acc |= s
        c = C()
        n = len(acc) + c.foo + c.bar + c.baz
    """
    result = determinism_census(dedent(source))
    assert result.ok is True, result.reason


# -- [internal ref] part 5 (MG): stop the whack-a-mole -- close the CLASS of bug -----------
# Part 4 fixed the assignment/lambda forms of installing a hook, then MG found
# MORE forms it missed: a class-body `for __getattr__ in (...):`, a `match`
# capture, and dynamic class creation (`type(name, bases, ns)`). Fixing each
# one by one is whack-a-mole. Fixed generally in `_Census.check_sets`:
#   - `_hook_scope_names` enumerates every AST shape Python uses to bind a
#     name (an `ast.Name` with Store/Del ctx -- which covers plain/annotated/
#     augmented assign, a walrus, a for-loop or comprehension target, a
#     with-item, a del target and a `type X = ...` alias, unpacked to any
#     depth, in ONE check, since CPython gives every nested name in an
#     unpacking its own Store/Del ctx -- plus the four forms Python
#     represents as a plain str instead of a Name node: import/import-as,
#     except-as, a match capture, and def/class), checked for every node in
#     class-body scope;
#   - 3-argument `type(name, bases, ns)`, a `metaclass=` keyword, and
#     setattr/delattr on a class (not only with a hook-name literal) now all
#     set the SAME `hook_installed` flag, since none of them can be reasoned
#     about as a specific name binding at all.
# Verbatim from [host]:[local path]
# and rev/1093/p5_1093.py.
HOOK_DYNAMIC_NEVER_TRUE = {
    "for_target": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "class C:\n    for __getattr__ in (_ga,):\n        pass\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    "match_capture": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "class C:\n    match _ga:\n        case __getattr__:\n            pass\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    "type3": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "C = type('C', (), {'__getattr__': _ga})\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    "type3_kw": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "ns = {}\nns['__getattr__'] = _ga\nC = type('C', (), ns)\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
}


@pytest.mark.parametrize("name", sorted(HOOK_DYNAMIC_NEVER_TRUE))
def test_a_dynamically_bound_or_created_hook_is_never_wrongly_tracked(name):
    result = determinism_census(HOOK_DYNAMIC_NEVER_TRUE[name])
    assert result.ok is False, (name, result.reason)


def test_unrelated_classbody_for_and_match_targets_do_not_leak():
    """Safe mirror: a class-body for-loop target and match capture whose
    names never collide with a hook name or a tracked name/attribute must
    not affect that name's own tracking (and must not themselves be refused
    -- they are ordinary, unrelated bindings)."""
    for_source = """
        s = {"apple", "banana", "cherry"}
        class C:
            _junk = 0
            for _junk in (1, 2, 3):
                pass
        acc = set()
        acc |= s
        c = C()
        n = len(acc) + c._junk
    """
    match_source = """
        s = {"apple", "banana", "cherry"}
        class C:
            _junk = 0
            match 1:
                case _junk:
                    pass
        acc = set()
        acc |= s
        c = C()
        n = len(acc) + c._junk
    """
    for source in (for_source, match_source):
        result = determinism_census(dedent(source))
        assert result.ok is True, result.reason


# -- [internal ref] part 6 (MG): builtin `type` reached INDIRECTLY -------------------------
# Part 5's 3-arg `type()` check only caught the call written out literally
# (`node.func.id == "type" and len(args) == 3`). MG found the builtin `type`
# reached INDIRECTLY still creates a hooked class -- an alias (`mk = type`),
# a container lookup (`[type][0]`), a lambda wrapper splatting *args, a
# default-parameter value, or `type(A)(...)` (the safe 1-arg `type(A)` then
# immediately called). Fixed as a VALUE-FLOW rule: the ONLY safe use of the
# name `type` is as the direct callee of a call with exactly one positional
# argument and no `*args`; any other appearance sets `hook_installed`, and so
# does a call whose callee is itself a call to `type(...)`. Verbatim from
# [host]:[local path]
INDIRECT_TYPE_NEVER_TRUE = {
    "alias": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "mk = type\nC = mk('C', (), {'__getattr__': _ga})\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    "subscript": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "C = [type][0]('C', (), {'__getattr__': _ga})\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    "lambda_star": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "mk = lambda *a: type(*a)\nC = mk('C', (), {'__getattr__': _ga})\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    "default_param": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "def mk(n, b, ns, t=type):\n    return t(n, b, ns)\n"
        "C = mk('C', (), {'__getattr__': _ga})\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    "type_of_obj_then_call": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "class A:\n    pass\n"
        "C = type(A)('C', (), {'__getattr__': _ga})\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
}


@pytest.mark.parametrize("name", sorted(INDIRECT_TYPE_NEVER_TRUE))
def test_type_reached_indirectly_is_never_wrongly_tracked(name):
    result = determinism_census(INDIRECT_TYPE_NEVER_TRUE[name])
    assert result.ok is False, (name, result.reason)


def test_type_used_as_the_ordinary_get_type_idiom_does_not_leak():
    """Safe mirror: `type(x)` as the ordinary direct, single-argument
    "get an object's type" call must not affect an unrelated tracked set's
    own tracking -- only OTHER uses of the name `type` are unsafe."""
    source = """
        s = {"apple", "banana", "cherry"}
        acc = set()
        acc |= s
        if type(1) is int:
            pass
        n = len(acc)
    """
    result = determinism_census(dedent(source))
    assert result.ok is True, result.reason


# -- [internal ref] part 7 (MG): an ALLOWLIST, not another denylist entry ------------------
# Parts 5 and 6 kept adding one more unsafe SHAPE of `type` (a literal 3-arg
# call, an alias, a splat, a saved-and-later-called result...) and MG kept
# finding the next one: the RESULT of a legal 1-argument `type(x)` is a
# metaclass when `x` is a class, so `M = type(A); M('C', (), {...})` builds a
# class exactly like `type(...)` does, with no further mention of the name
# `type` at all. Fixed as an ALLOWLIST in `_Census.check_sets`: class
# machinery is safe ONLY in a small, explicit list of shapes -- `type(x)`
# (exactly one positional argument, no keywords, no `*args`) used DIRECTLY as
# an operand of is/is not/==/!=/in/not in, a direct argument to isinstance()/
# issubclass(), or the base of a `.__name__` attribute access; and a `class`
# statement with no `metaclass=` keyword. ANY other occurrence of `type` or
# its result vetoes tracking, with no more enumerating escape routes one at a
# time. Verbatim from [host]:[local path]
# {hook_type_of_saved,p8_1093}.py (p8_1093.py's class_attr case was already
# correctly refused, at the resolution layer, before part 7).
TYPE_RESULT_ESCAPE_NEVER_TRUE = {
    "type_of_saved": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "class A:\n    pass\nM = type(A)\nC = M('C', (), {'__getattr__': _ga})\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    "type_of_list": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "class A:\n    pass\nms = [type(A)]\nC = ms[0]('C', (), {'__getattr__': _ga})\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
    "type_of_int": (
        's = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"}\n'
        'w = ["banana", "fig", "grape", "elderberry", "date", "apple", "cherry"]\n'
        "class D:\n    pass\nd = D()\nd.items = set()\n"
        "def _ga(self, n):\n    return []\n"
        "M = type(type(1))\nC = M('C', (), {'__getattr__': _ga})\n"
        "x = C().items\nx += s\nassert x != w\n"
    ),
}


@pytest.mark.parametrize("name", sorted(TYPE_RESULT_ESCAPE_NEVER_TRUE))
def test_a_saved_type_result_is_never_wrongly_tracked(name):
    result = determinism_census(TYPE_RESULT_ESCAPE_NEVER_TRUE[name])
    assert result.ok is False, (name, result.reason)


def test_type_allowlist_positions_do_not_leak():
    """Safe mirrors: each explicitly-safe `type(x)` position (is/==/in
    comparison, a direct isinstance()/issubclass() argument) and a plain
    class statement with no metaclass= must not affect an unrelated tracked
    set's own tracking. (`.__name__` is the allowlist's third safe position,
    but no runnable safe mirror exists for it: `.__name__` is not resolvable
    on ANY expression in this codebase today, confirmed unrelated to this
    fix by reproducing the same "attribute __name__ resolves to nothing
    known" failure on the pre-part-7 code -- see [internal ref] notes.)"""
    sources = [
        """
            s = {"apple", "banana", "cherry"}
            acc = set()
            acc |= s
            if type(1) is int:
                pass
            n = len(acc)
        """,
        """
            s = {"apple", "banana", "cherry"}
            acc = set()
            acc |= s
            class A:
                pass
            if type(A()) == A:
                pass
            n = len(acc)
        """,
        """
            s = {"apple", "banana", "cherry"}
            acc = set()
            acc |= s
            if type(1) in (int, float):
                pass
            n = len(acc)
        """,
        """
            s = {"apple", "banana", "cherry"}
            acc = set()
            acc |= s
            class A:
                pass
            if isinstance(A(), type(A())):
                pass
            n = len(acc)
        """,
        """
            s = {"apple", "banana", "cherry"}
            acc = set()
            acc |= s
            class A:
                pass
            if issubclass(A, type(A())):
                pass
            n = len(acc)
        """,
        """
            s = {"apple", "banana", "cherry"}
            class C:
                pass
            acc = set()
            acc |= s
            c = C()
            n = len(acc)
        """,
    ]
    for source in sources:
        result = determinism_census(dedent(source))
        assert result.ok is True, result.reason
