"""The catalog of protocol-level exception swallowing in CPython 3.14.4 ([internal ref] batch 5, [internal ref]).

CPython's C code and bytecode silently swallow, redirect, or convert exceptions inside a dozen
protocols: a length hint ignores a ``TypeError`` from ``__len__`` (PEP 424), a sequence-protocol
iteration ends on ``IndexError``, ``hasattr`` swallows ``AttributeError``, ``__exit__`` returning
true swallows anything, a ``NotImplemented`` return is not an error until both operands have
returned it, and so on.  Every one of these is a place where a model that treats "an exception
raised in a callee ends the path" (or the opposite) gives a WRONG TRUE: the violation is reachable
only because CPython swallows, or is reachable only because CPython does NOT.

Each :class:`Case` is a small program with an exact, pinned native outcome on 3.14.4:

* ``raises`` is the exact class of the exception that ends the run, or ``""`` for a clean exit;
* ``violates`` says whether that exception is one of ``classes`` (the SV-COMP property).  A TRUE
  verdict on a violating case is a soundness bug; a TRUE on a non-violating case is correct (and a
  refusal is merely incomplete).

``tests/contracts/test_swallowing_native.py`` pins every ``raises`` against the interpreter;
``tests/contracts/test_swallowing_engines.py`` runs Tharkun's engines over the catalog.  The
catalog is data only: no engine is imported here.

``mechanism`` names the rule; ``cpython`` cites the 3.14.4 source (``file::function``, never a
line number); ``vector`` is the draw vector of the run (``_sv_verifier.nondet_int()`` /
``nondet_bool()`` consume it in call order, then default to 0 / False).
"""

from __future__ import annotations

import builtins
import textwrap
from dataclasses import dataclass

from tests.contracts._shape import benchmark_shape

AR = ("ZeroDivisionError", "FloatingPointError")
DL = ("KeyError", "IndexError")
DT = ("TypeError", "AttributeError")
AS = ("AssertionError",)


@dataclass(frozen=True)
class Case:
    id: str
    group: str
    mechanism: str
    cpython: str
    classes: tuple
    src: str
    raises: str                      # exact native outcome: the uncaught class name, "" when clean
    vector: tuple = ()
    note: str = ""

    @property
    def violates(self) -> bool:
        if not self.raises:
            return False
        exc = getattr(builtins, self.raises)
        return any(issubclass(exc, getattr(builtins, c)) for c in self.classes)


def _s(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


#: a draw ``d`` that is False in the violating run; ``TAIL`` is a violation reachable only AFTER the
#: protocol under test, so a model that wrongly ends the path there never visits it.  The draw is a
#: ``nondet_bool`` so that every engine accepts the program's shape (``proof:stationary`` takes only
#: bool draws and trip counts)
HEAD = "import _sv_verifier\nd = _sv_verifier.nondet_bool()\n"
TAIL = "print(1 // d)\n"

CASES: list[Case] = []


def case(id, group, mechanism, cpython, classes, src, raises, vector=(), note=""):
    CASES.append(Case(id, group, mechanism, cpython, tuple(classes), benchmark_shape(_s(src)), raises,
                      tuple(vector), note))


# ======================================================================================
# LH: the length hint (PyObject_LengthHint): TypeError is swallowed, nothing else
# ======================================================================================
_LH = "Objects/abstract.c::PyObject_LengthHint"
_ITEMS = """
    class L:
        def __init__(self):
            self.items = [1, 2]
        def __iter__(self):
            return iter(self.items)
"""


def _lh(len_body: str, use: str, head=HEAD, tail=TAIL, extra=""):
    return _s(head) + extra + _s(_ITEMS) + textwrap.indent(_s(len_body), "    ") + _s(use) + tail


_LEN_TYPEERROR = "def __len__(self):\n    raise TypeError('no length')\n"
_LEN_VALUEERROR = "def __len__(self):\n    raise ValueError('boom')\n"
_LEN_ATTRERROR = "def __len__(self):\n    raise AttributeError('boom')\n"
_LEN_KEYERROR = "def __len__(self):\n    raise KeyError('boom')\n"

case("lh_list_typeerror_swallowed", "length_hint",
     "list() probes __len__; a TypeError from it means 'no hint' and iteration proceeds",
     _LH, AR, _lh(_LEN_TYPEERROR, "xs = list(L())\n"), "ZeroDivisionError", (False,))
case("lh_extend_typeerror_swallowed", "length_hint",
     "list.extend() probes __len__ the same way",
     _LH, AR, _lh(_LEN_TYPEERROR, "xs = [0]\nxs.extend(L())\n"), "ZeroDivisionError", (False,))
case("lh_sorted_typeerror_swallowed", "length_hint",
     "sorted() builds a list from the iterable, with the same probe",
     _LH, AR, _lh(_LEN_TYPEERROR, "xs = sorted(L())\n"), "ZeroDivisionError", (False,))
case("lh_iadd_typeerror_swallowed", "length_hint",
     "list += iterable is list.extend",
     _LH, AR, _lh(_LEN_TYPEERROR, "xs = [0]\nxs += L()\n"), "ZeroDivisionError", (False,))
case("lh_starred_list_typeerror_swallowed", "length_hint",
     "[*x] is LIST_EXTEND: the same probe",
     _LH, AR, _lh(_LEN_TYPEERROR, "xs = [*L()]\n"), "ZeroDivisionError", (False,))
case("lh_attributeerror_propagates", "length_hint",
     "an AttributeError from __len__ is NOT swallowed (only TypeError is): it ends the run, so the "
     "division after it is unreachable",
     _LH, AR, _lh(_LEN_ATTRERROR, "xs = list(L())\n"), "AttributeError", (False,))
case("lh_attributeerror_propagates_dt", "length_hint",
     "the same program under the dynamic-typing property: an uncaught AttributeError is a violation",
     _LH, DT, _lh(_LEN_ATTRERROR, "xs = list(L())\n", head="", tail=""), "AttributeError")
case("lh_valueerror_propagates", "length_hint",
     "a ValueError from __len__ propagates",
     _LH, AR, _lh(_LEN_VALUEERROR, "xs = list(L())\n"), "ValueError", (False,))
case("lh_keyerror_propagates_dl", "length_hint",
     "a KeyError from __len__ propagates (a data-lookup violation)",
     _LH, DL, _lh(_LEN_KEYERROR, "xs = list(L())\n", head="", tail=""), "KeyError")
case("lh_typeerror_swallowed_dt", "length_hint",
     "under dynamic-typing the swallowed TypeError is NOT a violation: the program is clean",
     _LH, DT, _lh(_LEN_TYPEERROR, "xs = list(L())\n", head="", tail=""), "")
case("lh_len_returns_none", "length_hint",
     "__len__ returning a non-int makes PyObject_Size raise TypeError, which the hint swallows",
     _LH, AR, _lh("def __len__(self):\n    return None\n", "xs = list(L())\n"), "ZeroDivisionError", (False,))
case("lh_len_returns_float", "length_hint",
     "a float __len__ result is the same TypeError, swallowed",
     _LH, AR, _lh("def __len__(self):\n    return 1.5\n", "xs = list(L())\n"), "ZeroDivisionError", (False,))
case("lh_len_wrong_arity", "length_hint",
     "a __len__ the probe cannot call (missing argument) raises TypeError: swallowed",
     _LH, AR, _lh("def __len__(self, extra):\n    return 1\n", "xs = list(L())\n"), "ZeroDivisionError", (False,))
case("lh_len_native_typeerror", "length_hint",
     "a TypeError raised by native code inside __len__ (len(5)) is swallowed too",
     _LH, AR, _lh("def __len__(self):\n    return len(5)\n", "xs = list(L())\n"), "ZeroDivisionError", (False,))
case("lh_len_negative_valueerror", "length_hint",
     "a negative __len__ is ValueError from PyObject_Size: it propagates",
     _LH, AR, _lh("def __len__(self):\n    return -1\n", "xs = list(L())\n"), "ValueError", (False,))
case("lh_len_too_big_overflowerror", "length_hint",
     "__len__ beyond sys.maxsize is OverflowError: it propagates",
     _LH, AR, _lh("def __len__(self):\n    return 2 ** 70\n", "xs = list(L())\n"), "OverflowError", (False,))
case("lh_len_indexerror_native", "length_hint",
     "an IndexError raised by native code inside __len__ propagates (data-lookup violation)",
     _LH, DL, _lh("def __len__(self):\n    return self.items[5]\n", "xs = list(L())\n", head="", tail=""),
     "IndexError")
case("lh_conditional_mutation", "length_hint",
     "the swallowing path keeps the side effects __len__ made before it raised TypeError",
     _LH, AR, _s("""
        import _sv_verifier
        g = 1
        class L:
            def __iter__(self):
                return iter([1])
            def __len__(self):
                global g
                if _sv_verifier.nondet_bool():
                    g = 0
                    raise TypeError("no length")
                return 1
        xs = list(L())
        print(1 // g)
     """), "ZeroDivisionError", (True,))
case("lh_unconditional_mutation", "length_hint",
     "the same with a raise on every path: the zero written to the global survives the swallow",
     _LH, AR, _s("""
        g = 1
        class L:
            def __iter__(self):
                return iter([1])
            def __len__(self):
                global g
                g = 0
                raise TypeError("no length")
        xs = list(L())
        print(1 // g)
     """), "ZeroDivisionError")
case("lh_conditional_mutation_attribute", "length_hint",
     "the same through an instance attribute",
     _LH, AR, _s("""
        import _sv_verifier
        class L:
            def __init__(self):
                self.k = 1
            def __iter__(self):
                return iter([1])
            def __len__(self):
                if _sv_verifier.nondet_bool():
                    self.k = 0
                    raise TypeError("no length")
                return 1
        l = L()
        xs = list(l)
        print(1 // l.k)
     """), "ZeroDivisionError", (True,))
case("lh_extend_conditional_mutation", "length_hint",
     "list.extend with a conditionally raising, mutating __len__",
     _LH, AR, _s("""
        import _sv_verifier
        g = 1
        class L:
            def __iter__(self):
                return iter([1])
            def __len__(self):
                global g
                if _sv_verifier.nondet_bool():
                    g = 0
                    raise TypeError("no length")
                return 1
        xs = []
        xs.extend(L())
        print(1 // g)
     """), "ZeroDivisionError", (True,))
case("lh_iter_runs_before_len", "length_hint",
     "__iter__ runs before the probe: its KeyError wins over a __len__ that would raise IndexError",
     _LH, DL, _s("""
        class L:
            def __iter__(self):
                raise KeyError("iter first")
            def __len__(self):
                raise IndexError("len")
        try:
            list(L())
        except IndexError:
            pass
     """), "KeyError")
case("lh_length_hint_typeerror_swallowed", "length_hint",
     "__length_hint__ (no __len__) raising TypeError is swallowed as well",
     _LH, AR, _s(HEAD) + _s("""
        class L:
            def __iter__(self):
                return iter([1, 2])
            def __length_hint__(self):
                raise TypeError("no hint")
        xs = list(L())
     """) + TAIL, "ZeroDivisionError", (False,))
case("lh_length_hint_notimplemented", "length_hint",
     "__length_hint__ returning NotImplemented is 'no hint': list() succeeds",
     _LH, AR, _s(HEAD) + _s("""
        class L:
            def __iter__(self):
                return iter([1, 2])
            def __length_hint__(self):
                return NotImplemented
        xs = list(L())
     """) + TAIL, "ZeroDivisionError", (False,))
case("lh_length_hint_negative", "length_hint",
     "a negative __length_hint__ is ValueError",
     _LH, AR, _s(HEAD) + _s("""
        class L:
            def __iter__(self):
                return iter([1, 2])
            def __length_hint__(self):
                return -1
        xs = list(L())
     """) + TAIL, "ValueError", (False,))
case("lh_length_hint_not_int", "length_hint",
     "a non-int __length_hint__ is TypeError raised by the hint itself: NOT swallowed",
     _LH, DT, _s("""
        class L:
            def __iter__(self):
                return iter([1, 2])
            def __length_hint__(self):
                return "two"
        xs = list(L())
     """), "TypeError")
case("lh_tuple_has_no_probe", "length_hint",
     "tuple() does not probe the length: a hostile __len__ is never called, so the division runs",
     _LH, AR, _lh(_LEN_VALUEERROR, "xs = tuple(L())\n"), "ZeroDivisionError", (False,))
case("lh_set_has_no_probe", "length_hint",
     "set() does not probe the length",
     _LH, AR, _lh(_LEN_VALUEERROR, "xs = set(L())\n"), "ZeroDivisionError", (False,))
case("lh_array_has_no_probe", "length_hint",
     "array.array(typecode, iterable) does not probe the length",
     _LH, AR, _lh(_LEN_VALUEERROR, "import array\nxs = array.array('i', L())\n"), "ZeroDivisionError", (False,))
case("lh_array_extend_has_no_probe", "length_hint",
     "array.extend(iterable) does not probe the length",
     _LH, AR, _lh(_LEN_VALUEERROR, "import array\nxs = array.array('i')\nxs.extend(L())\n"),
     "ZeroDivisionError", (False,))
case("lh_for_has_no_probe", "length_hint",
     "a for loop never probes the length",
     _LH, AR, _lh(_LEN_VALUEERROR, "for v in L():\n    pass\n"), "ZeroDivisionError", (False,))
case("lh_slice_assign_has_no_probe", "length_hint",
     "xs[:] = iterable probes the ITERATOR, not the original object: no __len__ call",
     _LH, AR, _lh(_LEN_VALUEERROR, "xs = [0]\nxs[:] = L()\n"), "ZeroDivisionError", (False,))
case("lh_bytes_probe", "length_hint",
     "bytes(iterable) probes (the hint propagates a KeyError)",
     _LH, DL, _s("""
        class L:
            def __iter__(self):
                return iter([1, 2])
            def __len__(self):
                raise KeyError("boom")
        xs = bytes(L())
     """), "KeyError")
case("lh_bytearray_ctor_has_no_probe", "length_hint",
     "bytearray(iterable) does not probe, bytearray.extend does",
     _LH, DL, _s("""
        class L:
            def __iter__(self):
                return iter([1, 2])
            def __len__(self):
                raise KeyError("boom")
        xs = bytearray(L())
     """), "")
case("lh_bytearray_extend_probe", "length_hint",
     "bytearray.extend(iterable) probes",
     _LH, DL, _s("""
        class L:
            def __iter__(self):
                return iter([1, 2])
            def __len__(self):
                raise KeyError("boom")
        xs = bytearray()
        xs.extend(L())
     """), "KeyError")
case("lh_join_probes_the_iterator_not_the_object", "length_hint",
     "str.join(x) builds its list from iter(x): the hint is asked of the iterator, never of x",
     _LH, DL, _s("""
        class L:
            def __iter__(self):
                return iter(["a", "b"])
            def __len__(self):
                raise KeyError("boom")
        xs = ",".join(L())
     """), "")
case("lh_operator_length_hint", "length_hint",
     "operator.length_hint is PyObject_LengthHint itself",
     _LH, DL, _s("""
        import operator
        class L:
            def __len__(self):
                raise KeyError("boom")
        operator.length_hint(L())
     """), "KeyError")
case("lh_len_is_not_swallowed", "length_hint",
     "len() itself is not a hint: its TypeError propagates (dynamic-typing violation)",
     _LH, DT, _s("""
        class L:
            def __len__(self):
                raise TypeError("no length")
        len(L())
     """), "TypeError")
case("lh_bool_does_not_swallow_typeerror", "length_hint",
     "truthiness calls __len__ without the swallow: the TypeError propagates",
     _LH, DT, _s("""
        class L:
            def __len__(self):
                raise TypeError("no length")
        if L():
            pass
     """), "TypeError")

# the extend() twins of the shapes list() refuses on: the same swallowed TypeError, through the
# consumer absint models for instances
for _cid, _body, _note in (
    ("lh_extend_len_wrong_arity", "def __len__(self, extra):\n    return 1\n",
     "a __len__ the probe cannot call (missing argument) raises TypeError: swallowed"),
    ("lh_extend_len_native_typeerror", "def __len__(self):\n    return len(5)\n",
     "a TypeError raised by native code inside __len__ is swallowed"),
    ("lh_extend_len_returns_none", "def __len__(self):\n    return None\n",
     "a non-int __len__ result is TypeError from PyObject_Size: swallowed"),
):
    case(_cid, "length_hint", _note, _LH, AR, _lh(_body, "xs = []\nxs.extend(L())\n"),
         "ZeroDivisionError", (False,))
case("lh_extend_attributeerror_propagates", "length_hint",
     "extend: an AttributeError from __len__ propagates (not swallowed)",
     _LH, AR, _lh(_LEN_ATTRERROR, "xs = []\nxs.extend(L())\n"), "AttributeError", (False,))
case("lh_extend_valueerror_propagates", "length_hint",
     "extend: a ValueError from __len__ propagates",
     _LH, AR, _lh(_LEN_VALUEERROR, "xs = []\nxs.extend(L())\n"), "ValueError", (False,))
case("lh_extend_conditional_typeerror_clean_after", "length_hint",
     "extend with a __len__ that may raise TypeError and mutates nothing: the swallow changes nothing "
     "observable, and the program is safe (a precise model proves it)",
     _LH, AR, _s("""
        import _sv_verifier
        class L:
            def __init__(self):
                self.items = [1, 2]
            def __iter__(self):
                return iter(self.items)
            def __len__(self):
                if _sv_verifier.nondet_bool():
                    raise TypeError("no length")
                return 2
        xs = []
        xs.extend(L())
        print(10 // (len(xs) + 1))
     """), "", (True,))


# ======================================================================================
# IT: iteration: StopIteration ends it, and so does IndexError from a sequence-protocol __getitem__
# ======================================================================================
_SEQ = """
    class Seq:
        def __getitem__(self, i):
            if i >= 3:
                raise {end}
            return i
"""


def _seq(use: str, end="IndexError('end')", head=HEAD, tail=TAIL):
    return _s(head) + _s(_SEQ.format(end=end)) + _s(use) + tail


_IT = "Objects/iterobject.c::iter_iternext"
for _cid, _use, _note in (
    ("it_getitem_for_ends", "for v in Seq():\n    pass\n", "a for loop over an object with only __getitem__"),
    ("it_getitem_list_ends", "xs = list(Seq())\n", "list() over a sequence-protocol object"),
    ("it_getitem_in_ends", "r = 7 in Seq()\n", "the 'in' operator falls back to the same iteration"),
    ("it_getitem_unpack_ends", "a, b, c = Seq()\n", "unpacking consumes the same iterator"),
    ("it_getitem_iter_ends", "it = iter(Seq())\nfor v in it:\n    pass\n", "iter() builds the sequence iterator"),
    ("it_getitem_sorted_ends", "xs = sorted(Seq())\n", "sorted()"),
    ("it_getitem_sum_ends", "s = sum(Seq())\n", "sum()"),
    ("it_getitem_tuple_ends", "s = tuple(Seq())\n", "tuple()"),
    ("it_getitem_extend_ends", "xs = []\nxs.extend(Seq())\n", "list.extend"),
    ("it_getitem_min_ends", "s = min(Seq())\n", "min()"),
    ("it_getitem_enumerate_ends", "for i, v in enumerate(Seq()):\n    pass\n", "enumerate()"),
    ("it_getitem_zip_ends", "for p in zip(Seq(), [0, 1, 2, 3, 4]):\n    pass\n", "zip()"),
):
    case(_cid, "iteration", f"{_note}: IndexError from __getitem__ ENDS the iteration silently", _IT, AR,
         _seq(_use), "ZeroDivisionError", (False,))
case("it_getitem_stopiteration_ends", "iteration",
     "StopIteration from a sequence-protocol __getitem__ ends it as well",
     _IT, AR, _seq("for v in Seq():\n    pass\n", end="StopIteration"), "ZeroDivisionError", (False,))
case("it_getitem_indexerror_subclass_ends", "iteration",
     "a subclass of IndexError matches too",
     _IT, AR, _s(HEAD) + _s("""
        class MyIndexError(IndexError):
            pass
        class Seq:
            def __getitem__(self, i):
                if i >= 3:
                    raise MyIndexError("end")
                return i
        for v in Seq():
            pass
     """) + TAIL, "ZeroDivisionError", (False,))
case("it_getitem_keyerror_propagates", "iteration",
     "a KeyError from __getitem__ propagates out of the iteration (data-lookup violation)",
     _IT, DL, _seq("for v in Seq():\n    pass\n", end="KeyError('k')", head="", tail=""), "KeyError")
case("it_getitem_indexerror_ends_clean_dl", "iteration",
     "under data-lookup the ending IndexError is NOT a violation: the program is clean",
     _IT, DL, _seq("for v in Seq():\n    pass\n", head="", tail=""), "")
case("it_iter_indexerror_propagates", "iteration",
     "an IndexError from __iter__ itself propagates (it is not the sequence protocol)",
     _IT, DL, _s("""
        class Seq:
            def __iter__(self):
                raise IndexError("from iter")
            def __getitem__(self, i):
                return i
        for v in Seq():
            pass
     """), "IndexError")
case("it_next_indexerror_propagates", "iteration",
     "an IndexError from __next__ propagates (only StopIteration ends an iterator protocol loop)",
     "Objects/typeobject.c::slot_tp_iternext", DL, _s("""
        class It:
            def __iter__(self):
                return self
            def __next__(self):
                raise IndexError("from next")
        for v in It():
            pass
     """), "IndexError")
case("it_next_stopiteration_ends", "iteration",
     "StopIteration from __next__ ends the loop; the code after it runs",
     "Python/bytecodes.c::FOR_ITER", AR, _s(HEAD) + _s("""
        class It:
            def __iter__(self):
                return self
            def __next__(self):
                raise StopIteration
        for v in It():
            pass
     """) + TAIL, "ZeroDivisionError", (False,))
case("it_next_stopiteration_ends_list", "iteration",
     "StopIteration from __next__ ends list() the same way",
     "Objects/listobject.c::list_extend_iter_lock_held", AR, _s(HEAD) + _s("""
        class It:
            def __iter__(self):
                return self
            def __next__(self):
                raise StopIteration
        xs = list(It())
     """) + TAIL, "ZeroDivisionError", (False,))
case("it_iter_none_typeerror", "iteration",
     "__iter__ = None blocks the sequence fallback: TypeError (dynamic-typing violation)",
     "Objects/abstract.c::PyObject_GetIter", DT, _s("""
        class Seq:
            __iter__ = None
            def __getitem__(self, i):
                raise IndexError("end")
        for v in Seq():
            pass
     """), "TypeError")
case("it_iter_returns_non_iterator", "iteration",
     "__iter__ returning a non-iterator is TypeError",
     "Objects/abstract.c::PyObject_GetIter", DT, _s("""
        class Seq:
            def __iter__(self):
                return 5
        for v in Seq():
            pass
     """), "TypeError")
case("it_next_default_swallows_stopiteration", "iteration",
     "next(it, default) swallows StopIteration",
     "Python/bltinmodule.c::builtin_next", AR, _s(HEAD) + _s("""
        x = next(iter([]), 5)
     """) + TAIL, "ZeroDivisionError", (False,))
case("it_next_default_propagates_keyerror", "iteration",
     "next(it, default) does NOT swallow anything but StopIteration",
     "Python/bltinmodule.c::builtin_next", DL, _s("""
        class It:
            def __iter__(self):
                return self
            def __next__(self):
                raise KeyError("n")
        x = next(It(), 0)
     """), "KeyError")
case("it_next_default_non_iterator_typeerror", "iteration",
     "next(obj, default) on a non-iterator is TypeError even with a default",
     "Python/bltinmodule.c::builtin_next", DT, _s("""
        class C:
            def __iter__(self):
                return iter([1])
        x = next(C(), 0)
     """), "TypeError")
case("it_callable_sentinel_stopiteration_ends", "iteration",
     "iter(callable, sentinel) ends when the callable raises StopIteration",
     "Objects/iterobject.c::calliter_iternext", AR, _s(HEAD) + _s("""
        n = [0]
        def f():
            n[0] += 1
            if n[0] > 2:
                raise StopIteration
            return n[0]
        xs = list(iter(f, None))
     """) + TAIL, "ZeroDivisionError", (False,))
case("it_callable_sentinel_keyerror_propagates", "iteration",
     "iter(callable, sentinel): any other exception propagates",
     "Objects/iterobject.c::calliter_iternext", DL, _s("""
        def f():
            raise KeyError("x")
        xs = list(iter(f, None))
     """), "KeyError")
case("it_map_stopiteration_ends", "iteration",
     "a StopIteration raised by map()'s function ends the map silently (the iteration protocol)",
     "Python/bltinmodule.c::map_next", AR, _s(HEAD) + _s("""
        def f(x):
            if x == 2:
                raise StopIteration
            return x
        xs = list(map(f, [1, 2, 3]))
     """) + TAIL, "ZeroDivisionError", (False,))
case("it_genexpr_first_iterable_eager", "iteration",
     "the first iterable of a generator expression is evaluated and iter()'d when the generator is "
     "CREATED: a TypeError there is not deferred to the consumer",
     "Python/bytecodes.c::GET_ITER", DT, _s("""
        g = (x for x in 5)
     """), "TypeError")
case("it_genexpr_body_lazy", "iteration",
     "the body of a generator expression runs only when consumed: never consumed, never raises",
     "Objects/genobject.c::gen_send_ex2", AR, _s("""
        g = (1 // x for x in [0])
     """), "")
case("it_genexpr_body_consumed", "iteration",
     "consumed, the body's ZeroDivisionError surfaces at the consumer",
     "Objects/genobject.c::gen_send_ex2", AR, _s("""
        g = (1 // x for x in [0])
        xs = list(g)
     """), "ZeroDivisionError")
case("it_map_lazy_never_consumed", "iteration",
     "map() applies its function only on consumption",
     "Python/bltinmodule.c::map_next", AR, _s("""
        m = map(lambda x: 1 // x, [0])
     """), "")
case("it_generator_stopiteration_runtimeerror", "iteration",
     "PEP 479: a StopIteration raised inside a generator body becomes RuntimeError",
     "Python/ceval.c::_PyEval_EvalFrameDefault (CALL_INTRINSIC_1 INTRINSIC_STOPITERATION_ERROR)", AR, _s("""
        def g():
            raise StopIteration
            yield 1
        xs = list(g())
     """), "RuntimeError")
case("it_reversed_getitem_indexerror_ends", "iteration",
     "reversed() over a sequence-protocol object ends on IndexError too",
     "Objects/enumobject.c::reversed_next", AR, _s(HEAD) + _s("""
        class Seq:
            def __len__(self):
                return 5
            def __getitem__(self, i):
                if i < 3:
                    raise IndexError("end")
                return i
        xs = list(reversed(Seq()))
     """) + TAIL, "ZeroDivisionError", (False,))
case("it_dict_ctor_keys_attributeerror_falls_back", "iteration",
     "dict(obj) asks for obj.keys, swallowing an AttributeError from that lookup and iterating pairs",
     "Objects/dictobject.c::dict_update_arg", AR, _s(HEAD) + _s("""
        class O:
            @property
            def keys(self):
                raise AttributeError("no keys")
            def __iter__(self):
                return iter([(1, 2)])
        x = dict(O())
     """) + TAIL, "ZeroDivisionError", (False,))
case("it_dict_ctor_keys_keyerror_propagates", "iteration",
     "any other exception from the keys lookup propagates",
     "Objects/dictobject.c::dict_update_arg", DL, _s("""
        class O:
            @property
            def keys(self):
                raise KeyError("k")
            def __iter__(self):
                return iter([(1, 2)])
        x = dict(O())
     """), "KeyError")
case("it_dict_star_attributeerror_becomes_typeerror", "iteration",
     "{**obj}: an AttributeError out of the mapping protocol is CONVERTED to TypeError ('not a mapping')",
     "Python/bytecodes.c::DICT_UPDATE", AR, _s("""
        class O:
            @property
            def keys(self):
                raise AttributeError("k")
        x = {**O()}
     """), "TypeError")


# ======================================================================================
# IN: the 'in' operator: __contains__, then __iter__, then the sequence protocol
# ======================================================================================
_IN = "Objects/abstract.c::PySequence_Contains, ::_PySequence_IterSearch"
case("in_fallback_iter", "contains",
     "no __contains__: 'in' iterates (no TypeError), and the code after it runs",
     _IN, AR, _s(HEAD) + _s("""
        class C:
            def __iter__(self):
                return iter([1, 2])
        r = 5 in C()
     """) + TAIL, "ZeroDivisionError", (False,))
case("in_fallback_getitem_ends_on_indexerror", "contains",
     "no __contains__, no __iter__: the sequence protocol, ended by IndexError (the answer is False)",
     _IN, AR, _s(HEAD) + _s("""
        class S:
            def __getitem__(self, i):
                if i >= 2:
                    raise IndexError("end")
                return i
        r = 7 in S()
     """) + TAIL, "ZeroDivisionError", (False,))
case("in_contains_wins_over_iter", "contains",
     "__contains__ is tried first: an __iter__ that would raise is never reached",
     _IN, DL, _s("""
        class C:
            def __contains__(self, x):
                return True
            def __iter__(self):
                raise KeyError("never")
        r = 5 in C()
     """), "")
case("in_contains_none_typeerror", "contains",
     "__contains__ = None blocks the fallbacks: TypeError",
     _IN, DT, _s("""
        class C:
            __contains__ = None
            def __iter__(self):
                return iter([1])
        r = 5 in C()
     """), "TypeError")
case("in_contains_indexerror_propagates", "contains",
     "an IndexError from __contains__ itself propagates: only the sequence protocol swallows it",
     _IN, DL, _s("""
        class C:
            def __contains__(self, x):
                raise IndexError("c")
        r = 5 in C()
     """), "IndexError")
case("in_iter_next_indexerror_propagates", "contains",
     "an IndexError from an iterator's __next__ propagates out of 'in'",
     _IN, DL, _s("""
        class It:
            def __iter__(self):
                return self
            def __next__(self):
                raise IndexError("n")
        class C:
            def __iter__(self):
                return It()
        r = 5 in C()
     """), "IndexError")
case("in_str_int_typeerror", "contains",
     "'in' on a str needs a str on the left: TypeError",
     "Objects/unicodeobject.c::PyUnicode_Contains", DT, _s("""
        r = 1 in "abc"
     """), "TypeError")
case("in_range_non_int_is_false", "contains",
     "'in' on a range with a non-int falls back to iteration and compares: no TypeError",
     "Objects/rangeobject.c::range_contains", DT, _s("""
        r = "a" in range(3)
        s = [] in range(3)
     """), "")
case("set_contains_set_key_converted", "contains",
     "set.__contains__ converts a set key to a frozenset when hashing raised TypeError",
     "Objects/setobject.c::set_contains_lock_held", DT, _s("""
        r = {1} in {frozenset({1})}
     """), "")
case("dict_contains_set_key_typeerror", "contains",
     "dict.__contains__ does NOT: an unhashable set key is TypeError",
     "Objects/dictobject.c::dict_contains", DT, _s("""
        r = {1} in {frozenset({1}): 1}
     """), "TypeError")
case("set_remove_set_key_converted", "contains",
     "set.remove / set.discard convert a set key too",
     "Objects/setobject.c::set_remove, ::set_discard", DL, _s("""
        s = {frozenset({1})}
        s.remove({1})
        s.discard({1})
     """), "")
case("set_subclass_hash_typeerror_converted", "contains",
     "the conversion keys on TypeError from the hash, even a user __hash__ on a set subclass",
     "Objects/setobject.c::set_contains_lock_held", DT, _s("""
        class S(set):
            def __hash__(self):
                raise TypeError("x")
        r = S({1}) in {frozenset({1})}
     """), "")
case("set_subclass_hash_keyerror_propagates", "contains",
     "but any other exception from the hash propagates",
     "Objects/setobject.c::set_contains_lock_held", DL, _s("""
        class S(set):
            def __hash__(self):
                raise KeyError("x")
        r = S({1}) in {frozenset({1})}
     """), "KeyError")


# ======================================================================================
# AT: attribute access: hasattr / getattr(default) / __getattr__ fallback / slots
# ======================================================================================
_AT = "Objects/object.c::PyObject_GetOptionalAttr"
_GA = "Objects/typeobject.c::slot_tp_getattr_hook"
case("hasattr_swallows_attributeerror", "attribute",
     "hasattr swallows an AttributeError from __getattr__: it answers False",
     _AT, DT, _s("""
        class O:
            def __getattr__(self, name):
                raise AttributeError(name)
        r = hasattr(O(), "x")
     """), "")
case("hasattr_propagates_keyerror", "attribute",
     "hasattr swallows ONLY AttributeError: a KeyError propagates",
     _AT, DL, _s("""
        class O:
            def __getattr__(self, name):
                raise KeyError(name)
        r = hasattr(O(), "x")
     """), "KeyError")
case("hasattr_propagates_typeerror", "attribute",
     "a TypeError from __getattr__ propagates through hasattr too",
     _AT, DT, _s("""
        class O:
            def __getattr__(self, name):
                raise TypeError(name)
        r = hasattr(O(), "x")
     """), "TypeError")
case("getattr_default_swallows_attributeerror", "attribute",
     "getattr(obj, name, default) returns the default on AttributeError",
     _AT, AR, _s(HEAD) + _s("""
        class O:
            pass
        x = getattr(O(), "missing", 0)
     """) + TAIL, "ZeroDivisionError", (False,))
case("getattr_default_propagates_keyerror", "attribute",
     "getattr's default does not cover other exceptions",
     _AT, DL, _s("""
        class O:
            def __getattr__(self, name):
                raise KeyError(name)
        x = getattr(O(), "missing", 0)
     """), "KeyError")
case("getattr_fallback_after_property_attributeerror", "attribute",
     "an AttributeError raised INSIDE a property getter makes CPython call __getattr__, whose "
     "value replaces the property's: the violation is reached only through that fallback",
     _GA, AR, _s(HEAD) + _s("""
        class O:
            @property
            def p(self):
                return self._q
            def __getattr__(self, name):
                return d
        x = 1 // O().p
     """), "ZeroDivisionError", (False,))
case("getattr_fallback_after_getattribute_attributeerror", "attribute",
     "an AttributeError from a user __getattribute__ also falls through to __getattr__",
     _GA, AR, _s(HEAD) + _s("""
        class O:
            def __getattribute__(self, name):
                raise AttributeError(name)
            def __getattr__(self, name):
                return d
        x = 1 // O().z
     """), "ZeroDivisionError", (False,))
case("getattribute_keyerror_propagates", "attribute",
     "any other exception from __getattribute__ propagates; __getattr__ is not consulted",
     _GA, DL, _s("""
        class O:
            def __getattribute__(self, name):
                raise KeyError(name)
            def __getattr__(self, name):
                return 7
        x = O().z
     """), "KeyError")
case("getattr_not_called_for_found_name", "attribute",
     "__getattr__ is only a fallback: a name that resolves normally never reaches it",
     _GA, DL, _s("""
        class O:
            y = 1
            def __getattr__(self, name):
                raise KeyError(name)
        x = O().y
     """), "")
case("slot_unset_attributeerror", "attribute",
     "reading a slot nothing was assigned to is AttributeError (and hasattr says False)",
     "Objects/descrobject.c::member_get", DT, _s("""
        class C:
            __slots__ = ("x",)
        r = hasattr(C(), "x")
        C().x
     """), "AttributeError")
case("slot_class_attribute_read", "attribute",
     "Cls.x = 5 replaces the slot descriptor: instances read the class value (#1072)",
     "Objects/typeobject.c::type_setattro", AR, _s("""
        class Cell:
            __slots__ = ("x",)
            def __init__(self):
                self.x = 0
        c = Cell()
        Cell.x = 5
        print(10 // (c.x - 5))
     """), "ZeroDivisionError")
case("slot_class_attribute_write", "attribute",
     "and the write to the vanished slot is AttributeError: attribute is read-only (#1072)",
     "Objects/object.c::_PyObject_GenericSetAttrWithDict", DT, _s("""
        class Cell:
            __slots__ = ("x",)
            def __init__(self):
                self.x = 0
        c = Cell()
        Cell.x = 5
        c.x = 1
     """), "AttributeError")
case("slot_class_variable_conflict", "attribute",
     "a class variable named like a slot, IN the class body, is ValueError (not a property class)",
     "Objects/typeobject.c::type_new_slots", DT, _s("""
        class C:
            __slots__ = ("x",)
            x = 5
     """), "ValueError")
case("delattr_missing_is_attributeerror", "attribute",
     "del obj.x for a missing x: the KeyError of the instance dict is CONVERTED to AttributeError",
     "Objects/object.c::_PyObject_GenericSetAttrWithDict", DL, _s("""
        class O:
            pass
        del O().x
     """), "AttributeError")
case("isinstance_class_attributeerror_swallowed", "attribute",
     "isinstance asks obj.__class__ and swallows an AttributeError from it: the answer is False",
     "Objects/abstract.c::object_isinstance", DT, _s("""
        class O:
            @property
            def __class__(self):
                raise AttributeError("c")
        r = isinstance(O(), int)
     """), "")
case("isinstance_class_keyerror_propagates", "attribute",
     "any other exception from obj.__class__ propagates",
     "Objects/abstract.c::object_isinstance", DL, _s("""
        class O:
            @property
            def __class__(self):
                raise KeyError("c")
        r = isinstance(O(), int)
     """), "KeyError")


# ======================================================================================
# WI: with / __exit__ / finally: swallowing by control flow
# ======================================================================================
_WI = "Python/bytecodes.c::WITH_EXCEPT_START"
_EXIT_TRUE = """
    class Sup:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return True
"""
case("with_exit_true_swallows", "with",
     "__exit__ returning true swallows ANY exception of the body: the code after the with runs",
     _WI, AR, _s(HEAD) + _s(_EXIT_TRUE) + _s("""
        with Sup():
            {}[1]
     """) + TAIL, "ZeroDivisionError", (False,))
case("with_exit_true_swallows_assertionerror", "with",
     "the same swallows an AssertionError: the program is clean for assertion-safety",
     _WI, AS, _s(_EXIT_TRUE) + _s("""
        with Sup():
            assert False
     """), "")
case("with_exit_true_swallows_zde", "with",
     "and a ZeroDivisionError: clean for arithmetic",
     _WI, AR, _s(_EXIT_TRUE) + _s("""
        with Sup():
            1 // 0
     """), "")
case("with_exit_false_propagates", "with",
     "__exit__ returning a false value re-raises",
     _WI, DL, _s("""
        class W:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
        with W():
            {}[1]
     """), "KeyError")
case("with_exit_truthy_object_swallows", "with",
     "any true value suppresses, not only True",
     _WI, DL, _s("""
        class W:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return [1]
        with W():
            {}[1]
     """), "")
case("with_exit_result_bool_raises", "with",
     "the truth test of __exit__'s result runs, and may raise (AttributeError here)",
     _WI, DT, _s("""
        class Bad:
            def __bool__(self):
                raise AttributeError("b")
        class W:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return Bad()
        with W():
            {}[1]
     """), "AttributeError")
case("with_exit_raises_replaces", "with",
     "an exception raised by __exit__ replaces the body's",
     _WI, DL, _s("""
        class W:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                raise IndexError("x")
        with W():
            {}[1]
     """), "IndexError")
case("with_exit_result_ignored_without_exception", "with",
     "without an exception the result is ignored",
     _WI, DL, _s("""
        class W:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return True
        with W():
            pass
        {}[1]
     """), "KeyError")
case("with_enter_raises_exit_not_called", "with",
     "when __enter__ raises, __exit__ is never called: nothing is swallowed",
     "Python/bytecodes.c::BEFORE_WITH", DL, _s("""
        class W:
            def __enter__(self):
                raise KeyError("e")
            def __exit__(self, *exc):
                return True
        with W():
            pass
     """), "KeyError")
case("with_missing_exit_typeerror", "with",
     "an object without __exit__ is a TypeError at the with statement",
     "Python/bytecodes.c::BEFORE_WITH", DT, _s("""
        class W:
            def __enter__(self):
                return self
        with W():
            pass
     """), "TypeError")
case("with_suppress", "with",
     "contextlib.suppress(KeyError) swallows a KeyError and only that",
     "Lib/contextlib.py::suppress.__exit__", AR, _s(HEAD) + _s("""
        import contextlib
        with contextlib.suppress(KeyError):
            {}[1]
     """) + TAIL, "ZeroDivisionError", (False,))
case("with_suppress_other_class_propagates", "with",
     "a different class passes through",
     "Lib/contextlib.py::suppress.__exit__", DL, _s("""
        import contextlib
        with contextlib.suppress(KeyError):
            [][1]
     """), "IndexError")
case("with_contextmanager_generator_swallows", "with",
     "a @contextmanager generator that catches the exception thrown in at its yield suppresses it",
     "Lib/contextlib.py::_GeneratorContextManager.__exit__", AR, _s(HEAD) + _s("""
        import contextlib
        @contextlib.contextmanager
        def cm():
            try:
                yield
            except KeyError:
                pass
        with cm():
            {}[1]
     """) + TAIL, "ZeroDivisionError", (False,))
case("finally_return_swallows", "with",
     "a return inside finally swallows the exception in flight",
     "Python/compile.c (finally block): POP_EXCEPT / RETURN", AR, _s(HEAD) + _s("""
        def f():
            try:
                {}[1]
            finally:
                return 0
        x = f()
     """) + TAIL, "ZeroDivisionError", (False,))
case("assert_tuple_is_always_true", "with",
     "assert (cond, msg) tests a non-empty TUPLE, which is true: it never fails",
     "Python/codegen.c::codegen_assert", AS, _s("""
        assert (False, "never checked")
     """), "")
case("assert_message_raises_replaces", "with",
     "a message that raises replaces the AssertionError (here with a ZeroDivisionError)",
     "Python/ceval.c: RAISE_VARARGS 1 after the message is built", AR, _s("""
        assert False, 1 // 0
     """), "ZeroDivisionError")
case("assert_message_not_evaluated_when_true", "with",
     "the message of a passing assert is never evaluated",
     "Python/codegen.c::codegen_assert", AR, _s("""
        assert True, 1 // 0
     """), "")
case("except_non_exception_class_lazy", "with",
     "an except clause naming a non-exception is a TypeError only when an exception is matched",
     "Python/ceval.c::_PyEval_CheckExceptTypeValid", DT, _s("""
        try:
            pass
        except 5:
            pass
     """), "")
case("except_non_exception_class_hit", "with",
     "and then it REPLACES the exception in flight with a TypeError",
     "Python/ceval.c::_PyEval_CheckExceptTypeValid", DT, _s("""
        try:
            {}[1]
        except 5:
            pass
     """), "TypeError")
case("finalizer_exception_is_unraisable", "with",
     "an exception raised by __del__ is reported and discarded: it never ends the run",
     "Objects/typeobject.c::slot_tp_finalize, Python/errors.c::_PyErr_WriteUnraisableMsg", AR, _s("""
        class D:
            def __del__(self):
                1 // 0
        D()
     """), "")
case("thread_exception_does_not_end_the_run", "with",
     "an exception that ends a thread is reported and discarded: the main thread is unaffected",
     "Lib/threading.py::Thread._bootstrap_inner", AR, _s("""
        import threading
        t = threading.Thread(target=lambda: 1 // 0)
        t.start()
        t.join()
     """), "")


# ======================================================================================
# TR: truthiness: __bool__, then __len__, then true; nothing is swallowed
# ======================================================================================
_TR = "Objects/object.c::PyObject_IsTrue"
case("truth_bool_false_takes_else", "truthiness",
     "__bool__ returning False sends control to the else branch",
     _TR, AR, _s(HEAD) + _s("""
        class B:
            def __bool__(self):
                return False
        if B():
            pass
        else:
            print(1 // d)
     """), "ZeroDivisionError", (False,))
case("truth_bool_wins_over_len", "truthiness",
     "__bool__ is consulted before __len__",
     _TR, AR, _s(HEAD) + _s("""
        class B:
            def __len__(self):
                return 0
            def __bool__(self):
                return True
        if B():
            print(1 // d)
     """), "ZeroDivisionError", (False,))
case("truth_len_zero_is_false", "truthiness",
     "without __bool__ a zero __len__ is false",
     _TR, AR, _s(HEAD) + _s("""
        class B:
            def __len__(self):
                return 0
        if not B():
            print(1 // d)
     """), "ZeroDivisionError", (False,))
case("truth_bool_non_bool_typeerror", "truthiness",
     "__bool__ must return a bool: an int is TypeError",
     _TR, DT, _s("""
        class B:
            def __bool__(self):
                return 1
        if B():
            pass
     """), "TypeError")
case("truth_bool_none_typeerror", "truthiness",
     "__bool__ = None is not 'absent': it is TypeError (and __len__ is not consulted)",
     _TR, DT, _s("""
        class B:
            __bool__ = None
            def __len__(self):
                return 1
        if B():
            pass
     """), "TypeError")
case("truth_len_negative_valueerror", "truthiness",
     "a negative __len__ in a truth test is ValueError",
     _TR, AR, _s(HEAD) + _s("""
        class B:
            def __len__(self):
                return -1
        if B():
            pass
     """) + TAIL, "ValueError", (False,))
case("truth_len_typeerror_propagates", "truthiness",
     "unlike the length hint, a truth test does not swallow a TypeError from __len__",
     _TR, DT, _s("""
        class B:
            def __len__(self):
                raise TypeError("t")
        if B():
            pass
     """), "TypeError")
case("truth_bool_keyerror_propagates", "truthiness",
     "an exception raised by __bool__ propagates",
     _TR, DL, _s("""
        class B:
            def __bool__(self):
                raise KeyError("b")
        if B():
            pass
     """), "KeyError")
case("truth_not_implemented_typeerror", "truthiness",
     "NotImplemented in a boolean context is TypeError since 3.14",
     "Objects/object.c::notimplemented_bool", DT, _s("""
        class A:
            def __eq__(self, other):
                return NotImplemented
        a = A()
        if a.__eq__(a):
            pass
     """), "TypeError")


# ======================================================================================
# NI: NotImplemented dispatch of operators and comparisons
# ======================================================================================
_NI = "Objects/abstract.c::binary_op1, Objects/object.c::do_richcompare"
case("ni_eq_both_not_implemented_is_identity", "notimplemented",
     "== with both __eq__ returning NotImplemented is identity: False, never TypeError",
     _NI, AR, _s(HEAD) + _s("""
        class E:
            def __eq__(self, other):
                return NotImplemented
        a = E()
        b = E()
        if a == b:
            pass
        else:
            print(1 // d)
     """), "ZeroDivisionError", (False,))
case("ni_ne_both_not_implemented_is_identity", "notimplemented",
     "!= likewise: the identity comparison",
     _NI, AR, _s(HEAD) + _s("""
        class E:
            def __eq__(self, other):
                return NotImplemented
        a = E()
        b = E()
        if a != b:
            print(1 // d)
     """), "ZeroDivisionError", (False,))
case("ni_lt_both_not_implemented_typeerror", "notimplemented",
     "an ordering comparison with both sides NotImplemented is TypeError",
     _NI, DT, _s("""
        class E:
            def __lt__(self, other):
                return NotImplemented
        E() < E()
     """), "TypeError")
case("ni_lt_reflects_to_gt", "notimplemented",
     "a < b tries b.__gt__(a) when a.__lt__ returns NotImplemented",
     _NI, AR, _s(HEAD) + _s("""
        class A:
            def __lt__(self, other):
                return NotImplemented
        class B:
            def __gt__(self, other):
                return True
        if A() < B():
            print(1 // d)
     """), "ZeroDivisionError", (False,))
case("ni_add_reflects_to_radd", "notimplemented",
     "a + b tries b.__radd__(a) when a.__add__ returns NotImplemented",
     _NI, AR, _s(HEAD) + _s("""
        class A:
            def __add__(self, other):
                return NotImplemented
        class B:
            def __radd__(self, other):
                return 1
        x = A() + B()
     """) + TAIL, "ZeroDivisionError", (False,))
case("ni_add_typeerror_is_not_not_implemented", "notimplemented",
     "a TypeError RAISED by __add__ propagates: __radd__ is not tried",
     _NI, DT, _s("""
        class A:
            def __add__(self, other):
                raise TypeError("no")
        class B:
            def __radd__(self, other):
                return 1
        x = A() + B()
     """), "TypeError")
case("ni_add_both_not_implemented_typeerror", "notimplemented",
     "both sides NotImplemented is TypeError",
     _NI, DT, _s("""
        class A:
            def __add__(self, other):
                return NotImplemented
        x = A() + A()
     """), "TypeError")
case("ni_int_plus_instance_uses_radd", "notimplemented",
     "int.__add__ returns NotImplemented for an instance: the instance's __radd__ answers",
     _NI, AR, _s(HEAD) + _s("""
        class B:
            def __radd__(self, other):
                return 1
        x = 3 + B()
     """) + TAIL, "ZeroDivisionError", (False,))
case("ni_list_plus_instance_radd_first", "notimplemented",
     "list + instance: the instance's __radd__ runs BEFORE list's own sequence concatenation",
     _NI, AR, _s(HEAD) + _s("""
        class B:
            def __radd__(self, other):
                return 1
        x = [1] + B()
     """) + TAIL, "ZeroDivisionError", (False,))
case("ni_iadd_not_implemented_falls_back_to_add", "notimplemented",
     "x += y falls back to __add__ when __iadd__ returns NotImplemented",
     _NI, AR, _s(HEAD) + _s("""
        class A:
            def __iadd__(self, other):
                return NotImplemented
            def __add__(self, other):
                return 1
        a = A()
        a += 1
     """) + TAIL, "ZeroDivisionError", (False,))
case("ni_iadd_typeerror_propagates", "notimplemented",
     "a TypeError raised by __iadd__ propagates: no __add__ fallback",
     _NI, DT, _s("""
        class A:
            def __iadd__(self, other):
                raise TypeError("x")
            def __add__(self, other):
                return 1
        a = A()
        a += 1
     """), "TypeError")
case("ni_subclass_reflected_first", "notimplemented",
     "a subclass that overrides the reflected method is tried FIRST",
     _NI, AR, _s(HEAD) + _s("""
        class A:
            def __add__(self, other):
                return 1
        class B(A):
            def __radd__(self, other):
                return 0
        x = 1 // (A() + B())
     """), "ZeroDivisionError", (False,))
case("ni_eq_raise_has_no_reflected_retry", "notimplemented",
     "an exception from __eq__ propagates: the reflected __eq__ is not tried",
     _NI, DL, _s("""
        class E:
            def __eq__(self, other):
                raise KeyError("e")
        E() == E()
     """), "KeyError")
case("ni_identity_shortcut_in_containers", "notimplemented",
     "containers compare by identity first: a raising __eq__ is never called on the same object",
     "Objects/object.c::PyObject_RichCompareBool", DL, _s("""
        class E:
            def __eq__(self, other):
                raise KeyError("e")
            __hash__ = object.__hash__
        x = E()
        r = [x] == [x]
        s = x in [x]
        t = [x].index(x)
        u = {x: 1}[x]
     """), "")
case("ni_container_compare_calls_eq_for_distinct_objects", "notimplemented",
     "but for two distinct objects __eq__ runs and its exception propagates",
     "Objects/object.c::PyObject_RichCompareBool", DL, _s("""
        class E:
            def __eq__(self, other):
                raise KeyError("e")
            __hash__ = object.__hash__
        r = [E()] == [E()]
     """), "KeyError")
case("ni_max_uses_gt_with_reflection", "notimplemented",
     "max() compares with >, reflected to __lt__ when the left operand lacks __gt__",
     "Python/bltinmodule.c::min_max", AR, _s(HEAD) + _s("""
        class A:
            def __lt__(self, other):
                return True
        m = max(A(), A())
     """) + TAIL, "ZeroDivisionError", (False,))
case("ni_sorted_needs_lt_or_its_reflection", "notimplemented",
     "sorted() compares with <: __gt__ alone suffices (a < b reflects to b > a) but __le__ alone does not",
     "Objects/listobject.c::list_sort_impl", DT, _s("""
        class A:
            def __gt__(self, other):
                return True
        class B:
            def __le__(self, other):
                return True
        xs = sorted([A(), A()])
        ys = sorted([B(), B()])
     """), "TypeError")


# ======================================================================================
# HD: hashing and dict / set lookups
# ======================================================================================
_HD = "Objects/dictobject.c::_Py_dict_lookup"
case("hd_eq_only_class_is_unhashable", "hashing",
     "a class defining __eq__ without __hash__ is unhashable: using an instance as a key is TypeError",
     "Objects/typeobject.c::type_new_set_hash (inherit_slots)", DT, _s("""
        class C:
            def __eq__(self, other):
                return True
        d = {}
        d[C()] = 1
     """), "TypeError")
case("hd_hash_raises_propagates", "hashing",
     "an exception from __hash__ propagates out of the lookup",
     _HD, DL, _s("""
        class E:
            def __hash__(self):
                raise IndexError("h")
        x = {}.get(E())
     """), "IndexError")
case("hd_eq_called_on_hash_collision", "hashing",
     "__eq__ runs when two distinct keys have the same hash, and its exception propagates",
     _HD, DL, _s("""
        class E:
            def __eq__(self, other):
                raise KeyError("e")
            def __hash__(self):
                return 1
        d = {E(): 1}
        d[E()]
     """), "KeyError")
case("hd_eq_not_called_for_identical_key", "hashing",
     "the identical key object is found without calling __eq__",
     _HD, DL, _s("""
        class E:
            def __eq__(self, other):
                raise KeyError("e")
            def __hash__(self):
                return 1
        k = E()
        d = {k: 1}
        x = d[k]
     """), "")
case("hd_missing_hook_redirects_keyerror", "hashing",
     "a dict subclass with __missing__ turns d[k]'s KeyError into the hook's value",
     "Objects/dictobject.c::dict_subscript", AR, _s(HEAD) + _s("""
        class D(dict):
            def __missing__(self, key):
                return d
        x = 1 // D()["a"]
     """), "ZeroDivisionError", (False,))
case("hd_missing_hook_not_used_by_get", "hashing",
     "get(), pop(), setdefault() and 'in' never call __missing__",
     "Objects/dictobject.c::dict_subscript", DL, _s("""
        class D(dict):
            def __missing__(self, key):
                return 0
        a = D().get("a")
        b = "a" in D()
        c = D().setdefault("a")
        D().pop("a")
     """), "KeyError")
case("hd_missing_hook_exception_propagates", "hashing",
     "an exception raised by __missing__ propagates unchanged",
     "Objects/dictobject.c::dict_subscript", DL, _s("""
        class D(dict):
            def __missing__(self, key):
                raise IndexError("m")
        D()["a"]
     """), "IndexError")
case("hd_defaultdict_has_no_keyerror", "hashing",
     "defaultdict[k] builds the value with its factory instead of raising KeyError",
     "Modules/_collectionsmodule.c::defdict_missing", AR, _s(HEAD) + _s("""
        import collections
        dd = collections.defaultdict(int)
        x = 1 // dd["k"]
     """), "ZeroDivisionError", (False,))
case("hd_defaultdict_without_factory_keyerror", "hashing",
     "with no factory it is the plain KeyError",
     "Modules/_collectionsmodule.c::defdict_missing", DL, _s("""
        import collections
        collections.defaultdict()["a"]
     """), "KeyError")
case("hd_counter_missing_is_zero", "hashing",
     "Counter[k] of an absent key is 0, not KeyError",
     "Lib/collections/__init__.py::Counter.__missing__", AR, _s(HEAD) + _s("""
        import collections
        x = 1 // collections.Counter()["a"]
     """), "ZeroDivisionError", (False,))
case("hd_dict_pop_empty_never_hashes", "hashing",
     "{}.pop(key) of an unhashable key raises KeyError, not TypeError: the empty dict never hashes",
     "Objects/dictobject.c::dict_pop_impl", DT, _s("""
        {}.pop([])
     """), "KeyError")
case("hd_abc_mapping_contains_swallows_keyerror", "hashing",
     "collections.abc.Mapping.__contains__ and .get swallow a KeyError from the user's __getitem__",
     "Lib/_collections_abc.py::Mapping.__contains__, ::Mapping.get", DL, _s("""
        import collections.abc
        class M(collections.abc.Mapping):
            def __getitem__(self, key):
                raise KeyError(key)
            def __iter__(self):
                return iter(())
            def __len__(self):
                return 0
        r = "a" in M()
        s = M().get("a", 1)
     """), "")
case("hd_abc_sequence_iter_ends_on_indexerror", "hashing",
     "collections.abc.Sequence.__iter__ / __contains__ / __reversed__ stop on IndexError from __getitem__",
     "Lib/_collections_abc.py::Sequence.__iter__", DL, _s("""
        import collections.abc
        class S(collections.abc.Sequence):
            def __getitem__(self, i):
                if i >= 2:
                    raise IndexError("end")
                return i
            def __len__(self):
                return 2
        xs = list(S())
        r = 1 in S()
        ys = list(reversed(S()))
     """), "")


# ======================================================================================
# IX: conversions of the exception class
# ======================================================================================
case("ix_huge_index_is_indexerror", "conversion",
     "a list index too large for a C ssize_t is IndexError ('cannot fit'), not OverflowError",
     "Objects/abstract.c::PyNumber_AsSsize_t (called with PyExc_IndexError)", DL, _s("""
        x = [1][10 ** 30]
     """), "IndexError")
case("ix_huge_negative_index_is_indexerror", "conversion",
     "so is a hugely negative one",
     "Objects/abstract.c::PyNumber_AsSsize_t", DL, _s("""
        x = [1][-(10 ** 30)]
     """), "IndexError")
case("ix_huge_str_index_is_indexerror", "conversion",
     "the same conversion for a str index too large for a C ssize_t",
     "Objects/unicodeobject.c::unicode_subscript", DL, _s("""
        x = "abc"[10 ** 30]
     """), "IndexError")
case("ix_huge_slice_bound_clamps", "conversion",
     "a huge slice bound clamps instead of raising",
     "Objects/sliceobject.c::PySlice_Unpack", DL, _s("""
        x = [1, 2][:10 ** 30]
     """), "")
case("ix_huge_pop_index_is_overflowerror", "conversion",
     "list.pop(huge) is OverflowError (argument conversion, not item access)",
     "Objects/listobject.c::list_pop", DL, _s("""
        [1].pop(10 ** 30)
     """), "OverflowError")
case("ix_huge_range_index_is_indexerror", "conversion",
     "range()[huge] is IndexError",
     "Objects/rangeobject.c::compute_item", DL, _s("""
        x = range(10)[10 ** 30]
     """), "IndexError")
case("ix_bytes_index_typeerror_falls_through", "conversion",
     "bytes(obj) tries __index__ first and, on TypeError from it, FALLS THROUGH to iteration",
     "Objects/bytesobject.c::_PyBytes_FromObject-path (bytes_new_impl)", DT, _s("""
        class I:
            def __index__(self):
                raise TypeError("x")
            def __iter__(self):
                return iter([1, 2])
        x = bytes(I())
     """), "")
case("ix_bytes_index_keyerror_propagates", "conversion",
     "any other exception from __index__ propagates",
     "Objects/bytesobject.c (bytes_new_impl)", DL, _s("""
        class I:
            def __index__(self):
                raise KeyError("x")
            def __iter__(self):
                return iter([1, 2])
        x = bytes(I())
     """), "KeyError")
