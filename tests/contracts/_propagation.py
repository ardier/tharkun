"""Shared stub-object builders for testing dunder-propagation claims ([internal ref]).

CPython dispatches these callables to a receiver dunder via the type's C slot, not a
``getattr``/``hasattr`` probe -- so a dunder that raises *any* exception propagates
unchanged, whatever exception the callable's own logic would otherwise raise. This
module builds one stub receiver per (dunder name, exception class) pair so every
contract's propagation claims can be exercised with the same, reviewed code instead of
each test file rolling its own throwaway classes.
"""

from __future__ import annotations

PROPERTY_CLASSES = {
    "TypeError": TypeError,
    "AttributeError": AttributeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "ZeroDivisionError": ZeroDivisionError,
    "FloatingPointError": FloatingPointError,
    "AssertionError": AssertionError,
}


def raiser(dunder_name: str, exc_name: str, *, nargs: int = 0):
    """A class whose ``dunder_name`` method raises ``PROPERTY_CLASSES[exc_name]``.

    ``nargs`` is the number of arguments the dunder takes besides ``self`` (e.g. 1 for
    ``__pow__``/``__divmod__`` which take the other operand); those extra positional
    arguments are accepted and ignored.
    """
    exc = PROPERTY_CLASSES[exc_name]

    def method(self, *_a, **_k):
        raise exc(f"boom-{dunder_name}-{exc_name}")

    return type(f"Raiser_{dunder_name.strip('_')}_{exc_name}", (), {dunder_name: method})


def nonproperty_raiser(dunder_name: str, exc: type[BaseException], *, nargs: int = 0):
    """Like :func:`raiser`, but for an arbitrary exception NOT in :data:`PROPERTY_CLASSES`.

    Backs the ``may_raise_other`` open-world flag: a receiver dunder is arbitrary user
    code and can raise *any* exception, not just one of the 7 property classes -- a
    contract may not set ``may_raise_other=False`` just because its ``raises`` claims
    (built from the 7 property classes) are individually sound.
    """

    def method(self, *_a, **_k):
        raise exc(f"boom-{dunder_name}-{exc.__name__}")

    return type(f"NonPropertyRaiser_{dunder_name.strip('_')}_{exc.__name__}", (), {dunder_name: method})


def attr_raiser(attr_name: str, exc_name: str):
    """An instance whose ``attr_name`` attribute, when called, raises ``exc_name``
    (for protocol paths like ``print``'s ``file.write`` that are a plain attribute
    lookup + call, not a dunder slot).
    """
    exc = PROPERTY_CLASSES[exc_name]

    class _Obj:
        pass

    def method(*_a, **_k):
        raise exc(f"boom-{attr_name}-{exc_name}")

    obj = _Obj()
    setattr(obj, attr_name, method)
    return obj
