"""CPython-tested TypeError cases for the ``str`` methods of ``contracts/adapters/str_shapes.py``.

The rules of ``rules_str.py`` split a method's property classes in two: every class but ``TypeError`` is reachable only through a
USER hook on an argument (so it is "never" for an argument of a builtin scalar), and ``TypeError`` is the call's own logic, a
function of the type and, for padding fills, length of each argument.  This template states that function for the builtin scalars, the way the other templates
do (``rules_builtins.py``, ``rules_protocols.py``): one case per assignment of an exact scalar tag (``int``, ``bool``, ``float``,
``str``, ``NoneType``) to each argument, with the receiver a ``str``, so a case's domain is a conjunction of ``typeis`` facts that
a consumer's tag facts single out.  Inside a case the call raises ``TypeError`` exactly when some argument's tag is one its
parameter does not take (``_TAKES``: ``prefix`` takes a str; ``chars`` a str or None; ``start`` an int, bool or None; ``width`` an
int or bool; ...), and, for a str padding fill, when its length is not one.  A container, an instance or an unknown value is in no case, so the admission term is
false there and a consumer that needs a certificate refuses.

Binding convention: ``receiver`` is the str the method is called on, ``args`` the positional arguments in call order, ``op`` is
``None``.  A call with a keyword has no binding (the record schema has no field for one), and an arity the method does not take is
refused, not a case.  Every operand is a source variable or an exact numeric literal, as in the other templates.

``splitlines(keepends)`` reads its argument by truth value, so every scalar tag is taken there (``_TAKES_ANY``).
"""

from __future__ import annotations

from itertools import product

from tharkun.solve.contracts.adapters.str_shapes import SIGNATURES
from tharkun.solve.contracts.safety import Binding, Case
from tharkun.solve.contracts.safety.rules_protocols import _source_operand
from tharkun.solve.terms import Refused, and_, length, literal, ne, or_, typeis

_SCALARS = ("int", "bool", "float", "str", "NoneType")
#: kind -> the scalar tags a parameter of that kind takes without a TypeError.  ``L`` (join) takes a ``str``: it iterates its
#: characters; an int, bool, float or None is not iterable.
_TAKES = {
    "s": ("str",),
    "f": ("str",),
    "S": ("str",),
    "c": ("str", "NoneType"),
    "i": ("int", "bool"),
    "n": ("int", "bool", "NoneType"),
    "L": ("str",),
}
#: parameters CPython reads by truth value: every scalar is taken
_TAKES_ANY = {("splitlines", "keepends")}


def _takes(method: str, param) -> tuple[str, ...]:
    return _SCALARS if (method, param.name) in _TAKES_ANY else _TAKES[param.kind]


def str_method_cases(qualname: str, exception: str, binding: Binding):
    """One case per assignment of a scalar tag to every argument (the receiver is a ``str``); ``TypeError`` iff some argument's tag
    is not one its parameter takes."""
    if exception != "TypeError" or not qualname.startswith("str."):
        return Refused("str method cases cover TypeError only")
    method = qualname[4:]
    params = SIGNATURES.get(method)
    if params is None:
        return Refused("no str method shape for this qualname")
    receiver = binding.receiver
    if receiver is None or binding.op is not None:
        return Refused("str method binding requires a receiver and no operator")
    required = sum(p.required for p in params)
    if not required <= len(binding.args) <= len(params):
        return Refused("str method binding has an arity the method does not take")
    operands = (receiver, *binding.args)
    if any(not _source_operand(value) for value in operands):
        return Refused("str method operands must be source variables or exact numeric literals")
    taken = [_takes(method, p) for p in params[:len(binding.args)]]
    rows = []
    for tags in product(_SCALARS, repeat=len(binding.args)):
        domain = and_(*(typeis(value, tag) for value, tag in zip(operands, ("str", *tags))))
        if isinstance(domain, Refused):
            return domain
        bad_type = any(tag not in ok for tag, ok in zip(tags, taken))
        lengths = [ne(length(value), literal(1)) for p, value in zip(params, binding.args) if p.kind == "f"]
        raises = literal(True) if bad_type else or_(*lengths) if lengths else None
        rows.append(Case(domain, raises))
    return tuple(rows)


CASES_BUILDERS = {(f"str.{method}", "TypeError"): str_method_cases for method in SIGNATURES}
