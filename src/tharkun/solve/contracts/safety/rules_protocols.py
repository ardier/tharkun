"""CPython-tested TypeError cases for protocol operator rules."""

from __future__ import annotations

import keyword
from itertools import product

from tharkun.solve.terms import (
    BOOL, F64, INT, PY, Refused, Term, and_, as_bool, as_f64, as_int,
    eq, gt, literal, lt, not_, typeis,
)
from tharkun.solve.contracts.safety import Binding, Case
from tharkun.solve.contracts.safety._runtime import supports_cpython_314

SCALAR_TAG_ORDER = ("int", "bool", "float", "str", "NoneType")
_NUMERIC_TAGS = frozenset({"int", "bool", "float"})
_INTEGER_TAGS = frozenset({"int", "bool"})

_COMPARISON_OPS = frozenset({"Lt", "LtE", "Gt", "GtE", "Eq", "NotEq"})
_BINARY_OPS = frozenset({
    "Add", "Sub", "Mult", "MatMult", "Div", "FloorDiv", "Mod", "Pow",
    "LShift", "RShift", "BitAnd", "BitXor", "BitOr",
})
_UNARY_OPS = frozenset({"UAdd", "USub", "Invert"})

# CPython 3.14.4 exact-builtin instance attribute names, captured with dir()
# and independently checked against LOAD_ATTR/getattr. This is static data:
# no target attribute lookup or dir() call is performed by the solver.
_INSTANCE_ATTRS = {
    "int": frozenset({
        "__abs__", "__add__", "__and__", "__bool__", "__ceil__", "__class__", "__delattr__", "__dir__", "__divmod__", "__doc__", "__eq__", "__float__", "__floor__", "__floordiv__", "__format__", "__ge__", "__getattribute__", "__getnewargs__", "__getstate__", "__gt__", "__hash__", "__index__", "__init__", "__init_subclass__", "__int__", "__invert__", "__le__", "__lshift__", "__lt__", "__mod__", "__mul__", "__ne__", "__neg__", "__new__", "__or__", "__pos__", "__pow__", "__radd__", "__rand__", "__rdivmod__", "__reduce__", "__reduce_ex__", "__repr__", "__rfloordiv__", "__rlshift__", "__rmod__", "__rmul__", "__ror__", "__round__", "__rpow__", "__rrshift__", "__rshift__", "__rsub__", "__rtruediv__", "__rxor__", "__setattr__", "__sizeof__", "__str__", "__sub__", "__subclasshook__", "__truediv__", "__trunc__", "__xor__", "as_integer_ratio", "bit_count", "bit_length", "conjugate", "denominator", "from_bytes", "imag", "is_integer", "numerator", "real", "to_bytes",
    }),
    "bool": frozenset({
        "__abs__", "__add__", "__and__", "__bool__", "__ceil__", "__class__", "__delattr__", "__dir__", "__divmod__", "__doc__", "__eq__", "__float__", "__floor__", "__floordiv__", "__format__", "__ge__", "__getattribute__", "__getnewargs__", "__getstate__", "__gt__", "__hash__", "__index__", "__init__", "__init_subclass__", "__int__", "__invert__", "__le__", "__lshift__", "__lt__", "__mod__", "__mul__", "__ne__", "__neg__", "__new__", "__or__", "__pos__", "__pow__", "__radd__", "__rand__", "__rdivmod__", "__reduce__", "__reduce_ex__", "__repr__", "__rfloordiv__", "__rlshift__", "__rmod__", "__rmul__", "__ror__", "__round__", "__rpow__", "__rrshift__", "__rshift__", "__rsub__", "__rtruediv__", "__rxor__", "__setattr__", "__sizeof__", "__str__", "__sub__", "__subclasshook__", "__truediv__", "__trunc__", "__xor__", "as_integer_ratio", "bit_count", "bit_length", "conjugate", "denominator", "from_bytes", "imag", "is_integer", "numerator", "real", "to_bytes",
    }),
    "float": frozenset({
        "__abs__", "__add__", "__bool__", "__ceil__", "__class__", "__delattr__", "__dir__", "__divmod__", "__doc__", "__eq__", "__float__", "__floor__", "__floordiv__", "__format__", "__ge__", "__getattribute__", "__getformat__", "__getnewargs__", "__getstate__", "__gt__", "__hash__", "__init__", "__init_subclass__", "__int__", "__le__", "__lt__", "__mod__", "__mul__", "__ne__", "__neg__", "__new__", "__pos__", "__pow__", "__radd__", "__rdivmod__", "__reduce__", "__reduce_ex__", "__repr__", "__rfloordiv__", "__rmod__", "__rmul__", "__round__", "__rpow__", "__rsub__", "__rtruediv__", "__setattr__", "__sizeof__", "__str__", "__sub__", "__subclasshook__", "__truediv__", "__trunc__", "as_integer_ratio", "conjugate", "from_number", "fromhex", "hex", "imag", "is_integer", "real",
    }),
    "str": frozenset({
        "__add__", "__class__", "__contains__", "__delattr__", "__dir__", "__doc__", "__eq__", "__format__", "__ge__", "__getattribute__", "__getitem__", "__getnewargs__", "__getstate__", "__gt__", "__hash__", "__init__", "__init_subclass__", "__iter__", "__le__", "__len__", "__lt__", "__mod__", "__mul__", "__ne__", "__new__", "__reduce__", "__reduce_ex__", "__repr__", "__rmod__", "__rmul__", "__setattr__", "__sizeof__", "__str__", "__subclasshook__", "capitalize", "casefold", "center", "count", "encode", "endswith", "expandtabs", "find", "format", "format_map", "index", "isalnum", "isalpha", "isascii", "isdecimal", "isdigit", "isidentifier", "islower", "isnumeric", "isprintable", "isspace", "istitle", "isupper", "join", "ljust", "lower", "lstrip", "maketrans", "partition", "removeprefix", "removesuffix", "replace", "rfind", "rindex", "rjust", "rpartition", "rsplit", "rstrip", "split", "splitlines", "startswith", "strip", "swapcase", "title", "translate", "upper", "zfill",
    }),
    "NoneType": frozenset({
        "__bool__", "__class__", "__delattr__", "__dir__", "__doc__", "__eq__", "__format__", "__ge__", "__getattribute__", "__getstate__", "__gt__", "__hash__", "__init__", "__init_subclass__", "__le__", "__lt__", "__ne__", "__new__", "__reduce__", "__reduce_ex__", "__repr__", "__setattr__", "__sizeof__", "__str__", "__subclasshook__",
    }),
}
_ATTRIBUTE_TAG_ORDER = ("int", "bool", "float", "str", "NoneType")
_INT_FLOAT_OVERFLOW_BOUNDARY = (1 << 1024) - (1 << 970)


def _operands(binding: Binding, arity: int, operators: frozenset[str]):
    if binding.receiver is None or len(binding.args) != arity:
        return Refused("operator binding has the wrong operand count")
    if binding.op not in operators:
        return Refused("operator name is not in this tested template family")
    values = (binding.receiver, *binding.args)
    if any(not _source_operand(value) for value in values):
        return Refused("operator operands must be source variables or exact scalar literals")
    return values


def _source_operand(value: Term) -> bool:
    if value.op == "var":
        return value.sort == PY
    return value.op == "lit" and value.sort in (INT, BOOL, F64)


def _case_domain(values: tuple[Term, ...], tags: tuple[str, ...]) -> Term | Refused:
    return and_(*(typeis(value, tag) for value, tag in zip(values, tags)))


def _numeric_zero(value: Term, tag: str) -> Term | Refused:
    literal_tag = {INT: "int", BOOL: "bool", F64: "float"}.get(value.sort)
    if value.op == "lit" and literal_tag != tag:
        return literal(False)
    if tag == "int":
        return eq(value if value.op == "lit" else as_int(value), literal(0))
    if tag == "bool":
        return eq(value if value.op == "lit" else as_bool(value), literal(False))
    if tag == "float":
        return eq(value if value.op == "lit" else as_f64(value), literal(0.0))
    return Refused("numeric zero test requested for a nonnumeric tag")


def _numeric_zero_cases(binding: Binding) -> tuple[Case, ...] | Refused:
    values = _operands(binding, 1, frozenset({"FloorDiv", "Mod", "Div"}))
    if isinstance(values, Refused):
        return values
    left, right = values
    tags = ("int", "bool", "float", "str", "NoneType")
    result = []
    for left_tag, right_tag in product(tags, repeat=2):
        domain = _case_domain(values, (left_tag, right_tag))
        if isinstance(domain, Refused):
            return domain
        if left_tag not in _NUMERIC_TAGS or right_tag not in _NUMERIC_TAGS:
            result.append(Case(domain, None))
            continue
        zero = _numeric_zero(right, right_tag)
        if isinstance(zero, Refused):
            return zero
        if left_tag == "int" and right_tag == "float":
            literal_tag = {INT: "int", BOOL: "bool", F64: "float"}.get(left.sort)
            if left.op == "lit" and literal_tag != "int":
                in_float_range = literal(False)
            elif left.op == "lit":
                in_float_range = and_(
                    gt(left, literal(-_INT_FLOAT_OVERFLOW_BOUNDARY)),
                    lt(left, literal(_INT_FLOAT_OVERFLOW_BOUNDARY)),
                )
            else:
                in_float_range = and_(
                    gt(as_int(left), literal(-_INT_FLOAT_OVERFLOW_BOUNDARY)),
                    lt(as_int(left), literal(_INT_FLOAT_OVERFLOW_BOUNDARY)),
                )
            result.append(Case(and_(domain, in_float_range), zero))
            result.append(Case(and_(domain, not_(in_float_range)), None))
        else:
            result.append(Case(domain, zero))
    return tuple(result)


def _truthiness_cases(binding: Binding) -> tuple[Case, ...] | Refused:
    if (binding.receiver is None or binding.args or binding.op is not None
            or not _source_operand(binding.receiver)):
        return Refused("truthiness binding requires one source receiver, no args, and no operator")
    return tuple(
        _make_case((binding.receiver,), (tag,), False)
        for tag in SCALAR_TAG_ORDER
    )


def _is_numeric_pair(left: str, right: str) -> bool:
    return left in _NUMERIC_TAGS and right in _NUMERIC_TAGS


def _is_integer_pair(left: str, right: str) -> bool:
    return left in _INTEGER_TAGS and right in _INTEGER_TAGS


def _pair_raises(qualname: str, op: str, left: str, right: str) -> bool | Refused:
    if qualname == "protocol.comparison":
        if op in {"Eq", "NotEq"}:
            return False
        return not (_is_numeric_pair(left, right) or (left == right == "str"))
    if qualname != "protocol.binary_operator":
        return Refused("binary case requested for a non-binary rule")
    if op == "Add":
        safe = _is_numeric_pair(left, right) or (left == right == "str")
    elif op in {"Sub", "Div", "FloorDiv", "Pow"}:
        safe = _is_numeric_pair(left, right)
    elif op == "Mult":
        safe = (
            _is_numeric_pair(left, right)
            or (left == "str" and right in _INTEGER_TAGS)
            or (right == "str" and left in _INTEGER_TAGS)
        )
    elif op in {"LShift", "RShift", "BitAnd", "BitXor", "BitOr"}:
        safe = _is_integer_pair(left, right)
    elif op == "Mod":
        # String interpolation is excluded by the case domain below.
        safe = _is_numeric_pair(left, right)
    elif op == "MatMult":
        safe = False
    else:
        return Refused("operator has no tested TypeError truth table")
    return not safe


def _make_case(values: tuple[Term, ...], tags: tuple[str, ...], raises: bool) -> Case:
    return Case(_case_domain(values, tags), literal(True) if raises else None)


def _iteration_cases(binding: Binding) -> tuple[Case, ...] | Refused:
    if (binding.receiver is None or binding.args or binding.op is not None
            or not _source_operand(binding.receiver)):
        return Refused("iteration binding requires one receiver, no args, and no operator")
    receiver = binding.receiver
    return tuple(
        _make_case((receiver,), (tag,), tag != "str")
        for tag in SCALAR_TAG_ORDER
    )


def attribute_cases(qualname: str, exception: str, binding: Binding) -> tuple[Case, ...] | Refused:
    """Return exact-builtin scalar attribute outcomes for an AST attribute read."""
    if not supports_cpython_314():
        return Refused("attribute template requires CPython 3.14")
    if qualname != "protocol.attribute_access" or exception not in {"AttributeError", "TypeError"}:
        return Refused("scalar attribute templates cover AttributeError and TypeError only")
    if binding.receiver is None or binding.args or not _source_operand(binding.receiver):
        return Refused("attribute binding must have one source receiver and no arguments")
    prefix = "Attribute:"
    if not isinstance(binding.op, str) or not binding.op.startswith(prefix):
        return Refused("attribute binding must retain the original AST identifier")
    name = binding.op[len(prefix):]
    if not name.isidentifier() or keyword.iskeyword(name):
        return Refused("attribute name is not a valid AST identifier")
    result = []
    for tag in _ATTRIBUTE_TAG_ORDER:
        domain = typeis(binding.receiver, tag)
        if isinstance(domain, Refused):
            return domain
        missing = name not in _INSTANCE_ATTRS[tag]
        # AttributeError is determined by the pinned exact-builtin instance
        # inventory. Exact scalar lookup never raises TypeError. NoneType stays
        # admitted internally even though the frozen witness grammar cannot
        # currently render a NoneType guard.
        raises = missing if exception == "AttributeError" else False
        result.append(Case(domain, literal(True) if raises else None))
    return tuple(result)


def protocol_cases(qualname: str, exception: str, binding: Binding) -> tuple[Case, ...] | Refused:
    """Return ordered, disjoint exact-scalar cases for one protocol operation."""
    if exception == "ZeroDivisionError":
        if qualname != "protocol.binary_operator":
            return Refused("zero-division cases cover binary operators only")
        return _numeric_zero_cases(binding)
    if exception != "TypeError":
        return Refused("protocol scalar templates cover TypeError only")
    if qualname == "protocol.truthiness":
        return _truthiness_cases(binding)
    if qualname == "protocol.iteration":
        return _iteration_cases(binding)
    if qualname == "protocol.comparison":
        values = _operands(binding, 1, _COMPARISON_OPS)
        if isinstance(values, Refused):
            return values
        return tuple(
            _make_case(values, pair, _pair_raises(qualname, binding.op, *pair))
            for pair in product(SCALAR_TAG_ORDER, repeat=2)
        )
    if qualname == "protocol.binary_operator":
        values = _operands(binding, 1, _BINARY_OPS)
        if isinstance(values, Refused):
            return values
        result = []
        for pair in product(SCALAR_TAG_ORDER, repeat=2):
            # str-left percent formatting depends on the format string and the
            # value (for example "%s" % x versus "%d" % x). It is not a case.
            if binding.op == "Mod" and pair[0] == "str":
                continue
            raises = _pair_raises(qualname, binding.op, *pair)
            if isinstance(raises, Refused):
                return raises
            result.append(_make_case(values, pair, raises))
        return tuple(result)
    if qualname == "protocol.unary_operator":
        values = _operands(binding, 0, _UNARY_OPS)
        if isinstance(values, Refused):
            return values
        result = []
        for tag in SCALAR_TAG_ORDER:
            safe = (
                tag in _NUMERIC_TAGS if binding.op in {"UAdd", "USub"}
                else tag in _INTEGER_TAGS
            )
            result.append(_make_case(values, (tag,), not safe))
        return tuple(result)
    return Refused("no scalar protocol TypeError template for this qualname")


CASES_BUILDERS = {
    ("protocol.attribute_access", "AttributeError"): attribute_cases,
    ("protocol.attribute_access", "TypeError"): attribute_cases,
    ("protocol.truthiness", "TypeError"): protocol_cases,
    ("protocol.iteration", "TypeError"): protocol_cases,
    ("protocol.comparison", "TypeError"): protocol_cases,
    ("protocol.binary_operator", "TypeError"): protocol_cases,
    ("protocol.binary_operator", "ZeroDivisionError"): protocol_cases,
    ("protocol.unary_operator", "TypeError"): protocol_cases,
}

__all__ = ["CASES_BUILDERS", "Case", "SCALAR_TAG_ORDER", "protocol_cases"]
