"""Independent literal cases for the MR1793 check_type behavior contract."""
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Generic, Literal, TypeVar, Union


class User:
    def __repr__(self):
        return "User()"


class Child(User):
    pass


class IterRaises(Iterable):
    def __iter__(self):
        raise RuntimeError("iter boom")

    def __repr__(self):
        return "IterRaises()"


class EqRaises:
    def __eq__(self, other):
        raise ValueError("eq boom")

    def __repr__(self):
        return "EqRaises()"


T = TypeVar("T")


class Box(Generic[T]):
    def __repr__(self):
        return "Box()"


for cls in (User, Child, IterRaises, EqRaises, Box):
    cls.__module__ = "sv_check_type_cases"


def function_with_wrong_signature(a, b):
    return 123


# Each row is (label, value, hint). Expected literal outcomes are recorded once
# from the pinned upstream function, in sv_check_type_outcomes.json.
CASES = [
    ("any", object(), Any),
    ("none", None, None),
    ("none_type", None, type(None)),
    ("none_mismatch", 0, None),
    ("plain_int", 3, int),
    ("plain_int_mismatch", "3", int),
    ("bool_is_int", True, int),
    ("int_not_bool", 1, bool),
    ("plain_list", [1, "x"], list),
    ("class_tuple_hint", "x", (int, str)),
    ("union", "x", Union[int, str]),
    ("union_mismatch", 1.5, Union[int, str]),
    ("pipe_union", None, int | None),
    ("pipe_mismatch", "x", int | None),
    ("literal_int", 1, Literal[1, "x"]),
    ("literal_str", "x", Literal[1, "x"]),
    ("literal_bool_not_int", True, Literal[1]),
    ("literal_int_not_bool", 1, Literal[True]),
    ("literal_eq_raises", EqRaises(), Literal[1]),
    ("tuple_fixed", (1, "x"), tuple[int, str]),
    ("tuple_length", (1,), tuple[int, str]),
    ("tuple_element", (1, 2), tuple[int, str]),
    ("tuple_variadic", (1, True, 3), tuple[int, ...]),
    ("tuple_variadic_mismatch", (1, "x"), tuple[int, ...]),
    ("tuple_empty_hint_empty", (), tuple[()]),
    ("tuple_empty_hint_nonempty", (1,), tuple[()]),
    ("tuple_wrong_origin", [], tuple[int]),
    ("list_nested", [[1, True], []], list[list[int]]),
    ("list_nested_mismatch", [[1], ["x"]], list[list[int]]),
    ("set", {1, 2}, set[int]),
    ("set_mismatch", {"x"}, set[int]),
    ("frozenset", frozenset({1}), frozenset[int]),
    ("frozenset_mismatch", frozenset({"x"}), frozenset[int]),
    ("dict", {"x": [1, 2]}, dict[str, list[int]]),
    ("dict_key_mismatch", {1: [2]}, dict[str, list[int]]),
    ("dict_value_mismatch", {"x": ["bad"]}, dict[str, list[int]]),
    ("mapping", {"x": (1, 2)}, Mapping[str, tuple[int, ...]]),
    ("iterable", [1, True], Iterable[int]),
    ("iterable_nested", [(1,), (2,)], Iterable[tuple[int, ...]]),
    ("iterable_mismatch", ["x"], Iterable[int]),
    ("iterator_consumed", iter([1, 2]), Iterable[int]),
    ("user_iter_raises", IterRaises(), Iterable[int]),
    ("callable_signature_ignored", function_with_wrong_signature, Callable[[str], str]),
    ("callable_ellipsis", function_with_wrong_signature, Callable[..., int]),
    ("callable_mismatch", 3, Callable[[int], str]),
    ("user_class", User(), User),
    ("user_subclass", Child(), User),
    ("user_class_mismatch", 3, User),
    ("unsupported_hint", Box(), Box[int]),
    ("unsupported_wrong_origin", 3, Box[int]),
    ("malformed_not_class", 1, 42),
    ("malformed_list_arity", [1], list[int, str]),
    ("malformed_mapping_short", {1: 2}, dict[int]),
    ("malformed_mapping_long", {1: 2}, dict[int, str, float]),
]
