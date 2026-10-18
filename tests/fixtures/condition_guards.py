"""Fixture for condition_solver tests — various guard patterns."""


def f(x, name, data, items, lines):
    if x > 100:
        pass
    if name == "hello":
        pass
    if data[:2] == b"\xfe\xff":
        pass
    if isinstance(items, list) and len(items) > 3:
        pass
    if lines[1].startswith("="):
        pass
    if x > 0 and x < 10 and x != 5:
        pass


def g(self_thing):
    if self_thing > 0:
        pass


class C:
    def m(self, flag):
        if self.allow_unicode:
            pass
        if flag == True:
            pass


def with_in(x):
    if x in (1, 2, 3):
        pass


def typed_params(x: int, name: str) -> None:
    if x > 50:
        pass


def str_methods(s):
    if s.startswith("prefix"):
        pass


def bytes_method(data):
    if data.startswith(b"\xff\xfe"):
        pass


def dict_method(d):
    if d.get("key"):
        pass


# --- IB-3 stress-test guards below ---


def flipped_float(ratio):
    if 0.0 < ratio <= 1.0:
        pass


def flipped_int(x):
    if 10 > x:
        pass


def len_constraint(password):
    if len(password) >= 8:
        pass


def startswith_endswith(x):
    if x.startswith("prefix") and x.endswith("suffix"):
        pass


def startswith_only(s):
    if s.startswith("http://"):
        pass


def data_subscript_list(data):
    if data[0] == "header" and data[-1] == "footer":
        pass


def is_not_none(x):
    if x is not None:
        pass


def multi_var_compound(username, password, role):
    if username and len(password) >= 8 and role in ("admin", "editor"):
        pass


def negated_isinstance(x):
    if not isinstance(x, str):
        pass


def is_none(x):
    if x is None:
        pass


def is_not_none_guard(x):
    if x is not None:
        pass


def method_chain(response):
    if response.json().get("status") == "ok":
        pass


def or_branches(x, y, z):
    if x < 0 or isinstance(y, str) or z is None:
        pass


def or_simple(a, b):
    if a > 10 or b == "yes":
        pass


def negative_literal(x):
    if x > -100 and x < -10:
        pass


import ast as _ast
import re

def regex_guard(ssn):
    if re.match(r"^\d{3}-\d{2}-\d{4}$", ssn):
        pass


def truthiness_and(x, y):
    if x and len(y) > 3:
        pass


def isinstance_ast_call(node):
    if isinstance(node, _ast.Call):
        pass


# --- Attribute access guards (Phase 5 fix) ---


def attr_in_set(c):
    if c.kind in {"method", "classmethod"}:
        pass


def attr_eq(p):
    if p.has_default == True:
        pass


def nested_attr(node):
    if node.value.id == "self":
        pass


def comprehension_guard(items):
    if all(x > 0 for x in items):
        pass


def hasattr_guard(obj):
    if hasattr(obj, "ready"):
        pass


def chained_isinstance(func):
    if isinstance(func.value, str):
        pass


# --- Bare attribute chain truthiness guards (deep attr fix) ---


def deep_attr_truthiness(frame):
    if frame.eval_ctx.volatile:
        pass


def negated_attr_truthiness(obj):
    if not obj.allow_unicode:
        pass


def triple_deep_attr(ctx):
    if ctx.md.htmlStash.html_counter:
        pass


# --- Math predicate guards ---

import math

def math_isnan(value):
    if math.isnan(value):
        pass


def math_isinf_neg(value):
    if math.isinf(value) and value < 0:
        pass


def not_isfinite(value):
    if not math.isfinite(value):
        pass


# --- Special patterns ---

def any_startswith(value):
    if any(value.startswith(p) for p in {"10.", "192.168.", "127."}):
        pass


def modular_membership(value):
    if value % 100 in (11, 12, 13):
        pass


def callable_check(fmt):
    if callable(fmt):
        pass


def string_containment(value):
    if "x" in value:
        pass


# --- Self.xxx compound patterns ---

class SelfCompound:
    def is_not_none(self):
        if self.encoding is not None:
            pass

    def compound_method(self, tried):
        if self.encoding is not None and self._usable(self.encoding, tried):
            pass

    def len_contents(self):
        if len(self.contents) != 1:
            pass

    def method_truthiness(self, e, tried):
        if self._usable(e, tried):
            pass


# --- Literal True ---

def while_true():
    while True:
        break


# --- Split/partition result constraints (OSS-Fuzz P1) ---


def split_eq_two(line):
    if len(line.split("=")) == 2:
        pass


def split_sep_gte(data):
    if len(data.split(",")) >= 3:
        pass


def split_maxsplit(line):
    if len(line.split(":", 1)) == 2:
        pass


def partition_truthy(line):
    key, sep, value = line.partition("=")
    if sep:
        pass


# --- Type coercion guards (OSS-Fuzz P2) ---


def float_coercion_guard(quality_str):
    if float(quality_str) > 0.5:
        pass


def int_coercion_guard(week_str):
    if int(week_str) > 10:
        pass


def float_coercion_eq(s):
    if float(s) == 1.0:
        pass


# --- Dict key shape synthesis (OSS-Fuzz P3) ---


def dict_subscript_guard(data):
    if data["name"] and data["age"] > 0:
        pass


def dict_get_guard(config):
    if config.get("host") and config.get("port"):
        pass


# --- Byte interior containment (OSS-Fuzz P4) ---


def byte_interior_guard(data):
    if b"\x1b" in data:
        pass


def byte_interior_null(data):
    if b"\x00" in data and len(data) > 4:
        pass


# --- Enum / constant attribute comparison (D1) ---

class Color:
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

class HTTPStatus:
    OK = 200
    NOT_FOUND = 404

def enum_eq_guard(color):
    if color == Color.RED:
        pass

def enum_reverse_guard(status_code):
    if HTTPStatus.OK == status_code:
        pass

def enum_neq_guard(status_code):
    if status_code != HTTPStatus.NOT_FOUND:
        pass
