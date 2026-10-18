"""Stress-test fixture for guard decomposer + runtime snapshot analyzer.

Covers: deep recursion, classmethods, lambdas, decorators, *args/**kwargs,
large locals, circular refs, loop snapshots, exception snapshots, complex types.
"""
from dataclasses import dataclass
from enum import Enum

# ===== DECOMPOSER SCENARIOS =====

# --- Scenario 1: 3-level recursive validators ---

_call_count = 0

def level3_deep(value):
    return value > 0 and value < 1000

def level2_intermediate(value, threshold):
    if threshold < 0:
        return False
    return level3_deep(value) and value >= threshold

def level1_top(value, threshold, strict):
    if strict and value == threshold:
        return False
    return level2_intermediate(value, threshold) and value < 500

def uses_deep_validators(value, threshold=10, strict=False):
    if level1_top(value, threshold, strict):
        pass


# --- Scenario 2: classmethod + staticmethod ---

class ConfigValidator:
    MIN_VERSION = "1.0"
    MAX_LOAD = 0.95

    @staticmethod
    def validate_version(version):
        return version >= ConfigValidator.MIN_VERSION

    @classmethod
    def validate_load(cls, load):
        if load < 0:
            return False
        return load <= cls.MAX_LOAD

def guard_class_validators(ver, load):
    return ConfigValidator.validate_version(ver) and ConfigValidator.validate_load(load)

def uses_class_validators(ver, load):
    if guard_class_validators(ver, load):
        pass


# --- Scenario 3: lambda + higher-order ---

def is_even(x):
    return x % 2 == 0

def apply_validator(value, validator):
    if value < 0:
        return False
    return validator(value)

def guard_with_lambda(x):
    even_check = lambda v: v % 2 == 0
    return x > 0 and even_check(x)

def guard_higher_order(x):
    return apply_validator(x, is_even)

def uses_lambda(x):
    if guard_with_lambda(x):
        pass

def uses_higher_order(x):
    if guard_higher_order(x):
        pass


# --- Scenario 4: decorated functions ---

def my_cache(fn):
    """Simple decorator that wraps fn."""
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

@my_cache
def cached_is_valid(x):
    return x > 0 and x < 1000

def guard_decorated(x):
    return cached_is_valid(x) and x % 7 != 0

def uses_decorated(x):
    if guard_decorated(x):
        pass


# --- Scenario 5: *args / **kwargs ---

def validate_all(*args):
    if not args:
        return False
    return all(isinstance(x, (int, float)) for x in args)

def validate_kwargs(**kwargs):
    if "required" not in kwargs:
        return False
    return kwargs["required"] > 0

def guard_varargs(x, y, z):
    return validate_all(x, y, z) and x + y + z > 100

def guard_kwargs(field):
    return validate_kwargs(required=field) and field < 1000

def uses_varargs(x, y, z):
    if guard_varargs(x, y, z):
        pass

def uses_kwargs(field):
    if guard_kwargs(field):
        pass


# ===== SNAPSHOT SCENARIOS =====

# --- Scenario 6: large locals ---

def validate_large_config(config):
    large_state = {f"key_{i}": i for i in range(1000)}
    if not config:
        return False
    required = ["version", "enabled"]
    if not all(k in config for k in required):
        return False
    return config.get("version") >= "1.0"

def uses_large_locals(config):
    if validate_large_config(config):
        pass


# --- Scenario 7: circular references ---

class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None

def validate_node(root):
    if root is None:
        return False
    node = Node(42)
    node.parent = node  # circular
    return root.value > 0

def uses_circular(root):
    if validate_node(root):
        pass


# --- Scenario 8: loop snapshots ---

def process_items(items):
    for i, item in enumerate(items):
        if item < 0:
            return False
        if item > 1000:
            return False
    return True

def uses_loop(items):
    if process_items(items):
        pass


# --- Scenario 9: exception at snapshot ---

class ValidationError(Exception):
    pass

def strict_validator(value, mode):
    if mode == "strict" and value == 0:
        raise ValidationError("zero not allowed")
    return value > 0

def guard_with_exception(value, mode):
    try:
        return strict_validator(value, mode)
    except ValidationError:
        return False

def uses_exception(value, mode):
    if guard_with_exception(value, mode):
        pass


# --- Scenario 10: complex types ---

class Status(Enum):
    PENDING = 1
    RUNNING = 2
    DONE = 3

@dataclass
class Request:
    id: int
    status: Status
    payload: dict

@dataclass
class AppConfig:
    version: str
    requests: list
    settings: dict

def validate_app_config(config, timeout):
    if not config:
        return False
    if timeout < 0:
        return False
    if not config.requests:
        return False
    req = config.requests[0]
    if req.status != Status.RUNNING:
        return False
    return req.id > 0 and timeout <= 30.0

def uses_complex_types(config, timeout):
    if validate_app_config(config, timeout):
        pass
