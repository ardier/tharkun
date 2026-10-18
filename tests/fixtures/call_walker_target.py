"""Fixture for call_walker tests.

Contains various call patterns for testing callee resolution.
"""


# --- Simple functions ---

def is_valid(x):
    return x > 0 and x < 100


def check_range(value):
    return is_valid(value)


# --- Class with methods ---

class Config:
    def __init__(self, name):
        self.name = name

    def validate(self):
        return self.name is not None

    @property
    def is_ready(self):
        return self.name is not None and len(self.name) > 0

    def run(self):
        if self.validate():
            return True
        return False


# --- Inheritance / super ---

class BaseProcessor:
    def process(self, data):
        return len(data) > 0


class SubProcessor(BaseProcessor):
    def process(self, data):
        if super().process(data):
            return data[0] > 0
        return False


# --- isinstance-typed receiver ---

def handle_config(obj):
    if isinstance(obj, Config):
        return obj.validate()
    return False


# --- Annotation-typed receiver ---

def use_config(cfg: Config):
    return cfg.validate()


# --- Guard clauses ---

def with_guards(x, y):
    if not x:
        return False
    if y < 0:
        return False
    return True


# --- If-true/else-false ---

def if_true_else_false(x):
    if x > 10:
        return True
    return False


# --- Complex return (unextractable) ---

def complex_return(x):
    result = x * 2
    if result > 0:
        return result
    else:
        return -result


# --- Class method call via ClassName ---

def use_class_method():
    c = Config("test")
    return Config.validate(c)
