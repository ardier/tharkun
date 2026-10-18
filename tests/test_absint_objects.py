"""User protocols must execute their bodies and preserve receiver heap facts."""
import ast
from textwrap import dedent

import pytest

from tharkun.solve.absint import prove

PROPS = ("TypeError", "AttributeError", "KeyError", "IndexError",
         "ZeroDivisionError", "AssertionError")

SAFE = [
    """
    class Scale:
        def __init__(self, factor):
            self.factor = factor
        def __call__(self, value, offset=1):
            return self.factor * value + offset
    assert Scale(2)(3, offset=4) == 10
    """,
    """
    from _sv_verifier import nondet_bool
    class Counter:
        def __init__(self):
            self.value = 1
        def __iadd__(self, delta):
            self.value += delta
            return self
    counter = Counter()
    while nondet_bool():
        counter += 1
    answer = counter.value + 2
    """,
    """
    class Shift:
        def __init__(self, amount):
            self.amount = amount
        def __add__(self, other):
            return Shift(self.amount + other.amount)
        def __sub__(self, other):
            return Shift(self.amount - other.amount)
        def __mul__(self, scale):
            return Shift(self.amount * scale)
    result = (Shift(2) + Shift(3) - Shift(1)) * 2
    assert result.amount == 8
    """,
    """
    class Bag:
        def __init__(self):
            self.items = [2, 4]
        def __getitem__(self, index):
            return self.items[index]
        def __setitem__(self, index, value):
            self.items[index] = value
        def __len__(self):
            return len(self.items)
        def __iter__(self):
            return iter(self.items)
    bag = Bag()
    bag[0] = 3
    assert len(bag) == 2
    for item in bag:
        assert item > 0
    """,
    """
    class Cell:
        def __init__(self, value):
            self.value = value
        def __eq__(self, other):
            return self.value == other.value
        def __lt__(self, other):
            return self.value < other.value
    assert Cell(2) == Cell(2)
    assert Cell(2) < Cell(3)
    assert Cell(2) != Cell(3)
    """,
    """
    class Counter:
        def __init__(self):
            self.value = 1
        def __iadd__(self, delta):
            self.value += delta
            return self
    counter = Counter()
    alias = counter
    counter += 2
    assert alias.value == 3
    """,
    """
    class Base:
        def __add__(self, other):
            return 1
    class Derived(Base):
        def __radd__(self, other):
            return 2
    assert Base() + Derived() == 2
    """,
    """
    class Empty:
        def __len__(self):
            return 0
    assert not Empty()
    """,
    """
    class Box:
        def __init__(self, *, value=3):
            self.value = value
        def __add__(self, other):
            return self.value + other
    assert Box() + 4 == 7
    """,
    """
    class Base:
        def __init__(self, **options):
            self.value = options.get('amount', 1)
    class Child(Base):
        def __init__(self, **options):
            Base.__init__(self, **options)
    box = Child(amount=2)
    result = box.value + 1
    """,
    """
    def choose(value, /, **options):
        return options['value']
    assert choose(1, **{'value': 2}) == 2
    """,
    """
    def fail(index):
        return 1 / 0
    class Box:
        def __getitem__(self, index):
            return 3
    box = Box()
    box.__getitem__ = fail
    assert box[0] == 3
    """,
    """
    def fail():
        return 1 / 0
    class Box:
        def __len__(self):
            return 2
    box = Box()
    box.__len__ = fail
    assert len(box) == 2
    """,
    """
    class Scale:
        def __init__(self, n):
            self.n = n
        def __mul__(self, k):
            return self.n * k
    Scale.unit = Scale(1)
    assert Scale.unit * 3 == 3
    """,
    """
    class Box:
        value = 1
    box = Box()
    Box.value = 2
    assert box.value == 2
    box.value = 3
    assert Box.value == 2
    """,
]

UNSAFE = [
    """
    def fail(index):
        return 1 / 0
    class Box:
        def __getitem__(self, index):
            return 3
    box = Box()
    box.__getitem__ = fail
    box.__getitem__(0)
    """,
    """
    class Bag:
        def __init__(self):
            self.items = [1]
        def __iter__(self):
            return iter(self.items)
    bag = Bag()
    for value in bag:
        result = value + 1
        bag.items.append('bad')
    """,
    """
    class Box:
        value = 1
    box = Box()
    Box.value = 'bad'
    value = box.value + 1
    """,
    """
    from _sv_verifier import nondet_bool
    class Box:
        pass
    if nondet_bool():
        Box.value = 1
    value = Box.value
    """,
    """
    class Box:
        pass
    def bad(self):
        return 1 / 0
    Box.__len__ = bad
    value = len(Box())
    """,
    """
    class Source:
        def __call__(self):
            return 1 / 0
    for value in iter(Source(), 0):
        pass
    """,
    """
    class Source:
        def __call__(self):
            value = int("10", 2)
            return 1 / 0
    Source()()
    """,
    """
    import random
    class Source:
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return 1 / 0
    random.choice(Source())
    """,
    """
    class Cell:
        def __radd__(self, other):
            return 1
    value = Cell() + Cell()
    """,
    """
    class Base:
        def __lt__(self, other):
            return True
        def __gt__(self, other):
            return 1 / 0
    class Derived(Base):
        pass
    value = Base() < Derived()
    """,
    """
    class Left:
        def __eq__(self, other):
            return 1 / 0
    class Right:
        def __ne__(self, other):
            return True
    value = Left() != Right()
    """,
    """
    class Cell:
        def __call__(self):
            return 1 / 0
    Cell()()
    """,
    """
    class Cell:
        def __eq__(self, other):
            return True
        def __repr__(self):
            return 1 / 0
    print([Cell()])
    """,
    """
    class Cell:
        def __len__(self):
            return 1 / 0
    print(1, flush=Cell())
    """,
    """
    class Cell:
        def __getattr__(self, name):
            return 1 / 0
    value = Cell().missing
    """,
    """
    class Meta(type):
        pass
    class Cell(metaclass=Meta):
        pass
    """,
    """
    class Cell:
        pass
    name = "value"
    setattr(Cell(), name, 1)
    """,
    """
    class Shift:
        def __add__(self, other):
            return 1 / 0
    value = Shift() + Shift()
    """,
    """
    class Bag:
        def __init__(self):
            self.items = [1]
        def __getitem__(self, index):
            return self.items[index]
    value = Bag()[1]
    """,
    """
    class Bag:
        def __init__(self):
            self.items = [1]
        def __setitem__(self, index, value):
            self.items[index] = value
        def __getitem__(self, index):
            return self.items[index]
    bag = Bag()
    alias = bag
    bag[0] = "wrong"
    value = alias[0] + 1
    """,
    """
    class Cell:
        def __init__(self, flag):
            if flag:
                self.value = 1
        def __add__(self, other):
            return self.value + other
    result = Cell(False) + 1
    """,
    """
    class Cell:
        def __eq__(self, other):
            return 1 / 0
    result = Cell() == Cell()
    """,
    """
    class Cell:
        def __eq__(self, other):
            return 1 / 0
    result = [Cell()] == [Cell()]
    """,
    """
    class Cell:
        def __eq__(self, other):
            return 1 / 0
    result = Cell() in [Cell()]
    """,
    """
    class Bag:
        def __iter__(self):
            return [1, 2]
    for item in Bag():
        pass
    """,
    """
    class Bag:
        def __iter__(self):
            return iter([1 / 0])
    for item in Bag():
        pass
    """,
    """
    class Bag:
        def __len__(self):
            return 1 / 0
    if Bag():
        pass
    """,
    """
    class Base:
        def __add__(self, other):
            return 1
    class Derived(Base):
        def __radd__(self, other):
            return 1 / 0
    value = Base() + Derived()
    """,
    """
    class First:
        def __add__(self, other):
            return NotImplemented
    class Second:
        def __radd__(self, other):
            return 1 / 0
    value = First() + Second()
    """,
    """
    class Bag:
        def __iter__(self):
            return iter(["bad"])
    items = [1]
    items.extend(Bag())
    value = items[1] + 1
    """,
    """
    class Bag:
        def __iter__(self):
            return iter([1])
        def __len__(self):
            return 1 / 0
    items = list(Bag())
    """,
    """
    class Bag:
        def __iter__(self):
            return iter([1 / 0])
        def __radd__(self, other):
            return [1]
    items = []
    items += Bag()
    """,
    """
    class Box:
        def __init__(self, **options):
            self.value = options.get('amount', 1)
    box = Box(amount='bad')
    result = box.value + 1
    """,
    """
    class Box:
        def __init__(self, *, value):
            self.value = value
    Box()
    """,
    """
    class Box:
        def __call__(self, *, value=1):
            return value + 1
    Box()(**{'value': 'bad'})
    """,
    """
    class Box:
        def __init__(self, **options):
            options['items'][0] = 'bad'
    values = [1]
    Box(items=values)
    result = values[0] + 1
    """,
    """
    class Cell:
        def __init__(self):
            self.value = 1
        def __call__(self, other):
            other.value = 0
            return 1 / self.value
    cells = []
    for index in range(2):
        cells.append(Cell())
    for cell in cells:
        cell(other=cell)
    """,
]


def renamed(source):
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    names |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
    names = {n: "renamed_" + n for n in names if not n.startswith("__")}
    keyword_names = {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                     for a in n.args.args + n.args.kwonlyargs}
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
    attributes |= {m.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                   for m in n.body if isinstance(m, ast.FunctionDef)}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, n):
            n.id = names.get(n.id, n.id)
            return n

        def visit_Attribute(self, n):
            self.generic_visit(n)
            if n.attr in attributes:
                n.attr = names.get(n.attr, n.attr)
            return n

        def visit_arg(self, n):
            n.arg = names.get(n.arg, n.arg)
            return n

        def visit_keyword(self, n):
            self.generic_visit(n)
            if n.arg is None and isinstance(n.value, ast.Dict):
                for key in n.value.keys:
                    if isinstance(key, ast.Constant) and key.value in keyword_names:
                        key.value = names.get(key.value, key.value)
            n.arg = names.get(n.arg, n.arg)
            return n

        def visit_FunctionDef(self, n):
            self.generic_visit(n)
            n.name = names.get(n.name, n.name)
            return n

        visit_ClassDef = visit_FunctionDef

    return ast.unparse(Rename().visit(tree))


@pytest.mark.parametrize("source", SAFE)
def test_user_protocols_safe(source):
    for program in (dedent(source), renamed(dedent(source))):
        result = prove(program, PROPS, time_budget=10)
        assert result.verdict == "TRUE", result.explain()


@pytest.mark.parametrize("source", UNSAFE)
def test_user_protocols_never_true(source):
    for program in (dedent(source), renamed(dedent(source))):
        result = prove(program, PROPS, time_budget=10)
        assert result.verdict != "TRUE", result.explain()



@pytest.mark.parametrize("statement", [
    "for value in source: pass", "values = list(source)",
    "values = []; values.extend(source)", "values = []; values[:] = source",
    "values = []; values += source", "a, b = source", "value = min(source)",
])
def test_iteration_effects_with_only_arithmetic_property(statement):
    source = "class Source:\n    def __iter__(self):\n        return iter([1 / 0])\nsource = Source()\n" + statement
    for program in (source, renamed(source)):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        result = prove(program, ("ZeroDivisionError",), time_budget=10)
        assert result.verdict != "TRUE", result.explain()


@pytest.mark.parametrize("decorator,parameters", [("staticmethod", ""), ("classmethod", "cls")])
def test_constructor_descriptor_effects(decorator, parameters):
    source = f"class Box:\n    @{decorator}\n    def __init__({parameters}):\n        1 / 0\nBox()"
    for program in (source, renamed(source)):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        result = prove(program, ("ZeroDivisionError",), time_budget=10)
        assert result.verdict != "TRUE", result.explain()


@pytest.mark.parametrize("change", [
    "cell.__dict__['value'] = 0",
    "cell.__dict__ = {'value': 0}",
])
def test_reflective_field_mutation_never_proves(change):
    source = "class Cell:\n    def __init__(self):\n        self.value = 1\ncell = Cell()\n" + change + "\nresult = 1 / cell.value"
    for program in (source, source.replace("Cell", "Parcel").replace("cell", "parcel")):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        assert prove(program, ("ZeroDivisionError",), time_budget=10).verdict != "TRUE"


def test_dynamic_receiver_class_never_proves():
    source = "class Clean:\n    def __add__(self, other):\n        return 1\nclass Bad:\n    def __add__(self, other):\n        return 1 / 0\ncell = Clean()\ncell.__class__ = Bad\nresult = cell + 1"
    for program in (source, renamed(source)):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        assert prove(program, ("ZeroDivisionError",), time_budget=10).verdict != "TRUE"


@pytest.mark.parametrize("method,call", [("__add__", "cell + 1"), ("method", "cell.method(1)")])
def test_lookup_limit_is_not_a_missing_method(method, call):
    source = f"class Base:\n    def {method}(self, other):\n        return 1 / 0\n"
    parent = "Base"
    for index in range(60):
        child = f"Child{index}"
        source += f"class {child}({parent}):\n    pass\n"
        parent = child
    source += f"cell = {parent}()\nresult = {call}"
    for program in (source, renamed(source)):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        assert prove(program, ("ZeroDivisionError",), time_budget=10).verdict != "TRUE"


@pytest.mark.parametrize("operation", ["values[key] = 1", "result = values[key]", "result = values.get(key)"])
@pytest.mark.parametrize("key", ["Key()", "(Key(),)"])
def test_equality_can_make_dictionary_keys_unhashable(operation, key):
    source = "class Key:\n    def __eq__(self, other):\n        return True\nvalues = {}\nkey = " + key + "\n" + operation
    for program in (source, renamed(source)):
        with pytest.raises(TypeError):
            exec(program, {})
        assert prove(program, ("TypeError",), time_budget=10).verdict != "TRUE"


@pytest.mark.parametrize("comparison", ["0 in values.values()", "0 in iter(values.values())"])
def test_dictionary_view_comparison_effects(comparison):
    source = "class Value:\n    def __eq__(self, other):\n        return 1 / 0\nvalues = {'x': Value()}\nresult = " + comparison
    for program in (source, renamed(source)):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        assert prove(program, ("ZeroDivisionError",), time_budget=10).verdict != "TRUE"


def test_string_mapping_protocol_is_not_reflected_modulo():
    source = "class Mapping:\n    def __getitem__(self, key):\n        return 1 / 0\n    def __rmod__(self, other):\n        return 'safe'\ntext = '%(x)s' % Mapping()"
    for program in (source, renamed(source)):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        assert prove(program, ("ZeroDivisionError",), time_budget=10).verdict != "TRUE"


@pytest.mark.parametrize("method,expression", [
    ("exchange", "cell.exchange(make())"),
    ("exchange", "cell.exchange(other=make())"),
    ("__add__", "cell + make()"),
    ("__getitem__", "cell[make()]"),
])
def test_argument_allocation_does_not_replace_an_earlier_receiver(method, expression):
    source = f"""class Cell:
    def __init__(self):
        self.value = 1
    def {method}(self, other):
        result = 1 / self.value
        other.value = 0
        return other
def make():
    return Cell()
cell = make()
for index in range(2):
    cell = {expression}
"""
    for program in (source, renamed(source)):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        result = prove(program, ("ZeroDivisionError",), time_budget=10)
        assert result.verdict != "TRUE", result.explain()


def test_later_list_element_does_not_replace_an_earlier_element():
    source = """class Cell:
    def __init__(self):
        self.value = 1
    def exchange(self, other):
        result = 1 / self.value
        other.value = 0
        return other
def make():
    return Cell()
cell = make()
for index in range(2):
    pair = [cell, make()]
    cell = pair[0].exchange(pair[1])
"""
    for program in (source, renamed(source)):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        result = prove(program, ("ZeroDivisionError",), time_budget=10)
        assert result.verdict != "TRUE", result.explain()


def test_later_argument_does_not_change_an_earlier_scalar_value():
    source = """value = 1
def clear():
    global value
    value = 0
    return 0
def divide(first, ignored):
    return 1 / (first - 1)
answer = divide(value, clear())
"""
    for program in (source, renamed(source)):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        result = prove(program, ("ZeroDivisionError",), time_budget=10)
        assert result.verdict != "TRUE", result.explain()


def test_keyword_mapping_roots_values_before_later_arguments():
    source = '''class Cell:
    def __init__(self):
        self.value = 1
def make():
    return Cell()
def exchange(*, prior, other):
    result = 1 / prior.value
    other.value = 0
    return other
packed = {'prior': make()}
for index in range(2):
    packed['prior'] = exchange(**packed, other=make())
'''
    for program in (source, source.replace('Cell', 'Box').replace('value', 'amount')
                    .replace('packed', 'arguments').replace('prior', 'earlier')):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        result = prove(program, ('ZeroDivisionError',), time_budget=10)
        assert result.verdict != 'TRUE', result.explain()
