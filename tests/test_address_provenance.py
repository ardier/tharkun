"""Address provenance: generic examples, alpha renamings, and broken guarantees."""
import ast
from textwrap import dedent

import pytest

from tharkun.solve.determinism import determinism_census

SAFE = [
    """
    def banner(width):
        prefix = "ab"
        return prefix * width + "!"
    print(banner(3))
    """,
    """
    class Grid:
        def __init__(self):
            self.cells = [[1, 2], [3, 4]]
        def __getitem__(self, index):
            return self.cells[index]
        def replace(self, value):
            self.cells[0] = [value]
    grid = Grid()
    grid.replace(7)
    print(str(grid[0][0]) * 2)
    """,
    """
    class Label:
        def __init__(self):
            self.value = 4
        def __repr__(self):
            return "Label(%s)" % self.value
    print(Label())
    """,
]

SAFE += [
    "import string\nprint('unused module does not taint this text')",
    "import collections\nprint('unused module does not taint this text' * 2)",
    "print(list(iter([1, 2])))",
    "print(tuple(value for value in [1, 2]))",
    "value = 4\ndef show(value=value):\n    print(value)\nshow(1)\nshow()",
    "value = 'x'\ndef show(*, value=value):\n    print(value)\nshow(value=2)\nshow()",
    "print(str(object=17) * 3)",
    "value = 2\ndef change():\n    global value\n    value = 3\nchange()\nprint(value)",
    "class Base:\n    def __repr__(self):\n        return 'safe'\nclass Child(Base):\n    pass\nprint(Child())",
    "def show(a, b):\n    print(a, b)\nvalues = (show, 1, 'x')\nvalues[0](*values[1:])",
    "def show(*args):\n    print(args)\nshow(1, 'x')",
    "class Unrelated:\n    pass\nvalue = Unrelated()\nprint('safe' * 3)",
    "print(repr(divmod(17, 3)))",
]

UNSAFE = [
    "from array import array\ndata = array('i', [1])\naddress = data.buffer_info()[0]\n[][0] if address % 8192 == 0 else None",
    "import array\ndata = array.array('d', [1.0])\nmethod = getattr(data, 'buffer_info')\naddress = method()[0]\n[][0] if address % 8192 == 0 else None",
    "values = {}; value = values[object()]",
    "values = {}; value = values.pop(object())",
    "value = [1].index(object())",
    "def values():\n    yield 1\n    return object()\niterator = values()\nnext(iterator)\nnext(iterator)",
    "from string import Template\ntext = Template('$value').substitute(value=object())\n[][0] if text[-9] == 'a' else None",
    "from collections import UserString\ntext = UserString(object()).data\n[][0] if text[-9] == 'a' else None",
    "assert False, object()",
    "import sys\nsys.exit(object())",
    "print(iter([1]))",
    "print(reversed([1]))",
    "print(value for value in [1])",
    "def values():\n    yield 1\nprint(values())",
    "# binding_order_control\nclass C:\n    int = object\n    value = int()\nprint(C.value)",
    "# binding_order_control\nclass C:\n    value = str(object())\n    def str(value):\n        return 'clean'\nprint(C.value)",
    "value = object()\ndef show(value=value):\n    print(value)\nshow(1)\nshow()",
    "value = object()\ndef show(*, value=value):\n    print(value)\nshow(value=2)\nshow()",
    "class Index:\n    def __index__(self):\n        return 2\nprint([[object()] * Index()])",
    "class Base:\n    def __add__(self, other):\n        return object()\nclass Other:\n    pass\nclass Derived(Base, Other):\n    pass\nprint([Derived() + 1])",
    "factory = type(object()); print([factory()])",
    "factory = type(object()); values = (factory(),); print(values)",
    "text = str(object=object()); [][0] if text[-9] == 'a' else None",
    "[][0] if repr(object())[-9] == 'a' else None",
    "print(str(**{'object': object()}))",
    "value = 2\ndef change():\n    global value\n    value = object()\nchange()\nprint(value)",
    "class Base:\n    pass\nclass Child(Base):\n    pass\nprint(Child())",
    "class Base:\n    def __repr__(self):\n        return 'safe'\nclass Unsafe:\n    pass\nBase = Unsafe\nclass Child(Base):\n    pass\nprint(Child())",
    "def show(a, b):\n    print(a, b)\nvalues = (show, 1, object())\nvalues[0](*values[1:])",
    "def show(*args):\n    print(args)\nshow(1, object())",
    "def show(a=object()):\n    print(a)\nargs = ()\nshow(*args)",
    "class Label:\n    def __mul__(self, value):\n        return object()\n    __rmul__ = __mul__\nprint([2 * Label()])",
    "def set_value(box, value):\n    box[0] = value\nbox = [1]\nset_value(*(box, object()))\nprint(box)",
    "class Base:\n    def __repr__(self):\n        return 'safe'\nclass Child(Base):\n    def __str__(self):\n        return str(object())\nprint(Child())",

    "# binding_order_control\nprint(str(object()))\ndef str(value):\n    return 'clean'",
    "# binding_order_control\nvalue = id(object())\ndef id(value):\n    return 1",
    "# binding_order_control\nprint(object())\nclass object:\n    def __repr__(self):\n        return 'clean'",
    "raise Exception(object())",
    "text = b'%r' % object(); assert text != b'other'",
    """
    class Number:
        def __repr__(self):
            return "clean"
        def __iadd__(self, other):
            return object()
    value = Number()
    value += 1
    print(value)
    """,
    """
    def extend(items):
        items += [object()]
    values = [1]
    extend(values)
    print(values[0])
    """,
    """
    values = [1]
    def key(value):
        values.append(value)
        return 0
    key(1)
    min([object()], key=key)
    print(values[0])
    """,
    """
    class Start:
        def __add__(self, value):
            return object()
    print(sum([1], start=Start()))
    """,
    "values = {'key': 1}; values.update({'key': object()}); print(values['key'])",
    """
    class Label:
        def __str__(self):
            return "clean"
    print("{!r}".format(Label()))
    """,
    """
    class Label:
        def __format__(self, spec):
            return "clean"
    print(f"{Label()!s}")
    """,
    """
    class Mapping:
        def __repr__(self):
            return "clean"
        def __getitem__(self, key):
            return object()
    print("{value}".format_map(Mapping()))
    """,
    "print(object())",
    "print(str(object()) * 2)",
    "text = repr(object()); print('prefix' + text)",
    "print([object()])",
    "print(hash(object()))",
    "print(id(object()))",
    """
    class Label:
        def __str__(self):
            return "clean"
    print(repr(Label()))
    """,
    """
    class Label:
        def __str__(self):
            return "clean"
    print([Label()])
    """,
    """
    class Cell:
        def __init__(self):
            self.value = 1
        def change(self):
            self.value = object()
    cell = Cell()
    cell.change()
    print(str(cell.value) * 2)
    """,
    """
    values = [1]
    alias = values
    alias.append(object())
    print(values[0])
    """,
    """
    class Multiplier:
        def __rmul__(self, text):
            return object()
    print("safe" * Multiplier())
    """,
    """
    def convert(value):
        return str(value)
    convert(1)
    print(convert(object()) * 3)
    """,
    """
    class Label:
        def __repr__(self):
            return repr(object())
    print(Label())
    """,
    """
    class Sink:
        def write(self, text):
            print(object())
    print("hello", file=Sink())
    """,
    """
    values = [1]
    def callback(value):
        values.append(object())
        return 0
    values.sort(key=callback)
    print(values[0])
    """,
    """
    class Values:
        def __init__(self):
            self.items = [1]
        def __getitem__(self, index):
            return self.items[index]
        def change(self, value):
            self.items[0] = value
    values = Values()
    values.change(object())
    print(str(values[0]) * 2)
    """,
]

# -- [internal ref] negative controls (owner review: these must stay refused) ------------------
UNSAFE += [
    # A bare default `repr()` must still refuse even with no surrounding expression --
    # earlier drafts only caught `repr(object())` when it fed a slice/concat/assignment.
    "print(repr(object()))",

    # A comprehension variable named `hash` must not rebind the builtin outside the
    # comprehension (PEP 289: comprehensions open their own scope). If the resolver ever
    # hand-rolled this instead of trusting the symtable-checked scope model, `hash` used
    # after the comprehension would stop looking like the builtin and the denied-builtin
    # check in determinism.py would silently not fire -- a wrong TRUE, since real `hash()`
    # is PYTHONHASHSEED-dependent. See tests/test_definition_scopes.py for the header-level
    # (decorator/default) version of this same scoping guarantee, oracle-checked against
    # `symtable`; this is the plain code-body form.
    "# binding_order_control\nvalues = [hash for hash in range(3)]\nprint(hash(42))",

    # `dict(**C())` where C is a user mapping (a real `keys()` + `__getitem__` protocol,
    # not the actual `dict`/mapping-subtype fast path) must stay covered: the resulting
    # dict's values carry C's provenance and must not be silently dropped because the
    # construction went through `dict(...)` instead of a literal `{...}`.
    """
    # binding_order_control
    class C:
        def keys(self):
            return ['a']
        def __getitem__(self, key):
            return object()
    d = dict(**C())
    print(str(d['a']) * 2)
    """,
]

# -- [internal ref] uncalled-method parameter poisoning (raytrace root cause) ------------------
# An uncalled method's never-bound parameter must stay EMPTY, not UNKNOWN: it never
# runs, so it must not poison the flat ("attr", name) bucket a DIFFERENT, live-called
# method also writes through the same attribute name. Found via bm_raytrace_det.py
# (SV-Benchmarks): Scene.moveTo(self, p) is never called; its "self.position = p"
# was making every read of .position UNKNOWN, including Scene.render's, which calls
# nothing but the constructor's self.position = Point(...).
SAFE += [
    """
    class Box:
        def __init__(self):
            self.value = 1
        def unused_setter(self, v):
            self.value = v
    b = Box()
    print(str(b.value) * 2)
    """,
    # The raytrace shape: the uncalled method's target attribute name is shared with
    # an UNRELATED class, so a flat-namespace collision alone must not be required
    # to trip this -- the uncalled parameter is the whole story.
    """
    class Vector:
        def __init__(self, x):
            self.x = x
    class Mover:
        def __init__(self):
            self.pos = Vector(1)
        def moveTo(self, p):
            self.pos = p
    m = Mover()
    print(str(m.pos.x) * 2)
    """,
]

UNSAFE += [
    # NEGATIVE CONTROL: same shape as the SAFE case above, but the unsafe value
    # reaches .value through a REAL, live write -- must still refuse.
    """
    class Box:
        def __init__(self):
            self.value = 1
        def unused_setter(self, v):
            self.value = v
    b = Box()
    b.value = object()
    print(str(b.value) * 2)
    """,
    # NEGATIVE CONTROL: the setter is NOT uncalled here -- it IS invoked, with an
    # unsafe argument -- must still refuse (self.invoked must not blanket-clear
    # a function's parameters just because SOME OTHER function is uncalled).
    """
    class Box:
        def __init__(self):
            self.value = 1
        def setter(self, v):
            self.value = v
    b = Box()
    b.setter(object())
    print(str(b.value) * 2)
    """,
]


# These two constants are scalar values, not permission to trust their module.
UNSAFE += [
    "from string import digits as bool\nprint(bool[2])",
    "import string\nstring = object()\nprint(string.digits)",
    "import string as alphabet\nalphabet = object()\nprint(alphabet.ascii_uppercase)",
    "from string import digits\ndigits = object()\nprint(digits)",
    "from string import ascii_uppercase as letters\nletters = object()\nprint(letters)",
    "import string\nstring.digits = object()\nprint(string.digits)",
    "import string\ndel string.ascii_uppercase\nprint(string.ascii_uppercase)",
    "import string\nsetattr(string, 'digits', object())\nprint(string.digits)",
    "import string\nprint(getattr(string, 'digits'))",
    "import string\nmodule = string\nprint(module.digits)",
    "import string\ndef read(module): return module.digits\nprint(read(string))",
    "from string import *\nprint(digits)",
    "class string:\n    digits = object()\nprint(string.digits)",
    "digits = object()\nprint(digits)",
    "import collections as string\nprint(string.UserString(object()))",
    "from string import Template as digits\nprint(digits('$x').substitute(x=object()))",
    "from string import digits\nprint(digits + str(object()))",
    "import string\ntext = string.digits + repr(object())\nassert text[-9] != 'a'",
    "import string\nprint(string.ascii_lowercase)",
    "import string\nprint(string.hexdigits)",
    "import string\nprint(string.printable)",
    "import string\nprint(string.Formatter().format('{0}', object()))",
    "import collections\nprint(collections.Counter('a'))",
    "import collections\nprint(collections.defaultdict(int))",
    "import collections\nprint(collections.deque([1]))",
]

SAFE += [
    "import heapq\nvalues = [2, 1]\nheapq.heapify(values)",
    "import string\nprint(string.digits)",
    "import string\nprint(string.ascii_uppercase)",
    "import string as alphabet\nprint(alphabet.digits + alphabet.ascii_uppercase)",
    "from string import digits\nprint(digits[3])",
    "from string import ascii_uppercase as letters\nprint(letters[1:3])",
    "from string import digits as numbers, ascii_uppercase as letters\ntext = numbers + letters\nprint(text[15])",
    "import string\ndef text(): return string.digits + string.ascii_uppercase\nprint(text()[2])",
    "from string import ascii_uppercase as _digits\nprint(_digits[2:4])",
]


def renamed(source):
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    names |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
    names = {name: "renamed_" + name for name in names if not name.startswith("__") and name not in {"write", "flush"}}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = names.get(node.id, node.id)
            return node

        def visit_Attribute(self, node):
            self.generic_visit(node)
            node.attr = names.get(node.attr, node.attr)
            return node

        def visit_Global(self, node):
            node.names = [names.get(name, name) for name in node.names]
            return node

        def visit_keyword(self, node):
            self.generic_visit(node)
            node.arg = names.get(node.arg, node.arg)
            return node

        def visit_arg(self, node):
            node.arg = names.get(node.arg, node.arg)
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.name = names.get(node.name, node.name)
            return node

        visit_ClassDef = visit_FunctionDef

    return ast.unparse(Rename().visit(tree))


@pytest.mark.parametrize("source,expected", [(s, True) for s in SAFE] + [(s, False) for s in UNSAFE])
def test_address_provenance_and_renaming(source, expected):
    source = dedent(source)
    variants = (source,) if "binding_order_control" in source else (source, renamed(source))
    for program in variants:
        result = determinism_census(program)
        assert result.ok is expected, (program, result.reason)

@pytest.mark.parametrize("source,expected", [(s, "true") for s in SAFE[:2]] + [
    ("print(object())", "unknown"),
    ("print(str(object()) * 2)", "unknown"),
])
def test_certificate_controls_verifier(tmp_path, source, expected):
    from tharkun.sv.engine import verify

    path = tmp_path / "program.py"
    path.write_text(dedent(source))
    verdict = verify(str(path), ("TypeError",), budget_seconds=10, jobs=1, prove=False)
    assert verdict.verdict == expected, verdict


DEFAULTDICT_SAFE = [
    "from collections import defaultdict\nd = defaultdict(int)\nprint(d[7])",
    "import collections as c\nd = c.defaultdict(int)\na = d\na[1] = d[2] + 3\nprint(d[1])",
    "from collections import defaultdict as mapping\nd = mapping(int)\nk = (1, 2)\nd[k] = 4\nprint(d[2, 3])",
    "from collections import defaultdict\nd = defaultdict(int)\nd[()] = 1\nx = d[()]",
    """
    from collections import defaultdict
    def total(limit, entries):
        memo = defaultdict(int)
        alias = memo
        for i in range(1, len(entries) + 1):
            weight, amount = entries[i - 1]
            for j in range(1, limit + 1):
                alias[i, j] = max(memo[i - 1, j], amount + memo[i - 1, j - weight])
        return memo[len(entries), limit]
    def check(data, expected):
        assert total(*data) == expected
    check([3, [[1, 4], [2, 5]]], 9)
    check([2, [[1, 2]]], 2)
    """,
    """
    from collections import defaultdict
    def use(key=1, /, *, value=2):
        d = defaultdict(int)
        d[key] = value
        print(d[key - 1])
    use()
    use(*[3], value=4)
    """,
]


@pytest.mark.parametrize("source", DEFAULTDICT_SAFE)
def test_integer_defaultdict_and_renaming(source):
    source = dedent(source)
    for program in (source, renamed(source)):
        result = determinism_census(program)
        assert result.ok, (program, result.reason)


@pytest.mark.parametrize("statement", [
    "d.default_factory", "d.default_factory = float", "d.default_factory = int",
    "getattr(d, 'default_factory')", "setattr(d, 'default_factory', int)",
    "d.get(1)", "d.setdefault(1, 2)", "d.update({1: 2})", "method = d.get",
    "iter(d)", "len(d)", "1 in d", "d |= {1: 2}", "d[1] += 2", "del d[1]",
    "box = [d]", "box = {'value': d}", "box = (d,)",
    "def consume(value): pass\nconsume(d)",
    "def expose(): return d\nexpose()",
    "def read(): return d[1]\nread()",
    "class Box: pass\nbox = Box()\nbox.value = d",
    "alias = d\nalias.default_factory = object",
    "alias = d\nalias[object()] = 1",
    "d = {}\nd[1] = 2",
    "repr(d)", "repr([d])", "repr({'nested': [d]})", "str(d)", "format(d)",
    "f'{d!r}'", "'{}'.format(d)", "Exception(d)",
])
def test_integer_defaultdict_refuses_unsupported_uses_without_text_sink(statement):
    source = "from collections import defaultdict\nd = defaultdict(int)\n" + statement
    for program in (source, renamed(source)):
        assert not determinism_census(program).ok, program


@pytest.mark.parametrize("expression", [
    "True", "1.0", "object()", "1 / 2", "not 1", "2 ** -1",
    "(1.5).real", "(1j).imag", "[1, 2]", "{1, 2}", "(1, True)", "(1, (2,))",
    "tuple([1, 2])", "max(1, 2, key=repr)", "max([1, 2])", "int('1')",
])
@pytest.mark.parametrize("use", ["d[value] = 1", "d[0] = value", "d[value]"])
def test_integer_defaultdict_rejects_inexact_producers_through_aliases(expression, use):
    source = ("from collections import defaultdict\nd = defaultdict(int)\n"
              f"original = {expression}\nvalue = original\n" + use)
    for program in (source, renamed(source)):
        assert not determinism_census(program).ok, program


@pytest.mark.parametrize("producer", [
    "value = 1\nvalue /= 2", "value = 2\nvalue **= -1", "value = 1\nvalue += 2",
    "value = 1\nvalue = 1 / 2", "value = missing", "value = value",
    "class Count:\n    def __len__(self): return 1\nvalue = len(Count())",
    "class Index:\n    def __index__(self): return 1\nvalue = range(Index())",
    "def choose(x): return x\nvalue = choose(1 / 2)",
    "value = max(1, 1 / 2)",
])
def test_integer_defaultdict_all_writes_and_callbacks_remain_unknown(producer):
    source = ("from collections import defaultdict\nd = defaultdict(int)\n"
              + producer + "\nd[value] = 1")
    for program in (source, renamed(source)):
        assert not determinism_census(program).ok, program


@pytest.mark.parametrize("calls", [
    "use(1)\nuse(1 / 2)", "use()", "use(*[1 / 2])", "use(**{'key': 1 / 2})",
])
def test_integer_defaultdict_joins_all_arguments_and_defaults(calls):
    source = ("from collections import defaultdict\n"
              "def use(key=1 / 2):\n    d = defaultdict(int)\n    d[key] = 1\n" + calls)
    for program in (source, renamed(source)):
        assert not determinism_census(program).ok, program


@pytest.mark.parametrize("source", [
    "from collections import defaultdict\nd = defaultdict(float)\nd[1]",
    "from collections import defaultdict\nfactory = int\nd = defaultdict(factory)\nd[1]",
    "from collections import defaultdict\nmake = defaultdict\nd = make(int)\nd[1]",
    "from collections import defaultdict\nd = defaultdict(int, {1: 2})\nd[1]",
    "from collections import defaultdict\nd = defaultdict(int, value=2)\nd[1]",
    "from collections import defaultdict\ndef use(int):\n    d = defaultdict(int)\n    d[1]\nuse(float)",
    "from collections import defaultdict\nclass Mapping(defaultdict): pass\nd = Mapping(int)\nd[1]",
    "from collections import defaultdict\ndef use():\n    global d\n    d = defaultdict(int)\nuse()",
])
def test_integer_defaultdict_requires_direct_factory_and_local_ownership(source):
    assert not determinism_census(source).ok, source


def test_integer_defaultdict_certificate_does_not_hide_missing_zero(tmp_path):
    from tharkun.sv.engine import verify

    source = "from collections import defaultdict\nd = defaultdict(int)\nvalue = 1 // d[7]"
    assert determinism_census(source).ok
    path = tmp_path / "missing_zero.py"
    path.write_text(source)
    result = verify(str(path), ("ZeroDivisionError",), budget_seconds=10, jobs=1, prove=False)
    assert result.verdict == "false", result
