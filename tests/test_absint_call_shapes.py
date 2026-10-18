"""Every call shape of the absint models that take arguments: the native run and absint agree on what the call does ([internal ref] siblings).

The [internal ref] wrong TRUE was a call SHAPE the model did not know (a keyword, a second positional argument) treated as an arity error
that ends the path.  This file is the sweep over every other ``_arity`` / ``kwargs`` site of ``solve/absint/interp.py`` and
``reducers.py`` (builtins, ``list``/``dict``/``str``/``array`` methods, ``math``, ``re``, ``io.StringIO``, ``array``): for each site, calls
whose shape is VALID natively (a keyword the function takes, an optional argument) and calls whose shape is a native ``TypeError`` (a
keyword a positional-only function does not take, an extra argument).

* a shape that is valid natively reaches the sentinel ``1 // 0``: absint must not say TRUE for ``ZeroDivisionError`` (a TRUE means the
  call ended the path);
* a shape that raises ``TypeError`` natively: absint must not say TRUE for ``TypeError``.

Each program is one fixed line, run natively and by absint (no generated programs); the lists were classified by a native run, which
this test repeats.  The ``random`` functions have their own file, ``tests/test_absint_random.py``: they were the only sites whose shape
handling was wrong.
"""

from __future__ import annotations

import io
import sys
import warnings

import pytest

from tharkun.solve.absint import prove

HEAD = "import math\nimport re\nimport io\nimport random\nfrom array import array\nclass A:\n    pass\n"
SENTINEL = "reached = True\nc = 1 // 0\n"

VALID = [
    'x = iter(lambda: 1, 1)',
    "x = int('10', 2)",
    "x = int('10', base=2)",
    'x = range(1, 3, 1)',
    'x = sorted([1, 2], reverse=True)',
    'x = sorted([1, 2], key=None)',
    'x = min([1, 2], default=0)',
    'x = min([1, 2], key=None)',
    'x = min(1, 2, key=None)',
    'x = max([1, 2], default=0)',
    'x = max([], default=0)',
    'x = sum([1, 2], start=1)',
    'x = sum([1, 2], 1)',
    'x = any([0, 1])',
    "x = print('a', sep=',')",
    "x = print('a', file=None)",
    'x = isinstance(1, int)',
    'x = round(1.5, ndigits=1)',
    'x = round(number=1.5)',
    'x = pow(2, 3, 5)',
    'x = pow(2, 3, mod=5)',
    'x = pow(base=2, exp=3)',
    'x = enumerate([1], start=1)',
    'x = enumerate([1], 1)',
    'x = zip([1], [2], strict=True)',
    'x = divmod(5, 2)',
    'x = dict(a=1)',
    'x = dict([(1, 2)], a=1)',
    'x = list(map(abs, [1]))',
    'x = str(1)',
    'x = str(object=1)',
    "x = str(b'a', 'utf-8')",
    'x = bool(1)',
    'x = next(iter([1]), None)',
    'x = open',
    "x = getattr([1], 'x', 3)",
    'l = [1, 2]; l.sort(key=None, reverse=True)',
    'l = [1, 2]; l.sort(reverse=True)',
    'l = [1, 2]; l.index(1, 0, 2)',
    'l = [1, 2]; l.count(1)',
    'l = [1, 2]; l.remove(1)',
    'l = [1, 2]; l.copy()',
    'l = [1, 2]; l.reverse()',
    'l = [1, 2]; l.clear()',
    'd = {1: 2}; d.get(1, 3)',
    'd = {1: 2}; d.pop(1, 3)',
    'd = {1: 2}; d.setdefault(1, 3)',
    'd = {1: 2}; d.update(a=1)',
    'd = {1: 2}; d.update({3: 4})',
    'd = {1: 2}; d.update([(3, 4)], b=2)',
    'd = {1: 2}; d.items()',
    'd = {1: 2}; d.popitem()',
    'd = {1: 2}; d.copy()',
    'd = {1: 2}; d.clear()',
    'd = {1: 2}; d.keys()',
    'd = {1: 2}; d.values()',
    'd = dict.fromkeys([1], 0)',
    "s = 'a b'; x = s.split(sep=' ')",
    "s = 'a b'; x = s.split(maxsplit=1)",
    "s = 'a b'; x = s.split(' ', 1)",
    "s = 'a b'; x = s.rsplit(maxsplit=1)",
    "s = 'a b'; x = s.replace(' ', '', count=1)",
    "s = 'a b'; x = s.replace(' ', '', 1)",
    "s = 'a b'; x = s.strip(' ')",
    "s = 'a b'; x = s.encode(encoding='utf-8')",
    "s = 'a b'; x = s.encode('utf-8', 'strict')",
    "s = 'a b'; x = s.format()",
    "s = '{a}'; x = s.format(a=1)",
    "s = '{}'; x = s.format(1)",
    "s = 'a b'; x = s.startswith('a', 0, 1)",
    "s = 'a b'; x = s.find('a', 0)",
    "s = 'a b'; x = s.join(['a'])",
    "s = 'a b'; x = s.splitlines(keepends=True)",
    "s = 'a b'; x = s.center(5, '*')",
    "s = 'a b'; x = s.expandtabs(tabsize=4)",
    "s = 'a b'; x = s.count('a', 0)",
    "s = 'a b'; x = s.index('a', 0)",
    "s = 'a b'; x = s.partition(' ')",
    "s = 'a b'; x = s.zfill(5)",
    "s = 'a b'; x = s.upper()",
    "s = 'a b'; x = s.removeprefix('a')",
    "s = 'a b'; x = s.translate({97: 98})",
    "s = 'a b'; x = s.casefold()",
    'x = math.log(8, 2)',
    'x = math.isclose(1.0, 1.0, rel_tol=0.1)',
    'x = math.pow(2.0, 3.0)',
    'x = math.hypot(3.0, 4.0)',
    'x = math.gcd(4, 6)',
    'x = math.fsum([1.0])',
    'x = math.prod([1], start=1)',
    'x = math.copysign(1.0, 2.0)',
    'x = math.atan2(1.0, 2.0)',
    'x = math.factorial(5)',
    'x = math.comb(5, 2)',
    'x = math.isqrt(4)',
    'x = math.ldexp(1.0, 2)',
    'x = math.fmod(1.0, 2.0)',
    'x = math.dist([1.0], [2.0])',
    "x = re.compile('a')",
    "x = re.compile('a', flags=0)",
    "x = re.compile(pattern='a')",
    "x = re.match('a', 'a', flags=re.I)",
    "x = re.match('a', 'a', re.I)",
    "x = re.match(pattern='a', string='a')",
    "x = re.sub('a', 'b', 'a', count=1)",
    "x = re.sub('a', 'b', 'a', 1, 0)",
    "x = re.sub(pattern='a', repl='b', string='a')",
    "x = re.split('a', 'bab', maxsplit=1)",
    "x = re.findall('a', 'bab', flags=0)",
    "x = re.search('a', 'bab', 0)",
    "x = re.fullmatch('a', 'a', flags=0)",
    "p = re.compile('a'); x = p.match('a', 0)",
    "p = re.compile('a'); x = p.match('a', pos=0)",
    "p = re.compile('a'); x = p.match(string='a')",
    "p = re.compile('a'); x = p.sub('b', 'a', count=1)",
    "p = re.compile('a'); x = p.sub(repl='b', string='a')",
    "p = re.compile('a'); x = p.split('bab', maxsplit=1)",
    "p = re.compile('a'); x = p.findall('bab', 0, 3)",
    "p = re.compile('a'); x = p.search('bab', endpos=2)",
    "p = re.compile('a'); x = p.fullmatch('a', 0, 1)",
    'x = io.StringIO()',
    "x = io.StringIO('abc')",
    "x = io.StringIO(initial_value='abc')",
    "x = io.StringIO('abc', newline=None)",
    "f = io.StringIO(); f.write('a')",
    "f = io.StringIO('a\\nb'); x = f.readline()",
    "f = io.StringIO('a\\nb'); x = f.readline(1)",
    "f = io.StringIO('a\\nb'); x = f.read()",
    "f = io.StringIO('a\\nb'); x = f.read(1)",
    "f = io.StringIO('a\\nb'); x = f.getvalue()",
    "f = io.StringIO('a\\nb'); x = f.seek(0)",
    "f = io.StringIO('a\\nb'); x = f.seek(0, 0)",
    "f = io.StringIO('a\\nb'); x = f.tell()",
    "f = io.StringIO('a\\nb'); x = f.truncate()",
    "f = io.StringIO('a\\nb'); x = f.truncate(1)",
    "f = io.StringIO('a\\nb'); x = f.readlines()",
    "f = io.StringIO('a\\nb'); x = f.readlines(1)",
    "f = io.StringIO('a\\nb'); f.close()",
    "f = io.StringIO('a\\nb'); x = f.writelines(['a'])",
    "f = io.StringIO('a\\nb'); x = f.flush()",
    "f = io.StringIO('a\\nb'); x = f.isatty()",
    "f = io.StringIO('a\\nb'); x = f.readable()",
    "a = array('i', [1]); a.append(1)",
    "a = array('i', [1]); a.extend([1])",
    "a = array('i', [1]); a.pop()",
    "a = array('i', [1]); a.pop(0)",
    "a = array('i', [1]); a.insert(0, 1)",
    "a = array('i', [1]); a.index(1)",
    "a = array('i', [1]); a.index(1, 0)",
    "a = array('i', [1]); a.count(1)",
    "a = array('i', [1]); a.remove(1)",
    "a = array('i', [1]); a.reverse()",
    "a = array('i', [1]); a.tolist()",
    "a = array('i', [1]); x = a.itemsize",
    "a = array('i', [1]); x = a.typecode",
    "a = array('i', [1]); a.fromlist([1])",
    "a = array('i', [1]); x = a.buffer_info()",
    "a = array('i', [1]); x = a.tobytes()",
    "a = array('i', [1]); a.byteswap()",
    "x = array('i')",
    "x = array('i', [1])",
    "x = array('i', b'')",
    "x = array('i', range(3))",
    "x = array('u', 'abc')",
    "x = array('i', array('i', [1]))",
    'x = float()',
    'x = int()',
    'x = list()',
    'x = tuple()',
    'x = abs(True)',
    'x = range(1, 2, 3)',
    'x = iter([1])',
    "x = len('abc')",
    'x = isinstance(1, (int, str))',
    'x = min(1, 2)',
    'x = max([1, 2], key=abs)',
    'x = sorted([3, 1, 2], key=abs, reverse=True)',
    'x = sum([1.5, 2.5])',
    "x = min('ab')",
    'x = any([])',
    'x = all([])',
    "x = print('a', end='')",
    "x = print(1, 2, sep='', end='')",
]

RAISES_TYPE_ERROR = [
    'x = iter([1], 1)',
    "x = int(x='5')",
    'x = int(5, base=2)',
    'x = float(x=1.5)',
    'x = abs(x=1)',
    'x = len(obj=[1])',
    'x = range(stop=3)',
    'x = list(iterable=[1])',
    'x = list([1], 2)',
    'x = tuple(iterable=[1])',
    'x = sorted(iterable=[1, 2])',
    'x = sorted([1, 2], None)',
    'x = sum(iterable=[1, 2])',
    'x = all(iterable=[1])',
    'x = isinstance(obj=1, class_or_tuple=int)',
    "x = ord(c='a')",
    'x = chr(i=65)',
    'x = len([1], 2)',
    'x = divmod(x=5, y=2)',
    'x = bool(x=1)',
    'x = set(iterable=[1])',
    'x = frozenset(iterable=[1])',
    'x = reversed(seq=[1])',
    'x = hash(obj=1)',
    'x = id(obj=1)',
    'x = repr(obj=1)',
    'x = next(iter([]), default=0)',
    "x = hasattr(obj=[1], name='x')",
    'l = [1, 2]; l.append(object=3)',
    'l = [1, 2]; l.insert(index=0, object=3)',
    'l = [1, 2]; l.pop(index=0)',
    'l = [1, 2]; l.sort(None)',
    'l = [1, 2]; l.extend(iterable=[3])',
    'l = [1, 2]; l.index(1, start=0)',
    'd = {1: 2}; d.get(key=1)',
    'd = {1: 2}; d.pop(key=1)',
    "s = 'a b'; x = s.strip(chars=' ')",
    "s = 'a b'; x = s.lower(1)",
    'x = math.sqrt(x=4.0)',
    'x = math.log(8, base=2)',
    'x = math.floor(x=1.5)',
    "f = io.StringIO(); f.write(s='a')",
    "f = io.StringIO('a\\nb'); x = f.read(size=1)",
    "f = io.StringIO('a\\nb'); x = f.seek(offset=0, whence=0)",
    "a = array('i', [1]); a.append(x=1)",
    "a = array('i', [1]); a.extend(iterable=[1])",
    "x = array(typecode='i')",
    "x = array('i', initializer=[1])",
    "x = array('i', [1], 3)",
    "x = array('i', 'abc')",
    'l = [1]; l.append(1, 2)',
    'l = [1]; l.append()',
    'l = [1]; l.insert(0)',
    'l = [1]; l.insert(0, 1, 2)',
    'l = [1]; l.pop(0, 1)',
    'l = [1]; l.extend([1], [2])',
    'l = [1]; l.extend()',
    'l = [1]; l.reverse(1)',
    'l = [1]; l.clear(1)',
    "a = array('i', [1]); a.append(1, 2)",
    "a = array('i', [1]); a.append()",
    "a = array('i', [1]); a.extend([1], [2])",
    "a = array('i', [1]); a.extend()",
    'd = {1: 2}; d.get()',
    'd = {1: 2}; d.get(1, 2, 3)',
    'd = {1: 2}; d.keys(1)',
    'd = {1: 2}; d.values(1)',
    'd = {1: 2}; d.items(1)',
    'x = abs()',
    'x = abs(1, 2)',
    'x = ord()',
    "x = ord('a', 'b')",
    'x = chr()',
    'x = chr(1, 2)',
    'x = len()',
    'x = float(1, 2)',
    'x = float(1.5, x=2)',
    'x = int(1, 2, 3)',
    'x = int(1, base=2, x=3)',
    'x = isinstance(1)',
    'x = isinstance(1, int, int)',
    'x = range()',
    'x = range(1, 2, 3, 4)',
    'x = iter()',
    'x = iter(1, 2, 3)',
    'x = any()',
    'x = all([1], [2])',
    'x = sum()',
    'x = sum([1], 2, 3)',
    'x = min()',
    'x = max()',
    'x = sorted()',
    'x = sorted([1], [2])',
    'x = list(1, 2)',
    'x = tuple([1], [2])',
    'x = array()',
    "x = array('i', [1], [2])",
]


def _native(src: str):
    ns: dict = {"__name__": "__main__"}
    warnings.simplefilter("ignore")
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(compile(src, "<program>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        return bool(ns.get("reached")), type(e).__name__
    finally:
        sys.stdout = old
    return False, None


def _program(line: str) -> str:
    return HEAD + line.replace("; ", "\n") + "\n" + SENTINEL


@pytest.mark.parametrize("line", VALID)
def test_a_valid_shape_never_ends_the_path(line):
    src = _program(line)
    reached, raised = _native(src)
    assert reached, ("must be valid natively", line, raised)
    assert prove(src, ["ZeroDivisionError"], filename="<program>", time_budget=20.0).verdict != "TRUE", line


@pytest.mark.parametrize("line", RAISES_TYPE_ERROR)
def test_an_invalid_shape_is_never_proven_free_of_type_error(line):
    src = _program(line)
    reached, raised = _native(src)
    assert not reached and raised == "TypeError", ("must raise TypeError natively", line, reached, raised)
    assert prove(src, ["TypeError"], filename="<program>", time_budget=20.0).verdict != "TRUE", line
