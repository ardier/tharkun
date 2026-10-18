"""``site_census`` -- can ANY construct of a program raise one of these exceptions? ([internal ref])

SOLVING, by ``docs/LAYERS.md``'s rule: the input is a program's source and a
set of exception class names, the output a claim computed from the AST alone.
The claim is ``no_site``: no expression, statement or library call anywhere in
the program -- reachable or not, with any values -- can raise an instance of
those classes.  Then ``! uncaught(E)`` holds for every execution, whatever the
draws: the proof ``proof:no-site`` of ``tharkun.sv``.

It is a census, not a value analysis, so it is sound by construction when its
tables are: every place that CAN raise counts as a site, whatever the operands.
The exceptions it knows (``SITE_KINDS``) and their sites:

* ``ZeroDivisionError`` -- ``/``, ``//``, ``%`` and ``**`` (and their augmented
  forms) unless the right operand is a literal that makes them safe (a nonzero
  number; for ``**`` a non-negative one; ``%`` with a string literal on the
  left is formatting); ``divmod``, ``pow``, ``math.log`` with a base, a
  ``log`` method, and the ``random``
  functions that divide by an argument; a numeric dunder called by name;
  ``_sv_verifier.check_type`` with a closed hint (the value may run callbacks).
* ``FloatingPointError`` -- only an explicit ``raise`` (CPython raises it from
  nowhere else since ``fpectl`` was removed).
* ``AssertionError`` -- ``assert`` (unless its test is a true literal), an
  explicit ``raise``.
* ``KeyError`` / ``IndexError`` -- every subscript (for ``IndexError`` alone, a
  plain slice is exempt; a slice is a hashable dict key since 3.12),
  ``del x[k]``, the methods that look up or remove (``pop``, ``popitem``,
  ``remove``, ``group``, ``format``, ...), ``random.choice`` / ``sample`` /
  ``choices``, ``%``-formatting with a mapping key, ``str.format`` calls.

Everywhere, an explicit ``raise`` of a property class, of a program class that
derives from one, or of an expression the census cannot name (not a builtin
class, not a caught exception being re-raised) is a site.

It is sound only as an ALLOWLIST (lane F reached ``__floordiv__`` through
``__getattribute__`` and ``int.__dict__`` behind the first version's back): the
program must RESOLVE (``solve/pyresolve.py::Resolver``) -- every import in
:data:`MODULES`, every attribute one the program binds or an allowlisted type or
module has, nothing reflective (dunder attributes, ``getattr`` with a computed
name, ``vars`` / ``globals`` / ``__builtins__``, ``eval`` / ``exec`` /
``__import__``) -- or the census REFUSES.  It also refuses an exception class
it has no table for (``TypeError``, ``AttributeError``, ...).  A site-raising
builtin or library callable used as a VALUE (``f = divmod``) counts as a site.
Defs never referenced by name are dead (reflection being refused, nothing else
can call them) and are skipped.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass

from tharkun.solve.contracts.adapters import obligations as _contracts
from tharkun.solve.check_type_hints import closed_hint_effects
from tharkun.solve.pyresolve import Resolver, Top

#: library modules the census has tables for; other imports refuse
MODULES = frozenset({"_sv_verifier", "math", "random", "re", "io", "array", "string",
                     "os", "os.path", "typing"})

SITE_KINDS = ("ZeroDivisionError", "FloatingPointError", "AssertionError", "KeyError", "IndexError")

_DYNAMIC = frozenset({"eval", "exec", "compile", "__import__", "globals", "vars", "__builtins__",
                      "breakpoint"})

_NUMERIC_DUNDERS = frozenset({
    "__truediv__", "__rtruediv__", "__itruediv__", "__floordiv__", "__rfloordiv__",
    "__ifloordiv__", "__mod__", "__rmod__", "__imod__", "__divmod__", "__rdivmod__",
    "__pow__", "__rpow__", "__ipow__"})

#: callables (builtin names, ``module.function`` names, or method names on any object) that may
#: raise each class.  A method name counts whatever object it is called on.
_ZERO_DIV_CALLS = frozenset({
    "divmod", "pow", "math.log", "log",
    "triangular", "expovariate", "paretovariate", "weibullvariate", "betavariate",
    "gammavariate", "vonmisesvariate", "lognormvariate", "normalvariate", "gauss",
    "binomialvariate", "choices", "sample"}) | _NUMERIC_DUNDERS
_LOOKUP_CALLS = frozenset({
    "pop", "popitem", "remove", "group", "groups", "groupdict", "start", "end", "span",
    "expand", "format", "format_map", "substitute", "choice", "choices", "sample",
    "__getitem__", "__delitem__", "__setitem__", "__missing__", "heappop", "heapreplace",
    "setdefault", "getrandbits", "fromkeys", "vformat", "get_value", "get_field", "sub", "subn"})

#: raising INTO a generator or coroutine: any class can come out
_THROW_CALLS = frozenset({"throw", "athrow"})
_CHECK_TYPE_CALLS = frozenset({"_sv_verifier.check_type"})

_CALL_SITES = {
    "ZeroDivisionError": _ZERO_DIV_CALLS | _CHECK_TYPE_CALLS,
    "FloatingPointError": _CHECK_TYPE_CALLS,
    "AssertionError": _CHECK_TYPE_CALLS,
    "KeyError": _LOOKUP_CALLS | _CHECK_TYPE_CALLS,
    "IndexError": _LOOKUP_CALLS | _CHECK_TYPE_CALLS,
}


@dataclass(frozen=True)
class Census:
    no_site: bool                                  # True: no construct can raise the classes
    reason: str = ""                               # why not (first site, or the refusal)
    sites: tuple[tuple[int, int, str], ...] = ()   # (line, col, what) -- the first few
    #: (line, col, Obligation) for sites a TESTED contract explains precisely ([internal ref] step 2).
    obligations: tuple[tuple[int, int, "_contracts.Obligation"], ...] = ()


class _Refuse(Exception):
    pass


def _builtin_exception(name: str):
    obj = getattr(builtins, name, None)
    return obj if isinstance(obj, type) and issubclass(obj, BaseException) else None


def _bound(node: ast.AST) -> list[str]:
    """Names (re)bound by ``node``."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return [node.id]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.ExceptHandler) and node.name:
        return [node.name]
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        return [node.name]
    return []


def site_census(source: str, classes, filename: str = "<program>", limit: int = 8,
                *, declared_files=()) -> Census:
    """Is there no site for any of ``classes`` in ``source``?  (See the module docstring.)

    [internal ref] (read-and-recurse): an import ``Resolver`` follows into code bundled with the task
    (``solve/extern.py``, ``pyresolve.py::Resolver``) is program code too (condition (d)) --
    every file ``res.all_resolvers()`` reaches (this program's own, plus every recursively-
    read one) is scanned for sites the same way, sharing one ``user_classes`` table across
    all of them so a property class DEFINED in one file and RAISED in another (e.g. a custom
    exception in a package sibling) still classifies correctly."""
    classes = tuple(classes)
    unknown = [c for c in classes if c not in SITE_KINDS]
    if unknown:
        return Census(False, f"no site table for {', '.join(unknown)}")
    try:
        tree = ast.parse(source, filename)
    except (SyntaxError, ValueError) as e:
        return Census(False, f"parse: {e}")
    try:
        # [internal ref] (read-and-recurse): scan EVERY file all_resolvers() reaches (this
        # program's own, plus every recursively-read one), not just the top-level
        # tree -- condition (d): imported code is program code, so its own sites
        # count too. user_classes is built across ALL of them first (a class DEFINED
        # in one recursively-read file and RAISED in another must still classify),
        # then each file gets its own _Scan (its own filename, for correct
        # per-file attribution -- census.py's own _enclosing_function/obligation
        # file_name -- matching what a single-file _Scan always did).
        res = Resolver(tree, MODULES, filename=filename, _declared_files=declared_files)
        resolvers = res.all_resolvers()
        user_classes: dict[str, ast.ClassDef] = {}
        for sub in resolvers:
            for node in ast.walk(sub.tree):
                if isinstance(node, ast.ClassDef):
                    user_classes[node.name] = node
        sites: list[tuple[int, int, str]] = []
        obligations: list[tuple[int, int, "_contracts.Obligation"]] = []
        for sub in resolvers:
            scan = _Scan(sub.tree, classes, sub, sub.filename, user_classes)
            sites.extend(scan.run())
            obligations.extend(scan.obligations)
    except Top as e:
        return Census(False, f"does not resolve: {e}")
    except _Refuse as e:
        return Census(False, str(e))
    if sites:
        line, col, what = sites[0]
        return Census(False, f"line {line}: {what}", tuple(sites[:limit]), tuple(obligations[:limit]))
    return Census(True, f"no construct can raise {', '.join(classes)}")


class _Scan:
    def __init__(self, tree: ast.Module, classes: tuple[str, ...], res: Resolver,
                 filename: str = "<program>",
                 user_classes: dict[str, ast.ClassDef] | None = None):
        self.tree = tree
        self.res = res
        self.filename = filename
        self.classes = classes
        self.want = {c: _builtin_exception(c) for c in classes}
        self.calls = frozenset().union(_THROW_CALLS, *(_CALL_SITES[c] for c in classes))
        self.zero_div = "ZeroDivisionError" in classes
        self.lookup = "KeyError" in classes or "IndexError" in classes
        self.asserts = "AssertionError" in classes
        self.modules: set[str] = set()
        self.from_names: dict[str, str] = {}     # local name -> "module.name"
        # [internal ref]: shared across every file one site_census() call scans (see site_census's own
        # docstring) so a class DEFINED in one recursively-read file and RAISED in another
        # still classifies; defaults to a fresh, single-file table for any other caller
        # (obligation_census) that still scans exactly one tree, unchanged.
        self.user_classes: dict[str, ast.ClassDef] = user_classes if user_classes is not None else {}
        self.sites: list[tuple[int, int, str]] = []
        self.nodes: list[ast.AST] = []
        self.obligations: list[tuple[int, int, "_contracts.Obligation"]] = []

    def _enclosing_function(self, node: ast.AST) -> str:
        """Nearest enclosing ``def``/``async def`` name, or ``"<module>"`` at top
        level -- a location label, not OB's fuller dotted class.method scheme
        ([internal ref] slice 0's ``source_points``), which this doesn't attempt to
        replicate."""
        p = self.parent.get(node)
        while p is not None:
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return p.name
            p = self.parent.get(p)
        return "<module>"

    def _record_obligation(self, node: ast.Call, qualname: str) -> None:
        """A checked, witness-format safety obligation for the call site just
        recorded ([internal ref] step 2 redesign) -- a ``python_expression``
        ``location_invariant``, or an explicit :class:`Gap`, never a dropped or
        English-text claim. Only ``ast.Call`` sites with an exactly resolved
        qualname reach here (a bare name "used as a value" has no call
        arguments to build a safety expression from, so it is skipped, not
        guessed)."""
        ob = _contracts.obligation_for_call(
            qualname, node, file_name=self.filename, function=self._enclosing_function(node),
            classes=self.classes)
        if ob is not None:
            self.obligations.append((node.lineno, node.col_offset, ob))

    def site(self, node: ast.AST, what: str) -> None:
        self.sites.append((getattr(node, "lineno", 0), getattr(node, "col_offset", 0), what))
        self.nodes.append(node)

    # -- tables ----------------------------------------------------------------
    def run(self) -> list[tuple[int, int, str]]:
        # [internal ref]: import resolution is `self.res` (a pyresolve.py::Resolver, already built and
        # already checked -- Resolver.__init__ raises Top on anything that does not resolve,
        # INCLUDING via read-and-recurse) -- this no longer re-derives or re-checks it. A
        # local name resolved via read-and-recurse (``res.externs``) is simply not in
        # MODULES's own tables (no call/site name ever matches it, since those tables are all
        # ``stdlib_module.func``-shaped), which is correct: this census reasons about calls to
        # KNOWN dangerous stdlib entry points, never about an imported program's own code --
        # THAT code is scanned in its own pass, as its own tree, by site_census's caller.
        self.modules = set(self.res.imports.modules.keys())
        self.from_names = {alias: f"{mod}.{attr}" for alias, (mod, attr) in self.res.imports.names.items()
                           if not mod.startswith("$extern:")}
        parent = {}
        for node in ast.walk(self.tree):
            for ch in ast.iter_child_nodes(node):
                parent[ch] = node
            if isinstance(node, ast.ClassDef):
                self.user_classes[node.name] = node
        self.bound: dict[str, int] = {}
        for node in ast.walk(self.tree):
            for name in _bound(node):
                self.bound[name] = self.bound.get(name, 0) + 1
        self.parent = parent
        for node in ast.walk(self.tree):
            if not self.res.live(node):
                continue
            self.res.check(node)              # anything unresolved or reflective: Top
            self.visit(node)
        return self.sites

    def raises_property(self, name: str, seen=()) -> bool | None:
        """Does class ``name`` derive from a property class?  None: cannot tell."""
        exc = _builtin_exception(name)
        if exc is not None and not self.bound.get(name):
            return any(w is not None and issubclass(exc, w) for w in self.want.values())
        cls = self.user_classes.get(name)
        if cls is None or name in seen or self.bound.get(name) != 1:
            return None
        verdicts = []
        for b in cls.bases:
            if not isinstance(b, ast.Name):
                return None
            verdicts.append(self.raises_property(b.id, seen + (name,)))
        if any(v is None for v in verdicts):
            return None
        return any(verdicts)

    # -- one node --------------------------------------------------------------
    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name) and node.id in _DYNAMIC:
            raise _Refuse(f"line {node.lineno}: uses {node.id}")
        if isinstance(node, ast.Name) and node.id in self.modules:
            par = self.parent.get(node)
            if not (isinstance(par, ast.Attribute) and par.value is node):
                raise _Refuse(f"line {node.lineno}: module {node.id} used as a value")
        if isinstance(node, ast.Raise) and node.exc is not None:
            self.visit_raise(node)
        if self.asserts and isinstance(node, ast.Assert):
            if not (isinstance(node.test, ast.Constant) and node.test.value):
                self.site(node, "assert")
        if self.zero_div and isinstance(node, (ast.BinOp, ast.AugAssign)):
            self.visit_arith(node)
        if self.lookup:
            self.visit_lookup(node)
        if isinstance(node, ast.Call):
            self.visit_call(node)
        elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            # a site-raising callable used as a value can be called from anywhere
            name = self.callable_name(node)
            par = self.parent.get(node)
            if name is not None and not (isinstance(par, ast.Call) and par.func is node) \
                    and not (isinstance(par, ast.Attribute) and par.value is node) \
                    and self.is_site_callable(name, node):
                self.site(node, f"{name} used as a value")
                # No call arguments exist at a bare reference site, so there is
                # nothing to build a safety expression from -- no obligation call.

    def visit_raise(self, node: ast.Raise) -> None:
        exc = node.exc
        target = exc.func if isinstance(exc, ast.Call) else exc
        if isinstance(target, ast.Name):
            if not isinstance(exc, ast.Call) and self.reraises(node, target.id):
                return                             # re-raising the exception being handled
            verdict = self.raises_property(target.id)
            if verdict is False:
                return
            self.site(node, f"raise {target.id}" + ("" if verdict else " (unknown class)"))
            return
        if isinstance(exc, (ast.Constant, ast.JoinedStr)) and not self.want.get("TypeError"):
            return                                 # raising a str/number raises TypeError
        self.site(node, f"raise {ast.unparse(exc)[:40]} (unknown class)")

    def reraises(self, node: ast.Raise, name: str) -> bool:
        """``raise NAME`` inside the ``except ... as NAME`` handler that binds it, with no
        other binding of ``NAME`` in the program: the exception being handled, re-raised."""
        if self.bound.get(name) != 1:
            return False
        p = self.parent.get(node)
        while p is not None:
            if isinstance(p, ast.ExceptHandler) and p.name == name:
                return True
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                return False
            p = self.parent.get(p)
        return False

    def visit_arith(self, node: ast.AST) -> None:
        op = node.op
        right = node.right if isinstance(node, ast.BinOp) else node.value
        left = node.left if isinstance(node, ast.BinOp) else None
        num = right.value if isinstance(right, ast.Constant) and not isinstance(
            right.value, (bool, str, bytes)) and right.value is not None else None
        if isinstance(op, (ast.Div, ast.FloorDiv)):
            if not (isinstance(num, (int, float, complex)) and num != 0):
                self.site(node, "division")
        elif isinstance(op, ast.Mod):
            if isinstance(left, (ast.Constant, ast.JoinedStr)) and isinstance(
                    getattr(left, "value", ""), (str, bytes)):
                return                             # string formatting
            if not (isinstance(num, (int, float)) and num != 0):
                self.site(node, "modulo")
        elif isinstance(op, ast.Pow):
            if not (isinstance(num, (int, float)) and num >= 0):
                self.site(node, "power")

    def visit_lookup(self, node: ast.AST) -> None:
        # A slice never raises IndexError, but slices are hashable (3.12+): ``d[a:b]`` on a
        # dict raises KeyError.
        if isinstance(node, ast.Subscript) and (
                not isinstance(node.slice, ast.Slice) or "KeyError" in self.classes):
            self.site(node, "subscript")
        op = getattr(node, "op", None)
        if isinstance(op, ast.Mod) and isinstance(node, (ast.BinOp, ast.AugAssign)):
            left = node.left if isinstance(node, ast.BinOp) else None
            if not (isinstance(left, ast.Constant) and not isinstance(left.value, bytes)
                    and "%(" not in str(left.value)):
                # a string on the left may be %-formatting with a mapping key: KeyError
                self.site(node, "modulo (may be %-formatting with a mapping)")

    def callable_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            if node.id in self.from_names:
                return self.from_names[node.id]
            return node.id
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in self.modules:
                module = self.res.imports.modules.get(node.value.id, node.value.id)
                return f"{module}.{node.attr}"
            return node.attr
        return None

    def is_site_callable(self, name: str, node: ast.AST) -> bool:
        short = name.rsplit(".", 1)[-1]
        if isinstance(node, ast.Name) and "." not in name and name not in dir(builtins):
            return False                           # a program name, not a library callable
        return name in self.calls or short in self.calls

    def visit_call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name) and f.id in ("getattr", "setattr", "delattr", "hasattr"):
            if len(node.args) < 2 or not (isinstance(node.args[1], ast.Constant)
                                          and isinstance(node.args[1].value, str)):
                raise _Refuse(f"line {node.lineno}: {f.id} with a computed name")
            if node.args[1].value in self.calls:
                self.site(node, f"{f.id}(..., {node.args[1].value!r})")
            return
        name = self.callable_name(f)
        if name is None:
            return                                 # calling a computed value: see "used as a value"
        if name == "_sv_verifier.check_type":
            if (len(node.args) != 2 or node.keywords or
                    any(isinstance(arg, ast.Starred) for arg in node.args) or
                    closed_hint_effects(node.args[1], self.res) is None):
                raise _Refuse(f"line {node.lineno}: check_type hint or call shape is not closed")
        if name == "math.log" and len(node.args) == 1 and not node.keywords \
                and not isinstance(node.args[0], ast.Starred):
            return                                 # one-argument log: ValueError, never a division
        if self.is_site_callable(name, f):
            self.site(node, f"call {name}")
            self._record_obligation(node, name)




def _is_literal(node: ast.AST) -> bool:
    """Is node a literal constant with no runtime execution or calls?"""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
        return isinstance(node.operand.value, (int, float, complex))
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_literal(k) and _is_literal(v) for k, v in zip(node.keys, node.values))
    return False


def _scan_unmodelled(tree: ast.AST, res: Resolver):
    """Yield (node, method_name, description) for unmodelled implicit calls / definition-time evals."""
    for node in ast.walk(tree):
        if not res.live(node):
            continue
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg is None and not (isinstance(kw.value, ast.Dict) and all(k is not None and _is_literal(k) for k in kw.value.keys)):
                    yield (kw.value, "keys", f"unmodelled implicit call: keys() from mapping unpacking (**{ast.unparse(kw.value)})")
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is None and not (isinstance(v, ast.Dict) and all(k2 is not None and _is_literal(k2) for k2 in v.keys)):
                    yield (v, "keys", f"unmodelled implicit call: keys() from dictionary display unpacking (**{ast.unparse(v)})")
        elif isinstance(node, ast.Starred):
            if not (isinstance(node.value, (ast.Tuple, ast.List)) and all(_is_literal(elt) for elt in node.value.elts)):
                yield (node.value, "__iter__", f"unmodelled implicit call: __iter__() from sequence unpacking (*{ast.unparse(node.value)})")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                yield (item.context_expr, "__enter__", f"unmodelled implicit call: context manager ({ast.unparse(item.context_expr)})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                yield (dec, "decorator", f"unmodelled definition-time decorator: @{ast.unparse(dec)}")
            if getattr(node, "type_params", ()):
                yield (node, "type_params", "unmodelled definition-time type parameters")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None and not _is_literal(default):
                    yield (default, "default", f"unmodelled definition-time default: {ast.unparse(default)}")

def obligation_census(source: str, classes, filename: str = "<program>", *, declared_files=()):
    """Untruncated, class-specific witness predicates plus explicit coverage gaps.

    This reuses the census's audited coverage boundary; it is not a second proof
    of that boundary. The caller must check the predicates against source state.
    Unsupported safety rules remain open. Site IDs use AST paths, not line text.
    """
    import hashlib
    from tharkun.solve.obligations import Invariant, Obligation, ObligationCensus, source_points

    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    obligations, gaps = [], []

    def gap(exc, reason):
        gaps.append(f"{exc}: {reason}")
        obligations.append(Obligation(sha, "$coverage", exc,
            Invariant(filename, 1, 1, "<module>", "False"), reason))

    try:
        tree = ast.parse(source, filename)
    except (SyntaxError, ValueError) as e:
        for exc in sorted(set(classes)):
            gap(exc, f"parse: {e}")
        return ObligationCensus(tuple(obligations), tuple(gaps))
    points = source_points(tree, source, filename)
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    bound_builtins = {name for n in ast.walk(tree) for name in _bound(n)} & {"type", "len", "int", "float", "bool", "list", "tuple", "str", "bytes", "dict"}
    try:
        res = Resolver(tree, MODULES, filename=filename, _declared_files=declared_files)
    except (Top, _Refuse):
        res = None

    # shortcut: imported-file obligation identities omit filename; keep coverage open until identities are extended.
    has_imported_resolvers = bool(res is not None and len(res.all_resolvers()) > 1)

    for exc in sorted(set(classes)):
        if exc not in SITE_KINDS:
            gap(exc, f"no site table for {exc}")
            continue
        if has_imported_resolvers:
            gap(exc, "imported-module obligation locations are not represented")
        try:
            scan = _Scan(tree, (exc,), res or Resolver(
                tree, MODULES, filename=filename, _declared_files=declared_files))
            scan.run()
        except (Top, _Refuse) as e:
            gap(exc, f"coverage refused: {e}")
            continue
        seen = set()
        for node in scan.nodes:
            path, location = points[node]
            if path in seen:
                continue
            seen.add(path)
            value, reason = _safety_predicate(node, exc, parent)
            if bound_builtins:
                value, reason = "False", f"predicate builtins rebound: {', '.join(sorted(bound_builtins))}"
            from tharkun.solve.predicates import InvalidPredicate, parse_predicate
            try:
                parse_predicate(value)
            except InvalidPredicate as e:
                value, reason = "False", f"unsupported safety expression: {e}"
            obligations.append(Obligation(sha, path, exc, Invariant(*location, value), reason))

        if res is not None:
            for unmod_node, method, desc in _scan_unmodelled(tree, res):
                path, location = points.get(unmod_node, points.get(tree, ("$", (filename, 1, 1, "<module>"))))
                obligations.append(Obligation(sha, path, exc, Invariant(*location, "False"), desc))
                gaps.append(f"{exc}: {desc}")

    return ObligationCensus(tuple(obligations), tuple(gaps))


def _safety_predicate(node, exc, parent):
    """Sufficient safety conditions at a statement pre-state, never an eval string."""
    def atom(n):
        return isinstance(n, ast.Name) or (isinstance(n, ast.Constant) and type(n.value) in (int, float, bool))

    def direct(n):
        p = parent.get(n)
        return isinstance(p, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr)) and p.value is n

    if exc == "ZeroDivisionError" and isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
        a, b = (node.left, node.right) if isinstance(node, ast.BinOp) else (node.target, node.value)
        if atom(a) and atom(b) and (isinstance(node, ast.AugAssign) or direct(node)):
            left, right = ast.unparse(a), ast.unparse(b)
            numeric = f"(type({left}) is int or type({left}) is float or type({left}) is bool)"
            nonzero = (f"((type({right}) is int and {right} != 0) or "
                       f"(type({right}) is float and {right} != 0.0) or "
                       f"(type({right}) is bool and {right} != False))")
            return f"{numeric} and {nonzero}", ""
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and atom(node.slice) and direct(node):
        container, index = node.value.id, ast.unparse(node.slice)
        if exc == "IndexError":
            return " or ".join(f"(type({container}) is {kind} and type({index}) is int and -len({container}) <= {index} < len({container}))" for kind in ("list", "tuple", "str", "bytes")), ""
        if exc == "KeyError":
            return f"type({container}) is dict and {index} in {container}", "membership needs checked primitive key/element facts"
    if exc == "AssertionError" and isinstance(node, ast.Assert):
        from tharkun.solve.predicates import InvalidPredicate, parse_predicate
        value = ast.unparse(node.test)
        try:
            parse_predicate(value)
        except InvalidPredicate:
            pass
        else:
            return value, "assert test needs checked Boolean type, purity and totality"
    return "False", "no supported safety predicate for this operation/evaluation order"
