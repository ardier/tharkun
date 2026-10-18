"""VENDORED COPY -- github.com/ardier/py_method_lift @ e6ef8d3 (2026-09-29).

MG review (2026-09-29, [internal ref]): the sv_2027 competition wheel is
SEALED to tharkun + hypothesis + sortedcontainers + z3-solver
(docs/SV_2027_VALIDATION.md) -- no network access, no unpinned git
dependency at install time. ``py_method_lift`` (the shared pure-data-class
detector, also depended on directly by eregion-forward-passes -- see that
repo's own PR #25) is therefore VENDORED here verbatim, pinned to the exact
commit SHA above, instead of a ``pyproject.toml`` dependency. tharkun's own
``dependencies = []`` (core) list is UNCHANGED by this file's existence.

To update: copy ``src/py_method_lift/detect.py`` from that repo at a new
commit over this file's CODE (everything below this header), update the
SHA above, and diff the two thisfile against the repo's copy to confirm the
only difference is this header -- never hand-edit the code below.

Pure-data-class shape detection -- the ONE lifting ([internal ref]).

Two independent consumers grew their own version of "is this class a
straight-line, single-instance data container, and what does each method
do to its fields": Eregion's ``eregion-forward-passes`` (serving both
``inline_pure_data_class.py``'s class-inlining pass AND
``method_lift.py``'s free-function lift for the inversion engine) and
Tharkun's SV witness checker (``tharkun.sv._Checker``, needing to check a
candidate fact inside a class method). Two implementations of the same
accept/refuse decision can silently drift -- this module is the single,
dependency-free (stdlib ``ast`` + ``dataclasses`` + ``typing`` only, no
import of ``tharkun`` or any ``eregion_*`` package) source of that decision,
so it can be depended on directly by BOTH sides without violating either
side's layering (tharkun's charter: "tharkun never depends on eregion";
Eregion's forward-passes must work with tharkun absent).

What this module decides, and what it does NOT decide
-------------------------------------------------------
:func:`detect_pure_data_class` answers exactly one question: does this class
match the straight-line, single-instance, pure-data shape (see the class and
per-method rules below), and if so, what are its fields and what does each
method's body do (as a sequence of field stores / local stores, plus an
optional final return)? It does **not** rewrite any code, does not decide a
field's runtime TYPE, and does not build an SSA form, a return-tuple, or a
class-free ``FunctionDef`` -- those are consumer-specific renderings built
FROM this module's output (Eregion's own SSA/return-tuple renderer in
``method_lift.py``; Tharkun's own self__FIELD-substitution walk over the
ORIGINAL, still-node-identity-preserved ``PureDataMethod.method_node``, kept
outside this module specifically so ``tharkun.solve.obligations.source_points``'s
node-identity-keyed location table stays valid for whichever statements a
consumer decides to check).

The accepted shape (verbatim from eregion-forward-passes'
``inline_pure_data_class.py`` as of the 2026-09-29 extraction, bd
[internal ref]'s semantics-diff table has the full history):

* No inheritance other than implicit/explicit ``object``; no class
  decorators/keywords.
* ``__slots__`` (a literal tuple/list/single string) is the only other
  class-body statement besides ``def``, a docstring, and ``pass``.
* Dunders that intercept construction/attribute-access/finalization
  (``__new__``, ``__getattr__``, ... -- see :data:`UNINLINABLE_DUNDERS``)
  are refused outright; any OTHER dunder is allowed only because it is
  simply dropped along with the class if never called (a caller reaching it
  is each CONSUMER's own escape concern, not this module's).
* No ``*args``/``**kwargs``/keyword-only/positional-only/default parameters
  on ``__init__`` or any method; the first parameter must be named exactly
  ``self``.
* A method body is a sequence of PLAIN, single-target ``Assign`` statements
  (``self.FIELD = expr`` or ``local = expr``) followed, optionally, by ONE
  trailing ``return expr`` / ``return self`` -- nothing else (no control
  flow, no ``AugAssign``, no ``assert``, no bare expression statements, no
  multi-target/tuple/subscript assignment). ``Pass`` statements and a
  docstring are ignored, not counted against this rule.
* An expression may read a constant, an already-stored local, a method
  parameter, ``self.FIELD`` (a field ``__init__`` already established), a
  field of ANOTHER same-shaped instance passed as a parameter
  (``other.FIELD``, once that parameter is used ONLY this way -- never also
  as a bare value), or call a FREE function name (not ``self``, not a
  parameter, not a local) with positional-only, non-starred arguments;
  arithmetic, comparison, boolean and conditional (``a if c else b``)
  combinations of these are allowed. Any other expression shape (a
  subscript, a container literal, a lambda, a chained/nested attribute
  access, a call through anything but a bare free name) is refused.
* Only ``__init__`` may introduce a new field name; every other method may
  only ever store into a field ``__init__`` already established.
* A local variable name may not collide with ``self`` or with any
  parameter name, anywhere in the body (Python's own name-scoping rule,
  checked once up front).

Refuse, never guess: any construct outside this shape makes
:func:`detect_pure_data_class` return ``None`` for the WHOLE class -- there
is no partial/best-effort acceptance of some methods and not others. A
consumer that only cares about one method of a class still needs the WHOLE
class to pass ([internal ref]'s "refuse whenever either consumer's
original implementation would have refused" merge rule -- deliberately
narrower than Tharkun's own earlier per-method-refusal lift, which this
module replaces).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

__all__ = [
    "PureDataMethod",
    "PureDataClassInfo",
    "UNINLINABLE_DUNDERS",
    "detect_pure_data_class",
]


#: Dunders whose mere presence changes construction, attribute access or
#: finalization -- inlining/lifting the class away would silently drop them.
#: Every other dunder (``__repr__``, ``__eq__``, ``__call__``, ...) is only
#: reachable through the instance itself, and each consumer's own escape
#: check declines when an instance could reach one, so an uncalled one is
#: harmless here.
UNINLINABLE_DUNDERS = frozenset({
    "__new__", "__getattr__", "__getattribute__", "__setattr__",
    "__delattr__", "__del__", "__init_subclass__", "__class_getitem__",
    "__set_name__", "__get__", "__set__", "__delete__",
})


@dataclass
class PureDataMethod:
    """Metadata for one method (or ``__init__``) of a detected pure-data class."""

    name: str
    params: List[str]  # excluding ``self``
    # Accessor return template (uses self.<attr> + param names); ``None`` for
    # a pure mutator that never returns a value.
    return_expr: Optional[ast.expr] = None
    # Field reassignments in body order: ``self.<field> = <rhs>``.
    mutations: List[Tuple[str, ast.expr]] = field(default_factory=list)
    # The ORIGINAL ``ast.FunctionDef`` -- kept so a consumer that needs
    # node-identity-preserving rewriting (never ``copy.deepcopy``) can walk
    # ``method_node.body`` itself instead of reconstructing statements from
    # ``body`` below.
    method_node: ast.FunctionDef = field(repr=False, default=None)
    # The body, in order: ``("field", <f>, rhs)`` for ``self.<f> = rhs`` and
    # ``("local", <v>, rhs)`` for a method-local ``v = rhs``. ``mutations``
    # is its field-only projection. Every ``rhs`` is the ORIGINAL expression
    # node (never copied).
    body: List[Tuple[str, str, ast.expr]] = field(default_factory=list)
    # ``return self`` (the fluent chaining shape, e.g. ``r = a.maximize(b)``).
    returns_self: bool = False
    # Parameters read as ``<param>.<field>`` -- another instance of the class.
    instance_params: Set[str] = field(default_factory=set)
    # Names the body reads that are neither params, locals nor ``self``
    # (free functions such as ``sin``); resolved at the call site.
    free_names: Set[str] = field(default_factory=set)
    # Fields read through an instance parameter (``other.<f>``).
    other_attrs: Set[str] = field(default_factory=set)

    @property
    def has_statements(self) -> bool:
        """True when this method has a body beyond substituting a return
        expression (a consumer that only handles accessors uses this)."""
        return bool(self.body) or self.returns_self


@dataclass
class PureDataClassInfo:
    """A detected pure-data class: its fields, in ``__init__`` store order,
    and every method's :class:`PureDataMethod`, INCLUDING ``__init__`` under
    the key ``"__init__"`` as well as in :attr:`init` directly."""

    class_name: str
    init_params: List[str]  # excluding ``self``, in order
    init_field_map: Dict[str, ast.expr]  # field name -> assigned expr (original node)
    methods: Dict[str, PureDataMethod] = field(default_factory=dict)
    class_node: ast.ClassDef = field(repr=False, default=None)
    init: Optional[PureDataMethod] = None

    @property
    def fields(self) -> Tuple[str, ...]:
        return tuple(self.init_field_map)


def _slots_of(stmt: ast.stmt) -> Optional[Set[str]]:
    """The field names of a literal ``__slots__ = ('x', 'y')`` (tuple, list
    or a single string), or ``None`` when *stmt* is not such a declaration."""
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == "__slots__"
    ):
        return None
    value = stmt.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return {value.value}
    if isinstance(value, (ast.Tuple, ast.List)) and all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts
    ):
        return {e.value for e in value.elts}
    return None


def _plain_params(fn: ast.FunctionDef) -> Optional[List[str]]:
    """``self`` plus plain positional parameters -> the names after ``self``;
    ``None`` for defaults / ``*args`` / ``**kwargs`` / keyword-only /
    positional-only / no ``self`` as the first parameter."""
    args = fn.args
    if args.vararg is not None or args.kwarg is not None:
        return None
    if args.kw_defaults or args.kwonlyargs or args.defaults or args.posonlyargs:
        return None
    if len(args.args) < 1 or args.args[0].arg != "self":
        return None
    return [a.arg for a in args.args[1:]]


def _scan_method(
    fn: ast.FunctionDef,
    *,
    is_init: bool,
    fields: Optional[Set[str]],
    slots: Optional[Set[str]],
) -> Optional[PureDataMethod]:
    """Scan *fn* as a straight-line method body, or ``None`` when it does not
    match (see the module docstring for the exact accepted shape).

    *fields* is ``None`` while scanning ``__init__`` (it DEFINES the fields)
    and the ``__init__`` field set otherwise; a non-``__init__`` method may
    only store a field ``__init__`` already declared. *slots*, when the
    class declares ``__slots__``, bounds every field name.
    """
    params = _plain_params(fn)
    if params is None:
        return None
    pset = set(params)
    stmts = [
        s for s in fn.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        and not isinstance(s, ast.Pass)
    ]
    # Python scoping: a name stored anywhere in the body is local throughout.
    body_locals: Set[str] = set()
    for s in stmts:
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
        ):
            body_locals.add(s.targets[0].id)
    if "self" in body_locals or body_locals & pset:
        return None

    readable: Set[str] = set() if is_init else set(fields or ())
    stored_locals: Set[str] = set()
    instance_params: Set[str] = set()
    other_attrs: Set[str] = set()
    bare_params: Set[str] = set()
    free: Set[str] = set()

    def ok(e: ast.expr) -> bool:
        if isinstance(e, ast.Constant):
            return True
        if isinstance(e, ast.Name):
            if e.id == "self":
                return False
            if e.id in pset:
                bare_params.add(e.id)
                return True
            if e.id in body_locals:
                return e.id in stored_locals
            free.add(e.id)
            return True
        if isinstance(e, ast.Attribute):
            if not isinstance(e.value, ast.Name):
                return False
            if e.value.id == "self":
                return e.attr in readable
            if e.value.id in pset:
                instance_params.add(e.value.id)
                other_attrs.add(e.attr)
                return True
            return False
        if isinstance(e, ast.BinOp):
            return ok(e.left) and ok(e.right)
        if isinstance(e, ast.UnaryOp):
            return ok(e.operand)
        if isinstance(e, ast.Compare):
            return ok(e.left) and all(ok(c) for c in e.comparators)
        if isinstance(e, ast.BoolOp):
            return all(ok(v) for v in e.values)
        if isinstance(e, ast.IfExp):
            return ok(e.test) and ok(e.body) and ok(e.orelse)
        if isinstance(e, ast.Tuple):
            return all(ok(v) for v in e.elts)
        if isinstance(e, ast.Call):
            f = e.func
            if not isinstance(f, ast.Name) or f.id == "self":
                return False
            if f.id in pset or f.id in body_locals:
                return False
            if e.keywords or any(isinstance(a, ast.Starred) for a in e.args):
                return False
            free.add(f.id)
            return all(ok(a) for a in e.args)
        return False

    body: List[Tuple[str, str, ast.expr]] = []
    return_expr: Optional[ast.expr] = None
    returns_self = False
    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, ast.Return):
            if idx != len(stmts) - 1:
                return None  # statement after a return is unreachable
            if stmt.value is None:
                break
            if is_init:
                return None
            if isinstance(stmt.value, ast.Name) and stmt.value.id == "self":
                returns_self = True
                break
            if not ok(stmt.value):
                return None
            return_expr = stmt.value
            break
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            return None
        target = stmt.targets[0]
        if not ok(stmt.value):
            return None
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            fname = target.attr
            if not is_init and fname not in (fields or ()):
                return None  # only fields declared in __init__ may be stored
            if slots is not None and fname not in slots:
                return None
            body.append(("field", fname, stmt.value))
            readable.add(fname)
        elif isinstance(target, ast.Name):
            body.append(("local", target.id, stmt.value))
            stored_locals.add(target.id)
        else:
            return None

    if instance_params & bare_params:
        return None  # an instance parameter also used as a bare value
    return PureDataMethod(
        name=fn.name,
        params=params,
        return_expr=return_expr,
        mutations=[(n, e) for kind, n, e in body if kind == "field"],
        method_node=fn,
        body=body,
        returns_self=returns_self,
        instance_params=instance_params,
        free_names=free,
        other_attrs=other_attrs,
    )


def detect_pure_data_class(cls: ast.ClassDef) -> Optional[PureDataClassInfo]:
    """Detect whether *cls* follows the pure-data (straight-line) class shape
    documented in this module's docstring.

    Returns a populated :class:`PureDataClassInfo` on success, or ``None``
    when the shape doesn't match -- absence, never an exception: every
    caller treats "not this shape" as an ordinary, expected outcome.
    """
    if cls.bases:
        # ``class Foo(object):`` is fine; anything else is out of scope.
        if not (
            len(cls.bases) == 1
            and isinstance(cls.bases[0], ast.Name)
            and cls.bases[0].id == "object"
        ):
            return None
    if cls.keywords or cls.decorator_list:
        return None

    init_fn: Optional[ast.FunctionDef] = None
    method_fns: List[ast.FunctionDef] = []
    slots: Optional[Set[str]] = None
    for item in cls.body:
        if isinstance(item, ast.FunctionDef):
            if item.decorator_list:
                return None  # staticmethod / classmethod / property
            if item.name == "__init__":
                init_fn = item
            elif item.name.startswith("__") and item.name.endswith("__"):
                if item.name in UNINLINABLE_DUNDERS:
                    return None
                continue
            else:
                method_fns.append(item)
        elif isinstance(item, ast.Pass):
            continue
        elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            continue
        else:
            declared = _slots_of(item)
            if declared is None or slots is not None:
                return None
            slots = declared

    if init_fn is None:
        return None
    init = _scan_method(init_fn, is_init=True, fields=None, slots=slots)
    if init is None:
        return None
    init_field_map: Dict[str, ast.expr] = {}
    for kind, name, expr in init.body:
        if kind == "field":
            init_field_map[name] = expr
    if not init_field_map:
        return None
    fields = set(init_field_map)
    if init.other_attrs - fields:
        return None  # reads a field another instance never has

    methods: Dict[str, PureDataMethod] = {}
    for meth in method_fns:
        scanned = _scan_method(meth, is_init=False, fields=fields, slots=slots)
        if scanned is None or scanned.other_attrs - fields:
            return None
        if scanned.return_expr is None and not scanned.body and not scanned.returns_self:
            return None  # empty / unrecognized method body
        methods[meth.name] = scanned

    return PureDataClassInfo(
        class_name=cls.name,
        init_params=list(init.params),
        init_field_map=init_field_map,
        methods=methods,
        class_node=cls,
        init=init,
    )
