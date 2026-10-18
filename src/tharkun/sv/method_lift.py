"""tharkun's own rendering on top of the SHARED pure-data-class detector
([internal ref], the "one lifting" follow-up to [internal ref] slice 2b).

``py_method_lift`` (github.com/ardier/py_method_lift, zero dependencies,
imports neither ``tharkun`` nor any ``eregion_*`` package) is the ONE
accept/refuse decision and field/method model, shared with Eregion's own
``eregion-forward-passes`` (its class-inlining pass AND its inversion-engine
method lift). This module is tharkun's OWN, checker-specific RENDERING on
top of that shared decision -- turning an already-validated
``PureDataClassInfo``/``PureDataMethod`` into a plain, class-free,
self-free ``ast.FunctionDef`` the checker can walk directly. It does not
re-decide accept/refuse; ``lift_class`` below is a thin pass-through to
``py_method_lift.detect_pure_data_class``.

MG review (2026-09-29): the sv_2027 competition wheel is sealed to tharkun
+ hypothesis + sortedcontainers + z3-solver -- no unpinned git dependency,
no network access at install time. ``tharkun.sv._vendored_py_method_lift``
(imported below) is a VERBATIM, commit-pinned copy of
``py_method_lift/detect.py`` vendored directly into tharkun's own source
tree for exactly that reason; tharkun's ``pyproject.toml`` ``dependencies``
list is unchanged (still just hypothesis/z3-solver/PyYAML) -- see that
file's own header for the pinned SHA and how to re-vendor it.

Two things are deliberately tharkun-specific, not part of the shared
package:

* The substitution itself mutates the ORIGINAL method's statement/expression
  nodes IN PLACE (never ``copy.deepcopy``s them, via
  ``ast.NodeTransformer.generic_visit``'s own identity-preserving default)
  so ``tharkun.solve.obligations.source_points``'s node-identity-keyed
  location table -- already computed once over the whole original module,
  before any lift runs -- stays valid for the lifted body's own statements.
  A candidate targeting a class method therefore uses the method's REAL,
  ORIGINAL ``(file, line, column, function="Class.method")`` location, the
  same shape every other ``Invariant`` candidate already uses.
* A field's TYPE (needed to synthesize a free Z3 symbol for a field this
  method READS before ever writing it) is inferred from ``__init__``'s own
  annotated parameter, when that field is a direct, unmodified copy of one
  (``self.x = x`` where ``x: float`` -- see :func:`_init_field_types`).
  ``py_method_lift`` itself is type-agnostic by design (pure shape
  detection), and a class-level ``NAME: TYPE`` annotation is OUT OF the
  shared detector's accepted shape (not a ``def``/docstring/``pass``/literal
  ``__slots__`` -- see ``py_method_lift.detect``'s docstring), so this is
  the only place tharkun's checker recovers a field's sort at all. A field
  with no inferred type here REFUSES the method that reads it -- never
  guessed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

from tharkun.sv._vendored_py_method_lift import PureDataClassInfo, PureDataMethod, detect_pure_data_class

_SORT_NAMES = {"int", "float", "bool"}


@dataclass(frozen=True)
class LiftedFunction:
    qualname: str                  # "Canvas.__init__" -- also the Invariant.function value
    tree: ast.FunctionDef            # class-free, self-free; every param carries a sort annotation
    field_reads: tuple[str, ...]     # "self__x" / "other__y", ... -- read as an input
    field_writes: tuple[str, ...]    # "self__x", ... -- assigned somewhere in the body


def lift_class(class_def: ast.ClassDef) -> PureDataClassInfo | None:
    """The shared accept/refuse decision, unchanged: see
    ``py_method_lift.detect_pure_data_class``. Returns ``None`` for the
    WHOLE class on any refusal ([internal ref]'s merge rule -- refuse
    whenever either lift that used to exist would have refused -- happens to
    be per-CLASS, since that was already Eregion's own, stricter rule)."""
    return detect_pure_data_class(class_def)


def _init_field_types(info: PureDataClassInfo) -> dict[str, str]:
    """See the module docstring. Only a field ``__init__`` stores an
    annotated parameter's value into UNCHANGED gets a sort here."""
    out: dict[str, str] = {}
    params = {a.arg: a.annotation for a in info.init.method_node.args.args[1:]}
    for field, expr in info.init_field_map.items():
        ann = params.get(expr.id) if isinstance(expr, ast.Name) else None
        if isinstance(ann, ast.Name) and ann.id in _SORT_NAMES:
            out[field] = ann.id
    return out


class _FieldRewriter(ast.NodeTransformer):
    """Replaces every ``<param>.FIELD`` (``param`` in *instance_params*) with
    ``Name('<param>__FIELD')``. No ``visit_*`` override for anything else,
    so ``generic_visit``'s default behaviour leaves every other node --
    and therefore its ``source_points`` identity -- exactly as it was."""

    def __init__(self, instance_params: frozenset[str]):
        self.instance_params = instance_params

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in self.instance_params:
            return ast.copy_location(ast.Name(id=f"{node.value.id}__{node.attr}", ctx=node.ctx), node)
        self.generic_visit(node)
        return node


def lift_method(info: PureDataClassInfo, method_name: str) -> LiftedFunction | None:
    """Render one already-validated method (or ``"__init__"``) of *info* to
    a class-free, self-free function. Returns ``None`` when a field this
    method reads before ever writing it has no inferred type."""
    method: PureDataMethod | None = info.init if method_name == "__init__" else info.methods.get(method_name)
    if method is None:
        return None
    instance_params = frozenset({"self"} | method.instance_params)
    field_types = _init_field_types(info)
    # MG review, blocker fix (2026-09-29): a REAL local/parameter already
    # named e.g. "self__x" (or "other__y" for another instance parameter)
    # collides with the rewritten form of "self.x"/"other.y" -- after the
    # rewrite the checker's env can no longer tell them apart, so a fact
    # about one gets silently attributed to the other (confirmed a false
    # claim in the reported repro). Checked on the ORIGINAL, un-rewritten
    # body/params, before anything is mutated in place.
    prefixes = tuple(f"{p}__" for p in instance_params)
    if any(a.arg.startswith(prefixes) for a in method.method_node.args.args[1:]):
        return None
    if any(isinstance(n, ast.Name) and n.id.startswith(prefixes)
          for stmt in method.method_node.body for n in ast.walk(stmt)):
        return None
    # Field reads/writes, in body order -- a read before any write in this
    # method's own OWN body is a genuine input; over-including a field as an
    # input is always sound (one more unconstrained free symbol), it just
    # costs precision. Only 'self' can ever be a write target (the shared
    # detector refuses `<param>.FIELD = ...` outright for any other
    # instance parameter), so every other instance's fields are always reads.
    reads: dict[str, set] = {p: set() for p in instance_params}
    written: set = set()
    for kind, name, expr in method.body:
        for prefix in instance_params:
            for n in ast.walk(expr):
                if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                        and n.value.id == prefix
                        and not (prefix == "self" and n.attr in written)):
                    reads[prefix].add(n.attr)
        if kind == "field":
            written.add(name)
    for f in sorted(reads["self"]):
        if field_types.get(f) not in _SORT_NAMES:
            return None
    for other in method.instance_params:
        for f in sorted(reads[other]):
            if field_types.get(f) not in _SORT_NAMES:
                return None
    rewriter = _FieldRewriter(instance_params)
    new_body = [rewriter.visit(stmt) for stmt in method.method_node.body]
    field_params = [
        ast.arg(arg=f"{prefix}__{f}", annotation=ast.Name(id=field_types[f], ctx=ast.Load()))
        for prefix in sorted(instance_params) for f in sorted(reads[prefix])
    ]
    real_params = [a for a in method.method_node.args.args[1:] if a.arg not in method.instance_params]
    lifted_args = ast.arguments(posonlyargs=[], args=real_params + field_params, vararg=None,
                                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    lifted_name = f"{info.class_name}__{method_name}".replace("<", "_").replace(">", "_")
    tree = ast.FunctionDef(name=lifted_name, args=lifted_args, body=new_body,
                           decorator_list=[], returns=None, type_params=[])
    ast.copy_location(tree, method.method_node)
    # Only the NEW nodes this lift created (the wrapper, the field_params)
    # can be missing a location; every reused original node already has
    # one, and fix_missing_locations only ever fills gaps.
    ast.fix_missing_locations(tree)
    field_reads = tuple(f"{p}__{f}" for p in sorted(instance_params) for f in sorted(reads[p]))
    field_writes = tuple(f"self__{f}" for f in sorted(written))
    return LiftedFunction(qualname=f"{info.class_name}.{method_name}", tree=tree,
                          field_reads=field_reads, field_writes=field_writes)
