"""Source-only plain classes for the exact product proof.

No target class body is executed. Empty engine-owned class tokens identify heap
cells; the audited source supplies constants and methods to the existing object
interpreter. Unsupported Python protocols refuse the whole proof.
"""
from __future__ import annotations

import ast
import copy

from tharkun.sv.exact_subset import _Audit, _name_bindable, _no, type_identity_args


def hint_names(node, classes):
    if isinstance(node, ast.Name) and node.id in {"int", "bool", "str", *classes}:
        return (node.id,)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = hint_names(node.left, classes), hint_names(node.right, classes)
        if left is not None and right is not None:
            return left + right
    return None


def plain_classes(tree):
    """Validate declarations before normalization can erase an unsupported shape."""
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    if not classes:
        return {}
    prefix_tuples = {id(value) for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr in {"startswith", "endswith"}
                     and n.args for value in ast.walk(n.args[0]) if isinstance(value, ast.Tuple)}
    seen, executable = set(), False
    for st in tree.body:
        if isinstance(st, ast.ClassDef):
            if executable or st.name in seen:
                _no("plain classes must be declared once before executable statements", st)
            seen.add(st.name)
        elif not isinstance(st, (ast.Import, ast.ImportFrom, ast.FunctionDef)) and not (
                isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)):
            executable = True
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n not in classes.values():
            _no("nested class", n)
        if ((isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id in classes)
                or isinstance(n, ast.arg) and n.arg in classes
                or isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in classes):
            _no("plain class name rebound or shadowed", n)
        if (isinstance(n, (ast.For, ast.While, ast.Lambda, ast.List, ast.Dict, ast.Set,
                           ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
                or isinstance(n, ast.Tuple) and id(n) not in prefix_tuples):
            _no("plain class proof requires scalar fields and loop-free calls", n)
    for cls in classes.values():
        _name_bindable(cls.name, set(), cls)
        if (cls.decorator_list or cls.keywords or getattr(cls, "type_params", ())
                or len(cls.bases) > 1 or any(not isinstance(b, ast.Name) or b.id != "object" for b in cls.bases)):
            _no("class is not plain (base, decorator, keyword or type parameter)", cls)
        names = set()
        for st in cls.body:
            if isinstance(st, ast.FunctionDef):
                a = st.args
                name = st.name
                if (st.decorator_list or a.defaults or a.kw_defaults or a.kwonlyargs or a.posonlyargs
                        or a.vararg or a.kwarg or not a.args or getattr(st, "type_params", ())):
                    _no("method is not plain positional", st)
                if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                      ast.Yield, ast.YieldFrom, ast.Await)) for s in st.body for n in ast.walk(s)):
                    _no("nested or suspended method", st)
            elif (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                  and isinstance(st.value, ast.Constant) and type(st.value.value) in (int, bool, str, type(None))):
                name = st.targets[0].id
            elif isinstance(st, ast.Pass) or isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
                continue
            else:
                _no("plain class body must contain methods and scalar constants", st)
            if name in names or name.startswith("__") and not (name == "__init__" and isinstance(st, ast.FunctionDef)):
                _no("duplicate, private or protocol class member", st)
            names.add(name)
    return classes


class ClassAudit(_Audit):
    """Reuse the scalar audit for every method, including methods never called."""

    def __init__(self, tree, *args, **kwargs):
        self.classes = plain_classes(tree)
        self.methods = {s.name for c in self.classes.values() for s in c.body if isinstance(s, ast.FunctionDef)}
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used.update(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        body = []
        for st in tree.body:
            if not isinstance(st, ast.ClassDef):
                body.append(st)
                continue
            # Audit-only declarations: share method bodies so authenticated draw IDs survive.
            body.append(ast.copy_location(ast.Assign([ast.Name(st.name, ast.Store())], ast.Constant(0)), st))
            for method in st.body:
                if isinstance(method, ast.FunctionDef):
                    fd = copy.copy(method)
                    name = "tharkun_method"
                    while name in used:
                        name += "_"
                    used.add(name)
                    fd.name = name
                    body.append(fd)
        super().__init__(ast.Module(body=body, type_ignores=[]), *args, **kwargs)

    def _attr(self, node, scope):
        if node.attr.startswith("__"):
            _no("private or protocol attribute", node)
        self._expr(node.value, scope)

    def _stmt(self, st, scope, in_def):
        target = st.targets[0] if isinstance(st, ast.Assign) and len(st.targets) == 1 else (
            st.target if isinstance(st, ast.AugAssign) else None)
        if isinstance(target, ast.Attribute):
            self._attr(target, scope)
            if isinstance(st, ast.AugAssign):
                from tharkun.sv.exact_subset import _BINOPS
                if not isinstance(st.op, _BINOPS) or not isinstance(target.value, ast.Name):
                    _no("field augmented operator", st)
            self._expr(st.value, scope)
            return
        super()._stmt(st, scope, in_def)

    def _expr(self, node, scope, **kwargs):
        if isinstance(node, ast.Attribute):
            self._attr(node, scope)
            return
        if isinstance(node, ast.Constant) and node.value is None:
            return
        if (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot))
                and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None):
            self._expr(node.left, scope)
            return
        if (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot))
                and type_identity_args(node) is None):
            self._expr(node.left, scope)
            self._expr(node.comparators[0], scope)
            return
        super()._expr(node, scope, **kwargs)

    def _call(self, node, scope, statement=False, unpack=False, callee=False):
        f = node.func
        check = (isinstance(f, ast.Name) and f.id in self.check_type_functions) or (
            isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
            and f.value.id in self.verifier_modules and f.attr == "check_type")
        if (check and statement and not node.keywords and len(node.args) == 2
                and (isinstance(node.args[1], ast.Name) and node.args[1].id in self.classes
                     or isinstance(node.args[1], ast.BinOp) and hint_names(node.args[1], self.classes) is not None)
                and self.guarded_type_checks is not None):
            from tharkun.solve.contracts.rules import get_rule
            from tharkun.solve.contracts.visitors import ProductVisitor
            rule = get_rule("_sv_verifier.check_type")
            if rule is None or rule.accept(ProductVisitor()) is not True:
                _no("no tested check_type success contract", node)
            self._expr(node.args[0], scope)
            self.guarded_type_checks.add(id(node))
            return
        constructor = isinstance(f, ast.Name) and f.id in self.classes
        method = isinstance(f, ast.Attribute) and f.attr in self.methods and not f.attr.startswith("__")
        if isinstance(f, ast.Attribute) and f.attr in {"startswith", "endswith"} and not method:
            return super()._call(node, scope, statement, unpack, callee)
        length = isinstance(f, ast.Name) and f.id == "len" and len(node.args) == 1
        if constructor or method or length:
            if node.keywords:
                _no("plain class call keywords", node)
            if method:
                self._attr(f, scope)
            for arg in node.args:
                self._expr(arg, scope)
            return
        super()._call(node, scope, statement, unpack, callee)


def attach(runner, tree):
    """Install the audited source table, with no target imports or execution."""
    from tharkun.sv._symexec_objects import ClassInfo, ClassTable, Method, ObjectModel

    classes = plain_classes(tree)
    tokens = {name: type(name, (), {}) for name in classes}

    class SourceTable(ClassTable):
        def __init__(self):
            self.ast, self.g = classes, tokens
            self.infos = {}
            for name, node in classes.items():
                info = ClassInfo(tokens[name], node, None)
                for st in node.body:
                    if isinstance(st, ast.FunctionDef):
                        info.methods[st.name] = Method("function", st, info)
                    elif isinstance(st, ast.Assign):
                        info.data[st.targets[0].id] = runner.ev(st.value)
                self.infos[tokens[name]] = info

        def is_user_class(self, value):
            return isinstance(value, type) and value in self.infos

        def info(self, cls):
            return self.infos[cls]

        def imported(self, cls):
            return None

    class SourceObjects(ObjectModel):
        def load(self, base, name):
            value = self._load_value(base, name)
            if getattr(self.r, "identity_enabled", False):
                from tharkun.sv.exact_identity import KEY
                value = self.r.present(value)
                origin = self.r.present(self._load_value(base, KEY + name))
                self.r.set_origin(origin)
            return value

        def _load_value(self, base, name):
            from tharkun.sv._symexec import Union, _Dead, _bool, _conj
            if not isinstance(base, Union):
                return super().load(base, name)
            self._check_attr_name(name)
            saved_pc, results = list(self.r.pc), []
            try:
                for guard, member in self.r._work_members(base):
                    self.r.pc = [*saved_pc, guard]
                    try:
                        if member is None:
                            raise _Dead()  # None has no non-dunder attributes: AttributeError.
                        value = super().load(member, name)
                        results.append((_conj(self.r.pc[len(saved_pc):]), value))
                    except _Dead:
                        pass
            finally:
                self.r.pc = saved_pc
            if not results:
                raise _Dead()
            import z3
            self.r._holds(z3.Or(*[_bool(guard) for guard, _ in results]))
            value = results[-1][1]
            for guard, other in reversed(results[:-1]):
                value = self.r._merge_value(_bool(guard), other, value)
            return value

        def store(self, base, name, value):
            super().store(base, name, value)
            if getattr(self.r, "identity_enabled", False):
                from tharkun.sv.exact_identity import KEY
                super().store(base, KEY + name, self.r.pending_origin)

        def _check_value(self, name, value):
            from tharkun.sv._symexec import Union
            if getattr(self.r, "identity_enabled", False) and isinstance(value, Union):
                for _, member in self.r._work_members(value):
                    super()._check_value(name, member)
                return
            super()._check_value(name, value)

        def _fallback(self, key, name, heap):
            if getattr(self.r, "identity_enabled", False):
                from tharkun.sv.exact_identity import ABSENT
                info = self.obj_cls[key] if isinstance(key, int) else self.table.info(key)
                if self._chain(info, name, heap, skip_first_overlay=not isinstance(key, int)) is None:
                    return ABSENT
            return super()._fallback(key, name, heap)

        def _merge_value(self, guard, a, b):
            return self.r._merge_value(guard, a, b)

        def _require(self, callee, expect):
            # The whole-source audit fixes class names and permits only positional calls.
            return {"name": callee}

    objects = SourceObjects(runner, tree, tokens, [])
    objects.table = SourceTable()
    if getattr(runner, "identity_enabled", False):
        from tharkun.sv.exact_identity import KEY
        for info in objects.table.infos.values():
            info.data.update({KEY + name: runner.origin(value) for name, value in list(info.data.items())})
    objects.proven_names = set(tokens)
    runner.g = {**runner.g, **tokens}
    runner.objects = objects
    runner.reach = objects.build_reach(runner.defs, runner.targets)
