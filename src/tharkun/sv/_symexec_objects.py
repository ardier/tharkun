"""User-class objects in the whole-harness symbolic executor ([internal ref], docs/SV.md "The call_target fact").

``sv/_symexec.py`` inlined module-level functions and nothing else: every ``Canvas(...)`` or ``obj.method()``
ended the statement as ``unsupported`` and poisoned what it assigned.  This module gives ``_Runner`` the class
protocol the program's own classes need -- construction, instance and class attributes, methods,
``@staticmethod`` / ``@classmethod``, single inheritance with ``super()``, ``isinstance`` -- with EXACT
semantics, and a NAMED refusal (:class:`~tharkun.sv._symexec.Unsupported`) for everything else.  Nothing here
guesses: an object this model cannot follow makes the statement unmodelled, never a different program.

What it stands on
-----------------
* **Which call is a class call** is the ``call_target`` fact (``solve/call_targets.py``): ``Canvas(...)`` is
  constructed only when the record of ``(scope, "Canvas")`` says ``user_class`` and is ``proven``; a name used
  any other way (``isinstance(x, Canvas)``, ``Canvas.count``) must be a proven class name: the callee of a proven
  ``user_class`` record or a class-origin ``module_constant`` that is proven (:attr:`ObjectModel.proven_names`).
  An unproven or missing record is a refusal that says so.
* **What a class is** is the LIVE class: ``module_globals`` is the harness module as the worker loaded it, so
  ``module_globals["Canvas"]`` is the class after every import-time statement ran (``Cls.m = f``, a decorator,
  a class-body ``if``).  :class:`ClassTable` checks each ``def`` of the live ``__dict__`` against the harness
  AST by first line, so a patched or generated method is refused; only the harness AST can be inlined, the live
  object decides what is in the class.
* **Objects** are :class:`~tharkun.sv._symexec.Ref` handles to cells of a per-path :class:`Heap`.  A cell's
  attribute dict is replaced on write, never mutated, so forking a path is a dict copy; the heap is forked and
  merged with the environment at every place the executor forks one (``_merge_if``, ``_exec_fn``,
  ``_guarded``), attribute by attribute with the same typed ``ite`` as the environment.  A value that is a
  different object on the two paths, or an attribute only one path defines, is refused when read.

What is refused (each a named ``Unsupported``)
----------------------------------------------
metaclasses, multiple inheritance, a base that is not a class of the program, ``__slots__`` /
``__new__`` / ``__getattr__`` / ``__getattribute__`` / ``__setattr__`` / ``__delattr__`` / ``__del__`` /
``__init_subclass__`` / ``__set_name__`` / ``__class_getitem__``, a descriptor (``property`` included), a
generator or async method, a decorated class, dunder and ``__private`` attribute names, a method used as a
value, an attribute holding a mutable container, ``list`` / ``dict`` / ``set`` or any value whose type depends
on the path, keyword arguments and defaults at a call, any operator, conversion or format applied to an object
(``a + b``, ``str(a)``, ``a < b``, ``a == b`` when the class defines ``__eq__``), and ``super()`` outside an
instance method.  Reading an attribute that does not exist ends the path (``AttributeError``), exactly as the
program does.

Soundness of the heap under skipped statements
----------------------------------------------
``_Runner.run`` skips a statement it cannot model.  A skipped statement may have changed any object or class (an
attribute store, a call to code the executor did not run), so unless it provably cannot
(:meth:`ObjectModel._heap_neutral`: no store, no call but a pure builtin, no operator on an object) the heap of
that path is STALE: every attribute read, store, method call, construction and ``isinstance`` on an object or a
class is refused from then on (the rest of the path keeps running on plain values).
"""
from __future__ import annotations

import ast
import types

import z3

from tharkun.solve.call_targets import call_target_index
from tharkun.sv._symexec import POISON, Ref, SymList, Union, Unsupported, _Dead, _typed_ite

#: ``cls.__dict__`` keys CPython puts in every class
_STANDARD_KEYS = frozenset({
    "__module__", "__dict__", "__weakref__", "__doc__", "__qualname__", "__firstlineno__",
    "__static_attributes__", "__annotations__", "__annotate_func__", "__annotations_cache__",
    "__classdictcell__", "__annotate__",
})
#: hooks that change how an object is created, looked up or stored: refused where the class defines them
_REFUSED_HOOKS = frozenset({
    "__new__", "__getattr__", "__getattribute__", "__setattr__", "__delattr__", "__slots__", "__init_subclass__",
    "__set_name__", "__del__", "__class_getitem__", "__prepare__", "__instancecheck__", "__subclasscheck__",
})
_SCALARS = (bool, int, float, str, bytes, type(None))
_GENERATOR_FLAGS = 0x20 | 0x80 | 0x100 | 0x200      # CO_GENERATOR, CO_COROUTINE, CO_ITERABLE_COROUTINE, CO_ASYNC_GENERATOR
#: calls a skipped statement may contain without having run user code (a builtin of this list that no name of
#: the program rebinds); ``str``/``repr``/``format`` are NOT here: they run ``__str__``/``__repr__``/``__format__``
_PURE_BUILTINS = frozenset({
    "print", "len", "abs", "min", "max", "sum", "int", "float", "bool", "range", "sorted", "enumerate", "zip",
    "reversed", "isinstance", "ord", "chr", "round", "divmod", "pow", "tuple", "list", "set", "dict",
})


class _Opaque:
    """A class attribute whose value this model cannot hold (a list, a dict, an instance, ...): reading it is
    refused, so a class with such an attribute is usable as long as the program never reads it."""

    def __init__(self, what: str):
        self.what = what


class Method:
    """One ``def`` of a class body: ``kind`` is ``function`` / ``static`` / ``class``."""

    __slots__ = ("kind", "fd", "owner")

    def __init__(self, kind: str, fd: ast.FunctionDef, owner: "ClassInfo"):
        self.kind, self.fd, self.owner = kind, fd, owner


class ClassInfo:
    """A validated user class: its live class object, its AST, its base and what its own body binds."""

    __slots__ = ("cls", "name", "node", "base", "methods", "data", "dunders")

    def __init__(self, cls, node, base):
        self.cls, self.name, self.node, self.base = cls, cls.__name__, node, base
        self.methods: dict = {}
        self.data: dict = {}
        self.dunders: frozenset = frozenset()


class Heap:
    """The objects of one path: ``cells[key]`` is the attribute dict of an instance (key: its ``oid``) or the
    OVERLAY of a class (key: the class; what a ``Cls.x = v`` stored at run time).  A dict in ``cells`` is
    never mutated -- a write replaces it -- so :meth:`fork` is a shallow copy."""

    __slots__ = ("cells", "stale")

    def __init__(self, cells=None, stale: "str | None" = None):
        self.cells = {} if cells is None else cells
        #: why nothing is known (the first skipped statement that may have changed an object or a class), or None
        self.stale = stale

    def fork(self) -> "Heap":
        return Heap(dict(self.cells), self.stale)


class ClassTable:
    """The live classes of the harness module, validated once each (see the module docstring)."""

    def __init__(self, tree: ast.Module, globals_: dict):
        self.ast = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        self.g = globals_
        self._done: dict = {}

    def is_user_class(self, v) -> bool:
        return isinstance(v, type) and v.__name__ in self.ast and self.g.get(v.__name__) is v

    def info(self, cls) -> ClassInfo:
        got = self._done.get(cls)
        if got is None:
            try:
                got = self._build(cls)
            except Unsupported as e:
                got = str(e)
            self._done[cls] = got
        if isinstance(got, str):
            raise Unsupported(got)
        return got

    def _build(self, cls) -> ClassInfo:
        name = cls.__name__
        node = self.ast.get(name)
        if node is None or self.g.get(name) is not cls:
            raise Unsupported(f"class {name} is not a module-level class of the program")
        if type(cls) is not type:
            raise Unsupported(f"class {name} has a metaclass ({type(cls).__name__})")
        if node.decorator_list:
            raise Unsupported(f"class {name} is decorated")
        if len(cls.__bases__) != 1:
            raise Unsupported(f"class {name}: multiple inheritance")
        base = None
        if cls.__bases__[0] is not object:
            if not self.is_user_class(cls.__bases__[0]):
                raise Unsupported(f"class {name}: its base {cls.__bases__[0].__name__} is not a class of the program")
            base = self.info(cls.__bases__[0])
        info = ClassInfo(cls, node, base)
        dunders = set(base.dunders) if base else set()
        for key, val in cls.__dict__.items():
            if key in _STANDARD_KEYS:
                continue
            if key in _REFUSED_HOOKS:
                raise Unsupported(f"class {name} defines {key}")
            if key.startswith("__") and key.endswith("__"):
                dunders.add(key)
            if isinstance(val, types.FunctionType):
                kind, fn = "function", val
            elif isinstance(val, staticmethod) and isinstance(val.__func__, types.FunctionType):
                kind, fn = "static", val.__func__
            elif isinstance(val, classmethod) and isinstance(val.__func__, types.FunctionType):
                kind, fn = "class", val.__func__
            elif any(hasattr(type(val), hook) for hook in ("__get__", "__set__", "__delete__")):
                raise Unsupported(f"class {name}.{key} is a descriptor ({type(val).__name__})")
            else:
                info.data[key] = val if _holdable(val) else _Opaque(type(val).__name__)
                continue
            fd = self._def_of(node, key, fn)
            if fd is None:
                raise Unsupported(f"class {name}.{key}: the live method is not the def of the class body "
                                  f"(patched, wrapped or generated)")
            if fn.__code__.co_flags & _GENERATOR_FLAGS:
                raise Unsupported(f"class {name}.{key} is a generator or async function")
            info.methods[key] = Method(kind, fd, info)
        info.dunders = frozenset(dunders)
        return info

    @staticmethod
    def _def_of(node: ast.ClassDef, key: str, fn) -> "ast.FunctionDef | None":
        for st in node.body:
            if isinstance(st, ast.FunctionDef) and st.name == key:
                first = min([d.lineno for d in st.decorator_list] + [st.lineno])
                if fn.__code__.co_firstlineno == first:
                    return st
        return None


def _holdable(v) -> bool:
    """May ``v`` live in an attribute or a class body entry: an immutable scalar, a tuple of such, a z3 term,
    an object handle or a class of the program (a mutable container never may: it would alias)."""
    if type(v) in _SCALARS or isinstance(v, (z3.ExprRef, Ref)):
        return True
    if type(v) is tuple:
        return all(_holdable(x) for x in v)
    return isinstance(v, type)


def _kind_name(v) -> str:
    if isinstance(v, SymList):
        return "a symbolic list"
    if isinstance(v, Union):
        return "a value whose type depends on the path"
    return type(v).__name__


class ObjectModel:
    """The class protocol of one :class:`~tharkun.sv._symexec._Runner`: its heap, its live classes and the fact."""

    def __init__(self, runner, tree: ast.Module, globals_: dict, records, proven_classes=()):
        self.r = runner
        self.table = ClassTable(tree, globals_)
        self.index = call_target_index(records)
        #: the class names a program name may denote: a class-origin ``module_constant`` that is proven
        #: (``proven_classes``), and the callee of every proven ``user_class`` call_target
        self.proven_names = set(proven_classes) | {
            rec["callee"] for rec in self.index.values() if rec["target"] == "user_class" and rec["proven"]}
        self.heap = Heap()
        self.next_oid = 0
        self.version = 0                    # every heap write, to tell whether a skipped statement wrote
        self.obj_cls: dict = {}             # oid -> ClassInfo (never changes, so it is not forked)
        self.refs: dict = {}                # oid -> its one Ref (interned: ``is`` on Refs is object identity)

    # -- the fact ---------------------------------------------------------------------------------

    def _record(self, callee: str):
        return self.index.get((self.r.scope_stack[-1], callee))

    def _require(self, callee: str, expect: str) -> dict:
        ct = self._record(callee)
        if ct is None:
            raise Unsupported(f"call {callee}: no call_target fact for it")
        if ct["target"] != expect or not ct["proven"]:
            raise Unsupported(f"call {callee}: the call_target is {ct['target']}, "
                              f"proven={ct['proven']} ({ct['reason']})")
        return ct

    def check_class_name(self, name: str, v) -> None:
        """A program name that evaluates to a user class must be a proven one."""
        if self.table.is_user_class(v) and name not in self.proven_names:
            raise Unsupported(f"class {name}: its name is not proven constant (no proven call_target or "
                              f"module_constant of class origin names it)")

    # -- names and values ---------------------------------------------------------------------------

    @staticmethod
    def _check_attr_name(name: str) -> None:
        if name.startswith("__"):
            raise Unsupported(f"attribute {name}: dunder and private-mangled names are not modelled")

    @staticmethod
    def _check_method_name(name: str) -> None:
        """A call may name a dunder method (``A.__init__(self)``, ``super().__init__()``): only a private-mangled name
        (``__helper``: the compiler renames it per class) is refused."""
        if name.startswith("__") and not name.endswith("__"):
            raise Unsupported(f"method {name}: private-mangled names are not modelled")

    @staticmethod
    def _missing(name: str):
        """The attribute is on no class of the chain: ``AttributeError`` ends the path -- unless ``object`` has it
        (every dunder may be), which this model does not follow."""
        if name.startswith("__") and name.endswith("__"):
            raise Unsupported(f"attribute {name} may be provided by object")
        raise _Dead()

    @staticmethod
    def _check_value(name: str, v) -> None:
        if not _holdable(v):
            raise Unsupported(f"attribute {name} would hold {_kind_name(v)}: a mutable container, list or "
                              f"path-dependent value in an attribute is not modelled")

    def _fresh(self, ref: Ref) -> None:
        if self.heap.stale:
            raise Unsupported(f"object state is unknown after an unmodelled statement ({self.heap.stale})")
        if ref.oid not in self.heap.cells:
            raise Unsupported("an object that does not exist on this path")


    def _classes_known(self) -> None:
        if self.heap.stale:
            raise Unsupported(f"class state is unknown after an unmodelled statement ({self.heap.stale})")

    # -- allocation -------------------------------------------------------------------------------

    def _alloc(self, info: ClassInfo) -> Ref:
        oid = self.next_oid
        self.next_oid += 1
        self.version += 1
        self.heap.cells[oid] = {}
        self.obj_cls[oid] = info
        ref = self.refs[oid] = Ref(oid, info.cls, info.dunders)
        return ref

    # -- lookup ---------------------------------------------------------------------------------------

    def _chain(self, info: ClassInfo, name: str, heap: Heap, skip_first_overlay: bool = False):
        """``("value", v)`` (a class overlay), ``("method", Method)``, ``("data", v)`` or ``None``."""
        cur, first = info, True
        while cur is not None:
            if not (first and skip_first_overlay):
                overlay = heap.cells.get(cur.cls)
                if overlay is not None and name in overlay:
                    return "value", overlay[name]
            first = False
            if name in cur.methods:
                return "method", cur.methods[name]
            if name in cur.data:
                return "data", cur.data[name]
            cur = cur.base
        return None

    def _held(self, hit, what: str):
        kind, v = hit
        if kind == "method":
            raise Unsupported(f"{what} is a method used as a value")
        if isinstance(v, _Opaque):
            raise Unsupported(f"{what} holds a {v.what}")
        if v is POISON:
            raise Unsupported(f"{what}: the paths disagree on what it holds")
        return v

    # -- attribute loads and stores -------------------------------------------------------------------

    def load(self, base, name: str):
        """``base.name`` for ``base`` an object or a class of the program."""
        self._check_attr_name(name)
        if isinstance(base, Ref):
            self._fresh(base)
            cell = self.heap.cells[base.oid]
            what = f"{self.obj_cls[base.oid].name} object attribute {name}"
            if name in cell:
                return self._held(("value", cell[name]), what)
            self._classes_known()
            hit = self._chain(self.obj_cls[base.oid], name, self.heap)
            if hit is None:
                raise _Dead()                   # AttributeError: the path ends
            return self._held(hit, what)
        if self.table.is_user_class(base):
            self._classes_known()
            hit = self._chain(self.table.info(base), name, self.heap)
            if hit is None:
                raise _Dead()
            return self._held(hit, f"class {base.__name__} attribute {name}")
        raise Unsupported(f"attribute {name} of {_kind_name(base)}")

    def store(self, base, name: str, value) -> None:
        self._check_attr_name(name)
        self._check_value(name, value)
        if isinstance(base, Ref):
            self._fresh(base)
            self.heap.cells[base.oid] = {**self.heap.cells[base.oid], name: value}
        elif self.table.is_user_class(base):
            self.table.info(base)
            self.heap.cells[base] = {**self.heap.cells.get(base, {}), name: value}
        else:
            raise Unsupported(f"attribute store on {_kind_name(base)}")
        self.version += 1

    # -- isinstance -------------------------------------------------------------------------------------

    def isinstance_ref(self, ref: Ref, types_) -> bool:
        self._fresh(ref)
        out = False
        for t in (types_ if isinstance(types_, tuple) else (types_,)):
            if not isinstance(t, type) or type(t) is not type:
                raise Unsupported("isinstance of an object against something that is not a plain class")
            out = out or issubclass(ref.cls, t)
        return out

    # -- calls ---------------------------------------------------------------------------------------------

    def call(self, n: ast.Call):
        """``(True, value)`` for a call that is a user-class call, ``(False, None)`` when it is none of them."""
        v = self._call(n)
        return (False, None) if v is _NO else (True, v)

    def _call(self, n: ast.Call):
        f = n.func
        if isinstance(f, ast.Name):
            return self._call_name(n, f.id)
        if isinstance(f, ast.Attribute):
            return self._call_attribute(n, f)
        return _NO

    def _is_program_name(self, name: str) -> bool:
        r = self.r
        return name not in r.env and not (r.frames and name in r.frames[0])

    def _call_name(self, n: ast.Call, name: str):
        if not self._is_program_name(name):
            return _NO
        cls = self.table.g.get(name)
        if not self.table.is_user_class(cls):
            return _NO
        ct = self._require(name, "user_class")
        if ct["name"] != name:
            raise Unsupported(f"call {name}: the call_target names {ct['name']}")
        info = self.table.info(cls)
        args = [self.r.ev(a) for a in n.args]
        return self._construct(info, args, n)

    def _construct(self, info: ClassInfo, args: list, n: ast.Call):
        self._classes_known()
        hit = self._chain(info, "__init__", Heap())      # a dunder is never overlaid
        if hit is None:
            if args or n.keywords:
                raise _Dead()                    # TypeError: the class takes no arguments
            return self._alloc(info)
        kind, init = hit
        if kind != "method" or init.kind != "function":
            raise Unsupported(f"class {info.name}: __init__ is not a plain method")
        ref = self._alloc(info)
        res = self.r._inline(init.fd, [ref] + args, n, qualname=f"{init.owner.name}.__init__", ctx=(init.owner, "function"))
        if isinstance(res, Union):
            raise Unsupported("__init__ returns a value whose type depends on the path")
        if res is not None:
            raise _Dead()                        # TypeError: __init__() should return None
        return ref

    def _receiver(self, base: ast.expr):
        """``(value, stash)`` of a call's receiver expression, or ``(_NO, False)`` when the call is not an
        object call.  A receiver that runs code (a call) is evaluated exactly once: when it turns out not to be an
        object, ``stash`` says the value is handed to the generic code through ``_Runner._pre``."""
        r = self.r
        if isinstance(base, ast.Name):
            if self._is_program_name(base.id):
                v = self.table.g.get(base.id)
                if not self.table.is_user_class(v):
                    return _NO, False
                self.check_class_name(base.id, v)
                return v, False
            return r.env.get(base.id, _NO) if base.id in r.env else r.frames[0].get(base.id, _NO), False
        pure = not any(isinstance(x, ast.Call) for x in ast.walk(base))
        try:
            v = r.ev(base)
        except Unsupported:
            if pure:
                return _NO, False
            raise
        return v, not pure

    def _call_attribute(self, n: ast.Call, f: ast.Attribute):
        base = f.value
        if (isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == "super"
                and not base.args and not base.keywords and self._is_program_name("super")):
            return self._call_super(n, f)
        v, stash = self._receiver(base)
        if isinstance(v, Ref):
            return self._call_on_object(n, f, v)
        if self.table.is_user_class(v):
            return self._call_on_class(n, f, v)
        if stash:
            self.r._pre[(id(base), self.r.depth)] = v
        return _NO

    def _args(self, n: ast.Call) -> list:
        return [self.r.ev(a) for a in n.args]

    def _call_on_object(self, n: ast.Call, f: ast.Attribute, ref: Ref):
        name = f.attr
        self._check_method_name(name)
        self._fresh(ref)
        if name in self.heap.cells[ref.oid]:
            raise Unsupported(f"call {ast.unparse(f)}: an instance attribute shadows the method")
        self._classes_known()
        info = self.obj_cls[ref.oid]
        hit = self._chain(info, name, self.heap)
        if hit is None:
            self._missing(name)
        kind, m = hit
        if kind != "method":
            raise Unsupported(f"call {ast.unparse(f)}: a class attribute that is not a method")
        args = self._args(n)
        if m.kind == "function":
            args = [ref] + args
        elif m.kind == "class":
            args = [ref.cls] + args
        return self.r._inline(m.fd, args, n, qualname=f"{m.owner.name}.{name}", ctx=(m.owner, m.kind))

    def _call_on_class(self, n: ast.Call, f: ast.Attribute, cls):
        callee = ast.unparse(f)
        ct = self._require(callee, "user_method")
        self._check_method_name(f.attr)
        self._classes_known()
        hit = self._chain(self.table.info(cls), f.attr, self.heap)
        if hit is None:
            self._missing(f.attr)
        kind, m = hit
        if kind != "method":
            raise Unsupported(f"call {callee}: a class attribute that is not a method")
        if ct["name"] != f"{m.owner.name}.{f.attr}":
            raise Unsupported(f"call {callee}: the call_target names {ct['name']}, the live class has "
                              f"{m.owner.name}.{f.attr}")
        args = self._args(n)
        if m.kind == "class":
            args = [cls] + args
        return self.r._inline(m.fd, args, n, qualname=f"{m.owner.name}.{f.attr}", ctx=(m.owner, m.kind))

    def _call_super(self, n: ast.Call, f: ast.Attribute):
        callee = ast.unparse(f)
        ct = self._require(callee, "user_method")
        self._check_method_name(f.attr)
        ctx = self.r.ctx_stack[-1]
        if ctx is None or ctx[1] != "function":
            raise Unsupported("super() outside an instance method")
        owner, _kind, first = ctx
        recv = self.r.env.get(first)
        if not isinstance(recv, Ref):
            raise Unsupported("super() with a receiver that is not an object")
        self._fresh(recv)
        self._classes_known()
        if owner.base is None:
            raise Unsupported("super() of a class with no base of the program")
        hit = self._chain(owner.base, f.attr, self.heap)
        if hit is None:
            self._missing(f.attr)
        kind, m = hit
        if kind != "method":
            raise Unsupported(f"call {callee}: a class attribute that is not a method")
        if ct["name"] != f"{m.owner.name}.{f.attr}":
            raise Unsupported(f"call {callee}: the call_target names {ct['name']}, the live class has "
                              f"{m.owner.name}.{f.attr}")
        args = self._args(n)
        if m.kind == "function":
            args = [recv] + args
        elif m.kind == "class":
            args = [recv.cls] + args
        return self.r._inline(m.fd, args, n, qualname=f"{m.owner.name}.{f.attr}", ctx=(m.owner, m.kind))

    # -- forking and merging the heap -----------------------------------------------------------------

    def fork(self) -> Heap:
        return self.heap.fork()

    def merge(self, guard, a: Heap, b: Heap) -> Heap:
        """The heap of the path that took arm ``a`` when ``guard`` holds and arm ``b`` otherwise."""
        if a is b:
            return a
        out = {}
        for key in set(a.cells) | set(b.cells):
            ca, cb = a.cells.get(key), b.cells.get(key)
            if ca is cb:
                out[key] = ca
            elif ca is None or cb is None:
                if isinstance(key, int):
                    out[key] = ca if cb is None else cb          # an object born on one arm only
                else:
                    out[key] = self._merge_cell(key, guard, ca or {}, cb or {}, a, b)
            else:
                out[key] = self._merge_cell(key, guard, ca, cb, a, b)
        return Heap(out, a.stale or b.stale)           # the first reason either path has

    def _merge_cell(self, key, guard, ca: dict, cb: dict, a: Heap, b: Heap) -> dict:
        merged = {}
        for name in set(ca) | set(cb):
            va, vb = ca.get(name, _NO), cb.get(name, _NO)
            if va is _NO or vb is _NO:
                # one arm never stored it: that arm reads what the class says (or has no such attribute)
                va = self._fallback(key, name, a) if va is _NO else va
                vb = self._fallback(key, name, b) if vb is _NO else vb
            if va is POISON or vb is POISON:
                merged[name] = POISON
            elif va is vb:
                merged[name] = va
            else:
                try:
                    merged[name] = _typed_ite(guard, va, vb)
                except Unsupported:
                    merged[name] = POISON
        return merged

    def _fallback(self, key, name: str, heap: Heap):
        """What reading ``key.name`` gives on a path whose own cell has no ``name``."""
        if isinstance(key, int):
            hit = self._chain(self.obj_cls[key], name, heap)
        else:
            hit = self._chain(self.table.info(key), name, heap, skip_first_overlay=True)
        if hit is None or hit[0] == "method" or isinstance(hit[1], _Opaque):
            return POISON
        return hit[1]

    # -- a skipped statement ------------------------------------------------------------------------------

    def after_skip(self, st: ast.stmt, version_before: int, why: str) -> None:
        """The executor skipped ``st`` (for ``why``): the heap is unknown unless ``st`` provably did nothing."""
        if self.version == version_before and self._heap_neutral(st):
            return
        if not self.heap.stale:
            self.heap.stale = f"line {st.lineno}: {why}"

    def _heap_neutral(self, st: ast.stmt) -> bool:
        r = self.r
        for node in ast.walk(st):
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                return False
            if isinstance(node, (ast.Delete, ast.With, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom,
                                 ast.FormattedValue, ast.JoinedStr, ast.Global, ast.Nonlocal, ast.Import,
                                 ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Try)):
                return False
            if isinstance(node, ast.Call):
                f = node.func
                if not (isinstance(f, ast.Name) and f.id in _PURE_BUILTINS and f.id not in r.env
                        and f.id not in r.defs and f.id not in self.table.g
                        and not (r.frames and f.id in r.frames[0])):
                    return False
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                return False                    # ``"%s" % obj`` formats
            if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.AugAssign, ast.For, ast.comprehension,
                                 ast.BoolOp, ast.If, ast.While, ast.IfExp, ast.Assert, ast.Subscript)):
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Attribute, ast.Subscript, ast.Call)):
                        return False
                    if isinstance(sub, ast.Name) and isinstance(
                            r.env.get(sub.id, r.frames[0].get(sub.id) if r.frames else None),
                            (Ref, list, tuple, type)):
                        return False
        return True

    def may_run_user_code(self, st: ast.stmt) -> bool:
        """Does ``st`` call a class of the program or a method (so a skipped ``st`` may hide a site)?"""
        for node in ast.walk(st):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute) or (isinstance(f, ast.Name) and f.id in self.table.ast):
                    return True
        return False

    # -- which sites a statement can reach through classes ------------------------------------------------

    def build_reach(self, defs: dict, targets) -> dict:
        """``{function name: target lines in it or in anything it calls}`` -- the plain-function reach of
        ``_targets_by_function`` extended with the constructors and methods a call can run: a call of a class
        reaches all of its methods, a call of ``x.m(...)`` every method called ``m`` (over-approximations: this
        only decides which sites an unmodelled statement makes ``unsupported``)."""
        nodes: dict = {}
        for name, fd in defs.items():
            nodes[name] = fd
        methods_of: dict = {}
        by_method: dict = {}
        for cname, cnode in self.table.ast.items():
            for st in cnode.body:
                if isinstance(st, ast.FunctionDef):
                    key = f"{cname}.{st.name}"
                    nodes[key] = st
                    methods_of.setdefault(cname, []).append(key)
                    by_method.setdefault(st.name, []).append(key)
        reach = {k: {t for t in targets if fd.lineno <= t <= (fd.end_lineno or fd.lineno)} for k, fd in nodes.items()}
        edges = {k: self._callees(fd, defs, methods_of, by_method) for k, fd in nodes.items()}
        changed = True
        while changed:
            changed = False
            for k, outs in edges.items():
                for o in outs:
                    if not reach[o] <= reach[k]:
                        reach[k] |= reach[o]
                        changed = True
        self.class_reach = {c: set().union(*(reach[m] for m in ms)) for c, ms in methods_of.items()}
        self.method_reach = {m: set().union(*(reach[k] for k in ks)) for m, ks in by_method.items()}
        return {k: v for k, v in reach.items() if v and k in defs}

    @staticmethod
    def _callees(fd, defs, methods_of, by_method) -> set:
        out = set()
        for n in ast.walk(fd):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Name):
                if f.id in defs:
                    out.add(f.id)
                out.update(methods_of.get(f.id, ()))
            elif isinstance(f, ast.Attribute):
                out.update(by_method.get(f.attr, ()))
        return out

    def reach_of_call(self, n: ast.Call) -> set:
        f = n.func
        if isinstance(f, ast.Name):
            return self.class_reach.get(f.id, set())
        if isinstance(f, ast.Attribute):
            return self.method_reach.get(f.attr, set())
        return set()


class _No:
    """The "not applicable" / "missing" sentinel."""

    def __repr__(self):
        return "<no>"


_NO = _No()
