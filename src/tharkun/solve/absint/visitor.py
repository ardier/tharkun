"""absint as a visitor of the contract rules ([internal ref]): the entry points through which the interval/shape analysis
reads a contract, one ``visit_<kind>`` per rule kind it consumes.

``solve/contracts/rules.py`` gives every contract one :class:`~tharkun.solve.contracts.rules.Rule` class (its kind is
the visitor dispatch key).  absint reaches a contracted callable as a ``static`` tag (``interp.py::Analyzer.call_static``:
``mfunc`` for a ``math`` function, ``cbuiltin`` for a contract-modelled builtin, ``cfunc`` for any other module
function), resolves the tag to the rule's qualname and asks the rule to ``accept`` this visitor.  A rule kind with no
``visit_<kind>`` here is a NAMED refusal (:class:`~tharkun.solve.contracts.rules.Refused`), which the analyzer reports as
``Unsupported(str(refused))``: the engine says which rule it could not handle, never a silent pass.

The visits are the analyzer's existing transfer functions, unchanged -- this module only makes the dispatch a fact the
registry can be asked about (:meth:`Visitor.implements`, :meth:`Visitor.cell`), which is what the engine-by-contract
audit table (``tests/contracts/_engine_audit.py``) reads:

* ``math_function``: ``Analyzer._math_call`` (the contract's seven classes, the model table's shapes and ranges);
* ``math_constant`` (``pi``, ``e``, ``tau``, ``inf``, ``nan``): ``Analyzer.module_attr``, which reads the exact float of the contract;
* ``ord``, ``chr``, ``tuple``, ``open_output``, ``sum``, ``any``, ``all``, ``min``, ``max``, ``sorted``, and the builtins the
  analyzer modelled by hand before it read their contracts (``abs``, ``len``, ``range``, ``int``, ``float``, ``list``, ``print``,
  ``isinstance``; ``Analyzer._modelled``): ``Analyzer._contract_builtin_call`` (``c_<name>``: the contract is the
  authority on which property classes a call can raise, the model argues away what it can);
* every other MODULE function with a rule that the analyzer neither models by hand nor refuses (``serve_module_functions``,
  called by ``interp.py`` once ``MODULES`` exists): ``Analyzer._contract_call``, the generic contract-driven call (an
  honest, total check per property class, the hand-verified precise facts of ``adapters/common.py`` where one applies, a
  conservative havoc of a mutated receiver otherwise).  Whether such a call is REACHED depends on the analyzer admitting
  the module's import, which the audit probes at run time;
* the set family (``set.*``: the methods, ``set.__or__`` ... the operators and in-place forms, ``set.__lt__`` ... the comparisons,
  ``set.__contains__``; and the constructors ``set`` and ``frozenset``, which are builtin visits): ``Analyzer.set_method``,
  ``set_binop``, ``set_ordering``, ``set_equality`` and ``set_membership`` (``sets.py::SetsMixin``) each recognise a set operand
  and then ask THEIR rule to accept this visitor with a :class:`SetCtx` (``serve_set_operations`` gives every ``set.*`` rule its
  ``visit_<kind>``); the visit runs the transfer function, whose checks are the contract's seven classes upgraded by what the
  model proved;
* the object protocols the analyzer dispatches by hand for a user instance (``protocol.binary_operator``, ``inplace_operator``, ``comparison``,
  ``container_getitem``, ``container_len``, ``iteration``, ``truthiness``, ``attribute_access``; ``PROTOCOL_DISPATCHES``, a
  :class:`ProtocolCtx`): ``Analyzer._through_protocol`` runs the unchanged dispatch and records the seven classes at its node;
* the six methods of a write-only output file (``output_write`` ... ``output_exit``, text and binary handle alike):
  ``Analyzer._outfile_method``, reached from a method call on a handle and from a ``with``;
* ``protocol_length_hint`` (a protocol, so the context is not a call: :class:`LengthHintCtx`): ``Analyzer.iterable`` asks
  it whether a consumer probes ``__len__`` and, if so, runs the probe.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from tharkun.solve.contracts.adapters.str_shapes import INT_RESULT_LOWER, bind as bind_str_call
from tharkun.solve.contracts.rules import Refused, Rule, Visitor, all_rules, resolve_native

from .heap import ListObj, TupleObj

#: the modules whose functions the generic contract-driven call serves (``math`` has its own visit)
_MODULES = frozenset({"bisect", "collections", "copy", "functools", "heapq", "html", "io", "itertools", "operator", "random", "re", "textwrap", "string", "statistics"})
#: a shape contract's qualname -> the builtin the analyzer's model is named after (``common.BUILTIN_SHAPE_CONTRACTS``)
_BUILTIN_OF = {"open.output": "open"}

# Binary64 rounds this last integer to max finite; the next midpoint rounds to infinity and overflows.
_FLOAT_CONVERSION_MAX_INT = 2**1024 - 2**970 - 1


@dataclass(frozen=True)
class CallCtx:
    """What a visit needs to run one call in the analyzer: the abstract state (mutated by the call), the abstract
    argument values, the keyword values, and the AST node (where checks are recorded)."""

    state: Any
    args: tuple
    kwargs: dict
    node: ast.AST


@dataclass(frozen=True)
class MethodCtx:
    """One method call on a builtin heap kind the analyzer models by hand (``list``, ``dict``, ``io.StringIO``): the abstract
    state (mutated by the call), the receiver addresses, the method name and the call's values."""

    state: Any
    receivers: frozenset
    name: str
    args: tuple
    kwargs: dict
    node: ast.AST


@dataclass(frozen=True)
class SubscriptCtx:
    """A subscript (``x[k]`` or ``x[k] = v``) of a builtin container the analyzer has already modelled: the rule's qualname, the node,
    the checks ``(classes, ok)`` the model recorded at it, and the classes of the rule that are the call's own logic which the key
    shape settles (``IndexError`` for a slice key, or a key that is not an int)."""

    qualname: str
    node: ast.AST
    seen: tuple
    native: dict


@dataclass(frozen=True)
class ProtocolCtx:
    """The dispatch of an object protocol (``protocol.binary_operator`` ...) the analyzer implements by hand: the rule's qualname, the node
    where the dispatch happens, ``run`` (a zero-argument callable closed over the state and the operands: the unchanged dispatch) and
    the classes of the rule the dispatch argued without recording a check (``native``: class -> why)."""

    qualname: str
    node: ast.AST
    run: Any
    native: Any = None


@dataclass(frozen=True)
class OutfileCtx:
    """One method call on a write-only output file: the handle kind (``binary``), the receiver address (``None``
    for the ``__exit__`` of a ``with``, whose handle may be gone), the method name and the call's values."""

    state: Any
    binary: bool
    receiver: Any
    name: str
    args: tuple
    kwargs: dict
    node: ast.AST


@dataclass(frozen=True)
class LengthHintCtx:
    """A callable ``consumer`` (``"list"``, ``"sorted"``, ``"list.extend"``, ...) about to iterate an instance ``part``
    whose class defines ``__len__``: CPython's ``PyObject_LengthHint`` may call it first, and swallows a TypeError."""

    state: Any
    part: Any
    node: ast.AST
    consumer: str


@dataclass(frozen=True)
class SetCtx:
    """One operation on a set or frozenset that ``sets.py`` has recognised (a method, an operator, a comparison, ``in``):
    ``run`` is the transfer function for it, a zero-argument callable closed over the abstract state and the operands
    (the entry points differ in what they take and return -- a value, a ``(value, TypeError possible)`` pair, a
    bool -- so the context carries the call, not its arguments)."""

    node: ast.AST
    run: Any


def is_module_function(qualname: str) -> bool:
    """``module.function`` (exactly one dot, a module of the registry) -- not ``TYPE.method``, not ``a.b.c``."""
    head, dot, tail = qualname.partition(".")
    return bool(dot) and "." not in tail and head in _MODULES


def exact_int_value(v) -> bool:
    """``v`` is an int or a bool and nothing else: no ``top``, no object, no str, float, None or static part.  Such a value is read
    by ``__index__`` without running any user code."""
    return (not v.top and not v.st and not v.refs and v.s is None and not v.f and not v.n
            and (v.i is not None or bool(v.b)))


def int_or_none_value(v) -> bool:
    """``v`` is an int, a bool or None, and nothing else (a slice BOUND: ``s[:n]``, ``s[:None]``, ``s[:(1 if c else None)]``): no
    ``top``, object, static, str or float part."""
    return (not v.top and not v.st and not v.refs and v.s is None and not v.f
            and (v.i is not None or bool(v.b) or bool(v.n)))


def non_int_value(v) -> bool:
    """``v`` has no int, bool or None part: used as a slice bound it is a certain ``TypeError`` (a float, a str, a container, a builtin
    object; ``top`` and a value with no part at all are not)."""
    return (not v.top and v.i is None and not v.b and not v.n
            and (v.s is not None or bool(v.f) or bool(v.refs) or bool(v.st)))


#: the heap kinds a str subscript key (or slice bound) may be without running user code: builtin containers and objects, none of which has
#: ``__index__``, so using one is a ``TypeError`` raised by a type check, not by a call (a user instance or class is NOT here: its
#: ``__index__`` or metaclass could run)
NO_INDEX_KINDS = frozenset({"tuple", "list", "dict", "range", "array", "dictview", "iterator", "set"})

#: the static objects a str subscript key (or slice bound) may be: builtin functions, types, exception classes and instances, modules
#: (none has ``__index__``; ``native-flag`` is NOT here: a ``re.RegexFlag`` is an int, so ``s[re.I]`` is valid)
NO_INDEX_TAGS = frozenset({"builtin", "type", "exc", "excinst", "module", "mfunc", "cfunc", "cbuiltin"})


def hook_free_value(state, v) -> bool:
    """``v`` is a value that no user code can run on when it is used as a subscript bound or key: not ``top``, a static part only of a
    builtin function, type, exception or module, an object part only of a builtin container or object (``NO_INDEX_KINDS``): it has no
    ``__index__``, so using it is a ``TypeError``."""
    return (not v.top and all(tag[0] in NO_INDEX_TAGS for tag in v.st)
            and all(state.heap[r].kind in NO_INDEX_KINDS for r in v.refs))


def str_key_parts(v):
    """What a str subscript KEY may do ([internal ref]), or ``None`` when a part of it is not understood (``top``, a static object that is
    neither a slice tag nor in ``NO_INDEX_TAGS``): a triple ``(type_error, index_error, certain)``, whether some part of the key may
    raise a ``TypeError``, whether some part may be an int index (the ``IndexError`` of an index out of range), and whether the
    ``TypeError`` is CERTAIN (every part of the key is one that raises it).  No part runs user code: a key is an int or bool (read
    as itself), a float, str, None, builtin container or builtin static object (a ``TypeError``: ``str.__getitem__`` type-checks it
    without calling anything), or a slice.  The slice tags ``Analyzer.subscript_load`` builds: ``("native-slice", slice(...))`` for
    finitely many constant bounds (a ``TypeError`` unless every bound is an int, bool or None), ``("int-slice",)`` for bounds that are
    ints or None but not finitely many (``s[:len(t)]``, never raises), ``("any-slice",)`` for hook-free bounds that may not be ints
    (``s[:x]`` with ``x`` a float or an int: a ``TypeError`` only for the float, so not certain) and ``("bad-slice",)`` for a slice
    one of whose hook-free bounds has no int or None part at all (``s[:[1]]``, ``s[:range(2)]``: a certain ``TypeError``).  A slice never raises ``IndexError``
    (out-of-range bounds are clamped)."""
    if v.top:
        return None
    type_error = bool(v.s is not None or v.f or v.n or v.refs)
    index_error = bool(v.i is not None or v.b)
    good = index_error                      # a part that may be a valid key
    for tag in v.st:
        if tag[0] == "int-slice":
            good = True
            continue
        if tag[0] == "any-slice":
            type_error = good = True
            continue
        if tag[0] == "bad-slice":                 # some bound is certainly not an int or None: a certain TypeError
            type_error = True
            continue
        if tag[0] in NO_INDEX_TAGS:
            type_error = True
            continue
        if tag[0] != "native-slice":
            return None
        if all(x is None or type(x) in (int, bool) for x in (tag[1].start, tag[1].stop, tag[1].step)):
            good = True
        else:
            type_error = True
    return type_error, index_error, type_error and not good


def definite_str(v) -> bool:
    """``v`` is a builtin ``str`` and nothing else (no ``top``, object, static, int, float or None part)."""
    return not v.top and v.s is not None and v.replace(s=None).is_bot()


def _str_kind_ok(state, kind: str, v) -> bool:
    """Whether ``v`` is of the KIND ``contracts/adapters/str_shapes.py`` names, so that no user code can run on it as an argument of a
    ``str`` method (an ``__index__``, ``__iter__``, ``__bool__``): a builtin ``str``, ``int``, ``bool``, ``None``, or a ``list``/``tuple`` of
    ``str`` and nothing else.  A part of any other kind -- ``top``, a float where an int is taken, a static object, an instance of a
    program class, a subclass of a builtin, an ``array`` or set where a list is taken -- is not admitted."""
    if v.top or v.st:
        return False
    if kind == "f":
        from .aval import ANY_STR
        return definite_str(v) and v.s != ANY_STR and all(len(x) == 1 for x in v.s)
    if kind == "s":
        return definite_str(v)
    if kind == "c":
        return not v.refs and not v.f and v.i is None and not v.b and (v.s is not None or bool(v.n))
    if kind in ("i", "n"):
        return (not v.refs and not v.f and v.s is None and (kind == "n" or not v.n)
                and (v.i is not None or bool(v.b) or (kind == "n" and bool(v.n))))
    if v.i is not None or v.b or v.f or v.n:
        return False
    if kind == "S":
        if v.s is None and not v.refs:
            return False
        return all(type(state.heap[r]) is TupleObj and state.heap[r].items is not None
                   and all(definite_str(x) for x in state.heap[r].items) for r in v.refs)
    if kind == "L":
        # a list or tuple whose element summary (``elem``, plus ``pre`` of a segmented list, or the items of a tuple) is str only: the
        # summary is universal and the analysis keeps it true across aliases, calls and stores, so a later non-str element is a
        # different summary, not this one
        return v.s is None and bool(v.refs) and all(
            type(state.heap[r]) in (ListObj, TupleObj) and all(x.is_bot() or definite_str(x) for x in state.heap[r].values())
            for r in v.refs)
    return False


def str_call_admitted(state, method: str, args, kwargs):
    """The binding ``{name: (kind, value)}`` of ``<args[0]>.<method>(*args[1:], **kwargs)`` when the receiver is a definite builtin ``str`` and
    every argument has exactly the kind the method takes, else ``None``.  An admitted call runs no user code and raises none of the
    seven property classes (``contracts/adapters/str_shapes.py``; ``tests/contracts/test_str_shapes.py`` measures it on CPython)."""
    if not args or not definite_str(args[0]):
        return None
    bound = bind_str_call(method, args[1:], kwargs)
    if bound is None or not all(_str_kind_ok(state, kind, v) for kind, v in bound.values()):
        return None
    return bound


class AbsintVisitor(Visitor):
    engine = "absint"

    def __init__(self, analyzer) -> None:
        self.analyzer = analyzer

    def visit_math_function(self, rule: Rule, ctx: CallCtx):
        return self.analyzer._math_call(ctx.state, ctx.args, ctx.kwargs, ctx.node, rule.qualname.split(".", 1)[1])

    def _builtin(self, rule: Rule, ctx: CallCtx):
        name = _BUILTIN_OF.get(rule.qualname, rule.qualname)
        return self.analyzer._contract_builtin_call(ctx.state, name, ctx.args, ctx.kwargs, ctx.node)

    visit_ord = visit_chr = visit_tuple = visit_open_output = _builtin

    def visit_open_read(self, rule: Rule, ctx: CallCtx):
        from tharkun.solve.contracts.adapters.absint import contract_for_builtin
        from .interp import Unsupported
        contract = contract_for_builtin("open.read")
        if contract is None:
            raise Unsupported("open.read has no tested contract", ctx.node)
        result, proven = self.analyzer.c_world_open(ctx.state, ctx.args, ctx.kwargs, ctx.node)
        self.analyzer._contract_checks(ctx.node, "open.read", contract, proven)
        return result
    visit_set = visit_frozenset = _builtin
    visit_sum = visit_any = visit_all = _builtin
    visit_min = visit_max = visit_sorted = visit_divmod = visit_round = visit_pow = _builtin
    #: the builtins the analyzer modelled by hand before it read their contracts ([internal ref] step 2): the model is unchanged,
    #: the contract supplies the seven classes and the model upgrades the ones it has argued (``Analyzer._modelled``)
    visit_abs = visit_len = visit_range = visit_int = visit_float = visit_list = visit_print = visit_isinstance = _builtin
    visit_bool = visit_callable = visit_bin = visit_hex = visit_oct = visit_issubclass = visit_dict = _builtin
    visit_hasattr = visit_zip = visit_enumerate = visit_map = _builtin

    def visit_html_function(self, rule: Rule, ctx: CallCtx):
        from tharkun.solve.contracts.registry import get
        from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES
        from .aval import ANYSTR, BOT, const_bool
        from .interp import Unsupported
        from .native import abstract, concrete, product

        a, st, node = self.analyzer, ctx.state, ctx.node
        contract = get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine, rule.qualname, 'no tested contract')
        params = ('s', 'quote') if rule.qualname == 'html.escape' else ('s',)
        bound = dict(zip(params, ctx.args))
        valid = len(ctx.args) <= len(params)
        for key, value in ctx.kwargs.items():
            valid = valid and key in params and key not in bound
            bound[key] = value
        valid = valid and 's' in bound
        a.check(node, 'arg-type', {'TypeError'}, valid, 'HTML utility argument binding', st=st)
        proven = {exc: 'validated builtin str and bool operands run no target hook' for exc in ALL_PROPERTY_CLASSES}
        if not valid:
            proven['TypeError'] = None
            a._contract_checks(node, rule.qualname, contract, proven)
            st.set_bottom()
            return BOT
        text = bound['s']
        if text.s is None or not text.replace(s=None).is_bot():
            raise Unsupported(f'{rule.qualname}: requires exact builtin str operands', node)
        args = [bound['s']]
        if len(params) == 2:
            quote = bound.get('quote', const_bool(True))
            if not quote.b or not quote.replace(b=frozenset()).is_bot():
                raise Unsupported('html.escape: unmodelled quote truth hook or kind', node)
            args.append(quote)
        values = product([concrete(a, st, arg) for arg in args], analyzer=a)
        # shortcut: long/nonfinite text keeps only str type; add string relations when needed.
        # The length bound also avoids process-dependent decimal character-reference conversion.
        result = (abstract(a, st, [rule.native(*xs) for xs in values], node)
                  if values is not None and all(len(xs[0]) <= 640 for xs in values) else ANYSTR)
        a._contract_checks(node, rule.qualname, contract, proven)
        return result

    def visit_statistics_query(self, rule: Rule, ctx: CallCtx):
        import inspect
        from tharkun.solve.contracts.registry import get
        from .aval import BOT, FLOAT, int_iv
        from .errors import Unsupported
        from .native import abstract, concrete, product
        from .state import vjoin
        a,st,node=self.analyzer,ctx.state,ctx.node
        q=rule.qualname;contract=get(q)
        if contract is None or not contract.tested or contract.returns is None or not contract.returns.tested:
            return Refused(self.engine,q,'no tested query return contract')
        try:
            inspect.signature(resolve_native(q)).bind(*ctx.args,**ctx.kwargs)
        except TypeError:
            a.check(node,q+':TypeError',{'TypeError'},False,'statistics argument binding',st=st,contract_qualname=q)
            a._contract_checks(node,q,contract,{c.exc:'binding fails before operand hooks' for c in contract.raises},recorded={'TypeError'})
            st.set_bottom()
            return BOT
        def check_work():
            if a._clock() > a.deadline:
                raise Unsupported('statistics query time budget exhausted',node)

        check_work()
        # Preflight tuple slots before concrete() allocates their Cartesian pools.
        for value in (*ctx.args,*ctx.kwargs.values()):
            for address in value.refs:
                obj=st.heap[address]
                if type(obj) is TupleObj and obj.items is not None:
                    for item in obj.items:
                        check_work()
                        if item.refs:
                            return Refused(self.engine,q,'primitive data elements required')
        if q=='statistics.quantiles':
            from .aval import V, const_int, const_str
            count=ctx.kwargs.get('n',const_int(4))
            if not count.replace(i=None,b=frozenset()).is_bot():
                return Refused(self.engine,q,'quantile index hooks and noninteger n are not modelled')
            bounds=([a.ibounds(st,count.i)] if count.i is not None else [])
            if count.b:
                bounds.append((int(min(count.b)),int(max(count.b))))
            if not bounds or any(lo is None or lo<1 for lo,hi in bounds):
                return Refused(self.engine,q,'positive integer n required; qualified StatisticsError transfer is not modelled')
            lo=min(lo for lo,hi in bounds)
            hi=None if any(hi is None for lo,hi in bounds) else max(hi for lo,hi in bounds)
            data=ctx.args[0];elem=BOT
            if not data.refs or not data.replace(refs=frozenset()).is_bot():
                return Refused(self.engine,q,'builtin list/tuple data required; iterator hooks are not modelled')
            method=ctx.kwargs.get('method',const_str('exclusive'))
            for address in data.refs:
                check_work()
                obj=st.heap[address]
                if type(obj) not in (ListObj,TupleObj):
                    return Refused(self.engine,q,'builtin list/tuple data required')
                low,high=a.len_lin(st,obj)[1]
                if low is None or low<1:
                    return Refused(self.engine,q,'nonempty data required; qualified StatisticsError transfer is not modelled')
                part=a.all_elems(st,obj)
                if low==high==1 and part.replace(i=None,b=frozenset(),f=False,s=None,n=False).is_bot():
                    pass  # CPython repeats the single element without interpolation or checking method.
                elif (method.s is not None and method.s!='ANY'
                      and method.replace(s=None).is_bot()
                      and all(x in ('inclusive','exclusive') for x in method.s)
                      and not part.is_bot() and part.replace(i=None,b=frozenset(),f=False).is_bot()):
                    # Clamped interpolation indices stay in range and positive n cannot divide by zero.
                    # Successful builtin numeric true division returns float, without target hooks.
                    part=vjoin(part,st.zone,FLOAT,st.zone) if low==1 else FLOAT
                else:
                    return Refused(self.engine,q,'quantile ordering/arithmetic hooks or invalid method are not modelled')
                elem=vjoin(elem,st.zone,part,st.zone)
            check_work()
            length=int_iv(lo-1,None if hi is None else hi-1).i
            if count.only_int() and count.i.lin is not None:
                from .linear import IntInfo
                length=IntInfo(count.i.lin.shift(-1),length.lo,length.hi)
            addr=a.new_list(st,node,length,elem,tag='native:quantiles')
            check_work()
            a._contract_checks(node,q,contract,{c.exc:'closed builtin quantile ordering, clamped indices and positive integer divisor' for c in contract.raises})
            return V(refs=frozenset({addr}))
        keys=tuple(ctx.kwargs)
        try:
            rows=product([concrete(a,st,x) for x in (*ctx.args,*ctx.kwargs.values())], analyzer=a)
        except MemoryError:
            return Refused(self.engine,q,'statistics query memory exhausted')
        check_work()
        def safe(value):
            check_work()
            if type(value) in (list,tuple):
                return all(type(x) in (int,bool,float,str) and safe(x) for x in value)
            return value is None or type(value) in (bool,str,int,float)
        if rows is None or any(not all(safe(x) for x in row) for row in rows):
            return Refused(self.engine,q,'finite primitive data and configuration required')
        outputs=[]
        for row in rows:
            check_work()
            args=row[:len(ctx.args)];kwargs=dict(zip(keys,row[len(ctx.args):]))
            try:
                outputs.append(rule.native(*args,**kwargs))
                check_work()
            except MemoryError:
                return Refused(self.engine,q,'statistics query memory exhausted')
            except (TypeError,ValueError,OverflowError,ArithmeticError,AttributeError,IndexError,KeyError,AssertionError) as exc:
                # Qualified StatisticsError must retain its class; never relabel it as ValueError.
                return Refused(self.engine,q,'native '+type(exc).__module__+'.'+type(exc).__qualname__+' transfer is not modelled')
        try:
            result=abstract(a,st,outputs,node)
        except MemoryError:
            return Refused(self.engine,q,'statistics query memory exhausted')
        check_work()
        if result is None:
            return Refused(self.engine,q,'native query result is not represented')
        a._contract_checks(node,q,contract,{c.exc:'exhaustive context-independent primitive alternatives' for c in contract.raises})
        return result

    def visit_bisect_native(self, rule: Rule, ctx: CallCtx):
        import inspect
        import sys
        from tharkun.solve.contracts.registry import get
        from .aval import ANYSTR, NONE, const_int, int_iv
        from .heap import ArrayObj, RangeObj
        from .state import vjoin
        a,st,node=self.analyzer,ctx.state,ctx.node
        q=rule.qualname;insert=q=='bisect.insort';contract=get(q)
        def refused(reason):return Refused(self.engine,q,reason)
        def check_work():
            if a._clock()>a.deadline:
                from .errors import Unsupported
                raise Unsupported('bisection time budget exhausted',node)
        check_work()
        if (contract is None or not contract.tested or contract.returns is None or not contract.returns.tested
                or any(not c.tested for c in contract.raises)
                or contract.returns.type!=('None' if insert else 'int')
                or insert and not contract.effects.mutates_receiver):
            return refused('current tested bisection return, raise and effect claims required')
        a._in_try_refuse('bisection non-property exception routing',node)
        try:bound=inspect.signature(resolve_native(q)).bind(*ctx.args,**ctx.kwargs).arguments
        except TypeError:return refused('native bisection argument binding fails')
        recv,item=bound['a'],bound['x'];lo=bound.get('lo',const_int(0));hi=bound.get('hi',NONE)
        if not exact_int_value(lo) or not (exact_int_value(hi) or hi.n and hi.replace(n=False).is_bot()):
            return refused('open index conversion or invalid bound kind')
        low=a.ibounds(st,a._as_int(st,lo))
        if low[0] is None or low[0]<0 or low[1] is None or low[1]>sys.maxsize:
            return refused('native lo conversion or negative-lo exception is not represented')
        address=obj=None
        if definite_str(recv) and not insert:
            length=a.ibounds(st,a._as_int(st,a._len_model(st,[recv],{},node)));elem=ANYSTR
        elif (not insert and len(recv.st)==1 and recv.replace(st=frozenset(),top=False).is_bot()
              and next(iter(recv.st))[0]=='bytes_const'):
            n=len(next(iter(recv.st))[1]);length=(n,n);elem=int_iv(0,255)
        elif len(recv.refs)==1 and recv.replace(refs=frozenset()).is_bot():
            address=next(iter(recv.refs));obj=st.heap[address]
            if type(obj) in (ListObj,TupleObj,ArrayObj) and (not insert or type(obj) in (ListObj,ArrayObj)):
                length=a.len_lin(st,obj)[1];elem=a.all_elems(st,obj)
            elif type(obj) is RangeObj and not insert:
                start=st.zone.bounds(obj.lo) if isinstance(obj.lo,str) else obj.lo
                stop=st.zone.bounds(obj.hi) if isinstance(obj.hi,str) else obj.hi
                left,right=(start,stop) if obj.step>0 else (stop,start)
                step=abs(obj.step)
                upper=(max(0,(right[1]-left[0]+step-1)//step)
                       if right[1] is not None and left[0] is not None else None)
                if upper is None or upper>sys.maxsize:return refused('range length may overflow native indexing')
                lower=(max(0,(right[0]-left[1]+step-1)//step)
                       if right[0] is not None and left[1] is not None else 0)
                length=(lower,upper);elem=int_iv(None,None)
            else:return refused('custom sequence or insertion protocol is not represented')
        else:return refused('one represented builtin sequence required')
        if hi.n:high=length
        else:
            high=a.ibounds(st,a._as_int(st,hi))
            if (high[0] is None or high[0]<-sys.maxsize-1 or high[1] is None or high[1]>sys.maxsize
                    or length[0] is None or high[1]>length[0]):
                return refused('native hi conversion or possible out-of-range search')
            # The C implementation treats explicit -1 as its full-length sentinel too.
            if high[0]<=-1<=high[1]:
                high=(high[0],None if length[1] is None else max(high[1],length[1]))
        empty_search=length[1]==0 or high[1] is not None and low[0]>=high[1]
        key=bound.get('key',NONE)
        if not (key.n and key.replace(n=False).is_bot()) and (insert or not empty_search):
            return refused('key callback effects and exceptions are not represented')
        if not empty_search:
            combined=vjoin(a.summarize(st,elem),st.zone,a.summarize(st,item),st.zone)
            if not a._elem_sortable(combined):return refused('open or mixed comparison protocols')
        check_work()
        if insert:
            if type(obj) is ArrayObj:item=a._array_store_value(st,node,obj.typecode,item)
            merged=vjoin(a.summarize(st,elem),st.zone,a.summarize(st,item),st.zone)
            a.collapse(st,{address})
            a.update_obj(st,{address},lambda ob,strong:a.set_len(st,
                ob.replace(elem=merged,pre=None),strong,
                *a._len_plus(st,ob,1)))
            result=NONE
        else:result=int_iv(low[0],None if high[1] is None else max(low[1],high[1]))
        check_work()
        a._contract_checks(node,q,contract,{c.exc:'native binding, builtin indexing and primitive comparisons' for c in contract.raises})
        return result

    visit_bisect_bisect_left = visit_bisect_insort = visit_bisect_native

    def visit_heapq_native(self, rule: Rule, ctx: CallCtx):
        from tharkun.solve.contracts.registry import get
        from .heap import ListObj
        from .aval import BOT, NONE
        from .state import vjoin
        a,st,node=self.analyzer,ctx.state,ctx.node
        q=rule.qualname;name=q.split('.')[1]
        def unsupported(reason):
            # Keep the established generic effect/precondition transfer for its original three APIs.
            if name in ('heapify','heappush','heappop'):
                return a._contract_call(st,'heapq',name,ctx.args,ctx.kwargs,node)
            return Refused(self.engine,q,reason)
        contract=get(q)
        if contract is None or not contract.tested or contract.returns is None or not contract.returns.tested:
            return Refused(self.engine,q,'no tested heap return contract')
        count=2 if name.startswith(('heappush','heapreplace')) else 1
        proven={c.exc:'primitive comparable elements have no target comparison hook' for c in contract.raises}
        if name in ('heapify','heappush','heappop') and (ctx.kwargs or len(ctx.args)!=count):
            return unsupported('call shape is outside the primitive transfer')
        if not a._arity(node,ctx.args,ctx.kwargs,count,count,q):
            proven['TypeError']=None
            a._contract_checks(node,q,contract,proven)
            st.set_bottom()
            return BOT
        recv=ctx.args[0]
        if len(recv.refs)!=1 or not recv.replace(refs=frozenset()).is_bot():
            return unsupported('one native list receiver required')
        address=next(iter(recv.refs));ob=st.heap[address]
        if type(ob) is not ListObj:
            return unsupported('receiver is not a native list')
        elem=a.borrow(st,a.summarize(st,a.all_elems(st,ob)))
        item=a.borrow(st,a.summarize(st,ctx.args[1])) if count==2 else BOT
        combined=vjoin(elem,st.zone,item,st.zone)
        if not (combined.is_bot() or (not combined.top and not combined.refs and not combined.st and not combined.n
                and (combined.s is not None and combined.replace(s=None).is_bot() or combined.s is None))):
            return unsupported('primitive numeric or str elements required; open comparisons refused')
        lo,hi=a.len_lin(st,ob)[1]
        pops=name.startswith(('heappop','heapreplace'))
        recorded=set()
        if pops:
            args=({'IndexError':a.contract_exception_args(q,'IndexError','native-empty',(),node)} if lo==0 or lo is None else None)
            a.check(node,q+':IndexError',{'IndexError'},lo is not None and lo>=1,'heap must be nonempty',st=st,
                    exact_args=args,contract_qualname=q)
            recorded.add('IndexError')
            if hi==0:
                a._contract_checks(node,q,contract,proven,recorded=recorded)
                st.set_bottom()
                return BOT
        pushpop=name.startswith('heappushpop')
        result=combined if pushpop else elem if pops else NONE
        a.borrow(st,result)
        if not (pushpop and hi==0):
            a.collapse(st,{address})
            delta=1 if name.startswith('heappush') and not pushpop else -1 if name.startswith('heappop') else 0
            new_elem=combined if count==2 else elem
            a.update_obj(st,{address},lambda obj,strong:a.set_len(st,obj.replace(elem=new_elem),strong,*a._len_plus(st,obj,delta)))
        a._contract_checks(node,q,contract,proven,recorded=recorded)
        return a.mat(st,result)

    visit_heapq_heapify = visit_heapq_heappush = visit_heapq_heappop = visit_heapq_native

    def visit_string_capwords(self, rule: Rule, ctx: CallCtx):
        import inspect
        from tharkun.solve.contracts.registry import get
        from .aval import ANYSTR, ANY_STR, BOT, NONE, const_str
        from .native import abstract, concrete, product
        a,st,node=self.analyzer,ctx.state,ctx.node
        contract=get(rule.qualname)
        if contract is None or not contract.tested or contract.returns is None or not contract.returns.tested:
            return Refused(self.engine,rule.qualname,'no tested return contract')
        proven={c.exc:'builtin str input and str-or-None separator execute no target hook' for c in contract.raises}
        try:
            bound=inspect.signature(resolve_native(rule.qualname)).bind(*ctx.args,**ctx.kwargs)
        except TypeError:
            a.check(node,'arg-type',{'TypeError'},False,'capwords argument binding',st=st)
            proven['TypeError']=None
            a._contract_checks(node,rule.qualname,contract,proven)
            st.set_bottom()
            return BOT
        text=bound.arguments['s'];sep=bound.arguments.get('sep',NONE)
        if (text.s is None or not text.replace(s=None).is_bot() or not
                (sep.n and sep.replace(n=False).is_bot() or sep.s is not None and sep.replace(s=None).is_bot())):
            return Refused(self.engine,rule.qualname,'requires builtin str and str-or-None separator')
        can_empty=sep.s is not None and (sep.s==ANY_STR or '' in sep.s)
        if can_empty:
            case=next((c for c in contract.other_raises if c.exc=='ValueError' and c.case=='empty-separator' and c.tested),None)
            if case is None:
                return Refused(self.engine,rule.qualname,'no tested empty-separator payload case')
            a.check(node,rule.qualname+':ValueError',{'ValueError'},False,'separator may be empty',st=st,
                    exact_args={'ValueError':tuple(const_str(s) for s in case.args_template)},contract_qualname=rule.qualname)
        a._contract_checks(node,rule.qualname,contract,proven)
        if sep.s==frozenset({''}):
            st.set_bottom()
            return BOT
        rows=product([concrete(a,st,text),concrete(a,st,sep)], analyzer=a)
        # shortcut: bounded finite strings retain exact contents; otherwise retain str type.
        if rows is None or any(len(s)>640 or t is not None and len(t)>640 for s,t in rows):
            return ANYSTR
        return abstract(a,st,[rule.native(s,t) for s,t in rows if t!=''],node)

    def visit_numeric_division(self, rule: Rule, ctx: CallCtx):
        from tharkun.solve.contracts.registry import get
        from tharkun.solve.contracts.rules_numeric_division import DIVISION_METHODS
        from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, ConstantArgsRaiseClaim
        from .aval import BOT, V, const_str, int_iv
        from .native import abstract, concrete, product

        a, st, node = self.analyzer, ctx.state, ctx.node
        contract = get(rule.qualname)
        if contract is None or not contract.tested or contract.returns is None or not contract.returns.tested:
            return Refused(self.engine, rule.qualname, 'no tested numeric division contract')
        owner, name, reverse = DIVISION_METHODS[rule.qualname]
        if ctx.kwargs or len(ctx.args) != 2:
            return Refused(self.engine, rule.qualname, 'numeric division requires two positional primitive operands')
        recv, other = ctx.args
        if a.primitive_numeric_kind(recv) != owner or a.primitive_numeric_kind(other) is None:
            return Refused(self.engine, rule.qualname, 'native receiver and numeric other operand required')
        if owner == 'int' and not exact_int_value(other):
            return Refused(self.engine, rule.qualname, 'NotImplemented result is not represented')
        left, right = (other,recv) if reverse else (recv,other)
        op = {'truediv':ast.Div, 'floordiv':ast.FloorDiv, 'mod':ast.Mod}[name]()
        claim = contract.raise_claim('ZeroDivisionError')
        if not isinstance(claim,ConstantArgsRaiseClaim) or not claim.tested:
            return Refused(self.engine, rule.qualname, 'no tested zero-divisor condition and args')
        if owner == 'float':
            # Outside try, overflowing conversions end the path; retain every successful conversion.
            numeric = [left, right]
            limit = _FLOAT_CONVERSION_MAX_INT
            for index, value in enumerate(numeric):
                ii = a._as_int(st,value)
                if ii is not None:
                    lo,hi = a.ibounds(st,ii)
                    if lo is None or hi is None or lo < -limit or hi > limit:
                        a._in_try_refuse('numeric division may overflow int-to-float conversion',node)
                    lo = max(-limit, lo) if lo is not None else -limit
                    hi = min(limit, hi) if hi is not None else limit
                    if lo > hi:
                        st.set_bottom()
                        return BOT
                    numeric[index] = int_iv(lo, hi)
            left, right = numeric
        if owner == 'int':
            li,ri = a._as_int(st,left),a._as_int(st,right)
            bounds = a.ibounds(st,ri)
            facts=[]
            result,zero = a._int_binop(st,op,li,ri,node,facts)
            fact='; '.join(facts)
            if bounds == (0,0):result=BOT
        else:
            result_info,raises,fact = a._float_binop(st,op,a.float_view(st,left),a.float_view(st,right),node)
            zero = 'ZeroDivisionError' in raises
            result=V(f=result_info) if not result_info.fact.is_bottom() else BOT
        pools = product([concrete(a,st,recv),concrete(a,st,other)], analyzer=a)
        if pools is not None and all(type(x) is not int or x.bit_length()<=4096 for pair in pools for x in pair):
            outputs=[]
            zero=False
            for pair in pools:
                try:outputs.append(rule.native(*pair))
                except ZeroDivisionError:zero=True
                except OverflowError:
                    a._in_try_refuse('numeric division OverflowError',node)
            result=abstract(a,st,outputs,node)
        elif owner == 'int' and name == 'truediv' and bounds != (0,0):
            lo,hi = a.ibounds(st,li)
            if lo is None or hi is None or max(abs(lo),abs(hi))>2**1023:
                a._in_try_refuse('integer true division may overflow',node)
        divisor_node = node.right if isinstance(node,ast.BinOp) else node.value if isinstance(node,ast.AugAssign) else None
        residual = '('+ast.unparse(divisor_node)+') == 0' if divisor_node is not None else None
        # Operator source bindings name the binary protocol; direct slot calls name the native callable.
        operator_site = isinstance(node,(ast.BinOp,ast.AugAssign))
        contract_qualname = None if operator_site else rule.qualname
        op_name = 'divide' if operator_site else rule.qualname+':ZeroDivisionError'
        a.check(node,op_name,{'ZeroDivisionError'},not zero,fact,st=st,
                residual=residual,operand=divisor_node,contract_qualname=contract_qualname,
                exact_args={'ZeroDivisionError':tuple(const_str(x) for x in claim.args)})
        proven={exc:'native numeric operands execute no target hook' for exc in ALL_PROPERTY_CLASSES}
        a._contract_checks(node,rule.qualname,contract,proven,recorded={'ZeroDivisionError'})
        if result is None:
            return Refused(self.engine,rule.qualname,'numeric result is not represented')
        if result.is_bot():st.set_bottom()
        return result

    def visit_exception_constructor(self, rule: Rule, ctx: CallCtx):
        from tharkun.solve.contracts.registry import get
        from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES
        from .aval import BOT, V, const_str
        from .heap import ExceptionObj
        from tharkun.solve.contracts.types import ConstantArgsRaiseClaim

        a, st, node = self.analyzer, ctx.state, ctx.node
        contract = get(rule.qualname)
        if (contract is None or not contract.tested or contract.returns is None
                or not contract.returns.tested or contract.returns.type != rule.qualname):
            return Refused(self.engine, rule.qualname, 'no tested exception constructor contract')
        proven = {exc: 'canonical exception constructor stores positional references without hooks'
                  for exc in ALL_PROPERTY_CLASSES}
        if ctx.kwargs:
            claim = contract.raise_claim('TypeError')
            if not isinstance(claim, ConstantArgsRaiseClaim) or not claim.tested:
                return Refused(self.engine, rule.qualname, 'no tested constructor TypeError args')
            a.check(node, 'call-args', {'TypeError'}, False, claim.condition, st=st,
                    exact_args={'TypeError': tuple(const_str(x) for x in claim.args)})
            proven['TypeError'] = None
            a._contract_checks(node, rule.qualname, contract, proven)
            st.set_bottom()
            return BOT
        args = a.exception_args(st, ctx.args, node)
        a.borrow(st, args)
        addr = a.alloc(st, node, lambda sid: ExceptionObj(rule.qualname, args), 'exception:'+rule.qualname)
        a._contract_checks(node, rule.qualname, contract, proven)
        return V(refs=frozenset({addr}))

    def visit_textwrap_function(self, rule: Rule, ctx: CallCtx):
        return self._textwrap_call(rule, ctx, rule.qualname)

    def visit_textwrap_constructor(self, rule: Rule, ctx: CallCtx):
        import inspect
        from tharkun.solve.contracts.registry import get
        from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES
        from .aval import BOT, V
        from .heap import TextWrapperObj
        from .native import abstract

        a, st, node = self.analyzer, ctx.state, ctx.node
        contract = get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine, rule.qualname, 'no tested contract')
        proven = {exc: 'canonical TextWrapper initialization only stores its parameters' for exc in ALL_PROPERTY_CLASSES}
        try:
            bound = inspect.signature(resolve_native(rule.qualname)).bind(*ctx.args, **ctx.kwargs)
        except TypeError:
            a.check(node, 'arg-type', {'TypeError'}, False, 'TextWrapper constructor binding', st=st)
            proven['TypeError'] = None
            a._contract_checks(node, rule.qualname, contract, proven)
            st.set_bottom()
            return BOT
        bound.apply_defaults()
        attrs = {name: a.borrow(st, a.summarize(st, raw if type(raw) is V else abstract(a, st, [raw], node)))
                 for name, raw in bound.arguments.items()}
        addr = a.alloc(st, node, lambda sid: TextWrapperObj(attrs), 'textwrap.TextWrapper')
        a._contract_checks(node, rule.qualname, contract, proven)
        return V(refs=frozenset({addr}))

    def visit_textwrap_method(self, rule: Rule, ctx: MethodCtx):
        import inspect
        import textwrap
        from tharkun.solve.contracts.registry import get
        from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES
        from .aval import BOT
        from .interp import Unsupported
        from .native import concrete

        a, st, node = self.analyzer, ctx.state, ctx.node
        contract = get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine, rule.qualname, 'no tested contract')
        if len(ctx.receivers) != 1:
            raise Unsupported('TextWrapper method on joined receivers', node)
        try:
            inspect.signature(getattr(textwrap.TextWrapper, ctx.name)).bind(None, *ctx.args, **ctx.kwargs)
        except TypeError:
            a.check(node, 'arg-type', {'TypeError'}, False, 'TextWrapper method binding', st=st)
            proven = {exc: 'argument binding fails before any method body or target hook' for exc in ALL_PROPERTY_CLASSES}
            proven['TypeError'] = None
            a._contract_checks(node, rule.qualname, contract, proven)
            st.set_bottom()
            return BOT
        obj = st.heap[next(iter(ctx.receivers))]
        if obj.kind != 'textwrapper':
            raise Unsupported('TextWrapper method on an unmodelled receiver', node)
        defaults = inspect.signature(textwrap.TextWrapper).parameters
        options = {}
        for name, value in obj.attrs.items():
            values = concrete(a, st, value)
            default = defaults[name].default
            if name == 'width' or values is None or len(values) != 1 or type(values[0]) is not type(default) or values[0] != default:
                options[name] = value
        return self._textwrap_call(rule, CallCtx(st, ctx.args, dict(ctx.kwargs) | options, node), 'textwrap.'+ctx.name)

    def _textwrap_call(self, rule: Rule, ctx: CallCtx, native_qualname):
        import inspect
        from tharkun.solve.contracts.registry import get
        from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES
        from .aval import ANYSTR, BOT, V
        from .interp import Unsupported
        from .linear import IntInfo
        from .native import abstract, concrete, product

        a, st, node = self.analyzer, ctx.state, ctx.node
        contract = get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine, rule.qualname, 'no tested contract')
        reason = 'validated builtin text/configuration runs no target hook'
        proven = {exc: reason for exc in ALL_PROPERTY_CLASSES}
        try:
            bound = inspect.signature(resolve_native(native_qualname)).bind(*ctx.args, **ctx.kwargs)
        except TypeError:
            a.check(node, 'arg-type', {'TypeError'}, False, 'textwrap argument binding', st=st)
            proven['TypeError'] = None
            a._contract_checks(node, rule.qualname, contract, proven)
            st.set_bottom()
            return BOT
        keys = tuple(ctx.kwargs)
        pools = [concrete(a, st, value) for value in (*ctx.args, *ctx.kwargs.values())]
        rows = product(pools, analyzer=a)
        bounded = (rows is not None and all(
            all(type(value) in (str, int, bool, type(None))
                and (len(value) <= 640 if type(value) is str else abs(value) <= 640 if type(value) is int else True)
                for value in row) for row in rows))
        if bounded:
            values, errors = [], set()
            for row in rows:
                try:
                    values.append(resolve_native(native_qualname)(*row[:len(ctx.args)], **dict(zip(keys, row[len(ctx.args):]))))
                except (AssertionError, AttributeError, FloatingPointError, IndexError, KeyError, TypeError,
                        ZeroDivisionError, ValueError, OverflowError) as exc:
                    errors.add(type(exc).__name__)
            for exc in errors:
                if exc in proven:
                    proven[exc] = None
                else:
                    a.check(node, f'{rule.qualname}:{exc}', {exc}, False, 'closed builtin call raises '+exc, st=st)
            a._contract_checks(node, rule.qualname, contract, proven)
            if not values:
                st.set_bottom()
                return BOT
            result = abstract(a, st, values, node)
            if result is None:
                raise Unsupported(f'{rule.qualname}: unrepresentable native result', node)
            return result
        bound.apply_defaults()
        def value(name):
            raw = bound.arguments[name]
            return raw if type(raw) is V else abstract(a, st, [raw], node)
        text = value('text')
        if text.s is None or not text.replace(s=None).is_bot():
            raise Unsupported(f'{rule.qualname}: requires builtin str text', node)
        name = native_qualname.split('.')[1]
        if name == 'indent':
            prefix, predicate = value('prefix'), value('predicate')
            if (prefix.s is None or not prefix.replace(s=None).is_bot()
                    or not predicate.n or not predicate.replace(n=False).is_bot()):
                raise Unsupported('textwrap.indent: nonfinite calls require str prefix and no predicate', node)
        elif name != 'dedent':
            if bound.arguments['kwargs'] or not exact_int_value(value('width')):
                raise Unsupported(f'{rule.qualname}: nonfinite calls require default options and builtin int width', node)
            lo, _ = a.ibounds(st, a._as_int(st, value('width')))
            a.check(node, f'{rule.qualname}:ValueError', {'ValueError'}, lo is not None and lo >= (5 if name == 'shorten' else 1),
                    'positive width and fitting default placeholder', st=st)
        a._contract_checks(node, rule.qualname, contract, proven)
        # shortcut: nonfinite wrapping keeps line/str types; add line-width relations when callers need them.
        if name == 'wrap':
            addr = a.new_list(st, node, IntInfo.interval(0, None), ANYSTR, tag='textwrap.wrap')
            return V(refs=frozenset({addr}))
        return ANYSTR

    def visit_math_constant(self, rule: Rule, ctx: CallCtx):
        """``math.pi``, ``e``, ``tau``, ``inf``, ``nan`` (a module attribute read, so no call and no property class): the exact float
        CPython has, while the contract is TESTED (``adapters/math_model.py::constant``), else a named refusal."""
        from tharkun.solve import values as fv
        from tharkun.solve.contracts.adapters.math_model import constant
        from .aval import FloatInfo, V

        value = constant(rule.qualname.split(".", 1)[1])
        if value is None:
            return Refused(self.engine, rule.qualname, "no tested contract")
        return V(f=FloatInfo(fv.FloatFact.const(value)))

    def _protocol(self, rule: Rule, ctx: ProtocolCtx):
        """The seven classes of a protocol rule, recorded at the node of the dispatch (``Analyzer._through_protocol``)."""
        return self.analyzer._through_protocol(rule.qualname, ctx.node, ctx.run, ctx.native)

    def _random_function(self, rule: Rule, ctx: CallCtx):
        """``random.choice`` ... (``rand.py::RandomMixin``): the model is unchanged, the contract supplies the seven classes."""
        a = self.analyzer
        model = getattr(a, "m_random_" + rule.qualname.split(".", 1)[1])
        return a._contract_module_model(rule.qualname, ctx.state, model, ctx.args, ctx.kwargs, ctx.node)

    def visit_array_array(self, rule: Rule, ctx: CallCtx):
        from ..contracts import ALL_PROPERTY_CLASSES,get
        from .aval import BOT,const_str
        a=self.analyzer
        contract=get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine,rule.qualname,'no tested contract')
        if a._array_bad_typecode(ctx.args,ctx.kwargs) is not None:
            case=next((c for c in contract.other_raises if c.case=='native-bad-typecode' and c.exc=='ValueError' and c.tested),None)
            if case is None:return Refused(self.engine,rule.qualname,'no tested bad-typecode payload relation')
            payload=tuple(const_str(x.format()) for x in case.args_template)
            a._contract_checks(ctx.node,rule.qualname,contract,
                {exc:'native bad typecode fails before initializer/warning callbacks' for exc in ALL_PROPERTY_CLASSES})
            a.check(ctx.node,'array.array:ValueError',{'ValueError'},False,'one native invalid typecode',st=ctx.state,
                    exact_args={'ValueError':payload},contract_qualname=rule.qualname)
            ctx.state.set_bottom()
            return BOT
        return a._contract_module_model(rule.qualname,ctx.state,a.m_array_array,ctx.args,ctx.kwargs,ctx.node)

    visit_random_random = visit_random_randint = visit_random_randrange = _random_function
    visit_random_choice = visit_random_uniform = visit_random_seed = _random_function

    def _verifier_container(self, rule: Rule, ctx: CallCtx):
        if rule.qualname == "_sv_verifier.check_type":
            from ..check_type_hints import check_type_contract
            if check_type_contract() is None:
                return Refused(self.engine, rule.qualname, "no tested contract with checker None return")
        name = rule.qualname.rsplit(".", 1)[1]
        model = getattr(self.analyzer, "m__sv_verifier_" + name)
        return self.analyzer._contract_module_model(
            rule.qualname, ctx.state, model, ctx.args, ctx.kwargs, ctx.node
        )

    visit_verifier_nondet_list = visit_verifier_nondet_dict = _verifier_container
    visit_verifier_check_type = _verifier_container

    def _module_function(self, rule: Rule, ctx: CallCtx):
        module, name = rule.qualname.split(".")
        return self.analyzer._contract_call(ctx.state, module, name, ctx.args, ctx.kwargs, ctx.node)

    def visit_functools_reduce(self, rule: Rule, ctx: CallCtx):
        from ..contracts import get
        contract = get(rule.qualname)
        if (contract is None or not contract.tested or contract.returns is None or not contract.returns.tested
                or any(not claim.tested for claim in contract.raises)):
            return Refused(self.engine, rule.qualname, 'no tested reduce return/raise contract')
        result = self.analyzer._integer_reduce(ctx.state, ctx.args, ctx.kwargs, ctx.node)
        if result is None:
            return self._module_function(rule, ctx)
        value, proven = result
        self.analyzer._contract_checks(ctx.node, rule.qualname, contract, proven)
        return value

    def visit_operator_function(self, rule: Rule, ctx: CallCtx):
        """Consume callable aliases through the existing intrinsic transfers, including their heap effects."""
        from tharkun.solve.contracts.registry import get
        from .aval import BOOL, BOT, NONE, V, const_bool
        from .native import abstract, concrete, product

        contract = get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine, rule.qualname, "no tested contract")
        a, st, node = self.analyzer, ctx.state, ctx.node
        name = rule.qualname.split('.')[1]
        args, kwargs = ctx.args, ctx.kwargs
        binary = {'and_': ast.BitAnd, 'or_': ast.BitOr, 'xor': ast.BitXor,
                  'sub': ast.Sub, 'mul': ast.Mult, 'truediv': ast.Div, 'floordiv': ast.FloorDiv,
                  'mod': ast.Mod, 'pow': ast.Pow, 'lshift': ast.LShift, 'rshift': ast.RShift,
                  'matmul': ast.MatMult, 'concat': ast.Add, 'add': ast.Add}
        comparisons = {'eq': ast.Eq, 'ne': ast.NotEq, 'lt': ast.Lt, 'le': ast.LtE, 'gt': ast.Gt, 'ge': ast.GtE}
        unary = {'neg': ast.USub, 'pos': ast.UAdd, 'inv': ast.Invert, 'invert': ast.Invert}
        seen, errors = [], set()

        def run():
            if name in {'methodcaller', 'itemgetter', 'attrgetter'}:
                if not args:
                    a.check(node, 'call-args', {'TypeError'}, False, 'getter/caller factory requires positional arguments', st=st)
                    st.set_bottom()
                    return BOT
            else:
                count = 1 if name in {*unary, 'abs', 'index', 'truth', 'not_', 'length_hint', 'is_none', 'is_not_none'} else 3 if name == 'setitem' else 2
                if not a._arity(node, args, kwargs, count, 2 if name == 'length_hint' else count, rule.qualname):
                    st.set_bottom()
                    return BOT
            if name in {'is_', 'is_not', 'is_none', 'is_not_none'}:
                left, right = (args[0], NONE) if name in {'is_none', 'is_not_none'} else args
                same = a._class_identity(st, left, right)
                if same is None:
                    same = a._bool_singleton_identity(left, right)
                if same is None and not (left.top or right.top):
                    if left.n and left.replace(n=False).is_bot():
                        same = True if right.n and right.replace(n=False).is_bot() else False if not right.n else None
                    elif right.n and right.replace(n=False).is_bot() and not left.n:
                        same = False
                return BOOL if same is None else const_bool(not same if name in {'is_not', 'is_not_none'} else same)
            base = name[1:] if name.startswith('i') and name not in {'index', 'indexOf', 'inv', 'invert'} else name
            if base in binary:
                if base == 'concat':
                    for value in args:
                        if value.top or value.st or value.i is not None or value.b or value.f or value.n or any(st.heap[x].kind not in {'list', 'tuple'} for x in value.refs):
                            return Refused(self.engine, rule.qualname, 'concat requires modelled builtin sequence operands')
                return a.binop(st, binary[base](), args[0], args[1], node, inplace=base != name)
            if name in unary or name == 'abs':
                if a.instance_value(st, args[0]):
                    return Refused(self.engine, rule.qualname, 'custom unary hooks require a protocol model')
                return a._abs_model(st, args, kwargs, node) if name == 'abs' else a.unop(st, unary[name](), args[0], node)
            if name in {'truth', 'not_'}:
                yes, no = a.truth_split(st, args[0], node)
                return a._merge_branches(st, [(yes, const_bool(name == 'truth')), (no, const_bool(name != 'truth'))])
            if name in comparisons:
                yes, no = a.compare_split(st, comparisons[name](), args[0], args[1], node.args[0] if getattr(node, 'args', ()) else node, node.args[1] if len(getattr(node, 'args', ())) > 1 else node, node)
                return a._merge_branches(st, [(yes, const_bool(True)), (no, const_bool(False))])
            if name == 'getitem':
                return a.subscript_load(st, args[0], ('index', args[1]), node)
            if name == 'setitem':
                a.subscript_store(st, args[0], ('index', args[1]), args[2], node)
                return NONE
            if name == 'delitem':
                value = args[0]
                if not value.refs or not value.replace(refs=frozenset()).is_bot() or any(st.heap[x].kind != 'list' for x in value.refs):
                    return Refused(self.engine, rule.qualname, 'item deletion requires a modelled builtin list')
                a.list_method(st, value.refs, 'pop', [args[1]], {}, node)
                return NONE
            if name == 'index' and exact_int_value(args[0]):
                return V(i=a._as_int(st, args[0]))
            # Read-only finite builtin values cannot invoke target hooks. Iterator
            # receivers are excluded by concrete(), so searches consume no hidden cursor.
            pools = [concrete(a, st, value) for value in args]
            keys = list(kwargs)
            pools.extend(concrete(a, st, kwargs[k]) for k in keys)
            alternatives = product(pools, analyzer=a)
            if alternatives is None:
                return Refused(self.engine, rule.qualname, 'non-finite or callback-bearing operands')
            outputs = []
            from tharkun.solve.contracts.rules_strformat import native_limit
            for values in alternatives:
                refusal = native_limit(rule.qualname, values[:len(args)], dict(zip(keys, values[len(args):])))
                if refusal:
                    return Refused(self.engine, rule.qualname, refusal)
                try:
                    outputs.append(rule.native(*values[:len(args)], **dict(zip(keys, values[len(args):]))))
                except (TypeError, ValueError, KeyError, IndexError, AttributeError, OverflowError) as exc:
                    errors.update(c.__name__ for c in type(exc).__mro__)
            if not outputs:
                st.set_bottom()
                return BOT
            if name in {'methodcaller', 'itemgetter', 'attrgetter'}:
                # A known callable result, with invocation deliberately refused by call_static.
                return V(st=frozenset({('contract-callable', rule.qualname)}))
            result = abstract(a, st, outputs, node)
            return result if result is not None else Refused(self.engine, rule.qualname, 'native result type is not represented')

        a._taps.append((node, seen))
        try:
            result = run()
        finally:
            a._taps.pop()
        if isinstance(result, Refused):
            return result
        failures = errors | {exc for classes, ok in seen if not ok for exc in classes}
        proven = {claim.exc: None if claim.exc in failures else 'modelled intrinsic dispatch or exhaustive hook-free builtin alternatives'
                  for claim in contract.raises}
        a._contract_checks(node, rule.qualname, contract, proven)
        return result

    visit_operator_add = visit_operator_attrgetter = visit_operator_itemgetter = visit_operator_function

    def visit_collections_namedtuple(self, rule: Rule, ctx: CallCtx):
        return self.analyzer.c_collections_namedtuple(ctx.state, ctx.args, ctx.kwargs, ctx.node)

    def visit_collections_defaultdict(self, rule: Rule, ctx: CallCtx):
        return self.analyzer.c_collections_defaultdict(ctx.state, ctx.args, ctx.kwargs, ctx.node)

    def visit_copy_copy(self, rule: Rule, ctx: CallCtx):
        a = self.analyzer
        if a._copy_copy_list_receiver(ctx.state, ctx.args, ctx.kwargs):
            return a._contract_module_model(rule.qualname, ctx.state, a._copy_copy_list,
                                            ctx.args, ctx.kwargs, ctx.node)
        return a._contract_call(ctx.state, "copy", "copy", ctx.args, ctx.kwargs, ctx.node)

    def visit_copy_deepcopy(self, rule: Rule, ctx: CallCtx):
        a = self.analyzer
        if a._copy_deepcopy_atomic_args(ctx.args, ctx.kwargs):
            return a._contract_module_model(rule.qualname, ctx.state, a._copy_deepcopy_atomic,
                                            ctx.args, ctx.kwargs, ctx.node)
        return a._contract_call(ctx.state, "copy", "deepcopy", ctx.args, ctx.kwargs, ctx.node)

    def _list_method(self, rule: Rule, ctx: MethodCtx):
        return self.analyzer._contract_method(
            rule.qualname, ctx, lambda st, args, kwargs, node: self.analyzer._list_method_model(
                st, ctx.receivers, ctx.name, args, kwargs, node))

    def _dict_method(self, rule: Rule, ctx: MethodCtx):
        return self.analyzer._contract_method(
            rule.qualname, ctx, lambda st, args, kwargs, node: self.analyzer._dict_method_model(
                st, next(iter(ctx.receivers)), ctx.name, args, kwargs, node))

    visit_dict_setdefault = _dict_method
    visit_dict_pop = _dict_method

    def _stringio_method(self, rule: Rule, ctx: MethodCtx):
        return self.analyzer._contract_method(
            rule.qualname, ctx, lambda st, args, kwargs, node: self.analyzer._stringio_method_model(
                st, next(iter(ctx.receivers)), ctx.name, args, kwargs, node))

    visit_list_append = visit_list_extend = visit_list_insert = visit_list_pop = visit_list_reverse = _list_method
    visit_list_copy = visit_list_clear = visit_list_count = visit_list_index = visit_list_remove = visit_list_sort = _list_method

    def _tuple_method(self, rule: Rule, ctx: MethodCtx):
        return self.analyzer._contract_method(
            rule.qualname, ctx, lambda st, args, kwargs, node: self.analyzer._tuple_method_model(
                st, next(iter(ctx.receivers)), ctx.name, args, kwargs, node))

    visit_tuple_count = visit_tuple_index = _tuple_method
    visit_dict_get = visit_dict_keys = visit_dict_values = visit_dict_items = visit_dict_update = _dict_method
    #: the kind of ``IoStringIOWrite`` is ``io_string_iowrite`` (the snake case of a name with the capitals ``IO`` run together)
    visit_io_string_iowrite = visit_io_string_ioread = visit_io_string_iogetvalue = visit_io_string_ioseek = _stringio_method

    def visit_collections_defaultdict_missing(self, rule: Rule, ctx):
        if isinstance(ctx,CallCtx):
            if len(ctx.args)!=2 or ctx.kwargs:
                return Refused(self.engine,rule.qualname,'two positional arguments required')
            receiver=ctx.args[0]
            if receiver.top or not receiver.replace(refs=frozenset()).is_bot() or len(receiver.refs)!=1:
                return Refused(self.engine,rule.qualname,'one native defaultdict receiver required')
            ctx=MethodCtx(ctx.state,receiver.refs,'__missing__',ctx.args[1:],{},ctx.node)
        if not isinstance(ctx,MethodCtx):
            return Refused(self.engine,rule.qualname,'method context required')
        return self.analyzer.defaultdict_missing(ctx)

    def _subscript(self, rule: Rule, ctx: SubscriptCtx):
        """The model of a subscript runs once, over every container kind the receiver may be; each rule of those kinds records its
        seven classes (``Analyzer._record_subscript``)."""
        if not isinstance(ctx, SubscriptCtx):
            return Refused(self.engine, rule.qualname, "builtin subscript rule requires a SubscriptCtx")
        self.analyzer._record_subscript(rule.qualname, ctx)

    visit_list_getitem = visit_list_setitem = visit_tuple_getitem = visit_dict_getitem = visit_dict_setitem = _subscript
    visit_collections_defaultdict_getitem = _subscript

    def _set_operation(self, rule: Rule, ctx: SetCtx):
        return ctx.run()

    def _output_method(self, rule: Rule, ctx: OutfileCtx):
        return self.analyzer._outfile_method(ctx.state, ctx.binary, ctx.receiver, ctx.name, ctx.args, ctx.kwargs,
                                             ctx.node)

    visit_output_write = visit_output_writelines = visit_output_flush = _output_method
    visit_output_close = visit_output_enter = visit_output_exit = _output_method

    def visit_protocol_length_hint(self, rule: Rule, ctx: LengthHintCtx):
        """``"probe"`` (the analyzer ran ``__len__`` on the state, following the TypeError CPython swallows), ``"skip"``
        (the consumer never asks), or a named refusal: no verified rule for this consumer."""
        from tharkun.solve.contracts.adapters.absint import length_hint_rule

        decision = length_hint_rule(ctx.consumer)
        if decision == "probe":
            self.analyzer._probe_length(ctx.state, ctx.part, ctx.node)
        elif decision != "skip":
            return Refused(self.engine, rule.qualname,
                           f"{ctx.consumer}() of an instance defining __len__: no verified rule for the length probe")
        return decision

    def _range_query(self, rule: Rule, ctx: CallCtx):
        from .aval import BOT, int_iv
        from .heap import RangeObj
        from .native import concrete
        from .state import iv_of_len
        from tharkun.solve.contracts.adapters.absint import contract_for_method
        contract = contract_for_method("range", rule.qualname.split(".")[1])
        if contract is None:
            return Refused(self.engine, rule.qualname, "no tested contract")
        a, st, node = self.analyzer, ctx.state, ctx.node
        if len(ctx.args) != 2 or ctx.kwargs:
            a._contract_checks(node, rule.qualname, contract,
                               {c.exc: "argument parser runs no callbacks" for c in contract.raises if c.exc != "TypeError"})
            st.set_bottom()
            return BOT
        recv, needle = ctx.args
        if (recv.top or not recv.refs or not recv.replace(refs=frozenset()).is_bot()
                or any(type(st.heap[r]) is not RangeObj for r in recv.refs)):
            return Refused(self.engine, rule.qualname, "receiver is not a definite builtin range")
        if needle.top or not needle.replace(i=None, b=frozenset()).is_bot():
            return Refused(self.engine, rule.qualname, "only exact int/bool needles have a callback-free arithmetic search")
        a._contract_checks(node, rule.qualname, contract,
                           {c.exc: "range integer fast path runs no equality hooks" for c in contract.raises})
        values = concrete(a, st, needle)
        outputs = []
        if values is not None:
            for addr in recv.refs:
                ob = st.heap[addr]
                lo, hi = iv_of_len(ob.lo, st.zone), iv_of_len(ob.hi, st.zone)
                if lo[0] is None or lo[0] != lo[1] or hi[0] is None or hi[0] != hi[1]:
                    break
                native = range(lo[0], hi[0], ob.step)
                for value in values:
                    try:
                        outputs.append(rule.native(native, value))
                    except ValueError:
                        pass  # Missing indexes end this path; calls under try are refused.
            else:
                if not outputs:
                    st.set_bottom()
                    return BOT
                return int_iv(min(outputs), max(outputs))
        sizes = [a._range_len(st, st.heap[r])[1][1] for r in recv.refs]
        upper = None if None in sizes else max(sizes)
        if upper == 0:
            if rule.qualname == "range.count":
                return int_iv(0, 0)
            st.set_bottom()
            return BOT
        if rule.qualname == "range.count":
            return int_iv(0, 1)
        return int_iv(0, None if upper is None else upper - 1)

    def _scalar_query(self, rule: Rule, ctx: CallCtx):
        from .aval import BOOL, BOT, V, const_bool, const_int, int_iv
        from .native import abstract, concrete
        import math
        from tharkun.solve.contracts.adapters.absint import contract_for_method
        owner, method = rule.qualname.split(".")
        contract = contract_for_method(owner, method)
        if contract is None:
            return Refused(self.engine, rule.qualname, "no tested contract")
        a, st, node = self.analyzer, ctx.state, ctx.node
        shape_ok = len(ctx.args) == 1 and not ctx.kwargs
        if not shape_ok:
            a._contract_checks(node, rule.qualname, contract,
                               {c.exc: "argument parser runs no callbacks" for c in contract.raises if c.exc != "TypeError"})
            st.set_bottom()
            return BOT
        recv = ctx.args[0]
        valid = (not recv.top and (recv.replace(i=None, b=frozenset()).is_bot() if owner == "int"
                                  else bool(recv.f) and recv.replace(f=False).is_bot()))
        if not valid:
            return Refused(self.engine, rule.qualname, "receiver is not a definite builtin " + owner)
        a._contract_checks(node, rule.qualname, contract,
                           {c.exc: "builtin numeric query runs no user hooks" for c in contract.raises})
        if owner == "float":
            if method == "conjugate":
                return recv
            values = concrete(a, st, recv)
            if values is not None:
                outputs = []
                for value in values:
                    try:
                        outputs.append(rule.native(value))
                    except (ValueError, OverflowError):
                        pass  # NaN/infinity ratios have no successful continuation; try calls are refused.
                if not outputs:
                    st.set_bottom()
                    return BOT
                return abstract(a, st, outputs, node)
            fact = recv.f.fact
            if not fact.has_interval() or fact.lo == fact.hi and math.isinf(fact.lo):
                if method == "is_integer":
                    return const_bool(False)
                st.set_bottom()
                return BOT
            if method == "is_integer":
                return BOOL
            addr = a.new_list(st, node, None, BOT, tag="float-ratio", kind="tuple",
                              items=[int_iv(None, None), int_iv(1, None)])
            return V(refs=frozenset({addr}))
        if rule.qualname == "int.is_integer":
            return const_bool(True)
        if rule.qualname in {"int.conjugate", "int.as_integer_ratio"}:
            value = V(i=a._as_int(st, recv))
            if rule.qualname == "int.conjugate":
                return value
            addr = a.new_list(st, node, None, BOT, tag="int-ratio", kind="tuple", items=[value, const_int(1)])
            return V(refs=frozenset({addr}))
        values = concrete(a, st, recv)
        if values is not None:
            outputs = [rule.native(x) for x in values]
            return int_iv(min(outputs), max(outputs)) if outputs else BOT
        lo, hi = a.ibounds(st, recv.i)
        if recv.b:
            lo = min(lo, min(recv.b)) if lo is not None else None
            hi = max(hi, max(recv.b)) if hi is not None else None
        upper = max(abs(lo), abs(hi)).bit_length() if lo is not None and hi is not None else None
        least_abs = lo if lo is not None and lo > 0 else -hi if hi is not None and hi < 0 else 0
        lower = least_abs.bit_length() if rule.qualname == "int.bit_length" else int(least_abs > 0)
        return int_iv(lower, upper)

    def _str_join_key_view(self, ctx, contract):
        from .aval import ANYSTR
        from .heap import DictObj, DictView

        a, st = self.analyzer, ctx.state
        if a._try_stack or len(ctx.args) != 2 or ctx.kwargs or not definite_str(ctx.args[0]):
            return None
        value = ctx.args[1]
        if len(value.refs) != 1 or not value.replace(refs=frozenset()).is_bot():
            return None
        view = st.heap[next(iter(value.refs))]
        if type(view) is not DictView or view.which != 'keys' or len(view.dicts) != 1:
            return None
        source = st.heap[next(iter(view.dicts))]
        if type(source) is not DictObj or not source.exact or any(type(k) is not str for k in source.items):
            return None
        if contract.returns is None or not contract.returns.tested or contract.returns.type != 'str':
            return Refused(self.engine, 'str.join', 'key view requires its tested str return contract')
        a._contract_checks(ctx.node, 'str.join', contract,
                           {c.exc: 'builtin str separator and live closed string-key view run no user hook'
                            for c in contract.raises if c.tested and c.condition != 'unknown'})
        # shortcut: DictObj joins forget insertion order; keep content and length unknown.
        return ANYSTR

    def _native_string(self, rule: Rule, ctx: CallCtx):
        import re
        from .native import abstract, concrete, product
        from .aval import ANYSTR, BOOL, BOT, FLOAT, V, int_iv
        from tharkun.solve.contracts import get
        contract = get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine, rule.qualname, "no tested contract")
        a, st, node = self.analyzer, ctx.state, ctx.node
        within_budget = lambda: a._clock() <= a.deadline
        if not within_budget():
            return Refused(self.engine, rule.qualname, "native evaluation deadline exhausted")
        if rule.qualname == 'str.join':
            joined = self._str_join_key_view(ctx, contract)
            if joined is not None:
                return joined
        if rule.qualname.endswith(".scanner"):
            return Refused(self.engine, rule.qualname, "mutable scanner state is not modelled")
        keys = list(ctx.kwargs)
        pools = []
        try:
            for value in (*ctx.args, *ctx.kwargs.values()):
                if not within_budget():
                    return Refused(self.engine, rule.qualname, "native evaluation deadline exhausted")
                pools.append(concrete(a, st, value))
            if not within_budget():
                return Refused(self.engine, rule.qualname, "native evaluation deadline exhausted")
            alternatives = product(pools, analyzer=a)
        except MemoryError:
            return Refused(self.engine, rule.qualname, "native operand memory exhausted")
        if not within_budget():
            return Refused(self.engine, rule.qualname, "native evaluation deadline exhausted")
        if alternatives is None:
            return self._abstract_string(rule, ctx, contract)
        from tharkun.solve.contracts.rules_strformat import native_limit
        outputs, errors = [], set()
        for xs in alternatives:
            args, kwargs = xs[:len(ctx.args)], dict(zip(keys, xs[len(ctx.args):]))
            refusal = native_limit(rule.qualname,args,kwargs, _work_check=within_budget)
            if refusal:
                if refusal in {'native operand/format size exceeds 4096','native width exceeds 4096'}:
                    return self._abstract_string(rule,ctx,contract)
                return Refused(self.engine,rule.qualname,refusal)
            try:
                result = rule.native(*args, **kwargs)
                if rule.qualname.endswith("finditer"):
                    result = list(result)
                outputs.append(result)
            except MemoryError:
                return Refused(self.engine, rule.qualname, "native result memory exhausted")
            except (TypeError, ValueError, KeyError, IndexError, AttributeError, OverflowError, re.error) as exc:
                errors.update(c.__name__ for c in type(exc).__mro__)
        if not within_budget():
            return Refused(self.engine, rule.qualname, "native evaluation deadline exhausted")
        proven = {claim.exc: "native evaluation of every finite builtin alternative" for claim in contract.raises
                  if claim.tested and claim.condition != 'unknown' and claim.exc not in errors}
        recorded=set()
        if (rule.qualname=='str.__getitem__' and 'IndexError' in errors
                and len(ctx.args)==2 and not ctx.kwargs and exact_int_value(ctx.args[1])):
            payload=a.sequence_index_args(rule.qualname,st,a._as_int(st,ctx.args[1]),node)
            a.check(node,rule.qualname+':IndexError',{'IndexError'},False,'finite builtin string index is out of range',st=st,
                    exact_args={'IndexError':payload},contract_qualname=rule.qualname)
            recorded.add('IndexError')
        a._contract_checks(node, rule.qualname, contract, proven,recorded=recorded)
        if not outputs:
            st.set_bottom()
            return BOT
        result = abstract(a, st, outputs, node)
        if result is None:
            return Refused(self.engine, rule.qualname, "native result type is not represented")
        if rule.qualname.endswith("finditer"):
            from .heap import IteratorObj
            addr = a.alloc(st,node,lambda sid: IteratorObj(result),"regex-iterator")
            return V(refs=frozenset({addr}))
        return result

    def _str_format_auto_fields(self, ctx, contract):
        import _string
        from string import Formatter
        from .aval import ANYSTR, ANY_STR
        a=self.analyzer
        if a._try_stack or not ctx.args or not definite_str(ctx.args[0]):
            return None
        values=(*ctx.args[1:],*ctx.kwargs.values())
        if any(not v.replace(i=None,b=frozenset(),f=False,s=None,n=False).is_bot() for v in values):
            return None
        templates=ctx.args[0].s
        if templates==ANY_STR:
            fn=getattr(ctx.node,'func',None)
            literal=fn.value if isinstance(fn,ast.Attribute) and fn.attr=='format' else None
            if not isinstance(literal,ast.Constant) or type(literal.value) is not str:
                return None
            templates=(literal.value,)
        try:
            for template in templates:
                if a._clock()>a.deadline:
                    return Refused(self.engine,'str.format','format summary deadline exhausted')
                automatic=0;manual=False
                for _literal,field,spec,conversion in Formatter().parse(template):
                    if a._clock()>a.deadline:
                        return Refused(self.engine,'str.format','format summary deadline exhausted')
                    if field is None:
                        continue
                    if conversion not in (None,'r','s','a') or '{' in spec or '}' in spec:
                        return None
                    if field=='':
                        if manual:
                            return None
                        key=automatic;automatic+=1
                    else:
                        key,lookups=_string.formatter_field_name_split(field)
                        if any(True for _ in lookups):
                            return None
                        if type(key) is int:
                            if automatic:
                                return None
                            manual=True
                    if type(key) is int:
                        if key>=len(ctx.args)-1:
                            return None
                        value=ctx.args[key+1]
                    else:
                        if key not in ctx.kwargs:
                            return None
                        value=ctx.kwargs[key]
                    if value.n and spec and conversion is None:
                        return None
        except (ValueError,OverflowError,RecursionError,MemoryError):
            return None
        if a._clock()>a.deadline:
            return Refused(self.engine,'str.format','format summary deadline exhausted')
        if contract.returns is None or not contract.returns.tested or contract.returns.type!='str':
            return Refused(self.engine,'str.format','primitive fields require the tested str return contract')
        a._contract_checks(ctx.node,'str.format',contract,
                           {c.exc:'bound direct fields format only closed builtin primitives'
                            for c in contract.raises if c.tested and c.condition!='unknown'})
        return ANYSTR

    def _str_concat_edge(self, ctx, method):
        if len(ctx.args) != 2 or ctx.kwargs or not all(definite_str(x) for x in ctx.args):
            return False
        from .equality import Congruence, Theories
        a = self.analyzer
        backend = Congruence(a.deadline)
        theory = Theories(backend)
        whole, wanted = (theory.build(a._string_term(x)) for x in ctx.args)
        if backend.equivalent(whole, wanted):
            return True
        needle = backend.nodes[wanted]
        if needle.op == "literal" and needle.data == "":
            return True
        value = backend.nodes[whole]
        parts = value.args if value.op == "concat" else (whole,)
        edge = needle.args if needle.op == "concat" else (wanted,)
        if len(edge) > len(parts):
            return False
        parts = parts[:len(edge)] if method == "startswith" else parts[-len(edge):]
        return all(backend.equivalent(x, y) for x, y in zip(parts, edge))

    def _abstract_string(self, rule, ctx, contract):
        from .aval import ANYSTR, ANY_STR, BOOL, BOT, FLOAT, V, int_iv
        from .linear import IntInfo
        a, st, node = self.analyzer, ctx.state, ctx.node
        if rule.qualname == 'str.format':
            formatted = self._str_format_auto_fields(ctx, contract)
            if formatted is not None:
                return formatted
        if rule.qualname.startswith("re."):
            return self._abstract_regex(rule, ctx, contract)
        # Unknown target objects may execute arbitrary callbacks; never run them natively.
        key_parts = (str_key_parts(ctx.args[1]) if rule.qualname == "str.__getitem__" and len(ctx.args) == 2 and not ctx.kwargs else None)
        if any(x.top or (x.st and not (key_parts is not None and x is ctx.args[1]))
               or any(st.heap[r].kind not in (NO_INDEX_KINDS if key_parts is not None and x is ctx.args[1] else {"tuple", "list", "dict"}) for r in x.refs)
               for x in (*ctx.args, *ctx.kwargs.values())):
            return Refused(self.engine, rule.qualname, "non-builtin or unknown operand")
        if rule.qualname.startswith("str.") and rule.qualname != "str.maketrans" and (not ctx.args or ctx.args[0].s is None or not ctx.args[0].replace(s=None).is_bot()):
            return Refused(self.engine, rule.qualname, "receiver is not a definite builtin str")
        proven = {}
        method = rule.qualname[4:] if rule.qualname.startswith("str.") else ""
        admitted = str_call_admitted(st, method, ctx.args, ctx.kwargs) if method else None
        if rule.qualname == "str.__getitem__" and key_parts is not None:
            # a definite builtin str receiver (checked above) and a key none of whose parts runs user code (``str_key_parts``): the
            # five classes the contract admits only through the key's hooks cannot occur; the ``TypeError`` is proven unless some
            # part of the key is not an int or an int-bound slice, and the ``IndexError`` unless some part may be an index.  So a
            # float, str, None or container key is a certain, recorded ``TypeError`` and nothing else (every other class is proven:
            # a program that raises it natively is UNKNOWN only for the classes it raises), and a key that may be an int or a
            # float leaves just those two.  An object key never reaches here (refused above).
            type_error, index_error, certain_type_error = key_parts
            proven = {c.exc: "definite builtin str receiver and a key that runs no user code" for c in contract.raises
                      if not (c.exc == "TypeError" and type_error) and not (c.exc == "IndexError" and index_error)}
        elif rule.qualname == "str.__mod__":
            if len(ctx.args) == 2 and not ctx.kwargs and self._percent_safe(ctx):
                proven = {c.exc: "checked percent specifiers and builtin operand kinds" for c in contract.raises}
        elif rule.qualname == "str.__contains__" and len(ctx.args) == 2 and not ctx.kwargs and ctx.args[1].s is not None and ctx.args[1].replace(s=None).is_bot():
            proven = {c.exc: "substring test of two builtin strs" for c in contract.raises}
        elif admitted is not None:
            proven = {c.exc: "definite builtin str receiver and arguments of the kinds the method takes: no user hook runs" for c in contract.raises}
        elif rule.qualname.startswith("str.") and not ctx.kwargs and len(ctx.args) == 1 and rule.qualname.split(".")[1] in {"capitalize", "casefold", "lower", "upper", "title", "swapcase", "strip", "lstrip", "rstrip", "isalnum", "isalpha", "isascii", "isdecimal", "isdigit", "isidentifier", "islower", "isnumeric", "isprintable", "isspace", "istitle", "isupper"}:
            proven = {c.exc: "total zero-argument method of a builtin str" for c in contract.raises}
        elif rule.qualname in {"repr", "ascii", "str"} and len(ctx.args) == 1 and not ctx.kwargs:
            a.check_formattable(st, ctx.args[0], node)
            proven = {c.exc: "conversion of callback-free builtin values" for c in contract.raises}
        elif rule.qualname in {"format", "syntax.FormattedValue"} and (1 <= len(ctx.args) <= 2 if rule.qualname == "format" else len(ctx.args) == 3) and not ctx.kwargs:
            a.check_formattable(st, ctx.args[0], node)
            specs = ctx.args[1].s if len(ctx.args) > 1 else frozenset({""})
            kinds = set(ctx.args[0].kinds())
            codes = a.key_consts(st, ctx.args[2]) if rule.qualname == "syntax.FormattedValue" else {-1}
            conversion_ok = codes is not None and codes <= {-1,115,114,97}
            converted = codes is not None and codes <= {115,114,97}
            if conversion_ok and specs is not None and (len(ctx.args) < 2 or ctx.args[1].replace(s=None).is_bot()) and (converted or kinds <= {"int", "bool", "float", "str"} or specs == frozenset({""})):
                proven = {c.exc: "builtin conversion and string format spec" for c in contract.raises}
        elif rule.qualname == "syntax.JoinedStr" and not ctx.kwargs and all(x.s is not None and x.replace(s=None).is_bot() for x in ctx.args):
            proven = {c.exc: "joining already formatted str parts" for c in contract.raises}
        recorded=set()
        if (rule.qualname=='str.__getitem__' and key_parts is not None
                and len(ctx.args)==2 and not ctx.kwargs and exact_int_value(ctx.args[1])):
            ok, fact = (self._str_index_bounds(ctx) if any(
                c.exc == 'IndexError' and c.tested and c.condition != 'unknown' for c in contract.raises)
                else (False, 'str.__getitem__ IndexError claim is not tested'))
            payload = None if ok else {'IndexError': a.sequence_index_args(
                rule.qualname, st, a._as_int(st, ctx.args[1]), node)}
            a.check(node,rule.qualname+':IndexError',{'IndexError'},ok,fact,st=st,
                    exact_args=payload,contract_qualname=rule.qualname)
            recorded.add('IndexError')
        a._contract_checks(node, rule.qualname, contract, proven,recorded=recorded)
        if rule.qualname == "str.__getitem__" and key_parts is not None and key_parts[2]:
            # every part of the key raises TypeError (a float, str, None, container, builtin object, a slice with such a bound): the
            # exception is RECORDED above (unproven) and certain, so no run goes on past it -- what follows is not reached, exactly as
            # for a subscript of a finite str (``_native_string``: no output ends the path)
            st.set_bottom()
            return BOT
        kind = contract.returns.type if contract.returns is not None else ""
        if kind == "str":
            return ANYSTR
        if kind == "bool":
            if (method in {"startswith", "endswith"} and admitted is not None
                    and contract.returns.tested and self._str_concat_edge(ctx, method)):
                from .aval import const_bool
                return const_bool(True)
            return BOOL
        if kind == "int":
            return int_iv(INT_RESULT_LOWER[method] if admitted is not None and method in INT_RESULT_LOWER else None, None)
        if kind == "tuple[str,str,str]" and method in {"partition", "rpartition"}:
            addr = a.new_list(st, node, None, BOT, tag=method, kind="tuple", items=[ANYSTR] * 3)
            return V(refs=frozenset({addr}))
        if kind.startswith("list["):
            elem = ANYSTR if kind == "list[str]" else V(top=True)
            # ``s.split(sep)`` / ``s.rsplit(sep)`` with a str separator has at least one part ("" splits to [""]); an empty separator raises
            # ValueError, which ends the path.  Without a separator (None) the result may be empty.
            at_least_one = (admitted is not None and method in ("split", "rsplit") and "sep" in admitted
                            and definite_str(admitted["sep"][1]))
            addr = a.new_list(st, node, IntInfo.interval(1 if at_least_one else 0, None), elem)
            return V(refs=frozenset({addr}))
        return Refused(self.engine, rule.qualname, "non-finite operands have no exact result model")

    def _str_index_bounds(self, ctx):
        """The existing sequence bounds check over the evaluated builtin str."""
        from .aval import ANY_STR
        from .linear import lin_bounds

        a, st = self.analyzer, ctx.state
        value, key = ctx.args
        index = a._as_int(st, key)
        if value.s != ANY_STR and value.s:
            lengths = [len(s) for s in value.s]
            return a.index_ok(st, None, (min(lengths), max(lengths)), index)
        if a.record and a._value_equalities and value.value_term is not None:
            length = a._string_length_form(st, value)
            if length is not None:
                return a.index_ok(st, length, lin_bounds(length, st.zone), index)
        return False, 'string length or index bounds not established'

    def _abstract_regex(self, rule, ctx, contract):
        import re
        from .aval import ANYSTR, ANY_STR, NONE, BOT, V, int_iv
        from .linear import IntInfo
        from .native import concrete, abstract, product
        from .state import vjoin
        from .heap import DictObj
        a, st, node = self.analyzer, ctx.state, ctx.node
        name = rule.qualname
        args = list(ctx.args)
        owner = name.split(".")[1] if name.count(".") == 2 else ""
        pattern = None
        def representative(v):
            xs = concrete(a, st, v)
            if xs is not None and len(xs) == 1:
                return xs[0]
            if v.s is not None and v.replace(s=None).is_bot():
                return ""
            return NotImplemented
        if owner in {"Pattern","Match"}:
            if not args or len(args[0].st) != 1 or not args[0].replace(st=frozenset()).is_bot():
                return Refused(self.engine,name,"unknown regex receiver")
            tag = next(iter(args.pop(0).st))
            if tag[0] == "native-regex" and type(tag[1]) is re.Pattern:
                pattern = tag[1]
            elif tag[0] in {"abstract-pattern", "abstract-match"}:
                pattern = tag[1]
            else:
                return Refused(self.engine,name,"non-finite Match receiver")
            if owner == "Match":
                return self._abstract_match(rule,ctx,contract,pattern,args)
            native_receiver = pattern or re.compile("a")
        else:
            native_receiver = None
        values = [representative(v) for v in args]
        kwargs = {k:representative(v) for k,v in ctx.kwargs.items()}
        if any(x is NotImplemented for x in (*values,*kwargs.values())):
            return Refused(self.engine,name,"callback or unknown regex argument kind")
        from tharkun.solve.contracts.rules_strformat import native_limit
        refusal = native_limit(name, ([native_receiver] if owner else []) + values, kwargs)
        if refusal:
            return Refused(self.engine, name, refusal)
        if not owner and name not in {"re.escape","re.purge"} and values:
            candidates = concrete(a,st,args[0])
            if candidates is not None and len(candidates) == 1:
                try:
                    flags = kwargs.get("flags",values[-1] if name in {"re.compile","re.match","re.search","re.fullmatch","re.findall","re.finditer"} and len(values) > (1 if name == "re.compile" else 2) else 0)
                    pattern = re.compile(candidates[0],flags)
                except (TypeError,ValueError,re.error):
                    pass
        errors=set()
        try:
            rule.native(*([native_receiver] if owner else []),*values,**kwargs)
        except (TypeError,ValueError,KeyError,IndexError,re.error,OverflowError) as exc:
            errors.update(c.__name__ for c in type(exc).__mro__)
        # Every non-finite argument admitted above is a plain str or int. These calls
        # cannot invoke target callbacks. Replacement templates can still name missing groups.
        unsafe_template = name.endswith((".sub", ".subn")) and (pattern is None or any(v.s == ANY_STR for v in args))
        proven = {c.exc:"validated builtin regex argument kinds and call shape" for c in contract.raises if c.exc not in errors and not (unsafe_template and c.exc in {"IndexError","KeyError"})}
        a._contract_checks(node,name,contract,proven)
        method = name.rsplit(".",1)[1]
        if method == "compile":
            return V(st=frozenset({("abstract-pattern",pattern)}))
        if method in {"match","search","fullmatch"}:
            return vjoin(V(st=frozenset({("abstract-match",pattern)})),st.zone,NONE,st.zone)
        if method in {"sub","escape"}:
            return ANYSTR
        if method == "subn":
            return V(refs=frozenset({a.new_list(st,node,None,BOT,tag="regex-subn",kind="tuple",items=[ANYSTR,int_iv(0,None)])}))
        if method in {"findall","finditer","split"}:
            if method == "finditer":
                return Refused(self.engine,name,"non-finite regex iterator consumption is not modelled")
            if method == "split":
                elem = vjoin(ANYSTR,st.zone,NONE,st.zone) if pattern is None or pattern.groups else ANYSTR
            elif pattern is None:
                elem = V(top=True)
            elif pattern.groups <= 1:
                elem = ANYSTR
            else:
                addr=a.new_list(st,node,None,BOT,tag="regex-captures",kind="tuple",items=[ANYSTR]*pattern.groups)
                elem=V(refs=frozenset({addr}))
            addr=a.new_list(st,node,IntInfo.interval(1 if method == "split" else 0,None),elem)
            return V(refs=frozenset({addr}))
        return Refused(self.engine,name,"non-finite regex result is not modelled")

    def _abstract_match(self,rule,ctx,contract,pattern,args):
        from .aval import ANYSTR, NONE, BOT, V, int_iv
        from .native import concrete
        from .state import vjoin
        from .heap import DictObj
        a,st,node=self.analyzer,ctx.state,ctx.node
        method=rule.qualname.rsplit(".",1)[1]
        keys=[concrete(a,st,v) for v in args] if args else [[0]]
        shape = method == "group" or (method == "__getitem__" and len(args) == 1) or (method in {"start", "end", "span"} and len(args) <= 1)
        valid=shape and pattern is not None and all(xs is not None and all((type(x) in {int,bool} and 0 <= x <= pattern.groups) or (type(x) is str and x in pattern.groupindex) for x in xs) for xs in keys)
        proven={c.exc:"valid builtin match group keys" for c in contract.raises} if valid and method in {"group","__getitem__","start","end","span"} and not ctx.kwargs else {}
        a._contract_checks(node,rule.qualname,contract,proven)
        if method in {"start","end"}:
            return int_iv(-1,None)
        if method == "span":
            addr=a.new_list(st,node,None,BOT,tag="match-span",kind="tuple",items=[int_iv(-1,None)]*2)
            return V(refs=frozenset({addr}))
        if method in {"group","__getitem__","expand"}:
            elem=vjoin(ANYSTR,st.zone,NONE,st.zone) if method != "expand" else ANYSTR
            if method == "group" and len(args)>1:
                addr=a.new_list(st,node,None,BOT,tag="match-groups",kind="tuple",items=[elem]*len(args))
                return V(refs=frozenset({addr}))
            return elem
        if method == "groups" and pattern is not None:
            default=ctx.kwargs.get("default",args[0] if args else NONE)
            elem=vjoin(ANYSTR,st.zone,default,st.zone)
            addr=a.new_list(st,node,None,BOT,tag="match-groups",kind="tuple",items=[elem]*pattern.groups)
            return V(refs=frozenset({addr}))
        if method == "groupdict" and pattern is not None:
            default=ctx.kwargs.get("default",args[0] if args else NONE)
            elem=vjoin(ANYSTR,st.zone,default,st.zone)
            addr=a.alloc(st,node,lambda sid:DictObj({k:elem for k in pattern.groupindex}),"match-groupdict")
            return V(refs=frozenset({addr}))
        return Refused(self.engine,rule.qualname,"unknown group structure")

    def _percent_safe(self, ctx):
        import re
        from .aval import ANY_STR
        a, st = self.analyzer, ctx.state
        left, right = ctx.args
        if left.s is None or left.s == ANY_STR:
            return False
        if right.refs:
            # [internal ref]: a tuple ref does not cover scalar/static alternatives of the join.
            if len(right.refs) != 1 or not right.replace(refs=frozenset()).is_bot():
                return False
            obj = st.heap[next(iter(right.refs))]
            if obj.kind != "tuple" or obj.items is None:
                return False
            values = obj.items
        else:
            values = [right]
        for pattern in left.s:
            pos, used = 0, 0
            while pos < len(pattern):
                if pattern[pos] != "%":
                    pos += 1
                    continue
                if pattern[pos:pos+2] == "%%":
                    pos += 2
                    continue
                match = re.match(r"%[-+#0 ]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlL]?([diouxXeEfFgGcrsa])", pattern[pos:])
                if match is None or "*" in match[0] or used >= len(values):
                    return False
                value = values[used]
                kinds = set(value.kinds())
                conv = match[1]
                if conv in "diouxXeEfFgG" and not kinds <= {"int", "bool"}:
                    return False
                if conv == "c" and not kinds <= {"int", "bool"}:
                    return False
                if conv in "rsa" and not kinds <= {"int", "bool", "float", "str", "None"}:
                    return False
                used += 1
                pos += len(match[0])
            if used != len(values):
                return False
        return True


def serve_module_functions(is_served) -> None:
    """Give every MODULE function with a rule a ``visit_<kind>`` -- the generic contract-driven call -- unless the
    analyzer models it by hand or refuses it (``is_served(module, name)`` is False: ``interp.py::MODULES``), because
    a visit that is never called would claim a consumption that does not happen."""
    for rule in all_rules():
        if is_module_function(rule.qualname) and is_served(*rule.qualname.split(".")):
            if not hasattr(AbsintVisitor, f"visit_{rule.kind}"):
                setattr(AbsintVisitor, f"visit_{rule.kind}", AbsintVisitor._module_function)


def serve_set_operations() -> None:
    """Give every ``set.*`` rule (the methods, the operators, the comparisons, ``in``) its ``visit_<kind>``: the analyzer
    models the whole family (``sets.py``), and the rule it asks is the one named by the operation."""
    for rule in all_rules():
        if rule.qualname.startswith("set."):
            setattr(AbsintVisitor, f"visit_{rule.kind}", AbsintVisitor._set_operation)


__all__ = ["AbsintVisitor", "CallCtx", "MethodCtx", "SubscriptCtx", "OutfileCtx", "LengthHintCtx", "SetCtx", "ProtocolCtx", "PROTOCOL_DISPATCHES", "is_module_function", "serve_module_functions",
           "serve_set_operations"]


#: the object protocols the analyzer dispatches by hand (``Analyzer.visit_protocol``): the operators, the comparisons, ``__getitem__``,
#: ``len``, iteration, the truth of an instance and the read of one of its attributes
PROTOCOL_DISPATCHES = ("protocol.binary_operator", "protocol.inplace_operator", "protocol.comparison", "protocol.container_getitem",
                       "protocol.container_len", "protocol.iteration", "protocol.truthiness", "protocol.attribute_access")
for _rule in all_rules():
    if _rule.qualname in PROTOCOL_DISPATCHES:
        setattr(AbsintVisitor, "visit_" + _rule.kind, AbsintVisitor._protocol)

# One native visit implementation; dispatch still names each Rule's distinct kind.
for _rule in all_rules():
    if _rule.qualname.startswith(("str.", "re.", "syntax.")) or _rule.qualname in {"format", "repr", "ascii", "str"}:
        setattr(AbsintVisitor, "visit_" + _rule.kind, AbsintVisitor._native_string)


for _rule in all_rules():
    if _rule.qualname in {"int.bit_length", "int.bit_count", "int.conjugate", "int.is_integer", "int.as_integer_ratio",
                          "float.conjugate", "float.is_integer", "float.as_integer_ratio"}:
        setattr(AbsintVisitor, "visit_" + _rule.kind, AbsintVisitor._scalar_query)

for _rule in all_rules():
    if _rule.qualname in {"range.count", "range.index"}:
        setattr(AbsintVisitor, "visit_" + _rule.kind, AbsintVisitor._range_query)
