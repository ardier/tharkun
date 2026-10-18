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

from tharkun.solve.contracts.rules import Refused, Rule, Visitor, all_rules

#: the modules whose functions the generic contract-driven call serves (``math`` has its own visit)
_MODULES = frozenset({"bisect", "collections", "copy", "functools", "heapq", "io", "itertools", "operator", "random", "re"})
#: a shape contract's qualname -> the builtin the analyzer's model is named after (``common.BUILTIN_SHAPE_CONTRACTS``)
_BUILTIN_OF = {"open.output": "open"}


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
    visit_set = visit_frozenset = _builtin
    visit_sum = visit_any = visit_all = _builtin
    visit_min = visit_max = visit_sorted = visit_divmod = visit_round = _builtin
    #: the builtins the analyzer modelled by hand before it read their contracts ([internal ref] step 2): the model is unchanged,
    #: the contract supplies the seven classes and the model upgrades the ones it has argued (``Analyzer._modelled``)
    visit_abs = visit_len = visit_range = visit_int = visit_float = visit_list = visit_print = visit_isinstance = _builtin

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

    visit_random_random = visit_random_randint = visit_random_randrange = _random_function
    visit_random_choice = visit_random_uniform = visit_random_seed = _random_function

    def _verifier_container(self, rule: Rule, ctx: CallCtx):
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

    def visit_collections_defaultdict(self, rule: Rule, ctx: CallCtx):
        return self.analyzer.c_collections_defaultdict(ctx.state, ctx.args, ctx.kwargs, ctx.node)

    def visit_copy_copy(self, rule: Rule, ctx: CallCtx):
        a = self.analyzer
        if a._copy_copy_list_receiver(ctx.state, ctx.args, ctx.kwargs):
            return a._contract_module_model(rule.qualname, ctx.state, a._copy_copy_list,
                                            ctx.args, ctx.kwargs, ctx.node)
        return a._contract_call(ctx.state, "copy", "copy", ctx.args, ctx.kwargs, ctx.node)

    def _list_method(self, rule: Rule, ctx: MethodCtx):
        return self.analyzer._contract_method(
            rule.qualname, ctx, lambda st, args, kwargs, node: self.analyzer._list_method_model(
                st, ctx.receivers, ctx.name, args, kwargs, node))

    def _dict_method(self, rule: Rule, ctx: MethodCtx):
        return self.analyzer._contract_method(
            rule.qualname, ctx, lambda st, args, kwargs, node: self.analyzer._dict_method_model(
                st, next(iter(ctx.receivers)), ctx.name, args, kwargs, node))

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
    visit_dict_get = visit_dict_keys = visit_dict_values = visit_dict_items = _dict_method
    #: the kind of ``IoStringIOWrite`` is ``io_string_iowrite`` (the snake case of a name with the capitals ``IO`` run together)
    visit_io_string_iowrite = visit_io_string_ioread = visit_io_string_iogetvalue = visit_io_string_ioseek = _stringio_method

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

    def _native_string(self, rule: Rule, ctx: CallCtx):
        import re
        from .native import abstract, concrete, product
        from .aval import ANYSTR, BOOL, BOT, FLOAT, V, int_iv
        from tharkun.solve.contracts import get
        contract = get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine, rule.qualname, "no tested contract")
        a, st, node = self.analyzer, ctx.state, ctx.node
        if rule.qualname.endswith(".scanner"):
            return Refused(self.engine, rule.qualname, "mutable scanner state is not modelled")
        pools = [concrete(a, st, x) for x in ctx.args]
        keys = list(ctx.kwargs)
        pools += [concrete(a, st, ctx.kwargs[k]) for k in keys]
        alternatives = product(pools)
        if alternatives is None:
            return self._abstract_string(rule, ctx, contract)
        from tharkun.solve.contracts.rules_strformat import native_limit
        outputs, errors = [], set()
        for xs in alternatives:
            args, kwargs = xs[:len(ctx.args)], dict(zip(keys, xs[len(ctx.args):]))
            refusal = native_limit(rule.qualname,args,kwargs)
            if refusal:
                return Refused(self.engine,rule.qualname,refusal)
            try:
                result = rule.native(*args, **kwargs)
                if rule.qualname.endswith("finditer"):
                    result = list(result)
                outputs.append(result)
            except (TypeError, ValueError, KeyError, IndexError, AttributeError, OverflowError, re.error) as exc:
                errors.update(c.__name__ for c in type(exc).__mro__)
        proven = {claim.exc: "native evaluation of every finite builtin alternative" for claim in contract.raises if claim.exc not in errors}
        a._contract_checks(node, rule.qualname, contract, proven)
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

    def _abstract_string(self, rule, ctx, contract):
        from .aval import ANYSTR, ANY_STR, BOOL, BOT, FLOAT, V, int_iv
        from .linear import IntInfo
        a, st, node = self.analyzer, ctx.state, ctx.node
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
        a._contract_checks(node, rule.qualname, contract, proven)
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
            return BOOL
        if kind == "int":
            return int_iv(None, None)
        if kind.startswith("list["):
            elem = ANYSTR if kind == "list[str]" else V(top=True)
            addr = a.new_list(st, node, IntInfo.interval(0, None), elem)
            return V(refs=frozenset({addr}))
        return Refused(self.engine, rule.qualname, "non-finite operands have no exact result model")

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

