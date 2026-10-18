"""The symbolic executor's contract visitor ([internal ref]).

Visits use the existing transfer functions and adapter gates unchanged.
An absent visit returns Refused; the executor keeps its existing named
Unsupported diagnostic when that refusal reaches a call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tharkun.solve.contracts.rules import Refused, Rule, Visitor, all_rules, get_rule


@dataclass(frozen=True)
class CallCtx:
    node: Any
    receiver: Any = None
    method: str = ""
    args: tuple | None = None
    kwargs: dict | None = None


#: the methods of the builtin containers the executor runs NATIVELY on concrete operands, by their rules ([internal ref] step 2): qualname ->
#: (it mutates its receiver, it compares elements with ``==`` or ``<``).  ``_Exec._native_method`` is the one model of them all.
NATIVE_METHODS = {
    "list.count": (False, True), "list.index": (False, True), "list.copy": (False, False), "list.clear": (True, False),
    "list.extend": (True, False), "list.insert": (True, False), "list.pop": (True, False), "list.remove": (True, True),
    "list.reverse": (True, False), "list.sort": (True, True),
    "dict.items": (False, False), "dict.values": (False, False), "dict.copy": (False, False), "dict.clear": (True, False),
    "dict.pop": (True, False), "dict.popitem": (True, False), "dict.setdefault": (True, False), "dict.update": (True, False),
    "tuple.count": (False, True), "tuple.index": (False, True),
}


#: the functions of the standard-library modules the executor runs NATIVELY on concrete operands, by their rules ([internal ref] step 2): qualname
#: -> (it mutates its first argument, it compares elements).  ``_Runner._native_module_function`` is the one model of them all; the module
#: is the one the NAME is bound to in the harness (identity, never the spelling).
NATIVE_MODULE_FUNCTIONS = {
    "bisect.bisect_left": (False, True), "bisect.insort": (True, True),
    "heapq.heappush": (True, True), "heapq.heapify": (True, True), "heapq.nlargest": (False, True), "heapq.nsmallest": (False, True),
    "copy.copy": (False, False), "copy.deepcopy": (False, True),
    "itertools.chain": (False, False), "itertools.product": (False, False), "itertools.combinations": (False, False),
    "itertools.permutations": (False, False), "itertools.islice": (False, False),
}


class SymexecVisitor(Visitor):
    engine = "symexec"

    def __init__(self, executor):
        self.executor = executor

    def visit_math_function(self, rule: Rule, ctx: CallCtx):
        from tharkun.sv import _symexec_math
        return _symexec_math.call(self.executor, rule.qualname.split(".", 1)[1], ctx.node)

    def visit_math_constant(self, rule: Rule, ctx: CallCtx):
        from tharkun.solve.contracts.adapters.concolic import math_constant
        return math_constant(rule.qualname.split(".", 1)[1])

    def _fold(self, rule: Rule, ctx: CallCtx):
        return self.executor._contract_fold_model(rule.qualname, ctx.node)

    visit_ord = visit_chr = visit_tuple = visit_list = _fold
    visit_zip = visit_enumerate = visit_reversed = visit_type = _fold

    def _builtin(self, rule: Rule, ctx: CallCtx):
        """The builtins this executor evaluates itself (``_Exec._model_<name>``: ``len``, ``abs``, ``min``, ``max``, ``any``,
        ``all``, ``int``, ``float``, ``divmod``, ``pow``, ``print``, ``isinstance``, ``sum``, ``range``, ``sorted``) are read through
        their rules ([internal ref] step 2): the
        call is the contract's only while the contract is TESTED -- an untested or withdrawn one is a refusal, never the model
        of old -- and the model is the one transfer function of the call."""
        from tharkun.solve.contracts import get
        contract = get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine, rule.qualname, "no tested contract")
        return getattr(self.executor, "_model_" + rule.qualname.replace(".", "_"))(ctx.node, *(ctx.args or ()))

    visit_len = visit_abs = visit_min = visit_max = visit_any = visit_all = _builtin
    visit_int = visit_isinstance = visit_sum = visit_range = visit_sorted = _builtin
    visit_float = visit_divmod = visit_pow = visit_print = _builtin
    #: ``random.choice`` as the harness lifts it (``__sv_index``): the index is a draw, the element is the contract's ([internal ref])
    visit_random_choice = _builtin
    #: the containers' methods and subscripts (``_Runner``): a model per rule, on the operands already evaluated
    visit_list_append = visit_dict_get = visit_dict_keys = _builtin
    visit_list_getitem = visit_dict_getitem = visit_tuple_getitem = visit_list_setitem = visit_dict_setitem = _builtin

    def _module_call(self, rule: Rule, ctx: CallCtx):
        return self.executor._contract_call_model(ctx.node.func, ctx.node)

    visit_random_random = visit_heapq_heappop = _module_call

    def visit_open_output(self, rule: Rule, ctx: CallCtx):
        return self.executor._open_output_model(ctx.node)

    def _output_method(self, rule: Rule, ctx: CallCtx):
        return self.executor._handle_call_model(ctx.receiver, ctx.method, ctx.node)

    visit_output_write = visit_output_writelines = visit_output_flush = _output_method
    visit_output_close = visit_output_enter = visit_output_exit = _output_method


    def _native_string(self, rule: Rule, ctx: CallCtx):
        import re
        import z3
        from tharkun.solve.contracts import get
        from tharkun.sv._symexec import _LazyIter, Ref, Unsupported
        contract = get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine, rule.qualname, "no tested contract")
        if ctx.args is None:
            return Refused(self.engine, rule.qualname, "native visit requires evaluated operands")
        args, kwargs = ctx.args, ctx.kwargs or {}
        for value in (*args, *kwargs.values()):
            if type(value) is Ref:
                try:
                    Ref._refuse(value)
                except Unsupported as exc:
                    return Refused(self.engine, rule.qualname, str(exc))
        def symbolic(v):
            return any(type(v) is cls for cls in (z3.SeqRef, z3.ArithRef, z3.IntNumRef))
        def safe(v, seen=frozenset()):
            if id(v) in seen or len(seen) > 32:
                return False
            if any(type(v) is cls for cls in (str, bytes, int, bool, float, type(None), re.Pattern, re.Match, re.RegexFlag)):
                return True
            if type(v) is tuple or type(v) is list:
                return all(safe(x, seen | {id(v)}) for x in v)
            if type(v) is slice:
                return all(safe(x) for x in (v.start, v.stop, v.step))
            if type(v) is dict:
                return all(safe(k, seen | {id(v)}) and safe(x, seen | {id(v)}) for k, x in v.items())
            return False
        if all(safe(x) for x in (*args, *kwargs.values())):
            if rule.qualname.endswith(".scanner"):
                return Refused(self.engine, rule.qualname, "mutable scanner state is not modelled")
            from tharkun.solve.contracts.rules_strformat import native_limit
            refusal = native_limit(rule.qualname,args,kwargs)
            if refusal:
                return Refused(self.engine,rule.qualname,refusal)
            try:
                result = rule.native(*args, **kwargs)
                return _LazyIter(list(result)) if rule.qualname.endswith("finditer") else result
            except (TypeError, ValueError, KeyError, IndexError, AttributeError, OverflowError, re.error) as exc:
                return Refused(self.engine, rule.qualname, "native " + type(exc).__name__ + ": " + str(exc))
        if rule.qualname == "str.__contains__" and len(args) == 2 and not kwargs and all(type(x) is str or (symbolic(x) and x.sort() == z3.StringSort()) for x in args):
            return z3.Contains(args[0], args[1])
        if rule.qualname in {"str.startswith", "str.endswith"} and 2 <= len(args) <= 4 and not kwargs and symbolic(args[0]) and args[0].sort() == z3.StringSort():
            prefixes = args[1] if type(args[1]) is tuple else (args[1],)
            indices = args[2:]
            if all(type(x) is str for x in prefixes) and all(x is None or type(x) is int or type(x) is bool or (symbolic(x) and x.sort() == z3.IntSort()) for x in indices):
                length = z3.Length(args[0])
                def lower(x):
                    return z3.If(x < 0, z3.If(length + x < 0, 0, length + x), x)
                start = lower(indices[0]) if indices and indices[0] is not None else 0
                end = lower(indices[1]) if len(indices) > 1 and indices[1] is not None else length
                end = z3.If(end > length, length, end)
                text = z3.SubString(args[0], start, end - start)
                fn = z3.PrefixOf if rule.qualname.endswith("startswith") else z3.SuffixOf
                return z3.And(start <= length, start <= end, z3.Or(*[fn(x, text) for x in prefixes]))
        if rule.qualname == "str" and len(args) == 1 and not kwargs:
            return self._native_string(get_rule("format"), CallCtx(ctx.node, args=args, kwargs={}))
        if rule.qualname == "format" and len(args) in (1, 2) and (len(args) == 1 or (type(args[1]) is str and args[1] == "")) and not kwargs:
            value = args[0]
            if symbolic(value) and value.sort() == z3.StringSort():
                return value
            if symbolic(value) and value.sort() == z3.IntSort():
                if not hasattr(self.executor,"_holds"):
                    return Refused(self.engine,rule.qualname,"symbolic int conversion requires successful-path bounds")
                import sys
                digits = sys.get_int_max_str_digits()
                if digits:
                    ceiling = z3.IntVal("1" + "0" * digits)
                    self.executor._holds(z3.And(value > -ceiling, value < ceiling))
                return z3.If(value >= 0, z3.IntToStr(value), z3.Concat("-", z3.IntToStr(-value)))
        if rule.qualname == "syntax.FormattedValue" and len(args) == 3 and not kwargs and type(args[1]) is str and args[1] == "" and type(args[2]) is int and args[2] in {-1, 115}:
            return self._native_string(get_rule("format"), CallCtx(ctx.node, args=(args[0],), kwargs={}))
        return Refused(self.engine, rule.qualname, "symbolic operands require an exact encoding; no approximation")


for _rule in all_rules():
    if _rule.qualname.startswith(("str.", "re.", "syntax.")) or _rule.qualname in {"format", "repr", "ascii", "str"}:
        setattr(SymexecVisitor, "visit_" + _rule.kind, SymexecVisitor._native_string)


for _rule in all_rules():
    if _rule.qualname in NATIVE_METHODS or _rule.qualname in NATIVE_MODULE_FUNCTIONS:
        setattr(SymexecVisitor, "visit_" + _rule.kind, SymexecVisitor._builtin)
