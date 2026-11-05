"""Closed single-consumer map proof, loaded only when a map call is reached."""
import ast
import os

from .aval import BOT, V, const_bool
from .errors import Unsupported
from .heap import FuncObj, IteratorObj, TupleObj


def _site(self, node):
    """Authenticate one module binding/unpack using the frozen-list escape audit."""
    import copy
    from .interp import frozen_list_displays

    if self._clock() >= self.deadline:
        raise Unsupported('identity map source audit time budget exhausted', node)
    parents = {child: parent for parent in ast.walk(self.tree) for child in ast.iter_child_nodes(parent)}
    statement = parents.get(node)
    if (not isinstance(statement, ast.Assign) or statement.value is not node or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)):
        raise Unsupported('identity map requires one named binding and unpack', node)
    name = statement.targets[0].id
    uses = [n for n in ast.walk(self.tree) if isinstance(n, ast.Name) and n.id == name]
    loads = [n for n in uses if isinstance(n.ctx, ast.Load)]
    if len(uses) != 2 or len(loads) != 1:
        raise Unsupported('identity map aliases or repeated consumption need cursor state', node)
    consumer = parents.get(loads[0])
    if (not isinstance(consumer, ast.Assign) or consumer.value is not loads[0] or len(consumer.targets) != 1
            or not isinstance(consumer.targets[0], (ast.Tuple, ast.List))
            or any(isinstance(x, ast.Starred) for x in consumer.targets[0].elts)):
        raise Unsupported('identity map requires one plain unpack consumer', node)
    for stmt in (statement, consumer):
        parent = parents.get(stmt)
        while isinstance(parent, ast.If):
            parent = parents.get(parent)
        if not isinstance(parent, ast.Module):
            raise Unsupported('identity map consumption in loops or functions needs cursor state', node)
    clone = copy.deepcopy(self.tree)
    copies = dict(zip(ast.walk(self.tree), ast.walk(clone)))
    copies[statement].value = ast.copy_location(ast.List(elts=[], ctx=ast.Load()), node)
    resolver = self._extern_resolver()
    if resolver is None:
        raise Unsupported('identity map source closure is unavailable', node)
    others = [r.tree for key, r in self._ext_by_key.items() if key != '' and r is not resolver]
    positions = frozen_list_displays(clone, others, os.path.splitext(os.path.basename(self.filename))[0])
    if self._clock() >= self.deadline:
        raise Unsupported('identity map source audit time budget exhausted', node)
    if (node.lineno, node.col_offset) not in positions:
        raise Unsupported('identity map may escape through a binding or global channel', node)

def model(self, st, args, kwargs, node):
    fctx = self._fctx_stack[-1]
    if not fctx.is_module or fctx.module_key:
        raise Unsupported('identity map requires the program module context', node)
    if (len(args) != 2 or len(node.args) != 2 or set(kwargs) - {'strict'}
            or any(k.arg != 'strict' or not isinstance(k.value, (ast.Name, ast.Constant)) for k in node.keywords)):
        raise Unsupported('map needs a single iterable and effect-free keyword capture', node)
    source = node.args[1]
    # The category is an exact literal DISPLAY of pure reads (a list or a tuple display): the positional capture is the same either way.
    if not isinstance(source, (ast.List, ast.Tuple)) or any(not isinstance(e, (ast.Name, ast.Constant)) for e in source.elts):
        raise Unsupported('map needs construction-time positional capture for this iterable', node)
    function = args[0]
    if len(function.refs) != 1 or not function.replace(refs=frozenset()).is_bot():
        raise Unsupported('map callback needs lazy call state', node)
    function = st.heap[next(iter(function.refs))]
    if (type(function) is not FuncObj or function.flavor != 'plain'
            or function.captures or function.capture_frames):
        raise Unsupported('map callback lifetime or closure needs lazy call state', node)
    if any(v.top or v.refs or v.st for v in (*function.defaults, *function.kwdefaults.values())):
        raise Unsupported('map callback default cleanup is not modeled', node)
    definition = function.node
    if not isinstance(definition, (ast.FunctionDef, ast.Lambda)):
        raise Unsupported('map callback requires an ordinary source function', node)
    parameters = [*definition.args.posonlyargs, *definition.args.args]
    body = definition.body if isinstance(definition, ast.Lambda) else (
        definition.body[0].value if len(definition.body) == 1 and isinstance(definition.body[0], ast.Return) else None)
    identity = (len(parameters) == 1 and not definition.args.kwonlyargs
                and not definition.args.vararg and not definition.args.kwarg
                and isinstance(body, ast.Name) and body.id == parameters[0].arg)
    strict = kwargs.get('strict', const_bool(False))
    if strict.top or strict.refs or strict.st:
        raise Unsupported('map strict truth may execute a target hook', node)
    _site(self, node)
    self.check(node, 'call-args', {'TypeError'}, True, 'native single-source map binding and primitive strict', st=st)
    # Only pure module reads occur after the original argument evaluation.
    items = []
    for expression in source.elts:
        if self._clock() >= self.deadline:
            raise Unsupported('identity map positional capture time budget exhausted', node)
        value = self.eval(expression, st, fctx)
        if not identity and (value.top or value.refs or value.st):
            raise Unsupported('map callback input cleanup needs heap lifetime tracking', node)
        items.append(value)
    tag = 'identity-map' if identity else 'scalar-map'
    if not identity:
        items.insert(0, args[0])
    address = self.new_list(st, node, None, BOT, tag=tag+'-positions', kind='tuple', items=items)
    value = self.borrow(st, V(refs=frozenset({address})))
    address = self.alloc(st, node, lambda sid: IteratorObj(value), tag)
    return V(refs=frozenset({address}))



def consume(self, st, iterator, count, node):
    """Analyze deferred calls in order, before any unpack target is assigned."""
    packet = st.heap[next(iter(iterator.source.refs))]
    if type(packet) is not TupleObj or packet.items is None or not packet.items:
        raise Unsupported('map callback positional snapshot is not closed', node)
    callback, *operands = packet.items
    if len(operands) != count:
        # StopIteration from a callback can change whether a mismatch occurs.
        raise Unsupported('map callback length mismatch needs iterator exception routing', node)
    values = []
    for operand in operands:
        if self._clock() >= self.deadline:
            raise Unsupported('map callback consumption time budget exhausted', node)
        value = self.call(st, callback, [operand], {}, node)
        if st.bottom:
            return None
        if value.top or value.refs or value.st:
            raise Unsupported('map callback result needs heap reference remapping', node)
        values.append(self.mat(st, value))
    return values
