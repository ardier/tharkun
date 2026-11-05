"""File instances shared by search and proof: bytes plus provenance-bearing facts ([internal ref])."""
from __future__ import annotations

import ast
import base64
import io
from _io import BytesIO as _BytesIO, StringIO as _StringIO
import re
from dataclasses import asdict, dataclass, field, replace

from tharkun.facts import Fact
from tharkun.lattice import Clause, FactState, Origin


class WorldRefused(ValueError):
    """A refinement could not be materialized; the previous object is unchanged."""


def validate_world(raw):
    """Validate the public concrete-only request boundary, without touching the host."""
    if not isinstance(raw, dict):
        raise WorldRefused("world maps paths to {content: str}")
    for path, spec in raw.items():
        if (not isinstance(path, str) or not path or "\0" in path or not isinstance(spec, dict)
                or set(spec) not in ({"content"}, {"content", "kind"})
                or spec.get("kind", "file") not in ("file", "stream") or not isinstance(spec["content"], str)):
            raise WorldRefused("world maps nonempty paths to {content: str}; recipes are not supported")
        try:
            spec["content"].encode("utf-8")
        except UnicodeEncodeError as exc:
            raise WorldRefused("world content must encode as UTF-8") from exc
    return raw


class _Terms:
    """The search subset of file predicates; every model is checked with Python too."""

    def __init__(self, content):
        import z3
        self.z3, self.content, self.constraints = z3, content, []

    def term(self, n):
        z3 = self.z3
        if isinstance(n, ast.Constant) and type(n.value) in (str, int, bool):
            return n.value
        if isinstance(n, ast.Name):
            if n.id == "content":
                return self.content
            if n.id == "length":
                return z3.Length(self.content)
        if isinstance(n, ast.Subscript):
            if isinstance(n.value, ast.Name) and n.value.id == "lines":
                index = ast.literal_eval(n.slice)
                if type(index) is not int or index < 0:
                    raise WorldRefused("world line index must be nonnegative")
                text = self.content
                for _ in range(index):
                    end = z3.IndexOf(text, "\n")
                    self.constraints.append(end >= 0)
                    text = z3.SubString(text, end + 1, z3.Length(text))
                end = z3.IndexOf(text, "\n")
                return z3.If(end < 0, text, z3.SubString(text, 0, end + 1))
            text = self.term(n.value)
            if isinstance(n.slice, ast.Slice) and n.slice.step is None:
                lo = ast.literal_eval(n.slice.lower) if n.slice.lower else 0
                hi = ast.literal_eval(n.slice.upper) if n.slice.upper else None
                if type(lo) is not int or lo < 0 or (hi is not None and (type(hi) is not int or hi < lo)):
                    raise WorldRefused("world slices require nonnegative constant bounds")
                return z3.SubString(text, lo, z3.Length(text) if hi is None else hi - lo)
        if isinstance(n, ast.UnaryOp):
            value = self.term(n.operand)
            if isinstance(n.op, ast.Not):
                return z3.Not(value)
            if isinstance(n.op, ast.USub):
                return -value
        if isinstance(n, ast.BoolOp):
            return (z3.And if isinstance(n.op, ast.And) else z3.Or)(*[self.term(v) for v in n.values])
        if isinstance(n, ast.BinOp):
            a, b = self.term(n.left), self.term(n.right)
            if isinstance(n.op, ast.Add):
                return z3.Concat(a, b) if z3.is_string(a) or isinstance(a, str) else a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
        if isinstance(n, ast.Compare):
            values = [self.term(v) for v in (n.left, *n.comparators)]
            ops = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
                   ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
                   ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}
            if all(type(op) in ops for op in n.ops):
                return z3.And(*[ops[type(op)](a, b) for op, a, b in zip(n.ops, values, values[1:])])
        if isinstance(n, ast.Call) and not n.keywords:
            if isinstance(n.func, ast.Name) and len(n.args) == 1:
                text = self.term(n.args[0])
                if n.func.id == "len":
                    return z3.Length(text)
                if n.func.id == "int":
                    number = z3.Int(f"world_int_{len(self.constraints)}")
                    before, after = z3.Strings(f"world_ws_{len(self.constraints)}_a world_ws_{len(self.constraints)}_b")
                    whitespace = z3.Star(z3.Union(*[z3.Re(c) for c in " \t\r\n\v\f"]))
                    digits = z3.If(number < 0, z3.Concat("-", z3.IntToStr(-number)), z3.IntToStr(number))
                    self.constraints.extend((z3.InRe(before, whitespace), z3.InRe(after, whitespace),
                                             text == z3.Concat(before, digits, after)))
                    return number
            if isinstance(n.func, ast.Attribute) and n.func.attr == "isdigit" and not n.args:
                return z3.InRe(self.term(n.func.value), z3.Plus(z3.Range("0", "9")))
            if isinstance(n.func, ast.Attribute) and len(n.args) == 1:
                text, arg = self.term(n.func.value), self.term(n.args[0])
                if n.func.attr == "startswith":
                    return z3.PrefixOf(arg, text)
                if n.func.attr == "endswith":
                    return z3.SuffixOf(arg, text)
        raise WorldRefused(f"unsupported world predicate: {ast.unparse(n)}")


@dataclass
class WorldObject:
    path: str
    concrete: bytes = b""
    facts: FactState = field(default_factory=FactState)
    mode: str = "mocked"
    origins: dict[str, Origin] = field(default_factory=dict)
    kind: str = "file"

    @property
    def lines(self):
        return tuple(io.StringIO(self.concrete.decode("utf-8", errors="surrogateescape")).readlines())

    def _holds(self, data, facts):
        try:
            content = data.decode("utf-8")
            env = {"content": content, "length": len(content), "lines": list(io.StringIO(content)),
                   "len": len, "int": int, "re": re}
            for clause in facts.clauses.clauses:
                # Validate the shape before evaluating; no arbitrary target calls run here.
                if not _predicate_shape(ast.parse(clause.text, mode="eval").body):
                    raise WorldRefused(f"unsupported world predicate: {clause.text}")
                if not eval(compile(ast.parse(clause.text, mode="eval"), "<world fact>", "eval"),
                            {"__builtins__": {}}, env):
                    return False
            for subject, state in facts.vars.items():
                if subject not in env:
                    raise WorldRefused(f"unknown world subject: {subject}")
                value = env[subject]
                if state.values.values is not None and value not in state.values.values:
                    return False
                if state.types.types is not None and type(value).__name__ not in state.types.types:
                    return False
                interval = state.interval
                if ((interval.lo is not None and value < interval.lo)
                        or (interval.hi is not None and value > interval.hi) or value in interval.excluded):
                    return False
            return not facts.is_bottom()
        except (UnicodeError, ValueError, IndexError, TypeError):
            return False

    def materialize(self, facts=None):
        """Preserve satisfying bytes; otherwise minimize length change and changed positions."""
        import z3
        facts = self.facts if facts is None else facts
        if facts.is_bottom() or facts.clauses.contradictions():
            raise WorldRefused("contradictory world facts (bottom)")
        if facts.is_top() or self._holds(self.concrete, facts):
            return self.concrete
        numeric = self._numeric_candidate(facts)
        if numeric is not None:
            return numeric
        text = z3.String("world_content")
        terms = _Terms(text)
        constraints = []
        regexes = []
        for clause in facts.clauses.clauses:
            node = ast.parse(clause.text, mode="eval").body
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "re"
                    and node.func.attr == "fullmatch" and len(node.args) == 2 and not node.keywords
                    and isinstance(node.args[1], ast.Name) and node.args[1].id == "content"):
                regexes.append(ast.literal_eval(node.args[0]))
            else:
                constraints.append(terms.term(node))
        if facts.vars:
            for subject, state in facts.vars.items():
                value = terms.term(ast.parse(subject, mode="eval").body)
                if state.values.values is not None:
                    constraints.append(z3.Or(*[value == v for v in state.values.values]))
                if state.interval.lo is not None:
                    constraints.append(value >= state.interval.lo)
                if state.interval.hi is not None:
                    constraints.append(value <= state.interval.hi)
                constraints.extend(value != v for v in state.interval.excluded)
        if regexes:
            from tharkun.solve.condition import _generate_from_regex
            for pattern in regexes:
                for _ in range(8):
                    candidate = _generate_from_regex(pattern)
                    if isinstance(candidate, str) and self._holds(candidate.encode(), facts):
                        return candidate.encode()
            raise WorldRefused("regex producer could not satisfy all world facts")
        solver = z3.Optimize()
        solver.set(timeout=2000)
        solver.add(*constraints, *terms.constraints)
        old = self.concrete.decode("utf-8", errors="replace")
        delta = z3.Length(text) - len(old)
        solver.minimize(z3.If(delta >= 0, delta, -delta))
        if old:
            solver.minimize(z3.Sum(*[z3.If(z3.SubString(text, i, 1) == c, 0, 1) for i, c in enumerate(old)]))
        if solver.check() != z3.sat:
            raise WorldRefused("world facts have no materialization (unsat or solver timeout)")
        candidate = solver.model().eval(text, model_completion=True).as_string().encode("utf-8")
        if not self._holds(candidate, facts):
            raise WorldRefused("world model failed native predicate validation")
        return candidate

    def _numeric_candidate(self, facts):
        """Reuse the integer guard producer, then splice parsed fields into the current text."""
        from tharkun.solve.arith import solve_int_guard
        fields, guards = {}, []

        class Fields(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == "int" and len(node.args) == 1:
                    text = ast.unparse(node.args[0])
                    fields.setdefault(text, f"field{len(fields)}")
                    return ast.copy_location(ast.Name(fields[text], ast.Load()), node)
                return self.generic_visit(node)

        for clause in sorted(facts.clauses.clauses, key=lambda c: c.text):
            tree = Fields().visit(ast.parse(clause.text, mode="eval").body)
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            if names and names <= set(fields.values()):
                guards.append(ast.unparse(tree))
        if not guards:
            return None
        solved = solve_int_guard(" and ".join(f"({g})" for g in guards), fields.values())
        if solved.status != "sat":
            return None
        content = self.concrete.decode("utf-8", errors="replace")
        edits = []
        for text, name in fields.items():
            if name not in solved.witness:
                continue
            node = ast.parse(text, mode="eval").body
            if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
                return None
            start = 0
            base = node.value
            if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name) and base.value.id == "lines":
                index = ast.literal_eval(base.slice)
                lines = content.splitlines(keepends=True)
                if type(index) is not int or not 0 <= index < len(lines):
                    return None
                start = sum(map(len, lines[:index]))
                end = start + len(lines[index])
            elif isinstance(base, ast.Name) and base.id == "content":
                end = len(content)
            else:
                return None
            lo = ast.literal_eval(node.slice.lower) if node.slice.lower else 0
            hi = ast.literal_eval(node.slice.upper) if node.slice.upper else end - start
            if type(lo) is not int or type(hi) is not int or lo < 0 or hi < lo or node.slice.step:
                return None
            start, end = start + lo, start + hi
            old = content[start:end]
            trailing = old[len(old.rstrip()):]
            edits.append((start, end, str(solved.witness[name]) + trailing))
        for start, end, replacement in sorted(edits, reverse=True):
            content = content[:start] + replacement + content[end:]
        candidate = content.encode()
        return candidate if edits and self._holds(candidate, facts) else None

    def add(self, fact: str | Clause | FactState, origin: Origin = Origin()):
        incoming = fact if isinstance(fact, FactState) else FactState().with_clause(
            fact.text if isinstance(fact, Clause) else fact, fact.origin if isinstance(fact, Clause) else origin)
        merged = self.facts.meet(incoming)
        try:
            concrete = self.materialize(merged)
        except (ValueError, TypeError, SyntaxError) as exc:
            if isinstance(exc, WorldRefused):
                raise
            raise WorldRefused(f"unsupported world fact: {exc}") from exc
        if concrete != self.concrete:
            self.mode = "mocked"
        self.facts, self.concrete = merged, concrete
        self.origins = {**self.origins, **{subject: origin for subject in incoming.vars}}
        return self

    def record(self):
        try:
            content, binary = self.concrete.decode("utf-8"), {}
        except UnicodeDecodeError:
            content, binary = "", {"content_base64": base64.b64encode(self.concrete).decode("ascii")}
        return {"content": content, **binary, "mode": self.mode,
                **({"kind": self.kind} if self.kind != "file" else {}),
                "variables": {subject: {"types": sorted(state.types.types) if state.types.types is not None else None,
                                        "values": list(state.values.values) if state.values.values is not None else None,
                                        "lo": state.interval.lo, "hi": state.interval.hi,
                                        "excluded": sorted(state.interval.excluded),
                                        "origin": asdict(self.origins.get(subject, Origin()))}
                              for subject, state in self.facts.vars.items()},
                "facts": [{"text": c.text, "origin": asdict(c.origin)}
                          for c in sorted(self.facts.clauses.clauses, key=lambda c: c.text)]}


def _predicate_shape(node):
    if isinstance(node, (ast.Constant, ast.Load, ast.And, ast.Or, ast.Not, ast.USub,
                         ast.Add, ast.Sub, ast.Mult, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
        return True
    if isinstance(node, ast.Name):
        return node.id in {"content", "length", "lines", "int", "len", "re"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"startswith", "endswith", "isdigit", "fullmatch"} and _predicate_shape(node.value)
    if isinstance(node, (ast.Call, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Subscript, ast.Slice)):
        return all(_predicate_shape(c) for c in ast.iter_child_nodes(node))
    return False


@dataclass
class StreamObject(WorldObject):
    """Program-owned buffer state; writes replace facts rather than refine an input."""

    mode: str = "program"
    kind: str = "stream"
    binary: bool = False
    position: int = 0
    closed: bool = False
    newline: str | None = "\n"

    @classmethod
    def create(cls, initial="", *, binary=False, newline="\n", path="<stream>"):
        stream = _BytesIO(initial) if binary else _StringIO(initial, newline=newline)
        return cls.from_native(stream, path=path)

    @classmethod
    def from_native(cls, stream, *, path="<stream>", origin=Origin("program")):
        binary = type(stream) is _BytesIO
        state = stream.__getstate__()
        content, position = (state[0], state[1]) if binary else (state[0], state[2])
        facts = FactState().with_values("content", {content}).with_interval("length", len(content), len(content))
        return cls(path, content if binary else content.encode("utf-8", "surrogatepass"),
                   facts=facts, origins={"content": origin, "length": origin}, binary=binary,
                   position=position, newline="\n" if binary else state[1])

    def native(self):
        stream = _BytesIO() if self.binary else _StringIO()
        content = self.concrete if self.binary else self.concrete.decode("utf-8", "surrogatepass")
        stream.__setstate__((content, self.position, None) if self.binary else
                            (content, self.newline, self.position, None))
        if self.closed:
            stream.close()
        return stream

    @classmethod
    def restore(cls, content, position, *, binary=False, newline="\n"):
        stream = _BytesIO() if binary else _StringIO()
        stream.__setstate__((content, position, None) if binary else (content, newline, position, None))
        return cls.from_native(stream)

    def project(self, method, *args, origin=Origin("program")):
        stream = self.native()
        result = getattr(stream, method)(*args)
        if stream.closed:
            from dataclasses import replace
            return replace(self, closed=True), result
        return self.from_native(stream, path=self.path, origin=origin), result

    @property
    def lines(self):
        content = self.concrete if self.binary else self.concrete.decode("utf-8", "surrogatepass")
        return tuple(self.restore(content, 0, binary=self.binary, newline=self.newline).native().readlines())

    def _holds(self, data, facts):
        try:
            replace(self, concrete=data).materialize(facts)
            return True
        except (ValueError, TypeError, SyntaxError):
            return False

    def materialize(self, facts=None):
        facts = self.facts if facts is None else facts
        content = self.concrete if self.binary else self.concrete.decode("utf-8", "surrogatepass")
        env = {"content": content, "length": len(content), "len": len, "int": int,
               "lines": self.lines, "re": re}
        if facts.is_bottom() or facts.clauses.contradictions():
            raise WorldRefused("contradictory program stream facts")
        for name, state in facts.vars.items():
            value = env.get(name)
            if (name not in env or state.values.values is not None and value not in state.values.values
                    or state.types.types is not None and type(value).__name__ not in state.types.types
                    or state.interval.lo is not None and value < state.interval.lo
                    or state.interval.hi is not None and value > state.interval.hi
                    or value in state.interval.excluded):
                raise WorldRefused("program stream facts contradict its buffer")
        for clause in facts.clauses.clauses:
            node = ast.parse(clause.text, mode="eval")
            if not _predicate_shape(node.body) or not eval(compile(node, "<stream fact>", "eval"),
                                                          {"__builtins__": {}}, env):
                raise WorldRefused("program stream facts contradict its buffer")
        return self.concrete

    def same_state(self, other):
        return (isinstance(other, StreamObject) and
                (self.concrete, self.binary, self.position, self.closed, self.newline) ==
                (other.concrete, other.binary, other.position, other.closed, other.newline))


class WorldStore:
    """One store in a run's KnowledgeBase; never certified witness evidence."""

    def __init__(self, world=None, knowledge=None):
        self.objects = {path: WorldObject(path, spec["content"].encode(), mode="policy", kind=spec.get("kind", "file"))
                        for path, spec in validate_world(world or {}).items()}
        self.knowledge = knowledge
        if knowledge is not None:
            knowledge.add(Fact("world_fact", "files", self, source="world", certified=False))

    def get(self, path):
        return self.objects.get(path)

    def request(self):
        return {path: {"content": ob.concrete.decode("utf-8"),
                       **({"kind": ob.kind} if ob.kind != "file" else {})} for path, ob in self.objects.items()}

    def open_stream(self, name, *, binary=False):
        """A native io stream with its own cursor over this world's current bytes."""
        obj = self.objects[name]
        if obj.kind != "stream":
            raise WorldRefused(f"{name} is not a stream world object")
        from tharkun.solve.environment import world_record
        record = world_record()
        if record is not None:
            record.touch(name, obj)
        return io.BytesIO(obj.concrete) if binary else io.StringIO(obj.concrete.decode("utf-8"))

    def snapshot(self):
        return {path: obj.record() for path, obj in self.objects.items()}

    def update_snapshot(self, records):
        restored = self.from_snapshot(records)
        for path, obj in restored.objects.items():
            if path in self.objects:
                current = self.objects[path]
                if current.kind != obj.kind:
                    raise WorldRefused(f"worker snapshot changes world kind for {path}")
                merged = current.facts.meet(obj.facts)
                if merged.is_bottom() or (not merged.is_top() and not obj._holds(obj.concrete, merged)):
                    raise WorldRefused(f"worker snapshot contradicts existing facts for {path}")
                obj.facts = merged
                obj.origins = {**current.origins, **obj.origins}
        for path, obj in restored.objects.items():
            if path not in self.objects:
                self.objects[path] = obj
            else:
                current = self.objects[path]
                current.concrete, current.facts, current.mode, current.origins = (
                    obj.concrete, obj.facts, obj.mode, obj.origins)

    @classmethod
    def from_snapshot(cls, records, knowledge=None):
        store = cls({path: {"content": item["content"], **({"kind": item["kind"]} if "kind" in item else {})}
                     for path, item in records.items()}, knowledge)
        for path, record in records.items():
            obj = store.objects[path]
            if "content_base64" in record:
                obj.concrete = base64.b64decode(record["content_base64"], validate=True)
            for fact in record.get("facts", ()):
                obj.add(fact["text"], Origin(**fact["origin"]))
            for subject, spec in record.get("variables", {}).items():
                fact = FactState().with_interval(subject, spec["lo"], spec["hi"], spec["excluded"])
                if spec["types"] is not None:
                    fact = fact.with_types(subject, spec["types"])
                if spec["values"] is not None:
                    fact = fact.with_values(subject, spec["values"])
                obj.add(fact, Origin(**spec["origin"]))
            obj.mode = record.get("mode", "mocked")
        return store
