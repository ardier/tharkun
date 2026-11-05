"""Search-only file streams and fact projection ([internal ref]); never a witness channel."""
from __future__ import annotations

import ast
import copy
import io
from dataclasses import dataclass

import z3

from tharkun.lattice import FactState, Origin
from tharkun.solve.world import WorldRefused, WorldStore, _Terms, _predicate_shape


@dataclass(eq=False, frozen=True)
class ReadHandle:
    path: str
    key: str


class WorldReads:
    def __init__(self, store):
        self.store = store
        self.contents = {}
        self.expressions = {}
        self.nodes = {}
        self.serial = 0

    def bind(self, term, path, text):
        self.expressions[term.sexpr()] = (path, text)
        return term

    def open(self, runner, node):
        from tharkun.solve.environment import open_effect
        from tharkun.sv._symexec import Unsupported

        if open_effect(node) != "read":
            raise Unsupported("world open: not read-only")
        given = dict(zip(("file", "mode", "buffering", "encoding", "errors", "newline"), node.args))
        given.update({kw.arg: kw.value for kw in node.keywords})
        if set(given) - {"file", "mode", "encoding", "errors", "newline"}:
            raise Unsupported("world open: buffering/descriptor/opener is not modelled")
        mode = runner.ev(given["mode"]) if "mode" in given else "r"
        options = {key: runner.ev(value) for key, value in given.items() if key not in {"file", "mode"}}
        if (mode not in ("r", "rt", "tr") or options.get("encoding") not in (None, "utf-8", "utf8")
                or options.get("errors") not in (None, "strict") or options.get("newline") is not None):
            raise Unsupported("world open: only UTF-8 text with universal newlines is modelled")
        path = runner.ev(given["file"])
        if isinstance(path, z3.SeqRef) and len(self.store.objects) == 1:
            name = next(iter(self.store.objects))
            runner._holds(path == name)
            path = name
        if (not isinstance(path, str) or self.store.get(path) is None
                or self.store.get(path).kind != "file"):
            raise Unsupported("world open: path must name an existing file world object")
        if path not in self.contents:
            term = z3.FreshConst(z3.StringSort(), "world content")
            self.contents[path] = term
            self.bind(term, path, "content")
        self.serial += 1
        handle = ReadHandle(path, f"\0world:{self.serial}:")
        runner.env[handle.key + "pos"] = 0
        runner.env[handle.key + "line"] = 0
        runner.env[handle.key + "closed"] = False
        return handle

    def call(self, runner, handle, method, node):
        from tharkun.sv._symexec import Unsupported

        arity = {"read": (0, 1), "readline": (0, 1), "close": (0,), "__enter__": (0,), "__exit__": (3,)}
        if method not in arity or len(node.args) not in arity[method] or node.keywords:
            raise Unsupported(f"world file: unsupported {method} call shape")
        closed = runner.env[handle.key + "closed"]
        if method in {"close", "__exit__"}:
            runner.env[handle.key + "closed"] = True
            return None
        runner._holds(z3.Not(closed))
        if method == "__enter__":
            return handle
        size = runner.ev(node.args[0]) if node.args else -1
        if type(size) not in (int, bool):
            raise Unsupported("world read: size must be a concrete integer")
        text = self.contents[handle.path]
        pos = runner.env[handle.key + "pos"]
        length = z3.Length(text)
        end = length
        if method == "readline":
            newline = z3.IndexOf(text, "\n", pos)
            end = z3.If(newline < 0, length, newline + 1)
        if size >= 0:
            end = z3.If(pos + size < end, pos + size, end)
        result = z3.SubString(text, pos, end - pos)
        runner.env[handle.key + "pos"] = end
        line = runner.env[handle.key + "line"]
        if method == "readline" and size < 0 and type(line) is int:
            self.bind(result, handle.path, f"lines[{line}]")
            runner.env[handle.key + "line"] = line + 1
        else:
            runner.env[handle.key + "line"] = None
            if type(pos) is int:
                stop = "" if size < 0 else str(pos + size)
                self.bind(result, handle.path, f"content[{pos}:{stop}]")
        return result

    def remember(self, node, value, env):
        """Keep Python spellings for supported expressions over one file's projections."""
        if not isinstance(value, z3.ExprRef):
            return
        from tharkun.sv._symexec import _Lit
        if isinstance(node, _Lit):
            return
        known = self.expressions.get(value.sexpr())
        if known is None:
            paths = set()
            nodes = self.nodes

            class Project(ast.NodeTransformer):
                def visit(self, child):
                    if isinstance(child, _Lit):
                        return child
                    if child is not node and child in nodes:
                        path, text = nodes[child]
                        paths.add(path)
                        return ast.parse(text, mode="eval").body
                    if isinstance(child, ast.Name) and child.id in env and type(env[child.id]) in (str, int, bool):
                        return ast.Constant(env[child.id])
                    clone = copy.copy(child)
                    for name, value in ast.iter_fields(clone):
                        if isinstance(value, list):
                            setattr(clone, name, list(value))
                    return super().visit(clone)

            projected = Project().visit(node)
            if len(paths) == 1 and _predicate_shape(projected):
                known = (paths.pop(), ast.unparse(ast.fix_missing_locations(projected)))
                self.expressions[value.sexpr()] = known
        if known is not None:
            self.nodes[node] = known

    def constraints(self):
        constraints = []
        for path, text in self.contents.items():
            current = [z3.Not(z3.Contains(text, "\r"))]
            obj = self.store.get(path)
            terms = _Terms(text)
            for clause in obj.facts.clauses.clauses:
                current.append(terms.term(ast.parse(clause.text, mode="eval").body))
            for subject, state in obj.facts.vars.items():
                value = terms.term(ast.parse(subject, mode="eval").body)
                if state.values.values is not None:
                    current.append(z3.Or(*[value == v for v in state.values.values]))
                if state.interval.lo is not None:
                    current.append(value >= state.interval.lo)
                if state.interval.hi is not None:
                    current.append(value <= state.interval.hi)
                current.extend(value != v for v in state.interval.excluded)
            current.extend(terms.constraints)
            variables = {v.sexpr(): v for c in current for v in z3.z3util.get_vars(c) if not z3.eq(v, text)}
            replacements = [(v, z3.FreshConst(v.sort(), "world fact")) for v in variables.values()]
            constraints.extend(z3.substitute(c, *replacements) for c in current)
        return constraints

    def snapshot(self, model, goal, *, exact=False):
        """Meet path facts transactionally; optionally pin unprojectable model relations."""
        store = WorldStore.from_snapshot(self.store.snapshot())
        facts = {path: FactState() for path in self.contents}
        origin = Origin("symexec")

        def add_fact(path, text):
            node = ast.parse(text, mode="eval").body
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                for value in node.values:
                    add_fact(path, ast.unparse(value))
            else:
                facts[path] = facts[path].with_clause(text, origin)

        def collect(term, truth=True):
            known = self.expressions.get(term.sexpr())
            if known is not None and z3.is_bool(term):
                path, text = known
                add_fact(path, text if truth else f"not ({text})")
            elif z3.is_and(term) or z3.is_or(term):
                for child in term.children():
                    holds = z3.is_true(model.eval(child, model_completion=True))
                    if (z3.is_and(term) and truth) or (z3.is_or(term) and not truth) or holds == truth:
                        collect(child, holds)
                    if holds == truth and ((z3.is_or(term) and truth) or (z3.is_and(term) and not truth)):
                        break
            elif z3.is_not(term):
                collect(term.arg(0), not truth)
            elif z3.is_app_of(term, z3.Z3_OP_ITE):
                chosen = z3.is_true(model.eval(term.arg(0), model_completion=True))
                collect(term.arg(0), chosen)
                collect(term.arg(1 if chosen else 2), truth)
            else:
                for child in term.children():
                    collect(child, z3.is_true(model.eval(child, model_completion=True)) if z3.is_bool(child) else True)

        collect(goal)
        for path, term in self.contents.items():
            obj = store.get(path)
            text = model.eval(term, model_completion=True).as_string()
            # The runtime's universal-newline view must match the symbolic text.
            if "\r" in text:
                raise WorldRefused("world model needs unsupported newline translation")
            for clause in sorted(facts[path].clauses.clauses, key=lambda c: ("int(" in c.text, c.text)):
                if not obj._holds(text.encode("utf-8"), FactState().with_clause(clause.text, clause.origin)):
                    line_count = len(io.StringIO(text).readlines())
                    eof = any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                              and n.value.id == "lines" and isinstance(n.slice, ast.Constant)
                              and type(n.slice.value) is int and n.slice.value >= line_count
                              for n in ast.walk(ast.parse(clause.text, mode="eval")))
                    if not eof:
                        raise WorldRefused(f"world model violates projected fact: {clause.text}")
                    # EOF has no lines[index]; record its model projection explicitly.
                    obj.add(f"content == {text!r}", Origin("symexec", polarity=f"EOF projection: {clause.text}"))
                    continue
                try:
                    obj.add(clause)
                except WorldRefused as exc:
                    # The path solver already supplied bytes; validate them instead of solving twice.
                    merged = obj.facts.meet(FactState().with_clause(clause.text, clause.origin))
                    if not obj._holds(text.encode("utf-8"), merged):
                        raise WorldRefused(f"{exc}; projected fact: {clause.text}") from exc
                    candidate = copy.deepcopy(obj)
                    candidate.concrete, candidate.mode = text.encode("utf-8"), "mocked"
                    candidate.add(clause)
                    obj = store.objects[path] = candidate
            if exact:
                # shortcut: unprojectable relations pin model bytes; replace with
                # relational world clauses when that predicate language grows.
                obj.add(f"content == {text!r}", Origin("symexec"))
        return store.snapshot()
