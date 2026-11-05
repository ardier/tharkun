"""Structured source generator continuations for the proof engine."""
from __future__ import annotations

import ast
from dataclasses import dataclass


class Refused(ValueError):
    pass


@dataclass(frozen=True)
class Instruction:
    kind: str
    node: ast.AST
    next: int | None = None
    alternate: int | None = None
    loop: int | None = None


@dataclass(frozen=True)
class ContinuationGraph:
    entry: int
    code: tuple[Instruction, ...]


def compile_program(function, suspends) -> ContinuationGraph:
    """Keep each yield/consumer boundary; no loop-count or graph-size cutoff."""
    code = []

    def emit(kind, node, next=None, alternate=None, loop=None):
        code.append(Instruction(kind, node, next, alternate, loop))
        return len(code)-1

    def expression(node):
        if node is not None and suspends(ast.Expr(value=node)):
            raise Refused('generator suspension inside an expression')

    def block(statements, next, breaking=None, continuing=None):
        for statement in reversed(statements):
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Yield):
                expression(statement.value.value)
                next=emit('yield',statement.value,next)
            elif isinstance(statement, (ast.Assign,ast.AnnAssign)) and isinstance(statement.value,ast.Yield):
                expression(statement.value.value)
                targets=statement.targets if isinstance(statement,ast.Assign) else [statement.target]
                if any(suspends(target) for target in targets):
                    raise Refused('generator suspension inside an assignment target')
                next=emit('send-none',statement,next)
                next=emit('yield',statement.value,next)
            elif isinstance(statement, ast.If):
                expression(statement.test)
                yes=block(statement.body,next,breaking,continuing)
                no=block(statement.orelse,next,breaking,continuing)
                next=emit('test',statement.test,yes,no)
            elif isinstance(statement, ast.For):
                expression(statement.iter)
                if suspends(statement.target):
                    raise Refused('generator suspension inside a loop target')
                end=block(statement.orelse,next,breaking,continuing)
                test=emit('for-next',statement,alternate=end)
                bump=emit('for-bump',statement,test,loop=test)
                body=block(statement.body,bump,next,bump)
                code[test]=Instruction('for-next',statement,body,end,test)
                next=emit('for-enter',statement,test,loop=test)
            elif isinstance(statement, ast.While):
                expression(statement.test)
                end=block(statement.orelse,next,breaking,continuing)
                test=emit('test',statement.test,alternate=end)
                body=block(statement.body,test,next,test)
                code[test]=Instruction('test',statement.test,body,end)
                next=test
            elif isinstance(statement, ast.Break):
                if breaking is None:
                    raise Refused('generator break has no enclosing loop')
                next=breaking
            elif isinstance(statement, ast.Continue):
                if continuing is None:
                    raise Refused('generator continue has no enclosing loop')
                next=continuing
            elif isinstance(statement, ast.Return):
                expression(statement.value)
                next=emit('return',statement)
            else:
                if suspends(statement):
                    raise Refused('generator suspension inside an unsupported statement')
                if not isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and any(
                        isinstance(node,(ast.Break,ast.Continue)) for node in ast.walk(statement)):
                    raise Refused('generator jump across an unsupported statement')
                next=emit('exec',statement,next)
        return next

    terminal=emit('return',ast.copy_location(ast.Return(value=None),function))
    if isinstance(function,ast.Lambda):
        body=[ast.copy_location(ast.Expr(value=function.body),function.body)]
    else:
        body=function.body
    entry=block(body,terminal)
    return ContinuationGraph(entry,tuple(code))
