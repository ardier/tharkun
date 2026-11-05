"""Program-owned IO buffers: exact native states and path-local symbolic text ([internal ref])."""
from __future__ import annotations

import ast
import io
from _io import BytesIO as _BytesIO, StringIO as _StringIO
from dataclasses import dataclass
from types import ModuleType

import z3

from tharkun.lattice import Origin
from tharkun.solve.world import StreamObject


@dataclass(eq=False, frozen=True)
class StreamHandle:
    key: str
    binary: bool
    newline: str | None


class ProgramStreams:
    def __init__(self):
        self.serial = 0

    def constructor(self, runner, func):
        if isinstance(func, ast.Name) and func.id not in runner.env and func.id not in runner.defs:
            value = runner.g.get(func.id)
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
              and func.value.id not in runner.env and runner.g.get(func.value.id) is io
              and type(io) is ModuleType):
            value = io.__dict__.get(func.attr)
        else:
            return None
        return value if value is _StringIO or value is _BytesIO else None

    def create(self, runner, factory, node):
        from ._symexec import Unsupported, _Dead
        if any(k.arg is None for k in node.keywords):
            raise Unsupported("stream constructor: keyword expansion")
        args = [runner.ev(a) for a in node.args]
        kwargs = {k.arg: runner.ev(k.value) for k in node.keywords}
        if any(z3.is_expr(x) for x in [*args, *kwargs.values()]):
            # Only text initial content can be symbolic; newline still determines native IO semantics.
            if (factory is not _StringIO or len(args) > 1 or set(kwargs) - {"initial_value", "newline"}
                    or args and "initial_value" in kwargs):
                raise Unsupported("symbolic stream constructor shape")
            initial = args[0] if args else kwargs.get("initial_value", "")
            newline = kwargs.get("newline", "\n")
            if not z3.is_string(initial) or type(newline) is not str or newline != "\n":
                raise Unsupported("symbolic stream initial value/newline")
            content, position, binary = initial, 0, False
        else:
            if any(type(x) not in (str, bytes, int, bool, type(None)) for x in [*args, *kwargs.values()]):
                raise Unsupported("stream constructor: user protocols")
            try:
                native = factory(*args, **kwargs)
            except TypeError as exc:
                raise Unsupported(f"stream constructor: {exc}") from exc
            except ValueError:
                raise _Dead()
            world = StreamObject.from_native(native)
            content = native.getvalue()
            position, binary, newline = world.position, world.binary, world.newline
        self.serial += 1
        handle = StreamHandle(f"\0program stream:{self.serial}:", binary, newline)
        runner.env.update({handle.key + "content": content, handle.key + "pos": position,
                           handle.key + "closed": False})
        return handle

    @staticmethod
    def _constant(value):
        if not z3.is_expr(value):
            return value
        value = z3.simplify(value)
        if z3.is_string_value(value):
            return value.as_string()
        if z3.is_int_value(value):
            return value.as_long()
        if z3.is_true(value) or z3.is_false(value):
            return z3.is_true(value)
        return value

    def call(self, runner, handle, method, node):
        from ._symexec import Unsupported, _Dead
        arity = {"write": (1,), "getvalue": (0,), "read": (0, 1), "readline": (0, 1),
                 "seek": (1, 2), "tell": (0,), "close": (0,), "flush": (0,),
                 "__enter__": (0,), "__exit__": (3,)}
        if method not in arity or len(node.args) not in arity[method] or node.keywords:
            raise Unsupported(f"program stream: unsupported {method} shape")
        args = [self._constant(runner.ev(a)) for a in node.args]
        content, pos, closed = [self._constant(runner.env[handle.key + k]) for k in ("content", "pos", "closed")]
        if method == "write" and not (
                type(args[0]) is bytes if handle.binary else type(args[0]) is str or z3.is_string(args[0])):
            raise Unsupported("program stream write argument is not builtin text/bytes")
        if method in {"seek", "read", "readline"} and any(
                not (type(a) in (int, bool) or z3.is_int(a) or a is None and method == "read") for a in args):
            raise Unsupported("program stream size/position is not a builtin integer")
        if method in {"close", "__exit__"}:
            runner.env[handle.key + "closed"] = True
            return None
        runner._holds(z3.Not(closed))
        if method == "__enter__":
            return handle
        if method == "flush":
            return None
        if not any(z3.is_expr(x) for x in [content, pos, *args]):
            if (type(content) not in (str, bytes) or type(pos) is not int
                    or any(type(a) not in (str, bytes, int, bool, type(None)) for a in args)):
                raise Unsupported("program stream: user protocols")
            try:
                world = StreamObject.restore(content, pos, binary=handle.binary, newline=handle.newline)
                updated, result = world.project(method, *args, origin=Origin("symexec", getattr(node, "lineno", 0)))
            except TypeError as exc:
                raise Unsupported(f"program stream argument: {exc}") from exc
            except (ValueError, OverflowError):
                raise _Dead()
            runner.env[handle.key + "content"] = updated.native().getvalue()
            runner.env[handle.key + "pos"] = updated.position
            return result
        if handle.binary or handle.newline != "\n":
            raise Unsupported("symbolic binary/non-default-newline stream")
        if method == "getvalue":
            return content
        if method == "tell":
            return pos
        if method == "seek":
            offset, whence = args[0], args[1] if len(args) > 1 else 0
            if type(whence) not in (int, bool) or whence not in (0, 1, 2):
                raise Unsupported("symbolic stream seek whence")
            if not (type(offset) in (int, bool) or z3.is_int(offset)):
                raise Unsupported("symbolic stream seek offset")
            if whence:
                runner._holds(offset == 0)
                pos = pos if whence == 1 else z3.Length(content)
            else:
                runner._holds(offset >= 0)
                pos = offset
            runner.env[handle.key + "pos"] = pos
            return pos
        if method == "write":
            value = args[0]
            if not (type(value) is str or z3.is_string(value)):
                raise Unsupported("symbolic stream write requires str")
            length = z3.Length(value)
            pad = z3.FreshConst(z3.StringSort(), "stream padding")
            runner._holds(z3.And(z3.Length(pad) == z3.If(pos > z3.Length(content), pos - z3.Length(content), 0),
                                 z3.InRe(pad, z3.Star(z3.Re("\0")))))
            text = z3.Concat(z3.SubString(content, 0, pos), pad, value,
                             z3.SubString(content, pos + length, z3.Length(content)))
            runner.env[handle.key + "content"] = z3.If(length == 0, content, text)
            runner.env[handle.key + "pos"] = pos + length
            return length
        size = args[0] if args else -1
        if size is None:
            size = -1
        if not (type(size) in (int, bool) or z3.is_int(size)):
            raise Unsupported("symbolic stream read size")
        end = z3.Length(content)
        if method == "readline":
            found = z3.IndexOf(content, "\n", pos)
            end = z3.If(found < 0, end, found + 1)
        end = z3.If(z3.And(size >= 0, pos + size < end), pos + size, end)
        count = z3.If(end > pos, end - pos, 0)
        runner.env[handle.key + "pos"] = pos + count
        return z3.SubString(content, pos, count)

    def print(self, runner, node):
        from ._symexec import Unsupported, _Lit
        # Evaluate in Python's order: positional arguments, then keyword arguments.
        args = [runner.ev(a) for a in node.args]
        kwargs = {k.arg: runner.ev(k.value) for k in node.keywords}
        handle = kwargs.get("file")
        if not isinstance(handle, StreamHandle) or handle.binary or set(kwargs) - {"file", "sep", "end", "flush"}:
            raise Unsupported("print: unmodelled stream destination/keywords")
        sep, end = kwargs.get("sep"), kwargs.get("end")
        sep, end = " " if sep is None else sep, "\n" if end is None else end
        if type(sep) is not str or type(end) is not str or type(kwargs.get("flush", False)) is not bool:
            raise Unsupported("print: symbolic separator/end/flush")
        text = []
        for value in args:
            if type(value) in (str, bytes, int, bool, float, type(None)):
                text.append(str(value))
            elif z3.is_string(value):
                text.append(value)
            elif z3.is_int(value):
                text.append(z3.If(value < 0, z3.Concat("-", z3.IntToStr(-value)), z3.IntToStr(value)))
            elif z3.is_bool(value):
                text.append(z3.If(value, "True", "False"))
            else:
                raise Unsupported("print to stream: user or symbolic formatting")
        # Native print issues separate writes; preserve the same content and cursor transitions.
        for i, value in enumerate(text):
            if i:
                self.call(runner, handle, "write", ast.Call(_Lit(None), [_Lit(sep)], []))
            self.call(runner, handle, "write", ast.Call(_Lit(None), [_Lit(value)], []))
        self.call(runner, handle, "write", ast.Call(_Lit(None), [_Lit(end)], []))
        return None
