"""Conservative Python-regex to Z3 membership for SV string draws.

Only the regular subset below is translated. Unsupported syntax leaves the symbolic
site open; every proposed model is replayed by the SV engine.
"""

import re
from re import _constants as c
from re import _parser as parser

import z3


def _regex(items):
    out = []
    for op, arg in items:
        if op is c.LITERAL:
            out.append(z3.Re(chr(arg)))
        elif op is c.ANY:
            out.append(z3.Union(z3.Range("\x00", "\x09"), z3.Range("\x0b", "\U0010ffff")))
        elif op is c.CATEGORY:
            if arg is c.CATEGORY_DIGIT:
                out.append(z3.Range("0", "9"))
            elif arg is c.CATEGORY_WORD:
                out.append(z3.Union(z3.Range("a", "z"), z3.Range("A", "Z"),
                                    z3.Range("0", "9"), z3.Re("_")))
            elif arg is c.CATEGORY_SPACE:
                out.append(z3.Union(*[z3.Re(x) for x in " \t\n\r\f\v"]))
            else:
                raise ValueError("regex category")
        elif op is c.IN:
            chars = []
            for kind, val in arg:
                if kind is c.LITERAL:
                    chars.append(z3.Re(chr(val)))
                elif kind is c.RANGE:
                    chars.append(z3.Range(chr(val[0]), chr(val[1])))
                elif kind is c.CATEGORY:
                    chars.append(_regex([(kind, val)]))
                else:
                    raise ValueError("regex class")
            if not chars:
                raise ValueError("empty regex class")
            out.append(chars[0] if len(chars) == 1 else z3.Union(*chars))
        elif op is c.SUBPATTERN:
            if arg[1] or arg[2]:
                raise ValueError("regex flags")
            out.append(_regex(arg[-1]))
        elif op is c.BRANCH:
            branches = [_regex(branch) for branch in arg[1]]
            out.append(branches[0] if len(branches) == 1 else z3.Union(*branches))
        elif op in (c.MAX_REPEAT, c.MIN_REPEAT):
            lo, hi, body = arg
            part = _regex(body)
            if hi == c.MAXREPEAT:
                parts = [part] * lo + [z3.Star(part)]
                out.append(parts[0] if len(parts) == 1 else z3.Concat(*parts))
            elif hi <= 64:
                out.append(z3.Loop(part, lo, hi))
            else:
                raise ValueError("regex repeat")
        else:
            raise ValueError(f"regex token {op}")
    return z3.Concat(*out) if len(out) > 1 else out[0] if out else z3.Re("")


def regex_constraint(value, pattern: str, mode: str):
    """Positive re.match/fullmatch/search constraint, or Unsupported for a regex gap."""
    from tharkun.sv._symexec import Unsupported

    try:
        parsed_pattern = parser.parse(pattern)
        if parsed_pattern.state.flags & (re.IGNORECASE | re.MULTILINE | re.DOTALL | re.ASCII | re.VERBOSE):
            raise ValueError("regex flags")
        parsed = list(parsed_pattern)
        start = parsed and parsed[0] == (c.AT, c.AT_BEGINNING)
        end = parsed and parsed[-1] == (c.AT, c.AT_END)
        if start:
            parsed.pop(0)
        if end:
            parsed.pop()
        body = _regex(parsed)
    except (ValueError, re.error) as exc:
        raise Unsupported(f"regex {exc}") from None
    any_string = z3.Star(z3.AllChar(z3.ReSort(z3.StringSort())))
    prefix = [] if mode in ("match", "fullmatch") or start else [any_string]
    suffix = [] if mode == "fullmatch" or end else [any_string]
    parts = prefix + [body] + suffix
    return z3.InRe(value, parts[0] if len(parts) == 1 else z3.Concat(*parts))
