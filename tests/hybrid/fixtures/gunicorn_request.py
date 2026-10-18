r"""A faithful gunicorn HTTP request parse, reduced to its regex blockers.

gunicorn/http/message.py validates the request method and every header field-name against
the RFC9110 token grammar and the HTTP version against ``HTTP/(\d)\.(\d)``. Reaching the
accepted body past all three from a random fuzzer field is exactly what libFuzzer stalls
on: a token is sparse in the space of arbitrary bytes, and value-profile cannot copy a
"matches this regex" answer out of the comparison.

The patterns are the byte form of gunicorn's grammar, pinned to the fuzzer field's fixed
width (8) so a fullmatch is a well-posed request the codec can recode byte-identically via
``ConsumeBytes`` (no mode byte, unlike the unicode fields). ``VERSION_RE`` is a literal;
``TOKEN8_RE`` is a COMPUTED pattern (``re.compile`` over a %-format of an ``re.escape``'d
constant) — the AST-opaque shape the solver must read off the compiled object — and a
*bytes* pattern, the branch the regex inverter grew to handle.

Written as nested positive guards (not ``if not X: raise``) so each accepted path is a
branch *body* the coverage frontier finder can see; each field maps DIRECTLY to the guarded
value, so the regex solution recodes without inverting any derived local.
"""
import re

RFC9110_5_6_2_TOKEN_SPECIALS = rb"!#$%&'*+-.^_`|~"
TOKEN8_RE = re.compile(rb"[%s0-9A-Za-z]{8}" % re.escape(RFC9110_5_6_2_TOKEN_SPECIALS))
VERSION_RE = re.compile(rb"HTTP/(\d)\.(\d)")


def parse_request(method, name, version):
    accepted = []
    if TOKEN8_RE.fullmatch(method):          # blocker 1: method is an 8-char token
        accepted.append(method)
        if VERSION_RE.fullmatch(version):    # blocker 2: version is HTTP/d.d
            accepted.append(version)
            if TOKEN8_RE.fullmatch(name):    # blocker 3: header name is an 8-char token
                accepted.append(name)
                return (b"request", method, version, name)   # the accepted body
    return accepted
