# Faithful transcription of gunicorn's HTTP request-line / header parsing guards
# (gunicorn/http/message.py). The regexes are verbatim: VERSION_RE and the RFC9110
# TOKEN_RE (a COMPUTED pattern — re.compile over a %-format of an re.escape'd constant).
# These guards gate request-line and header parsing; the upstream OSS-Fuzz harness
# (fuzz_util.py -> util functions) never calls this module, so they sit at 0% coverage.
import re

RFC9110_5_6_2_TOKEN_SPECIALS = r"!#$%&'*+-.^_`|~"
TOKEN_RE = re.compile(r"[%s0-9a-zA-Z]+" % (re.escape(RFC9110_5_6_2_TOKEN_SPECIALS)))
VERSION_RE = re.compile(r"HTTP/(\d)\.(\d)")


def parse_request_method(method):
    # gunicorn: `if not TOKEN_RE.fullmatch(self.method): raise InvalidRequestMethod`
    if not TOKEN_RE.fullmatch(method):
        raise ValueError("invalid method")
    return method


def parse_header_name(name):
    # gunicorn: header field-name must be a token
    if not TOKEN_RE.fullmatch(name):
        raise ValueError("invalid header name")
    return name


def parse_http_version(version):
    # gunicorn: `if VERSION_RE.match(...) is None: raise InvalidHTTPVersion`
    if VERSION_RE.fullmatch(version):
        return version
    raise ValueError("invalid HTTP version")
