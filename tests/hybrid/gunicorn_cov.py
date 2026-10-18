"""Measure real-gunicorn line coverage of the FUZZABLE parser surface, per harness.

OSS-Fuzz's gunicorn job ships a single ``fuzz_util.py`` harness, so its coverage report
only ever sees ``gunicorn/util.py`` (+ a few tiny modules) — the HTTP/ASGI request parsers,
which are the security-critical surface, sit at 0% because nothing imports them. This tool
drives those parsers directly and reports coverage.py *statement* coverage per module, in
the same units the OSS-Fuzz report uses, so the numbers are directly comparable.

Standalone (fresh process, so module-level statements are counted once):

    python tests/hybrid/gunicorn_cov.py <suite>      # suite: junk | util | http | asgi | all

Prints a JSON blob {module: [covered, statements]} plus a human table on stderr. The test
``test_gunicorn_real_coverage.py`` invokes it per-suite in subprocesses and compares.

Gunicorn-gated: exits 2 with a message if gunicorn is not importable.
"""
import json
import os
import sys

# Coverage MUST start before gunicorn is imported, or its module-level statements (already
# executed at import) are never counted.
import coverage

COV = coverage.Coverage(source=["gunicorn"], branch=False)
COV.start()

try:
    from gunicorn.config import Config
    from gunicorn.http.parser import RequestParser
    from gunicorn.http.wsgi import create as wsgi_create
    from gunicorn import util
    try:
        from gunicorn.asgi.parser import PythonProtocol
    except Exception:
        PythonProtocol = None
    try:
        from gunicorn.http2.connection import HTTP2ServerConnection  # needs the h2 library
    except Exception:
        HTTP2ServerConnection = None
except Exception as exc:  # gunicorn not installed / import failed
    COV.stop()
    sys.stderr.write(f"gunicorn unavailable: {exc}\n")
    sys.exit(2)

CFG = Config()
ADDR = ("127.0.0.1", 0)
_DEVNULL = open(os.devnull, "w")

# Modules that make up the fuzzable parser surface (the rest of gunicorn is server/worker/
# socket/config code no data-in fuzzer can reach).
PARSER_FILES = (
    "util.py", "http/message.py", "http/body.py", "http/unreader.py",
    "http/parser.py", "http/wsgi.py", "asgi/parser.py", "asgi/unreader.py",
)

# ---- harnesses -------------------------------------------------------------


def run_util(data: bytes):
    # The OSS-Fuzz fuzz_util.py call set (string-input util functions).
    s = data.decode("latin-1")
    for call in (
        lambda: util.parse_address(s), lambda: util.is_ipv6(s),
        lambda: util.split_request_uri(s), lambda: util.to_bytestring(s),
        lambda: util.http_date(len(s)),
    ):
        try:
            call()
        except Exception:
            pass


def run_request(data: bytes):
    try:
        parser = RequestParser(CFG, [data], ADDR)
        for req in parser:
            try:
                req.body.read(65536)
            except Exception:
                pass
            try:
                wsgi_create(req, ("sock",), ADDR, ("127.0.0.1", 0), CFG)
            except Exception:
                pass
            break
    except Exception:
        pass


def run_asgi(data: bytes):
    if PythonProtocol is None:
        return
    noop = lambda *a, **k: None
    for proxy in ("off", "v1", "v2"):
        try:
            p = PythonProtocol(on_message_begin=noop, on_url=noop, on_header=noop,
                               on_headers_complete=lambda: False, on_body=noop,
                               on_message_complete=noop, proxy_protocol=proxy)
            p.feed(data)
        except Exception:
            pass


_CFG_KEYS = list(Config().settings.keys())


def run_config(data: bytes):
    # Exercise every setting's validator (validate_bool/pos_int/addr_list/... — the parsing
    # code in config.py) with several fuzzed value shapes.
    s = data.decode("latin-1")
    cfg = Config()
    for key in _CFG_KEYS:
        for v in (s, len(s), [s], s.split(",")):
            try:
                cfg.set(key, v)
            except Exception:
                pass


class _FakeSock:
    def sendall(self, d): pass
    def send(self, d): return len(d)
    def recv(self, n): return b""
    def close(self): pass


_H2_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"


def run_http2(data: bytes):
    if HTTP2ServerConnection is None:
        return
    try:
        conn = HTTP2ServerConnection(Config(), _FakeSock(), ADDR)
        conn.initiate_connection()
        conn.receive_data(_H2_PREFACE + data)
    except Exception:
        pass


# ---- corpora ---------------------------------------------------------------

JUNK = [b"", b"\x00\x00", b"garbage", b"GET", b"GET /\r\n", b"XXX / HTTP/9.9\r\n\r\n"]

REQUESTS = [
    b"GET / HTTP/1.1\r\nHost: x\r\n\r\n",
    b"POST /p HTTP/1.1\r\nContent-Length: 5\r\nHost: y\r\n\r\nhello",
    b"POST /c HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n",
    b"GET /a?b=c HTTP/1.0\r\nConnection: keep-alive\r\nX-H: 1\r\n\r\n",
    b"HEAD / HTTP/1.1\r\nHost: h\r\nExpect: 100-continue\r\n\r\n",
    b"DELETE /r HTTP/1.1\r\nHost: h\r\nContent-Length: 0\r\n\r\n",
    b"GET / HTTP/1.1\r\nHost: h\r\nCookie: a=b; c=d\r\nAccept: */*\r\n\r\n",
    b"PUT /x HTTP/1.1\r\nHost: h\r\nContent-Length: 3\r\n\r\nabc",
    b"OPTIONS * HTTP/1.1\r\nHost: h\r\n\r\n",
    b"GET http://h/abs HTTP/1.1\r\nHost: h\r\n\r\n",
    b"POST /c HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\nTrailer: v\r\n\r\n",
    b"GET / HTTP/1.1\r\n\r\n",
    b"GET / HTTP/1.1\r\nHost: h\r\nContent-Length: -1\r\n\r\n",
    b"GET / HTTP/1.1\r\nHost: h\r\nContent-Length: abc\r\n\r\n",
    b"POST / HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: chunked\r\nContent-Length: 5\r\n\r\n0\r\n\r\n",
    b"POST / HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: chunked\r\n\r\nA\r\n0123456789\r\n0\r\n\r\n",
    b"POST / HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: chunked\r\n\r\nZZ\r\nx\r\n",
    b"GET / HTTP/1.1\r\nHost: h\r\n" + b"X-Pad: " + b"z" * 9000 + b"\r\n\r\n",
    b"GET " + b"/" + b"a" * 9000 + b" HTTP/1.1\r\nHost: h\r\n\r\n",
    b"CONNECT h:443 HTTP/1.1\r\nHost: h\r\n\r\n",
    b"GET / HTTP/1.1\r\nHost: h\r\nExpect: 100-continue\r\nContent-Length: 3\r\n\r\nabc",
] + [(b"GET /%d HTTP/1.1\r\nHost: h%d\r\nX-N: %d\r\n\r\n" % (i, i, i)) for i in range(40)]

# tharkun's hybrid loop solves these sparse util guards random mutation never lands.
UTIL_SOLVED = [b"fd://0", b"tcp://host:9", b"unix://sock", b"unix:/s", b"[::1]:80", b""]

PROXY = [
    b"PROXY TCP4 1.2.3.4 5.6.7.8 1234 80\r\nGET / HTTP/1.1\r\nHost: h\r\n\r\n",
    b"PROXY TCP6 ::1 ::2 1 2\r\nGET / HTTP/1.1\r\nHost: h\r\n\r\n",
    b"PROXY UNKNOWN\r\nGET / HTTP/1.1\r\nHost: h\r\n\r\n",
    b"\r\n\r\n\x00\r\nQUIT\n" + b"\x21\x11\x00\x0c" + b"\x7f\x00\x00\x01" * 2 + b"\x00\x50\x00\x50",
]

CONFIG_VALS = [b"", b"1", b"-1", b"abc", b"127.0.0.1:80", b"a,b,c", b"0.0.0.0",
               b"1.5", b"true", b"unix:/x", b"fd://3", b"1048576", b"info",
               b"TLSv1_2", b"sync", b"*", b"[::1]:8000"]
H2_FRAMES = [
    b"", b"\x00\x00\x00\x04\x00\x00\x00\x00\x00",                        # empty SETTINGS
    b"\x00\x00\x08\x06\x00\x00\x00\x00\x00" + b"12345678",               # PING
    b"\x00\x00\x04\x08\x00\x00\x00\x00\x01\x00\x00\x00\x05",             # WINDOW_UPDATE
    b"\x00\x00\x05\x01\x05\x00\x00\x00\x01\x82\x84\x86\x41\x00",         # HEADERS-ish
    b"\xff" * 32,
]

SUITES = {
    "junk": [(run_request, JUNK), (run_util, JUNK), (run_asgi, JUNK)],
    "util": [(run_util, [s.encode() if isinstance(s, str) else s
                         for s in ["x", "host:80", "[::1]:9"]] + UTIL_SOLVED)],
    "http": [(run_request, JUNK + REQUESTS)],
    "asgi": [(run_asgi, JUNK + REQUESTS + PROXY)],
    "config": [(run_config, CONFIG_VALS)],
    "http2": [(run_http2, H2_FRAMES)],
    "all": [(run_request, JUNK + REQUESTS), (run_asgi, JUNK + REQUESTS + PROXY),
            (run_util, JUNK + UTIL_SOLVED), (run_config, CONFIG_VALS),
            (run_http2, H2_FRAMES)],
}


def main() -> None:
    suite = sys.argv[1] if len(sys.argv) > 1 else "all"
    _so, _se = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = _DEVNULL
    for fn, corpus in SUITES[suite]:
        for d in corpus:
            fn(d)
    sys.stdout, sys.stderr = _so, _se

    COV.stop()
    out = {}
    tc = tl = 0                       # parser-surface subtotal (the clean fuzzing metric)
    wc = wl = 0                       # whole-package total (every gunicorn/*.py)
    extra = {}                        # config / http2 modules, reported separately
    for f in COV.get_data().measured_files():
        if "/gunicorn/" not in f:
            continue
        rel = f.split("/gunicorn/")[-1]
        _, stmts, _, missing, _ = COV.analysis2(f)
        c = len(stmts) - len(missing)
        wc += c
        wl += len(stmts)
        if rel in PARSER_FILES:
            out[rel] = [c, len(stmts)]
            tc += c
            tl += len(stmts)
        elif rel == "config.py" or rel.startswith("http2/"):
            extra[rel] = [c, len(stmts)]
    out["_total"] = [tc, tl]
    out["_whole_package"] = [wc, wl]
    out.update(extra)
    sys.stderr.write(f"=== suite={suite} — gunicorn coverage (coverage.py statements) ===\n")
    for rel in sorted(k for k in out if not k.startswith("_")):
        c, l = out[rel]
        sys.stderr.write(f"  gunicorn/{rel:20s} {c:4d}/{l:4d}  {round(100*c/max(l,1)):3d}%\n")
    sys.stderr.write(f"  parser surface (http+asgi+util)  {tc}/{tl} = {100*tc/max(tl,1):.1f}%\n")
    sys.stderr.write(f"  WHOLE PACKAGE                    {wc}/{wl} = {100*wc/max(wl,1):.1f}%\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
