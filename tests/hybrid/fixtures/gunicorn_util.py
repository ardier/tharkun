# Real-world OSS-Fuzz target: gunicorn util.parse_address (verbatim). Its string-prefix
# guards (fd://, tcp://, unix:) are reached through a whole-buffer FDP unicode field —
# the byte->string packing defeats libFuzzer's value-profile, so plain atheris stalls
# and the concolic assist (solve + variable-width recode) is what unblocks them.
import re


def parse_address(netloc, default_port='8000'):
    if re.match(r'unix:(//)?', netloc):
        return re.split(r'unix:(//)?', netloc)[-1]

    if netloc.startswith("fd://"):
        fd = netloc[5:]
        try:
            return int(fd)
        except ValueError:
            raise RuntimeError("%r is not a valid file descriptor." % fd)

    if netloc.startswith("tcp://"):
        netloc = netloc.split("tcp://")[1]
    host, port = netloc, default_port

    if '[' in netloc and ']' in netloc:
        host = netloc.split(']')[0][1:]
        port = (netloc.split(']:') + [default_port])[1]
    elif ':' in netloc:
        host, port = (netloc.split(':') + [default_port])[:2]
    elif netloc == "":
        host, port = "0.0.0.0", default_port

    try:
        port = int(port)
    except ValueError:
        raise RuntimeError("%r is not a valid port number." % port)

    return host.lower(), port
