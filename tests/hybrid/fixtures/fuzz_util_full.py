# Faithful port of google/oss-fuzz projects/gunicorn/fuzz_util.py — the exact 7-function call
# set. parse_address sits at field 3 (behind is_ipv6 / warn / split_request_uri), so recoding
# a solved netloc into it is the mid-buffer variable-length case that needs the execution-
# verified recode (a "fd://" solution reads back with trailing bytes but still satisfies
# startswith). Requires gunicorn installed.
import atheris, sys
with atheris.instrument_imports():
    from gunicorn import util

def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    try: util.is_ipv6(fdp.ConsumeString(100))
    except Exception: pass
    try: util.warn(fdp.ConsumeString(100))
    except Exception: pass
    try: util.split_request_uri(fdp.ConsumeString(100))
    except Exception: pass
    try: util.parse_address(fdp.ConsumeString(100))
    except Exception: pass
    try: util.http_date(fdp.ConsumeInt(50))
    except Exception: pass
    try: util.to_bytestring(fdp.ConsumeString(100))
    except Exception: pass
    try: util.import_app(fdp.ConsumeString(100))
    except Exception: pass

if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
