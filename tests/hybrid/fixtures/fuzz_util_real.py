# The OSS-Fuzz gunicorn fuzz_util.py, reduced to parse_address (the util function with the
# sparse fd:// / tcp:// / unix: prefix guards). A real atheris harness so the hybrid loop can
# drive the REAL gunicorn.util.parse_address and recode the solved prefix back into the
# ConsumeString field. Requires gunicorn installed.
import atheris, sys
with atheris.instrument_imports():
    from gunicorn import util

def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    try:
        util.parse_address(fdp.ConsumeString(100))
    except Exception:
        pass

if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
