import atheris, sys, os
sys.path.insert(0, os.path.dirname(__file__))
with atheris.instrument_imports():
    import gunicorn_util
def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    netloc = fdp.ConsumeUnicodeNoSurrogates(sys.maxsize)   # whole buffer as a string
    try:
        gunicorn_util.parse_address(netloc)
    except (RuntimeError, ValueError, IndexError):
        pass
if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
