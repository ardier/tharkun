import atheris, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with atheris.instrument_imports():
    import gunicorn_request

def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    method = fdp.ConsumeBytes(8)     # field 0 -> parse_request.method
    name = fdp.ConsumeBytes(8)       # field 1 -> parse_request.name
    version = fdp.ConsumeBytes(8)    # field 2 -> parse_request.version
    gunicorn_request.parse_request(method, name, version)

if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
