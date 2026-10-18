import atheris, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with atheris.instrument_imports():
    import mixed_easy_target
def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    a = fdp.ConsumeUInt(4)
    b = fdp.ConsumeUInt(4)
    c = fdp.ConsumeBytes(8)
    mixed_easy_target.route(a, b, c)
if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput); atheris.Fuzz()
