import atheris, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with atheris.instrument_imports():
    import layers_target
def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    a = fdp.ConsumeUInt(4); b = fdp.ConsumeUInt(4); c = fdp.ConsumeUInt(4)
    layers_target.top(a, b, c)
if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput); atheris.Fuzz()
