import atheris, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with atheris.instrument_imports():
    import toy_target

def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    op = fdp.ConsumeUInt(4)
    x = fdp.ConsumeUInt(4)
    y = fdp.ConsumeUInt(4)
    z = fdp.ConsumeUInt(4)
    toy_target.dispatch(op, x, y, z)

if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
