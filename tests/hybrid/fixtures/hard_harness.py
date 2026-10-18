import atheris, sys, os
sys.path.insert(0, os.path.dirname(__file__))
with atheris.instrument_imports():
    import hard_target
def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    x = fdp.ConsumeUInt(4)
    y = fdp.ConsumeUInt(4)
    hard_target.hard(x, y)
if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
