import atheris, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with atheris.instrument_imports():
    import mixed_hard_target
def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    magic = fdp.ConsumeUInt(4)
    ver = fdp.ConsumeUInt(4)
    tag = fdp.ConsumeUInt(4)
    mod = fdp.ConsumeUInt(4)
    name = fdp.ConsumeUnicodeNoSurrogates(64)
    mixed_hard_target.parse(magic, ver, tag, mod, name)
if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput); atheris.Fuzz()
