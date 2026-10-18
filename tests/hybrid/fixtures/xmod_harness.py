import atheris, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with atheris.instrument_imports():
    from xpkg import mod_a, mod_b, mod_c

def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    mod_a.check_a(fdp.ConsumeUInt(4))                       # field 0 -> check_a.code
    mod_c.check_c(fdp.ConsumeUInt(4), fdp.ConsumeUInt(4))   # fields 1,2 -> check_c.n, m
    mod_b.check_b(fdp.ConsumeUnicodeNoSurrogates(32))       # field 3 -> check_b.name

if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput); atheris.Fuzz()
