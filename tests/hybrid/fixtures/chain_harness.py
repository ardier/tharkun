import atheris, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with atheris.instrument_imports():
    from chainpkg import chain


def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    try:
        chain.dispatch(fdp.ConsumeInt(4), fdp.ConsumeBytes(124))
    except Exception:
        pass
