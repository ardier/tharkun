import atheris, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import deep_target
def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    magic = fdp.ConsumeInt(4)
    kind = fdp.ConsumeIntInRange(0, 20)
    size = fdp.ConsumeInt(4)
    deep_target.parse(magic, kind, size)
if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
