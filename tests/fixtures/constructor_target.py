"""Fixture for constructor tests — classes with various __init__ signatures."""


class SimpleClass:
    def __init__(self):
        self.value = 0


class WithArgs:
    def __init__(self, name: str, count: int = 5):
        self.name = name
        self.count = count


class WithDefaults:
    def __init__(self, x: int = 10, y: float = 3.14, label: str = "default"):
        self.x = x
        self.y = y
        self.label = label


class WithOptional:
    def __init__(self, data: list = None, meta: dict = None):
        self.data = data or []
        self.meta = meta or {}


class NoInit:
    pass


class SignatureKinds:
    def __init__(
        self,
        token: str,
        /,
        count: int = 2,
        *labels: str,
        enabled: bool = True,
        metadata: dict[str, int] = {"answer": 42},
        **extras: object,
    ):
        self.token = token
        self.count = count
        self.labels = labels
        self.enabled = enabled
        self.metadata = metadata
        self.extras = extras


DEFAULT_VALUE = object()


class WithNonLiteralDefault:
    def __init__(self, value: int = DEFAULT_VALUE):
        self.value = value
