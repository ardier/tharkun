"""Fixtures for field_resolver tests.

Covers:
- Simple annotated fields (class-level and __init__)
- Deeply nested chains (3–4 levels)
- Mixed annotation styles: AnnAssign, param hint, constructor call, literal
- Self-referential / Optional fields
- Unannotated fields (value-inferred types)
- Method guards on nested fields
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Imported in cross-file tests; also defined here for in-file tests.


# ---------------------------------------------------------------------------
# Level 1: simple classes
# ---------------------------------------------------------------------------


class Config:
    """Simple config object with annotated fields."""
    timeout: int
    retries: int
    host: str
    debug: bool

    def __init__(self, timeout: int = 30, retries: int = 3,
                 host: str = "localhost", debug: bool = False):
        self.timeout = timeout
        self.retries = retries
        self.host = host
        self.debug = debug


class Credentials:
    token: str
    expires: int
    active: bool

    def __init__(self, token: str = "", expires: int = 0, active: bool = True):
        self.token = token
        self.expires = expires
        self.active = active


# ---------------------------------------------------------------------------
# Level 2: classes that hold Level-1 objects
# ---------------------------------------------------------------------------


class Connection:
    """Connection that owns a Config (class-level annotation)."""
    config: Config
    creds: Credentials
    port: int

    def __init__(self, config: Config, creds: Credentials, port: int = 443):
        self.config = config
        self.creds = creds
        self.port = port


class ConnectionParamAnnotated:
    """Same nesting, but the field type comes from __init__ param annotation."""
    def __init__(self, config: Config, name: str = "default"):
        self.config = config
        self.name = name


class ConnectionConstructorRhs:
    """Field assigned via constructor call (type inferred from call, not annotation)."""
    def __init__(self):
        self.config = Config()          # type inferred as "Config" from rhs
        self.count = 0                  # type inferred as "int" from literal
        self.label = "default"          # type inferred as "str" from literal
        self.flags: list = []           # AnnAssign with builtin annotation


# ---------------------------------------------------------------------------
# Level 3: chains three deep
# ---------------------------------------------------------------------------


class Session:
    """Owns a Connection, which owns a Config."""
    connection: Connection
    session_id: str

    def __init__(self, connection: Connection, session_id: str = ""):
        self.connection = connection
        self.session_id = session_id


class DeepNested:
    """4-level chain: DeepNested.a.b.c.d."""
    class _D:
        value: int

    class _C:
        d: "_D"  # forward reference (string annotation)

    class _B:
        c: "_C"

    class _A:
        b: "_B"

    a: "_A"


# ---------------------------------------------------------------------------
# Guard functions — the guards that field_resolver should help solve
# ---------------------------------------------------------------------------


class Client:
    """Realistic client: guards on nested config fields."""

    def __init__(self, connection: Connection):
        self.connection = connection

    def send(self, data: bytes) -> bool:
        # chain: connection -> config -> timeout  (int)
        if self.connection.config.timeout > 60:
            return False
        # chain: connection -> creds -> active  (bool)
        if not self.connection.creds.active:
            return False
        # chain: connection -> config -> host (str)
        if self.connection.config.host == "":
            return False
        return True

    def connect(self) -> None:
        # chain: connection -> port  (int)
        if self.connection.port < 1024:
            raise PermissionError("privileged port")

    def authenticate(self, min_ttl: int) -> bool:
        # chain: connection -> creds -> expires  (int)
        if self.connection.creds.expires < min_ttl:
            return False
        return True


class Pipeline:
    """Multi-field guard covering multiple chain paths."""

    class Stage:
        enabled: bool
        max_items: int
        label: str

        def __init__(self, enabled: bool = True, max_items: int = 100,
                     label: str = ""):
            self.enabled = enabled
            self.max_items = max_items
            self.label = label

    def __init__(self, stage: "Pipeline.Stage"):
        self.stage = stage
        self.version: int = 1
        self.active: bool = True

    def run(self, items: list) -> list:
        # chain: stage -> enabled  (bool)
        if not self.stage.enabled:
            return []
        # chain: stage -> max_items  (int)
        if len(items) > self.stage.max_items:
            items = items[:self.stage.max_items]
        # own field
        if not self.active:
            return []
        return items


# ---------------------------------------------------------------------------
# Hard cases
# ---------------------------------------------------------------------------


class MixedAnnotations:
    """Fields annotated via different mechanisms — tests priority ordering."""

    # Class-level annotation (highest priority)
    class_level_int: int
    class_level_str: str

    def __init__(
        self,
        param_annotated_float: float,       # param hint
        unannotated_with_literal=42,        # literal default
        unannotated_no_default=None,        # no annotation, None default
    ):
        self.class_level_int = 0            # annotation from class body wins
        self.class_level_str = ""
        self.param_annotated_float = param_annotated_float
        self.unannotated_with_literal = unannotated_with_literal
        self.unannotated_no_default = unannotated_no_default
        self.init_annassign: bytes = b""    # AnnAssign in __init__ body
        self.constructed = Config()         # constructor call on rhs


class CyclicHolder:
    """Self-referential class — tests cycle detection."""
    next: Optional["CyclicHolder"]
    value: int

    def __init__(self, value: int = 0, next: Optional["CyclicHolder"] = None):
        self.value = value
        self.next = next


class WrapperChain:
    """Tests a chain that includes Optional and typing generics."""
    inner: Optional[Config]
    items: list
    mapping: dict

    def __init__(self):
        self.inner = Config()
        self.items = []
        self.mapping = {}


class ForwardRefHolder:
    """Fields annotated as forward references (string literals)."""
    # These are string annotations — ast.unparse gives "'Config'" not "Config"
    config: "Config"
    connection: "Connection"

    def __init__(self):
        self.config = Config()
        self.connection = Connection(Config(), Credentials())


class GuardOnOptionalField:
    """Guard is on an Optional[Config] field — solver must handle Optional."""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg

    def is_slow(self) -> bool:
        # chain: cfg -> timeout  (Optional[Config], but still int underneath)
        if self.cfg is not None and self.cfg.timeout > 100:
            return True
        return False


@dataclass
class DataclassConfig:
    """Dataclass — annotations in class body, no explicit __init__."""
    host: str = "localhost"
    port: int = 8080
    tls: bool = False


class DataclassClient:
    """Client wrapping a dataclass config."""
    cfg: DataclassConfig

    def __init__(self, cfg: DataclassConfig):
        self.cfg = cfg

    def is_secure(self) -> bool:
        if self.cfg.tls and self.cfg.port == 443:
            return True
        return False
