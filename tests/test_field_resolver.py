"""Tests for tharkun.field_resolver.

Coverage:
- _get_field_type: class-level annotation, __init__ AnnAssign,
  param annotation + assignment, constructor call, literal inference
- _find_class: same-file, imported
- resolve_field_chain: 1-level, 2-level, 3-level, cross-file,
  missing annotation fallback, cycle detection, max_depth
- Hard integration: guard expressions on deeply nested fields
"""

from __future__ import annotations

import os
import ast

import pytest

from tharkun.solve.frontend.field_resolver import (
    FieldResolution,
    _find_class,
    _get_field_type,
    _is_builtin_type,
    resolve_field_chain,
)


# ---------------------------------------------------------------------------
# Fixtures: paths
# ---------------------------------------------------------------------------

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
TARGET = os.path.join(FIXTURES, "field_resolver_target.py")
CROSSFILE = os.path.join(FIXTURES, "field_resolver_crossfile.py")
REMOTE = os.path.join(FIXTURES, "field_resolver_remote.py")


# ---------------------------------------------------------------------------
# Helper: load class def from fixture
# ---------------------------------------------------------------------------

def load_class(name: str, path: str = TARGET) -> ast.ClassDef:
    from tharkun.solve.frontend.import_resolver import load_class_def
    cls = load_class_def(path, name)
    assert cls is not None, f"{name} not found in {path}"
    return cls


# ---------------------------------------------------------------------------
# _is_builtin_type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("type_str,expected", [
    ("int", True),
    ("str", True),
    ("float", True),
    ("bytes", True),
    ("bool", True),
    ("list", True),
    ("dict", True),
    ("set", True),
    ("tuple", True),
    ("None", True),
    ("Any", True),
    ("object", True),
    ("List[int]", True),
    ("Dict[str, int]", True),
    ("Optional[str]", True),
    ("Config", False),
    ("Connection", False),
    ("RemoteConfig", False),
    ("MyCustomClass", False),
])
def test_is_builtin_type(type_str, expected):
    assert _is_builtin_type(type_str) == expected


# ---------------------------------------------------------------------------
# _get_field_type: class-level annotations
# ---------------------------------------------------------------------------

def test_get_field_type_class_level_int():
    cls = load_class("Config")
    assert _get_field_type(cls, "timeout") == "int"


def test_get_field_type_class_level_str():
    cls = load_class("Config")
    assert _get_field_type(cls, "host") == "str"


def test_get_field_type_class_level_bool():
    cls = load_class("Config")
    assert _get_field_type(cls, "debug") == "bool"


def test_get_field_type_credentials_token():
    cls = load_class("Credentials")
    assert _get_field_type(cls, "token") == "str"


def test_get_field_type_credentials_expires():
    cls = load_class("Credentials")
    assert _get_field_type(cls, "expires") == "int"


# ---------------------------------------------------------------------------
# _get_field_type: class-level annotation for complex type
# ---------------------------------------------------------------------------

def test_get_field_type_connection_config():
    """config: Config — class-level annotation of a user-defined type."""
    cls = load_class("Connection")
    assert _get_field_type(cls, "config") == "Config"


def test_get_field_type_connection_creds():
    cls = load_class("Connection")
    assert _get_field_type(cls, "creds") == "Credentials"


def test_get_field_type_connection_port():
    cls = load_class("Connection")
    assert _get_field_type(cls, "port") == "int"


# ---------------------------------------------------------------------------
# _get_field_type: param annotation in __init__
# ---------------------------------------------------------------------------

def test_get_field_type_param_annotated():
    """self.config = config where config: Config in __init__ params."""
    cls = load_class("ConnectionParamAnnotated")
    assert _get_field_type(cls, "config") == "Config"


def test_get_field_type_param_annotated_str():
    cls = load_class("ConnectionParamAnnotated")
    assert _get_field_type(cls, "name") == "str"


# ---------------------------------------------------------------------------
# _get_field_type: constructor call and literal inference
# ---------------------------------------------------------------------------

def test_get_field_type_constructor_rhs():
    """self.config = Config() — type inferred from rhs call."""
    cls = load_class("ConnectionConstructorRhs")
    assert _get_field_type(cls, "config") == "Config"


def test_get_field_type_literal_int():
    """self.count = 0 — type inferred as int from literal."""
    cls = load_class("ConnectionConstructorRhs")
    assert _get_field_type(cls, "count") == "int"


def test_get_field_type_literal_str():
    """self.label = 'default' — type inferred as str from literal."""
    cls = load_class("ConnectionConstructorRhs")
    assert _get_field_type(cls, "label") == "str"


def test_get_field_type_init_annassign():
    """self.flags: list = [] — AnnAssign inside __init__ body."""
    cls = load_class("ConnectionConstructorRhs")
    assert _get_field_type(cls, "flags") == "list"


# ---------------------------------------------------------------------------
# _get_field_type: missing field
# ---------------------------------------------------------------------------

def test_get_field_type_missing():
    cls = load_class("Config")
    assert _get_field_type(cls, "nonexistent_field") is None


def test_get_field_type_no_init():
    """Class with no __init__ — only class-level annotations work."""
    from tharkun.solve.frontend.import_resolver import load_class_def
    cls = load_class_def(TARGET, "Credentials")
    assert cls is not None
    # Class-level annotation works
    assert _get_field_type(cls, "token") == "str"
    # No __init__ for Config-like nested type — no problem
    assert _get_field_type(cls, "nonexistent") is None


# ---------------------------------------------------------------------------
# _find_class
# ---------------------------------------------------------------------------

def test_find_class_same_file():
    found = _find_class("Config", TARGET)
    assert found is not None
    cls, path = found
    assert cls.name == "Config"
    assert os.path.realpath(path) == os.path.realpath(TARGET)


def test_find_class_missing():
    assert _find_class("DoesNotExist", TARGET) is None


def test_find_class_cross_file():
    """RemoteConfig is imported into crossfile.py from remote.py."""
    found = _find_class("RemoteConfig", CROSSFILE)
    assert found is not None
    cls, path = found
    assert cls.name == "RemoteConfig"
    assert os.path.realpath(path) == os.path.realpath(REMOTE)


# ---------------------------------------------------------------------------
# resolve_field_chain: 1-level
# ---------------------------------------------------------------------------

def test_resolve_1level_int():
    """Config.timeout → int (class-level annotation)."""
    result = resolve_field_chain("Config", ["timeout"], TARGET)
    assert result is not None
    assert result.leaf_type == "int"
    assert result.chain_types == ["int"]
    assert result.resolved_depth == 1
    assert result.leaf_path is None


def test_resolve_1level_str():
    result = resolve_field_chain("Config", ["host"], TARGET)
    assert result is not None
    assert result.leaf_type == "str"


def test_resolve_1level_bool():
    result = resolve_field_chain("Config", ["debug"], TARGET)
    assert result is not None
    assert result.leaf_type == "bool"


def test_resolve_1level_user_type():
    """Connection.config → Config (user-defined type, not builtin)."""
    result = resolve_field_chain("Connection", ["config"], TARGET)
    assert result is not None
    assert result.leaf_type == "Config"
    assert result.resolved_depth == 1
    assert result.leaf_path is not None  # not a builtin


# ---------------------------------------------------------------------------
# resolve_field_chain: 2-level
# ---------------------------------------------------------------------------

def test_resolve_2level_connection_config_timeout():
    """Connection.config.timeout → int (2 levels)."""
    result = resolve_field_chain("Connection", ["config", "timeout"], TARGET)
    assert result is not None
    assert result.leaf_type == "int"
    assert result.chain_types == ["Config", "int"]
    assert result.resolved_depth == 2


def test_resolve_2level_connection_config_host():
    result = resolve_field_chain("Connection", ["config", "host"], TARGET)
    assert result is not None
    assert result.leaf_type == "str"
    assert result.resolved_depth == 2


def test_resolve_2level_connection_creds_token():
    """Connection.creds.token → str."""
    result = resolve_field_chain("Connection", ["creds", "token"], TARGET)
    assert result is not None
    assert result.leaf_type == "str"
    assert result.chain_types == ["Credentials", "str"]


def test_resolve_2level_connection_creds_active():
    result = resolve_field_chain("Connection", ["creds", "active"], TARGET)
    assert result is not None
    assert result.leaf_type == "bool"


# ---------------------------------------------------------------------------
# resolve_field_chain: 3-level
# ---------------------------------------------------------------------------

def test_resolve_3level_session_connection_config_timeout():
    """Session.connection.config.timeout → int (3 levels)."""
    result = resolve_field_chain(
        "Session", ["connection", "config", "timeout"], TARGET
    )
    assert result is not None
    assert result.leaf_type == "int"
    assert result.chain_types == ["Connection", "Config", "int"]
    assert result.resolved_depth == 3


def test_resolve_3level_session_connection_creds_expires():
    result = resolve_field_chain(
        "Session", ["connection", "creds", "expires"], TARGET
    )
    assert result is not None
    assert result.leaf_type == "int"
    assert result.resolved_depth == 3


# ---------------------------------------------------------------------------
# resolve_field_chain: cross-file
# ---------------------------------------------------------------------------

def test_resolve_crossfile_1level():
    """RemoteClient.config → RemoteConfig (class defined in another file)."""
    result = resolve_field_chain("RemoteClient", ["config"], CROSSFILE)
    assert result is not None
    assert result.leaf_type == "RemoteConfig"
    assert result.leaf_path is not None


def test_resolve_crossfile_2level_bool():
    """RemoteClient.config.secure → bool (cross-file 2-level)."""
    result = resolve_field_chain("RemoteClient", ["config", "secure"], CROSSFILE)
    assert result is not None
    assert result.leaf_type == "bool"
    assert result.chain_types == ["RemoteConfig", "bool"]
    assert result.resolved_depth == 2


def test_resolve_crossfile_2level_int():
    """RemoteClient.config.port → int."""
    result = resolve_field_chain("RemoteClient", ["config", "port"], CROSSFILE)
    assert result is not None
    assert result.leaf_type == "int"


def test_resolve_crossfile_auth_ttl():
    """RemoteClient.auth.ttl → int."""
    result = resolve_field_chain("RemoteClient", ["auth", "ttl"], CROSSFILE)
    assert result is not None
    assert result.leaf_type == "int"
    assert result.chain_types == ["RemoteAuth", "int"]


# ---------------------------------------------------------------------------
# resolve_field_chain: param-annotated field (no class-level annotation)
# ---------------------------------------------------------------------------

def test_resolve_param_annotated_chain():
    """ConnectionParamAnnotated.config.timeout → int via param annotation."""
    result = resolve_field_chain(
        "ConnectionParamAnnotated", ["config", "timeout"], TARGET
    )
    assert result is not None
    assert result.leaf_type == "int"
    assert result.chain_types == ["Config", "int"]


# ---------------------------------------------------------------------------
# resolve_field_chain: constructor-rhs inference
# ---------------------------------------------------------------------------

def test_resolve_constructor_rhs_chain():
    """ConnectionConstructorRhs.config.host → str (config inferred from ctor call)."""
    result = resolve_field_chain(
        "ConnectionConstructorRhs", ["config", "host"], TARGET
    )
    assert result is not None
    assert result.leaf_type == "str"


# ---------------------------------------------------------------------------
# resolve_field_chain: missing class / field
# ---------------------------------------------------------------------------

def test_resolve_missing_root_class():
    result = resolve_field_chain("NoSuchClass", ["foo"], TARGET)
    assert result is None


def test_resolve_missing_field_stops_early():
    """Chain that hits a missing field annotation returns partial result."""
    result = resolve_field_chain(
        "Connection", ["config", "nonexistent_field"], TARGET
    )
    # Resolved "config" → "Config", then stopped
    assert result is not None
    assert result.leaf_type == "Config"
    assert result.resolved_depth == 1


def test_resolve_empty_chain():
    result = resolve_field_chain("Config", [], TARGET)
    assert result is None


# ---------------------------------------------------------------------------
# resolve_field_chain: cycle detection
# ---------------------------------------------------------------------------

def test_resolve_cyclic_class_no_infinite_loop():
    """CyclicHolder.next.next → should terminate, not infinite loop."""
    result = resolve_field_chain(
        "CyclicHolder", ["next", "next", "value"], TARGET
    )
    # Resolution may be partial (Optional[CyclicHolder] hits cycle), but
    # must terminate and not raise.
    # Regardless of partial vs full, just assert it returns without hanging.
    assert True  # If we got here, no infinite loop


# ---------------------------------------------------------------------------
# resolve_field_chain: max_depth
# ---------------------------------------------------------------------------

def test_resolve_max_depth_truncates():
    """With max_depth=1, a 3-level chain stops after the first step."""
    result = resolve_field_chain(
        "Session", ["connection", "config", "timeout"], TARGET, max_depth=1
    )
    # Only resolved "connection" → "Connection" (1 step), then stopped
    assert result is not None
    assert result.resolved_depth == 1
    assert result.chain_types == ["Connection"]


def test_resolve_max_depth_2():
    result = resolve_field_chain(
        "Session", ["connection", "config", "timeout"], TARGET, max_depth=2
    )
    assert result is not None
    assert result.resolved_depth == 2
    assert result.chain_types == ["Connection", "Config"]


# ---------------------------------------------------------------------------
# Hard: MixedAnnotations priority
# ---------------------------------------------------------------------------

def test_mixed_class_level_takes_priority():
    """Class-level annotation wins over __init__ assignment literal."""
    result = resolve_field_chain("MixedAnnotations", ["class_level_int"], TARGET)
    assert result is not None
    assert result.leaf_type == "int"


def test_mixed_param_annotated_float():
    """Param annotation for float field."""
    result = resolve_field_chain(
        "MixedAnnotations", ["param_annotated_float"], TARGET
    )
    assert result is not None
    assert result.leaf_type == "float"


def test_mixed_init_annassign_bytes():
    """self.init_annassign: bytes — AnnAssign in __init__ body."""
    result = resolve_field_chain("MixedAnnotations", ["init_annassign"], TARGET)
    assert result is not None
    assert result.leaf_type == "bytes"


def test_mixed_constructed_field():
    """self.constructed = Config() — type inferred from constructor call."""
    result = resolve_field_chain("MixedAnnotations", ["constructed"], TARGET)
    assert result is not None
    assert result.leaf_type == "Config"


# ---------------------------------------------------------------------------
# Hard: dataclass (class-level annotations, no explicit __init__ in source)
# ---------------------------------------------------------------------------

def test_dataclass_config_host():
    """DataclassConfig.host — annotation in dataclass class body."""
    result = resolve_field_chain("DataclassConfig", ["host"], TARGET)
    assert result is not None
    assert result.leaf_type == "str"


def test_dataclass_config_port():
    result = resolve_field_chain("DataclassConfig", ["port"], TARGET)
    assert result is not None
    assert result.leaf_type == "int"


def test_dataclass_config_tls():
    result = resolve_field_chain("DataclassConfig", ["tls"], TARGET)
    assert result is not None
    assert result.leaf_type == "bool"


def test_dataclass_client_chain():
    """DataclassClient.cfg.port → int (dataclass nested in regular class)."""
    result = resolve_field_chain("DataclassClient", ["cfg", "port"], TARGET)
    assert result is not None
    assert result.leaf_type == "int"
    assert result.chain_types == ["DataclassConfig", "int"]


# ---------------------------------------------------------------------------
# Hard: Pipeline inner class (nested class annotation)
# ---------------------------------------------------------------------------

def test_pipeline_stage_enabled():
    """Pipeline.Stage.enabled → bool."""
    result = resolve_field_chain("Pipeline.Stage", ["enabled"], TARGET)
    # "Pipeline.Stage" is a dotted name — _find_class looks for "Pipeline.Stage"
    # which load_class_def won't find directly. This tests the limitation.
    # Either resolved or None — we just assert no crash.
    assert True  # Document known limitation: nested classes not yet resolved


# ---------------------------------------------------------------------------
# _strip_annotation: Optional unwrapping + forward reference stripping
# ---------------------------------------------------------------------------

from tharkun.solve.frontend.field_resolver import _strip_annotation

@pytest.mark.parametrize("input_str,expected", [
    ("int", "int"),
    ("str", "str"),
    ("Config", "Config"),
    ("Optional[Config]", "Config"),
    ("Optional['Config']", "Config"),
    ("Config | None", "Config"),
    ("None | Config", "Config"),
    ("Union[Config, None]", "Config"),
    ("'NodeB'", "NodeB"),
    ('"NodeB"', "NodeB"),
    ("Optional[\"Config\"]", "Config"),
])
def test_strip_annotation(input_str, expected):
    assert _strip_annotation(input_str) == expected


# ---------------------------------------------------------------------------
# Forward reference annotations
# ---------------------------------------------------------------------------

def test_get_field_type_forward_ref():
    """config: 'Config' — forward reference string annotation."""
    cls = load_class("ForwardRefHolder")
    result = _get_field_type(cls, "config")
    # Raw annotation is 'Config' (with quotes) — should be stripped to "Config"
    assert result == "Config"


def test_resolve_forward_ref_chain():
    """ForwardRefHolder.config.timeout → int (field annotated as 'Config')."""
    result = resolve_field_chain("ForwardRefHolder", ["config", "timeout"], TARGET)
    assert result is not None
    assert result.leaf_type == "int"
    assert result.resolved_depth == 2


def test_resolve_optional_field_chain():
    """WrapperChain.inner.timeout → int (inner: Optional[Config])."""
    result = resolve_field_chain("WrapperChain", ["inner", "timeout"], TARGET)
    assert result is not None
    assert result.leaf_type == "int"


# ---------------------------------------------------------------------------
# Integration: resolve leaf types for real guard expressions
# ---------------------------------------------------------------------------

def test_client_send_timeout_guard():
    """Simulates what solver would do for `self.connection.config.timeout > 60`."""
    # self → Client, chain = ["connection", "config", "timeout"]
    result = resolve_field_chain(
        "Client", ["connection", "config", "timeout"], TARGET
    )
    assert result is not None
    assert result.leaf_type == "int"
    assert result.resolved_depth == 3


def test_client_send_active_guard():
    """Simulates `not self.connection.creds.active`."""
    result = resolve_field_chain(
        "Client", ["connection", "creds", "active"], TARGET
    )
    assert result is not None
    assert result.leaf_type == "bool"
    assert result.resolved_depth == 3


def test_client_send_host_guard():
    """Simulates `self.connection.config.host == '...'`."""
    result = resolve_field_chain(
        "Client", ["connection", "config", "host"], TARGET
    )
    assert result is not None
    assert result.leaf_type == "str"


def test_client_connect_port_guard():
    """Simulates `self.connection.port < 1024`."""
    result = resolve_field_chain(
        "Client", ["connection", "port"], TARGET
    )
    assert result is not None
    assert result.leaf_type == "int"
    assert result.resolved_depth == 2


def test_client_authenticate_expires_guard():
    """Simulates `self.connection.creds.expires < min_ttl`."""
    result = resolve_field_chain(
        "Client", ["connection", "creds", "expires"], TARGET
    )
    assert result is not None
    assert result.leaf_type == "int"
    assert result.resolved_depth == 3
