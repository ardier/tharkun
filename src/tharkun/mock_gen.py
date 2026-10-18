"""External call detection and ``@patch`` decorator generation (SG-5).

Scans function bodies via AST to detect calls to external services
(HTTP, filesystem, database, checksums/hashes), then generates
``unittest.mock.patch`` decorators for test emission.

Usage::

    from tharkun.mock_gen import detect_externals, generate_patches

    externals = detect_externals(source_path, "my_function")
    patches = generate_patches(externals)
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalCall:
    """A detected external call in a function body.

    Attributes:
        target:    Patch target string, e.g. ``"requests.get"``.
        category:  ``"http"`` | ``"filesystem"`` | ``"database"``
                   | ``"checksum"`` | ``"other"``.
        lineno:    Line number in the source file.
        context:   How the call is used: ``"direct"``, ``"context_manager"``,
                   ``"chain"``.
        return_mock: Suggested mock return value as source code.
    """

    target: str
    category: str
    lineno: int
    context: str = "direct"
    return_mock: str | None = None


@dataclass
class PatchSpec:
    """A generated ``@patch`` decorator specification.

    Attributes:
        target:         Patch target string.
        return_value:   Mock return value source (if simple).
        setup_lines:    Lines of mock setup code inside the test body.
        category:       Same as ``ExternalCall.category``.
    """

    target: str
    return_value: str | None = None
    setup_lines: list[str] = field(default_factory=list)
    category: str = "other"


@dataclass
class MockPlan:
    """Complete mock plan for a function."""

    callable_qualname: str
    source_path: str
    externals: list[ExternalCall] = field(default_factory=list)
    patches: list[PatchSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# External call detection registry
# ---------------------------------------------------------------------------

# HTTP libraries: module.function → category
_HTTP_CALLS: dict[str, str] = {
    "requests.get": "http",
    "requests.post": "http",
    "requests.put": "http",
    "requests.delete": "http",
    "requests.patch": "http",
    "requests.head": "http",
    "requests.options": "http",
    "requests.Session": "http",
    "urllib.request.urlopen": "http",
    "urllib.request.Request": "http",
    "httpx.get": "http",
    "httpx.post": "http",
    "httpx.Client": "http",
    "httpx.AsyncClient": "http",
}

# Filesystem: module.function → category
_FS_CALLS: dict[str, str] = {
    "builtins.open": "filesystem",
    "open": "filesystem",
    "os.path.exists": "filesystem",
    "os.path.isfile": "filesystem",
    "os.path.isdir": "filesystem",
    "os.listdir": "filesystem",
    "os.makedirs": "filesystem",
    "os.remove": "filesystem",
    "os.unlink": "filesystem",
    "os.rename": "filesystem",
    "shutil.copy": "filesystem",
    "shutil.copy2": "filesystem",
    "shutil.rmtree": "filesystem",
    "shutil.move": "filesystem",
    "pathlib.Path.read_text": "filesystem",
    "pathlib.Path.read_bytes": "filesystem",
    "pathlib.Path.write_text": "filesystem",
    "pathlib.Path.write_bytes": "filesystem",
    "pathlib.Path.exists": "filesystem",
    "pathlib.Path.is_file": "filesystem",
    "pathlib.Path.is_dir": "filesystem",
    "pathlib.Path.mkdir": "filesystem",
    "pathlib.Path.unlink": "filesystem",
}

# Database: module.function → category
_DB_CALLS: dict[str, str] = {
    "sqlite3.connect": "database",
    "psycopg2.connect": "database",
    "pymysql.connect": "database",
    "mysql.connector.connect": "database",
    "sqlalchemy.create_engine": "database",
}

# Checksum / hash: module.function → category
_HASH_CALLS: dict[str, str] = {
    "hashlib.md5": "checksum",
    "hashlib.sha1": "checksum",
    "hashlib.sha256": "checksum",
    "hashlib.sha512": "checksum",
    "hashlib.sha3_256": "checksum",
    "hashlib.sha3_512": "checksum",
    "hashlib.blake2b": "checksum",
    "hashlib.blake2s": "checksum",
    "hashlib.new": "checksum",
    "hmac.new": "checksum",
    "hmac.digest": "checksum",
    "hmac.compare_digest": "checksum",
    "zlib.crc32": "checksum",
    "zlib.adler32": "checksum",
}

# Merge all registries.
_ALL_EXTERNALS: dict[str, str] = {**_HTTP_CALLS, **_FS_CALLS, **_DB_CALLS, **_HASH_CALLS}

# Method names that indicate external calls when on known types.
_PATHLIB_METHODS = frozenset({
    "read_text", "read_bytes", "write_text", "write_bytes",
    "exists", "is_file", "is_dir", "mkdir", "unlink",
})

_SESSION_METHODS = frozenset({"get", "post", "put", "delete", "patch", "head", "options"})


# ---------------------------------------------------------------------------
# AST-based external call detection
# ---------------------------------------------------------------------------


def _resolve_call_target(node: ast.Call, imports: dict[str, str]) -> str | None:
    """Try to resolve a call node to a fully-qualified target string."""
    func = node.func

    # Direct name: open(...), requests.get(...)
    if isinstance(func, ast.Name):
        name = func.id
        if name == "open":
            return "builtins.open"
        # Check if it's an imported module call.
        if name in imports:
            return imports[name]
        return None

    # Attribute access: requests.get(...), os.path.exists(...)
    if isinstance(func, ast.Attribute):
        # module.func
        if isinstance(func.value, ast.Name):
            prefix = func.value.id
            full = f"{prefix}.{func.attr}"
            if full in _ALL_EXTERNALS:
                return full
            # Check imports: `import requests` → requests.get → requests.get
            if prefix in imports:
                resolved = f"{imports[prefix]}.{func.attr}"
                if resolved in _ALL_EXTERNALS:
                    return resolved

        # module.submodule.func: os.path.exists
        if (isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)):
            prefix = f"{func.value.value.id}.{func.value.attr}"
            full = f"{prefix}.{func.attr}"
            if full in _ALL_EXTERNALS:
                return full

        # Path(...).read_text() — method on constructed Path
        if func.attr in _PATHLIB_METHODS:
            # Check if value is a Path() constructor or a Name that could be a Path.
            if isinstance(func.value, ast.Call):
                call_func = func.value.func
                if isinstance(call_func, ast.Name) and call_func.id == "Path":
                    return f"pathlib.Path.{func.attr}"
            if isinstance(func.value, ast.Name):
                return f"pathlib.Path.{func.attr}"

        # self.session.get(...) — attribute chain for HTTP sessions
        if (func.attr in _SESSION_METHODS
                and isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "self"):
            attr_name = func.value.attr
            if attr_name == "session":
                return f"requests.Session.{func.attr}"

    return None


def _parse_imports_for_detection(tree: ast.Module) -> dict[str, str]:
    """Build a simple import map: {local_name: qualified_module_name}."""
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = f"{module}.{alias.name}" if module else alias.name
    return imports


def detect_externals(
    source_path: str,
    function_name: str,
) -> list[ExternalCall]:
    """Detect external calls in a function body.

    Args:
        source_path:   Path to the Python source file.
        function_name: Name of the function to analyze (simple name or qualname).

    Returns:
        List of ``ExternalCall`` objects for detected external calls.
    """
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    imports = _parse_imports_for_detection(tree)

    # Find the function definition.
    func_def = None
    parts = function_name.split(".")
    if len(parts) == 1:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                func_def = node
                break
    elif len(parts) == 2:
        cls_name, method_name = parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                        func_def = child
                        break
                break

    if func_def is None:
        return []

    externals: list[ExternalCall] = []
    seen_targets: set[str] = set()

    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call):
            continue

        target = _resolve_call_target(node, imports)
        if target is None:
            continue
        if target not in _ALL_EXTERNALS and not target.startswith("pathlib.Path.") and not target.startswith("requests.Session."):
            continue

        category = _ALL_EXTERNALS.get(target, "other")
        if target.startswith("pathlib.Path."):
            category = "filesystem"
        if target.startswith("requests.Session."):
            category = "http"

        if target in seen_targets:
            continue
        seen_targets.add(target)

        lineno = getattr(node, "lineno", 0)
        externals.append(ExternalCall(
            target=target,
            category=category,
            lineno=lineno,
        ))

    return externals


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

_DEFAULT_RETURN_VALUES: dict[str, str] = {
    # HTTP
    "requests.get": 'Mock(status_code=200, json=Mock(return_value={}), text="", raise_for_status=Mock())',
    "requests.post": 'Mock(status_code=200, json=Mock(return_value={}), text="")',
    "requests.put": 'Mock(status_code=200)',
    "requests.delete": 'Mock(status_code=200)',
    "requests.Session.get": 'Mock(status_code=200, json=Mock(return_value={}), raise_for_status=Mock())',
    "requests.Session.post": 'Mock(status_code=200, json=Mock(return_value={}))',
    "urllib.request.urlopen": 'Mock(read=Mock(return_value=b""), __enter__=Mock(), __exit__=Mock())',
    "httpx.Client": 'Mock()',
    # Filesystem
    "builtins.open": None,  # Uses mock_open
    "os.path.exists": "True",
    "os.path.isfile": "True",
    "os.path.isdir": "True",
    "os.listdir": "[]",
    "pathlib.Path.exists": "True",
    "pathlib.Path.is_file": "True",
    "pathlib.Path.read_text": '""',
    "pathlib.Path.read_bytes": 'b""',
    "pathlib.Path.write_text": "None",
    "pathlib.Path.write_bytes": "None",
    # Database
    "sqlite3.connect": 'Mock(cursor=Mock(return_value=Mock(fetchone=Mock(return_value=(0,)), fetchall=Mock(return_value=[]))))',
    # Checksum
    "hashlib.md5": 'Mock(hexdigest=Mock(return_value="d41d8cd98f00b204e9800998ecf8427e"))',
    "hashlib.sha256": 'Mock(hexdigest=Mock(return_value="e3b0c44298fc1c149afbf4c8996fb924" + "27ae41e4649b934ca495991b7852b855"))',
    "hashlib.sha1": 'Mock(hexdigest=Mock(return_value="da39a3ee5e6b4b0d3255bfef95601890afd80709"))',
    "hashlib.new": 'Mock(hexdigest=Mock(return_value="0" * 64))',
    "hmac.new": 'Mock(hexdigest=Mock(return_value="0" * 64))',
    "zlib.crc32": "0",
    "zlib.adler32": "1",
}


def generate_patches(externals: list[ExternalCall]) -> list[PatchSpec]:
    """Generate ``@patch`` specs from detected external calls."""
    patches: list[PatchSpec] = []
    for ext in externals:
        return_val = _DEFAULT_RETURN_VALUES.get(ext.target)
        setup: list[str] = []

        if ext.target == "builtins.open":
            patches.append(PatchSpec(
                target="builtins.open",
                return_value=None,
                setup_lines=['mock_open_obj = mock_open(read_data=b"")'],
                category=ext.category,
            ))
            continue

        patches.append(PatchSpec(
            target=ext.target,
            return_value=return_val,
            setup_lines=setup,
            category=ext.category,
        ))

    return patches


# ---------------------------------------------------------------------------
# Rendering helpers (for emit.py integration)
# ---------------------------------------------------------------------------


def render_patch_decorators(patches: list[PatchSpec]) -> list[str]:
    """Render ``@patch(...)`` decorator lines."""
    lines: list[str] = []
    for p in reversed(patches):  # reverse: innermost decorator = first param
        if p.target == "builtins.open":
            lines.append(f'@patch("builtins.open", new_callable=mock_open, read_data=b"")')
        elif p.return_value:
            lines.append(f'@patch("{p.target}", return_value={p.return_value})')
        else:
            lines.append(f'@patch("{p.target}")')
    return lines


def render_mock_params(patches: list[PatchSpec]) -> list[str]:
    """Render mock parameter names for the test function signature."""
    params: list[str] = []
    for p in patches:
        # Convert "requests.get" → "mock_requests_get"
        safe = p.target.replace(".", "_").replace("builtins_", "")
        params.append(f"mock_{safe}")
    return params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_mocking_needs(
    source_path: str,
    function_names: list[str] | None = None,
) -> list[MockPlan]:
    """Analyze a source file for external calls that need mocking.

    Args:
        source_path:    Path to the Python source file.
        function_names: Functions to analyze. If ``None``, analyzes all
                       top-level functions and methods.

    Returns:
        List of ``MockPlan`` objects, one per function with externals.
    """
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    if function_names is None:
        function_names = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        function_names.append(f"{node.name}.{child.name}")

    plans: list[MockPlan] = []
    for fn_name in function_names:
        externals = detect_externals(source_path, fn_name)
        if externals:
            patches = generate_patches(externals)
            plans.append(MockPlan(
                callable_qualname=fn_name,
                source_path=rp,
                externals=externals,
                patches=patches,
            ))

    return plans


def serialize_mock_plan(plan: MockPlan) -> dict[str, Any]:
    """Serialize a MockPlan for JSON output."""
    return {
        "callable": plan.callable_qualname,
        "source_path": plan.source_path,
        "externals": [
            {
                "target": e.target,
                "category": e.category,
                "lineno": e.lineno,
                "context": e.context,
            }
            for e in plan.externals
        ],
        "patches": [
            {
                "target": p.target,
                "return_value": p.return_value,
                "category": p.category,
            }
            for p in plan.patches
        ],
    }
