"""The ONE place the solving core imports or executes a target module (#0607, plan #0600).

Solving never needs to *run* the target: a predicate in, a value-set claim out.
The core still *loads* target modules in a few places -- to read a computed
``re.compile`` pattern, to construct an instance of a class named in an
``isinstance`` guard, to probe a constructor's ``TypeError``s -- and every one
of those goes through this module.  ``tests/test_layering.py`` forbids
``importlib`` / ``exec`` / ``subprocess`` / ``sys.settrace`` everywhere else under
``solve/``, and a caller can switch loading off entirely with :func:`no_targets`:
inside that block every loader returns its absent value and the callers fall
back to the purely static answer.

The three strategies (package import, ``spec_from_file_location``, ``exec`` of
the source) and their order are the ones ``branch_solver`` has always used; it
now delegates here, which removes the core's last import of a layer above it.

Project import (#1132b)
-----------------------
A module of a package (``pkg/mod.py`` with ``from ._c import K``) exists only
inside its package: ``import pkg.mod`` runs ``pkg/__init__.py`` first, then the
submodules in the order their imports ask for them, and what ``__init__`` does
(``_c.LIMIT = 5``) is what ``mod`` sees.  Loading the file as a bare module, or
under a synthetic parent, runs a program that never runs in reality.  Three
functions are the ONE answer, for every tool that must see the real thing:

* :func:`project_root_and_module` -- STATIC, executes nothing: the project root
  (the first ancestor that is not a package) and the dotted name, the way Python
  itself would import the file.  Avengers (``avengers/backends.py``) delegates
  here, so Pynguin and every in-process caller agree on what "the project" is.
* :func:`load_project_module` -- the real ``importlib.import_module`` of that
  dotted name, in the CALLING process, under a ``confine`` context the caller
  must supply (the environment model and file-system confinement of
  ``solve/environment.py``).  It runs the project's module-level code, so it is
  for a caller whose job is to run the program (the Eregion inverter's forward
  replay); a verifier that must not run it in its own process uses the
  subprocess variant :func:`import_project_isolated`.  The returned facade
  confines every later call of a module-level function the same way.
* :func:`import_project_isolated` -- the same import in a disposable child
  interpreter (scratch cwd, the environment model installed first), answering
  with a JSON-safe report only.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import dataclasses
import functools
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Sequence

_TARGETS_DISABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tharkun_solve_targets_disabled", default=False
)

EXEC_TARGET_NAME = "_tharkun_exec_target"


def targets_disabled() -> bool:
    """True inside a :func:`no_targets` block: no target module is imported or executed."""
    return _TARGETS_DISABLED.get()


@contextlib.contextmanager
def no_targets() -> Iterator[None]:
    """Forbid target loading for the duration of the block (re-entrant, task-local)."""
    token = _TARGETS_DISABLED.set(True)
    try:
        yield
    finally:
        _TARGETS_DISABLED.reset(token)


def file_path_to_dotted_module(file_path: str) -> str | None:
    """Infer a dotted module name from a file path using ``sys.path``.

    ``/site-packages/humanize/number.py`` -> ``humanize.number``;
    ``/site-packages/jinja2/__init__.py`` -> ``jinja2``; ``None`` when the file
    is under no ``sys.path`` entry.

    Two rules keep this from naming something that cannot be imported:

    * Every component must be a Python identifier.  A path can sit under a
      ``sys.path`` entry and still have no dotted name -- a directory called
      ``.venv`` or ``python3.12`` is not a package, however deeply the file
      nests inside it.
    * When several entries match, the LONGEST wins.  The current directory is
      usually on ``sys.path``, so a target under it -- a virtual environment
      inside a checkout, say -- would otherwise be named relative to the
      checkout instead of relative to its own site-packages, and import as
      something else entirely or not at all.
    """
    rp = os.path.realpath(file_path)
    best: tuple[int, str] | None = None
    for sp in sys.path:
        if not sp:
            continue
        sp_real = os.path.realpath(sp)
        if not rp.startswith(sp_real + os.sep):
            continue
        rel = rp[len(sp_real) + 1:]
        if rel.endswith(".py"):
            rel = rel[:-3]
        parts = rel.split(os.sep)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts or not all(part.isidentifier() for part in parts):
            continue
        if best is None or len(sp_real) > best[0]:
            best = (len(sp_real), ".".join(parts))
    return best[1] if best is not None else None


def import_dotted(dotted: str | None) -> types.ModuleType | None:
    """Import a target by dotted name; ``None`` when disabled, unnamed or failing."""
    if not dotted or targets_disabled():
        return None
    try:
        return importlib.import_module(dotted)
    except Exception:
        return None


def load_from_file(path: str, name: str = EXEC_TARGET_NAME, *, source_only: bool = False) -> types.ModuleType | None:
    """Execute *path* as a standalone module; ``None`` when disabled or failing."""
    if targets_disabled():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        from tharkun.solve.environment import _CONFINED

        if source_only or _CONFINED["depth"]:
            # A manual loader has no real-import exemption for its bytecode cache.
            # Read declared source under the existing guard; preserve module metadata.
            code = spec.loader.source_to_code(Path(path).read_bytes(), path)
            exec(code, mod.__dict__)  # noqa: S102 -- the sanctioned target loader
        else:
            spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def exec_source(path: str) -> dict[str, Any] | None:
    """Execute the source at *path* into a fresh globals dict (the last resort:
    relative imports fail); ``None`` when disabled or failing."""
    if targets_disabled():
        return None
    try:
        source = Path(path).read_text(encoding="utf-8", errors="ignore")
        module_globals: dict[str, Any] = {"__builtins__": __builtins__}
        exec(compile(source, path, "exec"), module_globals)  # noqa: S102 -- the one sanctioned exec
        return module_globals
    except Exception:
        return None


def _globals_of(mod: types.ModuleType) -> dict[str, Any]:
    module_globals = {k: getattr(mod, k) for k in dir(mod) if not k.startswith("__")}
    module_globals["__builtins__"] = __builtins__
    return module_globals


def load_module_for_execution(source_path: str, *, source_only: bool = False) -> tuple[Any, dict[str, Any]]:
    """Load a target module, returning ``(module, globals_dict)``.

    Tries in order: package import (the only strategy that handles relative
    imports), ``spec_from_file_location``, then ``exec`` of the source into a
    synthetic module.  Returns ``(None, {})`` if every strategy fails or
    loading is disabled.
    """
    if targets_disabled():
        return None, {}
    rp = os.path.realpath(source_path)
    for mod in (import_dotted(file_path_to_dotted_module(rp)), load_from_file(rp, source_only=source_only)):
        if mod is None:
            continue
        try:
            return mod, _globals_of(mod)
        except Exception:
            continue
    module_globals = exec_source(rp)
    if module_globals is not None:
        mod = types.ModuleType(EXEC_TARGET_NAME)
        for k, v in module_globals.items():
            if not k.startswith("__"):
                setattr(mod, k, v)
        return mod, module_globals
    return None, {}


# ---------------------------------------------------------------------------
# project import (#1132b)
# ---------------------------------------------------------------------------


class ProjectImportError(ImportError):
    """The project module could not be imported; the message says why and where."""


def relative_import_level(source_path: str) -> int:
    """The deepest relative import (``from . import x`` is 1, ``from .. import y`` is 2) in the file; 0 when
    it has none or does not parse.  Static."""
    try:
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return 0
    return max((n.level for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level), default=0)


def project_root_and_module(source_path: str, *, depth: int | None = None) -> tuple[str, str] | None:
    """``(project root, dotted module)`` for the file, the way Python would import it; ``None`` when no
    importable name exists.  STATIC: nothing is imported or executed.

    The root is the first ancestor directory that is not a package (has no ``__init__.py``), so a module inside
    a package is named by its full dotted path and its relative imports resolve.  A package's own
    ``__init__.py`` is the package.  A file whose relative imports reach above the packages found by that walk
    (``from .. import x`` in ``pkg/mod.py`` when ``pkg`` has no ``__init__.py``) is a member of a NAMESPACE
    package: the walk goes up through plain directories until the dotted name has as many package components as
    the deepest relative import needs (``depth``, by default read from the file itself).  Every component must be
    an identifier: ``my-dir/mod.py`` has no import name.
    """
    path = Path(os.path.realpath(source_path))
    if path.suffix != ".py" or not path.is_file():
        return None
    parts = [path.stem]
    directory = path.parent
    while (directory / "__init__.py").is_file() and directory.parent != directory:
        parts.append(directory.name)
        directory = directory.parent
    is_init = parts[0] == "__init__" and len(parts) > 1
    if is_init:
        parts = parts[1:]
    level = relative_import_level(str(path)) if depth is None else depth
    needed = level if is_init else level + 1
    while len(parts) < needed:
        if directory.parent == directory:
            return None
        parts.append(directory.name)
        directory = directory.parent
    if not all(part.isidentifier() for part in parts):
        return None
    return (str(directory), ".".join(reversed(parts)))


def is_package_member(source_path: str) -> bool:
    """True when the file only runs as a member of its package: its directory has an ``__init__.py`` (importing
    it runs that first) or it uses a relative import (it has no meaning outside a package).  Static."""
    path = Path(source_path)
    return (path.parent / "__init__.py").is_file() or relative_import_level(str(path)) > 0


@contextlib.contextmanager
def _importable(paths: Sequence[str]) -> Iterator[None]:
    """``paths`` at the front of ``sys.path`` for the block (those already there are left alone), removed after."""
    added = [p for p in paths if p not in sys.path]
    for entry in reversed(added):
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        for entry in added:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass


def _evict(top: str) -> dict[str, types.ModuleType]:
    """Take the cached package ``top`` and its submodules out of ``sys.modules``; return them so a failed load
    can put them back.  Two copies of one package (an original and a mutant) have the SAME dotted names, so a
    cached copy must never answer for the one being loaded."""
    taken = {k: sys.modules.pop(k) for k in [k for k in sys.modules if k == top or k.startswith(top + ".")]}
    return taken


_CALL_STATE = threading.local()


def _confined_call(fn: Callable[..., Any], confine: Callable[[], ContextManager[Any]]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        if getattr(_CALL_STATE, "depth", 0):
            return fn(*args, **kwargs)           # already inside a confined call on this thread
        _CALL_STATE.depth = 1
        try:
            with confine():
                return fn(*args, **kwargs)
        finally:
            _CALL_STATE.depth = 0

    return run


class ConfinedModule:
    """A facade over a loaded project module: every module-level FUNCTION fetched through it runs under the
    caller's ``confine`` context (the environment model: reads of undeclared files refused, every mutation of
    the file system confined to scratch).  Everything else passes through untouched, and an assignment writes
    through to the real module, so a caller can inject helpers into the namespace project code reads.

    Not confined: code reached through an OBJECT or CLASS a call returns (its methods run when the caller calls
    them, outside this facade).  The facade is the confinement of the program's FUNCTIONS, which is what a
    forward replay calls."""

    __slots__ = ("_raw", "_confine", "_wrapped")

    def __init__(self, raw: types.ModuleType, confine: Callable[[], ContextManager[Any]]):
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_confine", confine)
        object.__setattr__(self, "_wrapped", {})

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._raw, name)
        if isinstance(value, types.FunctionType):
            cached = self._wrapped.get(name)
            if cached is None or cached[0] is not value:
                cached = (value, _confined_call(value, self._confine))
                self._wrapped[name] = cached
            return cached[1]
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._raw, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._raw, name)

    def __dir__(self) -> list[str]:
        return dir(self._raw)

    def __repr__(self) -> str:
        return f"<confined {self._raw!r}>"


@dataclasses.dataclass(frozen=True)
class ProjectModule:
    """A project module loaded by :func:`load_project_module`."""

    module: ConfinedModule            # what a caller calls into: functions run under ``confine``
    raw: types.ModuleType             # the module itself (``module`` writes through to it)
    root: str                         # the project root, on ``sys.path`` whenever project code runs through the module
    dotted: str                       # the name it was imported under
    evicted: tuple[str, ...]          # cached modules of the same package it displaced


def load_project_module(source_path: str, roots: Sequence[str] = (), *,
                        confine: Callable[[], ContextManager[Any]]) -> ProjectModule:
    """Import the file as a member of its own package, for real, and return it behind a confining facade.

    The project root (:func:`project_root_and_module`) and any extra ``roots`` (``--package-root`` directories)
    are on ``sys.path`` whenever project code runs through this module -- the import, every function call made
    through the returned facade and the facade's ``_confine`` context -- and not otherwise, so a function that
    imports a project module by absolute name at call time finds it, as it would natively; the same-named
    package cached from an earlier load is evicted first (and put back if this load fails);
    ``importlib.import_module`` then runs ``__init__.py`` and the submodules in their real order, inside
    ``with confine():``.  ``confine`` is REQUIRED: a context-manager factory (``solve/environment.py``'s
    environment model with its file-system confinement) that is entered around the import and around every
    function call made through the returned facade.

    This runs the project's module-level code in the CALLING process.  Use it only where running the program is
    the caller's job (a forward replay); to check an import without running it in your own process, use
    :func:`import_project_isolated`.  Raises :class:`ProjectImportError` (with the cause chained) when the file
    has no import name or the import fails, and when loading targets is switched off (:func:`no_targets`).
    """
    if targets_disabled():
        raise ProjectImportError("loading target modules is switched off (no_targets())")
    split = project_root_and_module(source_path)
    if split is None:
        raise ProjectImportError(f"{source_path} has no import name (not a .py file, or a path component that is "
                                 "not an identifier)")
    root, dotted = split
    top = dotted.split(".")[0]
    taken = _evict(top)
    paths = (root, *(os.path.realpath(r) for r in roots))

    @contextlib.contextmanager
    def confine_on_path() -> Iterator[Any]:
        with confine() as confinement, _importable(paths):
            yield confinement

    try:
        with confine_on_path():
            raw = importlib.import_module(dotted)
    except BaseException as exc:
        _evict(top)
        sys.modules.update(taken)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ProjectImportError(f"importing {dotted} from {root} failed: {type(exc).__name__}: {exc}") from exc
    return ProjectModule(ConfinedModule(raw, confine_on_path), raw, root, dotted, tuple(sorted(taken)))


_ISOLATED_CHILD = r"""
import importlib, json, os, sys, traceback
req = json.loads(os.environ["THARKUN_PROJECT_IMPORT_REQ"])
flags = set()
report = {"ok": False, "error": None, "dotted": req["dotted"], "root": req["root"], "flags": [], "names": []}
try:
    # the environment model, imported by bare name with a self-cleaning sys.path insert (what _child.py does)
    sys.path.insert(0, req["solve_dir"])
    try:
        from environment import install_files
    finally:
        sys.path.remove(req["solve_dir"])
    install_files(req["files"], os.getcwd(), flags)
    for entry in reversed(req["paths"]):
        sys.path.insert(0, entry)
    mod = importlib.import_module(req["dotted"])
    report["names"] = sorted(n for n in vars(mod) if not n.startswith("__"))
    report["ok"] = True
except BaseException as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc()[-2000:]
report["flags"] = sorted(flags)
sys.stdout.write("\n" + json.dumps(report) + "\n")
"""


def import_project_isolated(source_path: str, roots: Sequence[str] = (), files: Sequence[str] = (),
                            timeout: float = 30.0) -> dict[str, Any]:
    """Import the file as a member of its own package in a DISPOSABLE child interpreter and report what happened.

    The child starts in a fresh scratch directory with stdin on ``/dev/null``, installs the environment model
    (``solve/environment.py::install_files``: reads of declared ``files`` only, writes confined to the scratch)
    BEFORE the project code runs, then performs the same real import as :func:`load_project_module`.  The answer
    is a JSON-safe report: ``ok``, ``error`` (``"Type: message"`` when the import raised), ``dotted``, ``root``,
    ``flags`` (the environment model's ``file_effect`` / ``file_unmodelled`` ... markers: the import touched the
    environment), ``names`` (the module's public names) -- and ``traceback`` on failure.  No project object ever
    reaches the caller's process; a hang is killed at ``timeout`` seconds (``error == "timeout"``).  ``files``
    defaults to the module itself.
    """
    out: dict[str, Any] = {"ok": False, "error": None, "dotted": None, "root": None, "flags": [], "names": []}
    if targets_disabled():
        out["error"] = "loading target modules is switched off (no_targets())"
        return out
    split = project_root_and_module(source_path)
    if split is None:
        out["error"] = f"{source_path} has no import name"
        return out
    root, dotted = split
    out["dotted"], out["root"] = dotted, root
    request = {
        "dotted": dotted, "root": root,
        "paths": [p for p in (root, *(os.path.realpath(r) for r in roots))],
        "files": [os.path.realpath(f) for f in (files or (source_path,))],
        "solve_dir": os.path.dirname(os.path.abspath(__file__)),
    }
    from tharkun.solve.environment import subprocess_environment
    env = dict(subprocess_environment(), THARKUN_PROJECT_IMPORT_REQ=json.dumps(request), PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="tharkun-project-import-") as scratch:
        try:
            proc = subprocess.run([sys.executable, "-c", _ISOLATED_CHILD], cwd=scratch, env=env,
                                  stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout,
                                  start_new_session=True)
        except subprocess.TimeoutExpired:
            out["error"] = "timeout"
            return out
        except OSError as exc:
            out["error"] = f"could not start the child: {exc}"
            return out
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                report = json.loads(line)
            except ValueError:
                break
            out.update(report)
            return out
    out["error"] = f"the child exited with status {proc.returncode} and no report: {proc.stderr.strip()[-300:]}"
    return out
