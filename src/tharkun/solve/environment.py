"""Shared SV task environment: declared files, argv candidates, and admissible reads.

Proof mode treats undeclared reads and stdin as arbitrary bytes or failure; callers
must leave a site open if they cannot prove both. Witness mode starts with only
declared file bytes and empty stdin. All writes are local to one scratch run.
"""

from __future__ import annotations

import builtins
import _io
import contextlib
import errno
import glob
import hashlib
import io
import os
import re
import shutil
import sys
import itertools
import tempfile
import threading
import types
from dataclasses import dataclass


def boxed_home() -> dict[str, str]:
    """Stable HOME/XDG paths for a disposable mocked execution."""
    return {"HOME": BOX_HOME, "XDG_CONFIG_HOME": BOX_HOME + "/.config",
            "XDG_DATA_HOME": BOX_HOME + "/.local/share", "XDG_CACHE_HOME": BOX_HOME + "/.cache",
            "XDG_STATE_HOME": BOX_HOME + "/.local/state"}


def stdin_bytes(world):
    """The boxed stdin stream's bytes, or None (ordinary SV still uses DEVNULL)."""
    objects = world.objects if hasattr(world, "objects") else (world or {})
    obj = objects.get("stdin")
    if obj is None or (obj.kind if hasattr(obj, "kind") else obj.get("kind")) != "stream":
        return None
    if hasattr(obj, "concrete"):
        return obj.concrete
    if "content_base64" in obj:
        import base64
        return base64.b64decode(obj["content_base64"], validate=True)
    return obj["content"].encode("utf-8")


@contextlib.contextmanager
def feed_stdin(data):
    """Feed bytes through the real fd 0; never replace sys.stdin or sys.__stdin__."""
    if data is None:
        yield
        return
    with tempfile.TemporaryFile() as source:
        source.write(data)
        source.seek(0)
        previous = os.dup(0)
        try:
            os.dup2(source.fileno(), 0)
            sys.stdin.seek(0)  # Discard buffered bytes from an earlier boxed input.
            yield
        finally:
            os.dup2(previous, 0)
            os.close(previous)
            try:
                sys.stdin.seek(0)
            except (OSError, ValueError):
                pass  # The original fd can be a pipe, restored intact.


def subprocess_environment() -> dict[str, str]:
    """Anchor inherited import paths before a child changes cwd; unset/empty stays so."""
    environment = dict(os.environ)
    if environment.get("PYTHONPATH"):
        cwd = os.getcwd()
        # Prefix paths without resolving symlinks or collapsing '..'; empty fields mean cwd.
        environment["PYTHONPATH"] = os.pathsep.join(
            os.path.join(cwd, entry) if entry else cwd
            for entry in environment["PYTHONPATH"].split(os.pathsep)
        )
    return environment


# [internal ref] regression (found by W65 on [internal ref]'s GUUT harness, fixed 2026-09-29;
# hardened 2026-09-29 against MG review of PR 115): importlib._bootstrap[_external]
# (frozen or not) resolve sys.path -- however the target set it up, including its own
# runtime sys.path.insert() -- and their OWN mechanical disk reads (get_data()'s
# _io.open_code() no-hook fallback to _io.open(), or _io.FileIO() for a
# sourceless/extension loader) land on this module's patched open()/FileIO exactly
# like any other call would. That read is not a program file read under the I/O
# model; it must not be refused or counted as a target file effect, the same
# reasoning as the existing program-file/infrastructure exemptions just below, just
# not expressible as a fixed path prefix.
#
# os.stat/os.listdir are NEVER affected by any of this: the loader's own path
# probing (_path_stat, FileFinder._fill_cache) calls the raw posix/nt extension
# module directly (``import posix as _os``), never this process's patched ``os``
# module -- confirmed by frame instrumentation finding zero calls into
# model_stat/model_enumerate for a plain sys.path.insert() + import.  The
# loader's .pyc write (_write_atomic) DOES reach the model since [internal ref] patched
# ``posix.open`` with model_os_open; it is exempted there by
# _import_machinery_read, the same mechanical need as the reads above.
# (linecache is a SEPARATE, narrower channel onto
# model_stat -- its checkcache/updatecache calls the module-level ``os.stat``,
# which DOES reach model_stat -- but the fix for that is not an exemption here:
# _child.py's own traceback.extract_tb(e.__traceback__), the only caller inside
# this harness that could trigger it, now passes lookup_lines=False so it never
# calls linecache at all; a target that imports linecache itself and calls it
# directly is correctly still refused/flagged like any other undeclared read.)
#
# MG review of PR 115 (2026-09-29): checking only the ONE immediate caller frame,
# as the first version of this fix did, also exempted a target driving a loader
# DIRECTLY on an arbitrary, undeclared path with no real import involved -- e.g.
# ``importlib.util.spec_from_file_location('x', p).loader.get_data(p)`` -- since
# that call reaches get_data(), and therefore _io.open/_io.FileIO, exactly the same
# way a real import does. Fixed by requiring _find_and_load specifically, not just
# "some importlib frame", somewhere in the walk: it is the one frame every REAL
# import passes through, no matter which of the three ways it started (a bare
# ``import`` statement, ``__import__()``, or ``importlib.import_module()`` all
# funnel through it) -- confirmed empirically for a sys.path.insert() import
# against CPython 3.14: get_data <- get_code <- exec_module <- _load_unlocked <-
# _find_and_load_unlocked <- _find_and_load <- the importing code, 6 frames, all
# within this module set. A direct ``loader.get_data(path)`` call has the
# caller's OWN frame immediately above get_data instead (no _find_and_load
# anywhere above it), so the bounded walk below exits the import-machinery frame
# chain without ever finding it, and the read is correctly refused like any other
# undeclared one.
_IMPORT_MACHINERY_MODULES = frozenset({
    "importlib._bootstrap", "importlib._bootstrap_external",
    "_frozen_importlib", "_frozen_importlib_external",
})
_IMPORT_ENTRY_POINT = "_find_and_load"
_IMPORT_WALK_LIMIT = 24  # generous: the real chain above is 6 frames deep


def _import_machinery_read() -> bool:
    """True when the caller's caller is CPython's import machinery reading a
    module's source/bytecode off disk as part of resolving an ACTUAL import --
    never the target PROGRAM's own open()/read() call, and never a loader
    method (get_data/get_source/exec_module) the target drove directly on an
    arbitrary path with no real import involved (see the module-level comment
    above for why _find_and_load is what tells the two apart).

    Walks up only the CONTIGUOUS run of import-machinery frames starting
    immediately above the caller of the patched entry point itself (stopping
    at the first frame outside that module set, never further): this is what
    keeps the target's own open() call made while ITS top-level code happens
    to run DURING another module's import from ever being mistaken for this --
    that frame belongs to the target module's own code, not to importlib, so
    the walk exits before it could matter. Call one frame down from the
    patched open()/FileIO entry point itself, e.g.
    ``if not writing and _import_machinery_read():``.
    """
    frame = sys._getframe(2)
    for _ in range(_IMPORT_WALK_LIMIT):
        if frame is None or frame.f_globals.get("__name__") not in _IMPORT_MACHINERY_MODULES:
            return False
        if frame.f_code.co_name == _IMPORT_ENTRY_POINT:
            return True
        frame = frame.f_back
    return False


class TaskMetadataError(ValueError):
    """An unreadable declaration, with input paths retained for output protection."""

    def __init__(self, reason, files):
        super().__init__(f"unsupported task metadata: {reason}")
        self.files = files


def _task_language_version(lines: list[str]) -> str:
    """Project ordinary block options and a quoted string (no escapes) or simple
    nonnumeric plain string. Refuse YAML forms this stdlib projection cannot bind."""
    version = "3.14"
    options = False
    seen_options = False
    seen_version = False
    child_indent = None
    children = 0
    root_sequence = None
    root_children = False
    key_pattern = r"""(?:([A-Za-z_][\w.-]*|<<)|'([^']*)'|"([^"\\]*)")\s*:\s*(.*)"""
    scalar_pattern = r"""(?:"([^"\\]*)"|'([^']*)'|([A-Za-z_][\w.-]*|\d+(?:\.\d+){2,}[\w.-]*))\s*(?:\#.*)?"""
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "\t" in line[:len(line) - len(line.lstrip())] or text in {"---", "..."}:
            raise ValueError("unsupported YAML structure")
        match = re.fullmatch(key_pattern, text)
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            if text.startswith("- ") and not options:
                if root_sequence is None:
                    raise ValueError("unsupported root sequence")
                continue
            options = False
            root_sequence = None
            if text.startswith(("{", "[", "&", "*")) or match is None:
                raise ValueError("expected a block task mapping")
        if indent and not options and not root_children:
            raise ValueError("indented child after scalar declaration")
        if match is None:
            if options:
                raise ValueError("expected a direct options mapping")
            continue
        key = next(v for v in match.groups()[:3] if v is not None)
        value = match.group(4)
        if not indent:
            root_children = not value.split("#", 1)[0].strip()
        if not indent and key in {"properties", "input_files", "required_files"} and not value.split("#", 1)[0].strip():
            root_sequence = key
        if key == "options":
            if indent or seen_options or value.split("#", 1)[0].strip():
                raise ValueError("ambiguous options declaration")
            seen_options = options = True
            continue
        if key == "language_version" and (not options or not indent):
            raise ValueError("misplaced language_version")
        if key == "<<" and (not indent or options):
            raise ValueError("merged task options")
        if not options:
            continue
        children += 1
        if child_indent is None:
            child_indent = indent
        if indent != child_indent or not value or value.startswith(("&", "*", "!", "{", "[", "|", ">")):
            raise ValueError("unsupported nested or aliased options")
        if key != "language_version":
            continue
        if seen_version:
            raise ValueError("duplicate language_version")
        seen_version = True
        scalar = re.fullmatch(scalar_pattern, value)
        if scalar is None:
            raise ValueError("language_version must be a plain string")
        version = next(v for v in scalar.groups() if v is not None)
        if scalar.group(3) is not None and version.lower() in {"null", "true", "false", "yes", "no", "on", "off"}:
            raise ValueError("language_version must be a plain string")
    if seen_options and not children:
        raise ValueError("options must be a mapping")
    return version



def _file_selector(value: str) -> str:
    """Only direct string selectors whose YAML spelling this projection can bind."""
    if value.startswith(("'", '"')):
        match = re.fullmatch(r"""(?:'([^'\\]*)'|"([^"\\]*)")\s*(?:#.*)?""", value)
        if match is None:
            raise ValueError("unsupported file selector")
        return next(v for v in match.groups() if v is not None)
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    if (not value or value.startswith(("[", "{", "&", "*", "!", "|", ">", "?", "@"))
            or any(c in value for c in ("'", '"', "\\", ","))
            or ": " in value or value.lower() in {"null", "true", "false", "yes", "no", "on", "off", "~"}
            or value.lower() in {".nan", ".inf", "+.inf", "-.inf"}
            or re.fullmatch(r"[-+]?(?:0[bB][01_]+|0[xX][0-9a-fA-F_]+|"
                            r"[0-9][0-9_]*(?::[0-5]?[0-9])+(?:\.[0-9_]*)?|"
                            r"(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)(?:[eE][-+]?[0-9]+)?)", value)
            or re.match(r"\d{4}-\d{1,2}-\d{1,2}(?:$|[Tt\s])", value)):
        raise ValueError("file selector must be a direct string")
    return value


def _task_file_selectors(lines: list[str]) -> dict[str, list[str]]:
    """Bind all input/required selectors, without letting helpers supply an input."""
    fields: dict[str, list[str]] = {"input_files": [], "required_files": []}
    seen = set()
    active = None
    list_indent = None
    properties_sequence = False
    root_children = False
    key_pattern = r"""(?:([A-Za-z_][\w.-]*|<<)|'([^']*)'|"([^"\\]*)")\s*:\s*(.*)"""
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if "\t" in line[:indent]:
            raise ValueError("unsupported file declaration indentation")
        if active and value.startswith("- "):
            if list_indent is None:
                list_indent = indent
            if indent != list_indent:
                raise ValueError("nested file declaration")
            fields[active].append(_file_selector(value[2:].strip()))
            continue
        if indent:
            if active or not root_children:
                raise ValueError("unsupported file declaration indentation")
            continue
        active = None
        list_indent = None
        if value.startswith("- "):
            if properties_sequence:
                continue
            raise ValueError("unsupported root sequence")
        match = re.fullmatch(key_pattern, value)
        if match is None:
            raise ValueError("expected a block task mapping")
        key = next(v for v in match.groups()[:3] if v is not None)
        root_children = not match.group(4).split("#", 1)[0].strip()
        properties_sequence = key == "properties" and not match.group(4).split("#", 1)[0].strip()
        if key == "<<":
            raise ValueError("merged task file declarations")
        if key not in fields:
            continue
        if key in seen:
            raise ValueError(f"duplicate {key}")
        seen.add(key)
        value = match.group(4).strip()
        if not value or value.startswith("#"):
            active = key
        elif value.startswith("["):
            flow = re.fullmatch(r"\[(.*)\]\s*(?:#.*)?", value)
            if flow is None:
                raise ValueError("unsupported file list")
            items = flow.group(1).strip()
            fields[key] = [_file_selector(v.strip()) for v in items.split(",")] if items else []
        else:
            fields[key] = [_file_selector(value)]
    if not fields["input_files"]:
        raise ValueError("input_files must declare a primary input")
    return fields


@dataclass(frozen=True)
class TaskEnvironment:
    files: tuple[str, ...]
    language_version: str = "3.14"
    world: object = None

    @classmethod
    def from_task(cls, task: str) -> "TaskEnvironment":
        task = os.path.abspath(task)
        if not task.endswith((".yml", ".yaml")):
            from tharkun.solve.extern import _task_import_closure
            return cls(_task_import_closure(task, _packages=True) or (task,))
        with open(task, encoding="utf-8") as fh:
            lines = fh.readlines()
        try:
            fields = _task_file_selectors(lines)
            paths = []
            base = os.path.dirname(task)
            for value in fields["input_files"]:
                found = sorted(glob.glob(os.path.join(base, value)))
                if not found or any(not os.path.isfile(path) for path in found):
                    raise ValueError(f"cannot resolve input_files selector: {value!r}")
                paths.extend(os.path.abspath(path) for path in found)
            bases = list(dict.fromkeys((base, os.path.dirname(paths[0]))))
            for value in fields["required_files"]:
                found = sorted({path for root in bases for path in
                                glob.glob(os.path.join(root, value), recursive=True)})
                if not found:
                    raise ValueError(f"cannot resolve required_files selector: {value!r}")
                for path in found:
                    if os.path.isfile(path):
                        paths.append(os.path.abspath(path))
                    elif os.path.isdir(path):
                        paths.extend(os.path.abspath(p) for p in
                                     sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True))
                                     if os.path.isfile(p))
                    else:
                        raise ValueError("unsupported declared file")
        except ValueError as error:
            raise TaskMetadataError(str(error), ()) from error
        files = tuple(dict.fromkeys(paths))
        try:
            version = _task_language_version(lines)
        except ValueError as error:
            raise TaskMetadataError(str(error), files) from error
        return cls(files, version)

    def read_effect(self, path: str, written: set[str] | None = None) -> str:
        path = os.path.abspath(os.path.join(os.path.dirname(self.files[0]), path)) if self.files else os.path.abspath(path)
        if written and path in written:
            return "generated"
        return "fixed" if path in self.files else "unknown"


def argv_candidates(strings=()):
    """Examples of an unbounded list of strings, never an exhaustive domain."""
    yield []
    for s in dict.fromkeys(("", "x", *strings)):
        yield [s]
    for n in range(2, 65):
        yield ["x"] * n


#: modules the install-time alias sweep leaves alone: CPython's import machinery reads the disk through ``posix``
#: itself (the module comment above), so rebinding ``posix.stat``/``open``/``listdir`` would flag every import's own
#: path probe; and the harness -- ``__main__`` (a child script such as ``sv/_child.py``, which runs before any target
#: code) and Tharkun's own modules, wherever installed -- keeps real primitives ON PURPOSE
#: (``sv/_child.py::_RESULT_FILE_IO`` writes the result file outside the model).
_ALIAS_SWEEP_SKIP = frozenset({"posix", "nt", "__main__", *_IMPORT_MACHINERY_MODULES})
_MODULE_DICT = types.ModuleType.__dict__["__dict__"]


def _rebind_aliases(pairs, *, interpreter_only: bool = False) -> None:
    """Make every module-level alias of an ``old`` object in ``pairs`` the matching ``new`` one ([internal ref]).

    The installers rebind each primitive on the module that owns it (``builtins.open``, ``os.stat``, ``os.remove``,
    ...), but a module imported EARLIER that bound the same object under its own name kept the real one:
    ``tokenize``, ``bz2`` and ``tarfile`` do ``from builtins import open as ...``, so ``tokenize.open`` and
    ``bz2.open`` read an undeclared host file unflagged.  ``interpreter_only`` (the install direction) limits the
    sweep to the interpreter's own modules -- the stdlib and installed packages, under ``sys.prefix`` /
    ``sys.base_prefix``, which is where a target's libraries live -- minus :data:`_ALIAS_SWEEP_SKIP` and Tharkun: a
    module elsewhere (the harness, a test, the program hosting an in-process :func:`confined`) keeps what it bound
    on purpose.  The restore direction sweeps every module, since no model object may outlive its window.
    Matched by identity, so no ``__eq__``/``__hash__`` of a target value runs, and read through the module type's
    own ``__dict__`` slot, so a lazy module stays unloaded.  shortcut: class attributes, default arguments and
    closures that captured a primitive are not swept (none in the stdlib), nor is a package installed in editable
    mode outside the prefixes; upgrade when a measured target hides one there."""
    swap = {id(old): (old, new) for old, new in pairs if old is not new}
    if not swap:
        return
    roots = tuple({os.path.join(sys.prefix, ""), os.path.join(sys.base_prefix, "")})
    for name, module in list(sys.modules.items()):
        if not issubclass(type(module), types.ModuleType):
            continue
        namespace = _MODULE_DICT.__get__(module)
        if interpreter_only:
            path = namespace.get("__file__")
            if (name in _ALIAS_SWEEP_SKIP or name == "tharkun" or name.startswith("tharkun.")
                    or not isinstance(path, str) or not path.startswith(roots)):
                continue
        for key, value in list(namespace.items()):
            hit = swap.get(id(value))
            if hit is not None and hit[0] is value:
                namespace[key] = hit[1]


#: Standard-library modules a confined child imports BEFORE it is confined (:func:`preload_infrastructure`), a fixed list.
#:
#: Why this list.  Importing every public standard-library module inside ``install_files`` ([internal ref]) flags exactly one
#: cause: ``sysconfig``'s import-time ``is_python_build()`` stats the interpreter's own directory, an undeclared path,
#: and ``model_stat`` flags it ``file_unmodelled``.  ``ctypes``, ``pydoc``, ``trace`` and ``zoneinfo`` flag only because
#: they import ``sysconfig``; ``platform`` reaches it from its functions (``py_obj_flatting`` records
#: ``platform.python_version()``).  A target that imports any of them first INSIDE the confinement used to be lost as
#: ``unmodelled`` for the interpreter's probe, not for anything the target does.  Importing them first, as trusted
#: preparation, leaves the module cached; the target's own import then reads nothing.
#:
#: What it does not do.  It exempts no path and weakens no check: ``model_stat`` flags an undeclared stat exactly as
#: before (``tests/test_environment_preload.py`` pins it), and a target that stats the interpreter directory itself is
#: still flagged.  It is a module list, not a path list; add a module (here and in :func:`preload_infrastructure`) only with a measurement like the one above.
INFRASTRUCTURE_MODULES = ("sysconfig", "platform")


def preload_infrastructure() -> tuple[str, ...]:
    """Import :data:`INFRASTRUCTURE_MODULES` now, finish ``sysconfig``'s lazy preparation, return their names.

    Call it in a child BEFORE :func:`install_files` / :func:`confined`, as the Avengers worker does
    (``avengers/_worker.py::_prepare_confinement`` and ``sv/_child.py``). The imports are spelled out, not looked up by name, because the
    core runs no import machinery of its own (``tests/test_layering.py``); the test pins list and imports together.

    ``get_config_vars()`` belongs to the same preparation ([internal ref]): the interpreter's config vars are computed LAZILY,
    on the first ``sysconfig.get_config_var`` call wherever that happens.  With the [internal ref] metadata model in place, a
    target's ``import zoneinfo`` (whose module init asks for ``TZPATH``) would run ``_init_config_vars`` inside the
    window, and its ``_safe_realpath`` walk (``posixpath.realpath``) probes every ancestor of the interpreter prefix
    with ``os.lstat`` -- undeclared paths that now flag.  Forced HERE, before the confinement, that state is already
    computed, exactly as for the imports themselves (measured: ``tests/test_environment_preload.py``).
    """
    import platform  # noqa: F401
    import sysconfig
    sysconfig.get_config_vars()

    return INFRASTRUCTURE_MODULES


#: The stdlib modules whose whole job is turning a path into a canonical one.  ``posixpath.realpath`` walks
#: EVERY component of the path with ``os.lstat``/``os.readlink`` (``pathlib.Path.resolve`` calls it, and the
#: mutation confinement realpaths a mutator's path): once the [internal ref] metadata model patches those, walking a
#: path the model does not know is a run of undeclared probes.
_RESOLUTION_MODULES = frozenset({"posixpath", "ntpath", "pathlib"})

#: The harness's own source tree (``.../src/tharkun``) with its separator: a frame whose file is under it is the
#: harness's own code whatever name it was imported under -- the package modules, the bare-name copies a confined
#: child imports (``environment``, ``extern``), and the files that run as ``__main__`` (``sv/_child.py``,
#: ``avengers/_worker.py``: a script or ``-m`` launch gives them ``__name__ == "__main__"``, so a module-identity
#: check would miss exactly the frames that must be exempted).  A target's files live in its own directory.
_HARNESS_SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _resolution_probe(frame) -> bool:
    """True when ``frame`` (the caller of a patched metadata primitive) is the host-resolution machinery of the
    HARNESS's own code -- ``posixpath.realpath``/``pathlib`` resolution for bookkeeping (a worker resolving the
    program's own path: ``solve/loader.py::file_path_to_dotted_module``; the mutation confinement resolving a
    mutator's path to catch a link out of the scratch).  That walk is not something the target observed: the
    resolution is a mechanical need, like :func:`_import_machinery_read`, and the real functions answer -- the
    resolution must still SEE the host (a symlinked checkout resolves the same way CPython would).  A call the
    TARGET itself made into ``realpath`` (or ``Path.resolve``) has the target's own code as the first
    non-resolution frame and stays modelled: its probes flag exactly as any other undeclared metadata read.
    Identity is the frame's FILE being inside :data:`_HARNESS_SOURCE_ROOT`, never a module name (the children run
    as ``__main__``) -- and a real target's file never lives inside the harness's own source tree."""
    while frame is not None and frame.f_globals.get("__name__") in _RESOLUTION_MODULES:
        frame = frame.f_back
    if frame is None:
        return False
    filename = frame.f_code.co_filename
    if not isinstance(filename, str) or filename.startswith("<"):
        return False
    return os.path.abspath(filename).startswith(_HARNESS_SOURCE_ROOT)


#: The fixed, absent HOME a boxed run gets ([internal ref]): never the random scratch path -- an object built under the
#: box (a class holding its config directory, say) must carry the same path on every host, or the same draw
#: vector would replay to different object state elsewhere.  Set by ``sv/engine.py::run_once``'s child
#: environment and by ``sv/_hybrid_worker.py::_install_environment``; the ``~/...`` form of a
#: :class:`WorldRecord` path is relative to it.
BOX_HOME = "/nonexistent/tharkun-home"

#: The touch record of the installed box, one per process like the model itself (:func:`world_record`).
_BOX_WORLD: list = [None]


class WorldRecord:
    """The undeclared paths a boxed run touched and the answer it got, in touch order ([internal ref]).

    A path under the run's HOME is recorded relative to it (``~/.httpie``): a boxed child's HOME is the fixed
    :data:`BOX_HOME`, so the record is host-independent and can travel in a witness.  The only answer a C6 box
    gives an undeclared path is ``"absent"``; the drawn file contents of [internal ref]'s later work extend
    :meth:`touch`."""

    def __init__(self, home: str | None):
        self.home = home
        self.touches: list = []

    def relative(self, path: str) -> str:
        """``path`` under the run's HOME, as ``~/...``; anything else stays as it is."""
        if self.home and (path == self.home or path.startswith(self.home + os.sep)):
            return "~" + path[len(self.home):]
        return path

    def touch(self, path: str, obj=None) -> None:
        """Record ONE touch of the undeclared ``path`` and the answer the box gave it."""
        self.touches.append([self.relative(path), "absent" if obj is None else (
            obj.record() if hasattr(obj, "record") else dict(obj))])

    def entries(self) -> list:
        """The record as JSON-ready lists, in touch order."""
        return [list(entry) for entry in self.touches]


def world_record() -> "WorldRecord | None":
    """The touch record of the installed box, or ``None`` without one."""
    return _BOX_WORLD[0]


def reset_world() -> None:
    """Start a fresh touch record (the hybrid worker calls this before every input); no-op without a box."""
    record = _BOX_WORLD[0]
    if record is not None:
        record.touches.clear()


def install_files(files: tuple[str, ...] | list[str], scratch: str, flags: set[str], *, box: bool = False,
                  world=None):
    """Install witness-mode open() for target code; return the original open.

    Both builtins.open and io.open are patched so pathlib uses the same state, and so is every module-level alias
    an earlier import bound to a patched primitive (:func:`_rebind_aliases`).
    A declared file is copied before mutation; its real bytes never change.
    ``os.lstat``/``os.readlink``/``os.access``/``os.statvfs`` are modelled exactly as ``os.stat`` is ([internal ref],
    :func:`stat_like`: an undeclared, non-infrastructure path is answered for real only after ``file_unmodelled``
    is recorded, and the harness's own resolution machinery -- :func:`_resolution_probe` -- is not charged to the
    target), and ``posix.open`` -- what ``import posix`` reaches -- is patched with the same ``model_os_open``
    (the import machinery's own ``.pyc`` write exempted, :func:`_import_machinery_read`).

    ``box`` ([internal ref], the hybrid search's mocked world) makes every undeclared path a MOCKED object:
    ``stat``/``lstat``/``readlink``/``access``/``statvfs``/``listdir``/``scandir``/``os.open`` answer
    "absent" for it -- no host call, no ``file_unmodelled`` -- and record the touch in :func:`world_record`
    (a read still raises the same ``FileNotFoundError`` ``open`` already raised for it, and writes stay exactly
    as above).  The import roots (the program's own directory, the interpreter's prefixes, every directory on
    ``sys.path`` -- never ``/`` or an ancestor of HOME or the scratch) answer for real, unflagged, and
    ``tempfile.tempdir`` points into the scratch.  ``box=False`` is the model above, unchanged.
    """
    before = [(owner, name, getattr(owner, name)) for owner, name in _patched_attributes() if hasattr(owner, name)]
    real = builtins.open
    real_stat, real_lstat, real_access = os.stat, os.lstat, os.access
    real_readlink, real_statvfs = getattr(os, "readlink", None), getattr(os, "statvfs", None)
    real_os_open, real_listdir, real_scandir = os.open, os.listdir, os.scandir
    declared = {os.path.abspath(p): os.path.abspath(p) for p in files}
    by_name = {}
    by_dir = {}
    # [internal ref]: files[0] is always the program itself (TaskEnvironment's own convention;
    # see also 'base' just below, already keyed off it). runpy.run_path(program, ...)
    # re-opens the program file internally to load/compile it -- that is this HARNESS's
    # own mechanical need to execute the target, not something the target program did,
    # so a plain read of exactly this path must not count toward file_effect below
    # (model_open), or every run gets flagged regardless of the target's own behavior.
    program_path = os.path.abspath(files[0]) if files else None
    base = os.path.dirname(files[0]) if files else os.getcwd()
    for p in files:
        name = os.path.relpath(p, base)
        by_name[name] = p if name not in by_name else None
        parent = os.path.dirname(name)
        while parent and parent not in {".", ".."}:
            by_dir[parent] = os.path.join(base, parent)
            parent = os.path.dirname(parent)
    by_dir_abs = set(by_dir.values())
    by_dir_abs.add(base)  # the declared files' own directory is known, not unmodelled
    created = set()
    scratch_fds = {}
    root = os.path.join(scratch, "sv-files")
    # [internal ref] Bug C follow-on: a custom PYTHONPYCACHEPREFIX (sys.pycache_prefix) relocates
    # EVERY .pyc cache read -- including runpy.run_path's own pkgutil/stdlib imports,
    # exactly the mechanical-need reads this tuple already exists to exempt -- from
    # alongside their source (under sys.prefix/base_prefix, already covered below) to a
    # mirrored tree under the custom prefix instead. Un-exempted, that shows up as a
    # plain undeclared file read: flags.add("file_unmodelled") on every single run under
    # such a prefix, discarding an otherwise-genuine FALSE (Outcome.violated() refuses a
    # "raised" status when "file_unmodelled" is set) -- confirmed empirically: run-tasks'
    # own campaign wrapper sets PYTHONPYCACHEPREFIX (to keep a shared Python install's
    # cache clean under heavy parallel benchmarking) and reproducibly broke dozens of
    # otherwise-passing tests this way, unrelated to argv specifically. sys.pycache_prefix
    # is None when unset, so this is a no-op everywhere else.
    infrastructure = (os.path.dirname(os.path.dirname(__file__)), os.sys.prefix, os.sys.base_prefix,
                      *((os.sys.pycache_prefix,) if os.sys.pycache_prefix else ()))
    os.makedirs(root, exist_ok=True)
    world_sources = {}
    objects = world.objects if hasattr(world, "objects") else (world or {})
    def world_bytes(obj):
        if hasattr(obj, "concrete"):
            return obj.concrete
        if "content_base64" in obj:
            import base64
            return base64.b64decode(obj["content_base64"], validate=True)
        return obj["content"].encode("utf-8")

    for path, obj in objects.items():
        if (obj.kind if hasattr(obj, "kind") else obj.get("kind", "file")) == "stream":
            continue
        source = os.path.join(root, "world-" + hashlib.sha256(path.encode()).hexdigest())
        data = world_bytes(obj)
        with real(source, "wb") as stream:
            stream.write(data)
        world_sources[source] = (path, obj, data)
        declared[os.path.abspath(os.path.join(base, path))] = source
        if not os.path.isabs(path):
            by_name[os.path.normpath(path)] = source
    box_world = None
    box_roots: tuple[str, ...] = ()
    if box or objects:
        home = os.environ.get("HOME")
        box_world = WorldRecord(os.path.abspath(home) if home else None)
    if box:
        # Every directory a plain ``python program`` could import from: the program's own directory, the
        # interpreter's prefixes, and every sys.path entry.  ``/`` and an ancestor of HOME or of the scratch
        # are dropped: trusting one would make that whole subtree -- the user's home, the run's own temporary
        # tree -- readable-for-real from a mocked start.
        candidates = ([os.path.abspath(base)] if files else []) + [sys.prefix, sys.base_prefix]
        candidates += [os.path.abspath(entry) for entry in sys.path if isinstance(entry, str) and entry]
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if candidate == os.sep or candidate in box_roots:
                continue
            if os.path.commonpath((candidate, scratch)) == candidate:
                continue
            if box_world.home is not None and os.path.commonpath((candidate, box_world.home)) == candidate:
                continue
            box_roots += (candidate,)
        temp = os.path.join(scratch, "tmp")
        os.makedirs(temp, exist_ok=True)
        tempfile.tempdir = temp
    _BOX_WORLD[0] = box_world

    def target_for(slot, key, source):
        if source:
            name = os.path.relpath(source, base)
            if name != ".." and not name.startswith(".." + os.sep):
                return os.path.join(scratch, name)
        elif os.path.commonpath((scratch, key)) == scratch:
            return key
        return os.path.join(root, hashlib.sha256(slot.encode()).hexdigest())

    class ModelFile:
        """Keep the path visible to target code when an absolute path is shadowed."""

        def __init__(self, fh, name):
            self._fh, self._name = fh, name

        @property
        def name(self):
            return self._name

        def __getattr__(self, name):
            if name in {"buffer", "raw", "fileno"}:
                flags.add("file_unmodelled")
            return getattr(self._fh, name)

        def __enter__(self):
            self._fh.__enter__()
            return self

        def __exit__(self, *args):
            return self._fh.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._fh)

    def resolve(file):
        path = os.fspath(file)
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        key = os.path.abspath(path)
        source = declared.get(key)
        if source is None and not os.path.isabs(path) and os.getcwd() == scratch:
            source = by_name.get(os.path.normpath(path))
        return path, key, source

    def reachable(key):
        """A path a stat-like probe answers for real in the box: the model's own paths (the scratch
        bookkeeping, the infrastructure), and any ancestor-or-equal of an import root -- resolving a root
        means walking its ancestors (``posixpath.realpath``), and those exist wherever the interpreter does.
        A box answers everything else "absent"."""
        if any(os.path.commonpath((probe, key)) == probe for probe in (scratch, *infrastructure)):
            return True
        return any(os.path.commonpath((root_, key)) == key for root_ in box_roots)

    def refresh_world(source):
        original_path, obj, previous = world_sources[source]
        data = world_bytes(obj)
        if data != previous:
            with real(source, "wb") as stream:
                stream.write(data)
            created.discard(source)
            world_sources[source] = (original_path, obj, data)
        box_world.touch(original_path, obj)

    def stat_like(real_fn, file, *args, absent=None, **kwargs):
        if isinstance(file, int) or _resolution_probe(sys._getframe(2)):
            return real_fn(file, *args, **kwargs)
        path, key, source = resolve(file)
        if source in world_sources:
            refresh_world(source)
        if source is None:
            directory = key if key in by_dir_abs else (
                by_dir.get(os.path.normpath(path)) if not os.path.isabs(path) else None)
            if directory is not None:
                return real_fn(directory, *args, **kwargs)
            if key in created:
                target = target_for(key, key, None)
                return real_fn(target, *args, **kwargs)
            if box:
                # [internal ref]'s box: an undeclared path is a mocked object -- absent, the host never consulted,
                # no ``file_unmodelled``.  The import roots and their resolution chain answer for real and
                # unflagged (``reachable``).
                if not reachable(key):
                    box_world.touch(key)
                    if absent is not None:
                        return absent
                    raise FileNotFoundError(2, "No such file or directory", path)
            elif all(os.path.commonpath((probe, key)) != probe for probe in (scratch, *infrastructure)):
                # Flag before the real call: an undeclared, non-infrastructure path's
                # outcome (success, or which OSError) depends on this host's filesystem,
                # so a raised error must not slip through unflagged as if it were modelled.
                flags.add("file_unmodelled")
            return real_fn(file, *args, **kwargs)
        target = target_for(source, key, source)
        return real_fn(target if source in created else source, *args, **kwargs)

    def model_stat(file, *args, **kwargs):
        return stat_like(real_stat, file, *args, **kwargs)

    # [internal ref]: the same model for the metadata primitives that used to reach the host unpatched.  A declared file (and
    # the scratch bookkeeping) answers for real; anything undeclared outside the infrastructure sets ``file_unmodelled``
    # BEFORE the host call -- success, and which OSError, are this host's own, exactly as for ``stat``.
    def model_lstat(file, *args, **kwargs):
        return stat_like(real_lstat, file, *args, **kwargs)

    def model_readlink(file, *args, **kwargs):
        return stat_like(real_readlink, file, *args, **kwargs)

    def model_access(file, *args, **kwargs):
        # ``os.access`` ANSWERS for an absent path where the metadata calls raise (in the box and on the host).
        return stat_like(real_access, file, *args, absent=False, **kwargs)

    def model_statvfs(file, *args, **kwargs):
        return stat_like(real_statvfs, file, *args, **kwargs)

    def under_roots(key):
        """A path ``open``/``listdir``/``os.open`` reads for real in the box: the scratch, the infrastructure,
        or an import root itself -- those trees are readable, statable and listable, unflagged."""
        return any(os.path.commonpath((probe, key)) == probe
                   for probe in (scratch, *infrastructure, *box_roots))

    def stage_dirs(target_dir):
        """Create scratch-side ancestor directories for staging, without the
        host's real ancestors (e.g. /tmp) leaking into the target's flags: this
        is our own bookkeeping, never something the target program observed."""
        saved = os.stat
        os.stat = real_stat
        try:
            os.makedirs(target_dir, exist_ok=True)
        finally:
            os.stat = saved

    def model_open(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            flags.add("file_effect")
            if box and file in scratch_fds:
                try:
                    info = os.fstat(file)
                    if (info.st_dev, info.st_ino) == scratch_fds[file]:
                        return real(file, mode, *args, **kwargs)
                except OSError:
                    pass
            flags.add("file_unmodelled")
            raise OSError("file descriptor reads are outside the SV file model")
        path, key, source = resolve(file)
        writing = any(c in mode for c in "wax+")
        if source in world_sources:
            refresh_world(source)
            if writing:
                flags.add("file_unmodelled")  # world facts describe read-only initial content
        # [internal ref] regression, fixed 2026-09-29 ([internal ref], W65's GUUT-harness evidence):
        # a THIRD exempt read, alongside the two below -- CPython's own import machinery
        # reading a module's source off ANY sys.path entry, however the target set it up
        # (including its own runtime sys.path.insert(0, <external dir>), the exact GUUT
        # harness / eregion-sv det_import_boundary_* pattern that regressed here). That
        # read is not a program file read under the I/O model: the static import-boundary
        # resolver (pyresolve.py) already decides what an import MEANS for proofs, so the
        # mechanical read that satisfies it must not be refused as an undeclared file
        # (below) or counted as a target file effect (immediately below), the same as the
        # two exemptions that already existed for this reason.
        import_read = not writing and _import_machinery_read()
        # [internal ref]: exempt two kinds of read that are the HARNESS's own mechanical need, not
        # something the target program did: (a) re-reading the program's own file (files[0]
        # by TaskEnvironment convention) -- runpy.run_path(program, ...) does this internally
        # to load/compile it; (b) anything under the Python installation itself (the existing
        # "infrastructure" roots, already used a few lines below to bypass the sandbox
        # entirely for these paths) -- runpy's own import machinery reads .pyc caches and
        # stdlib modules (e.g. pkgutil) as part of ordinary module loading, unrelated to
        # anything the target source references. Both were always exempt from SANDBOXING
        # (the infrastructure check below already passes them through to the real open());
        # the fix here is narrower -- they must also not count toward file_effect, which
        # otherwise fires on every single run regardless of the target's own behavior.
        exempt = import_read or (key == program_path and not writing) or (
            not writing and any(os.path.commonpath((base, key)) == base for base in infrastructure))
        if not exempt:
            flags.add("file_effect")
        if import_read or (source is None and not writing and any(
                os.path.commonpath((base, key)) == base for base in infrastructure)):
            return real(file, mode, *args, **kwargs)
        slot = source or key
        target = target_for(slot, key, source)
        parent = os.path.dirname(key)
        if key in by_dir_abs or (not os.path.isabs(path) and os.path.normpath(path) in by_dir):
            raise IsADirectoryError(21, "Is a directory", path)
        relative_parent = os.path.normpath(os.path.dirname(path)) if not os.path.isabs(path) else None
        if (parent in declared or (relative_parent in by_name and by_name[relative_parent] is not None)
                or os.path.isfile(parent)):
            raise NotADirectoryError(20, "Not a directory", path)
        if writing and source is None and os.path.commonpath((scratch, key)) != scratch:
            flags.add("file_unmodelled")
        known_dir = by_dir.get(relative_parent) if relative_parent is not None else None
        if writing and not source and not os.path.isdir(parent) and known_dir is None:
            raise FileNotFoundError(2, "No such file or directory", path)
        if slot not in created and source:
            stage_dirs(os.path.dirname(target))
            with real(source, "rb") as src, real(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            created.add(slot)
        if not writing and slot not in created:
            if box:
                box_world.touch(key)          # [internal ref]: the box's answer for the undeclared read (still absent)
            raise FileNotFoundError(2, "undeclared file", path)
        if writing and known_dir is not None:
            stage_dirs(os.path.dirname(target))
        fh = real(file if target == key else target, mode, *args, **kwargs)
        if writing:
            created.add(slot)
        return fh if target == key else ModelFile(fh, path)

    def model_os_open(file, *args, **kwargs):
        if kwargs.get("dir_fd") is not None or isinstance(file, int):
            flags.add("file_unmodelled")
            raise OSError("unmodelled file descriptor")
        # [internal ref]: the import machinery writes ``.pyc`` caches through ``posix.open``
        # (``_bootstrap_external._write_atomic``); that is the harness's own need, exempt
        # exactly as model_open/model FileIO exempt it, so it is not refused or flagged.
        if _import_machinery_read():
            return real_os_open(file, *args, **kwargs)
        _, key, source = resolve(file)
        if source in world_sources:
            original_path, obj, _ = world_sources[source]
            box_world.touch(original_path, obj)
            flags.add("file_unmodelled")
            raise OSError("world file descriptors are not modelled")
        if source or os.path.commonpath((scratch, key)) != scratch:
            # [internal ref]: a WRITE never leaves the scratch, boxed or not -- the one thing the box must not
            # loosen.  A READ of an import root is real; any other undeclared read is the box's mocked
            # "absent" (and today's refusal outside the box).
            opened = (args[0] if args else kwargs.get("flags", 0)) or 0
            writing = bool(opened & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if box and not writing and under_roots(key):
                return real_os_open(file, *args, **kwargs)
            if box and not writing:
                box_world.touch(key)
                raise FileNotFoundError(2, "No such file or directory", os.fsdecode(os.fspath(file)))
            flags.add("file_unmodelled")
            raise OSError("unmodelled os.open path")
        fd = real_os_open(file, *args, **kwargs)
        if box:
            info = os.fstat(fd)
            scratch_fds[fd] = (info.st_dev, info.st_ino)
        return fd

    class ModelFileIO(_io.FileIO):
        def __new__(cls, *args, **kwargs):
            # [internal ref] regression, fixed 2026-09-29: a sourceless/extension loader's
            # get_data(), and the namespace-package __init__ probes, construct
            # _io.FileIO directly instead of going through _io.open -- the same
            # import-machinery exemption as model_open's import_read, for the same
            # reason (see _import_machinery_read's docstring). super().__new__ (not
            # the module-level _io.FileIO, which by now IS this class) reaches the
            # real, unpatched FileIO this class was defined to extend; the caller's
            # own cls(*args, **kwargs) then runs its inherited __init__ normally.
            if _import_machinery_read():
                return super().__new__(cls, *args, **kwargs)
            flags.add("file_unmodelled")
            raise OSError("unmodelled FileIO")

    def model_enumerate(real_fn):
        def run(*args, **kwargs):
            flags.add("file_unmodelled")
            return real_fn(*args, **kwargs)
        return run

    def box_enumerate(real_fn):
        def run(path=None, *args, **kwargs):
            if isinstance(path, int):
                return real_fn(path, *args, **kwargs)
            directory = os.fspath(path) if path is not None else "."
            key = os.path.abspath(os.fsdecode(directory))
            if under_roots(key):
                return real_fn(path, *args, **kwargs) if path is not None else real_fn()
            box_world.touch(key)
            raise FileNotFoundError(2, "No such file or directory", directory)
        return run

    install_mutation_confinement(scratch, flags)
    install_spawn_confinement(flags)
    install_network_confinement(flags)
    builtins.open = io.open = _io.open = model_open
    io.FileIO = _io.FileIO = ModelFileIO
    os.stat = model_stat
    os.lstat = model_lstat
    os.access = model_access
    if real_readlink is not None:
        os.readlink = model_readlink
    if real_statvfs is not None:
        os.statvfs = model_statvfs
    os.open = model_os_open
    try:
        import posix as _posix_io
    except ImportError:               # pragma: no cover -- not a POSIX interpreter
        _posix_io = None
    if _posix_io is not None and getattr(_posix_io, "open", None) is real_os_open:
        _posix_io.open = model_os_open
    if box:
        os.listdir = box_enumerate(real_listdir)
        os.scandir = box_enumerate(real_scandir)
    else:
        os.listdir = model_enumerate(real_listdir)
        os.scandir = model_enumerate(real_scandir)
    _rebind_aliases(((original, getattr(owner, name)) for owner, name, original in before), interpreter_only=True)
    return real


#: the ``os`` functions that change the file system, with the names of the arguments that are PATHS (an ``int``
#: first argument is a file descriptor the program already holds, opened through the model, and passes).  Everything
#: else that mutates -- ``os.makedirs``/``removedirs``/``renames``, ``shutil.rmtree``/``move``/``copy*``/``copytree``/
#: ``chown``, ``pathlib.Path.unlink``/``rename``/``mkdir``/``rmdir``/``touch``/``chmod``/``symlink_to`` and
#: ``tempfile`` -- reaches the disk through these, looked up on the ``os`` module at call time.
_MUTATORS = {
    "remove": ("path",), "unlink": ("path",), "rmdir": ("path",), "mkdir": ("path",), "mkfifo": ("path",),
    "mknod": ("path",), "chmod": ("path",), "lchmod": ("path",), "chown": ("path",), "lchown": ("path",),
    "chflags": ("path",), "lchflags": ("path",), "truncate": ("path",), "utime": ("path",),
    "setxattr": ("path",), "removexattr": ("path",), "chroot": ("path",),
    "rename": ("src", "dst"), "replace": ("src", "dst"), "link": ("src", "dst"),
}


def install_mutation_confinement(scratch: str, flags: set[str]) -> None:
    """Confine every file-system-mutating ``os`` call to the scratch directory.

    ``install_files`` shadows what a program READS and WRITES through ``open`` (a write outside the scratch lands in a
    scratch-side shadow file), but ``os.remove``/``rename``/``mkdir``/``rmdir``/``chmod``/... went straight to the
    host: a replay (or the Atheris worker, which runs the program for real, millions of times) could delete, rename
    or re-permission a file outside the scratch.  A call whose path -- every path, symbolic links resolved -- is
    not under the scratch fails the way a read-only file system does (``OSError(EROFS)``, never a silent success) and
    sets ``file_effect`` and ``file_unmodelled`` (so the run can never be a violation or a TRUE).  A call on paths
    under the scratch runs for real.  Also refused: a ``dir_fd`` argument (a path relative to a descriptor the model
    cannot see), and a symbolic or hard link that would lead out of the scratch (a write through it would escape).
    The functions are replaced on ``os`` and on ``posix`` (what ``import posix`` reaches); CPython's import machinery
    expects ``EROFS`` when it cannot write a ``.pyc``, and ignores it."""
    scratch_real = os.path.realpath(scratch)
    real_realpath = os.path.realpath

    def inside(path) -> bool:
        try:
            p = real_realpath(os.fsdecode(os.fspath(path)))
        except (TypeError, ValueError):
            return False
        return p == scratch_real or p.startswith(scratch_real + os.sep)

    def refuse(name, path, why="outside the SV scratch directory"):
        flags.add("file_effect")
        flags.add("file_unmodelled")
        raise OSError(errno.EROFS, f"{os.strerror(errno.EROFS)} (os.{name}: {why})", os.fsdecode(path)
                      if isinstance(path, (str, bytes, os.PathLike)) else None)

    def confine(real_fn, name, params):
        def run(*args, **kwargs):
            if any(kwargs.get(k) is not None for k in ("dir_fd", "src_dir_fd", "dst_dir_fd")):
                refuse(name, args[0] if args else "", "a path relative to a descriptor is outside the SV file model")
            for i, param in enumerate(params):
                path = args[i] if i < len(args) else kwargs.get(param)
                if path is None or isinstance(path, int):
                    continue                       # a descriptor the program holds, or an omitted optional path
                if not inside(path):
                    refuse(name, path)
            return real_fn(*args, **kwargs)
        run.__name__ = name
        run.__qualname__ = name
        run.__wrapped__ = real_fn
        return run

    def confine_symlink(real_fn):
        def run(src, dst, *args, **kwargs):
            if kwargs.get("dir_fd") is not None:
                refuse("symlink", dst, "a path relative to a descriptor is outside the SV file model")
            if not inside(dst):
                refuse("symlink", dst)
            target = os.fsdecode(os.fspath(src))
            if not inside(target if os.path.isabs(target) else os.path.join(os.path.dirname(os.path.abspath(
                    os.fsdecode(os.fspath(dst)))), target)):
                refuse("symlink", src, "a link out of the SV scratch directory")
            return real_fn(src, dst, *args, **kwargs)
        run.__name__ = run.__qualname__ = "symlink"
        run.__wrapped__ = real_fn
        return run

    try:
        import posix as _posix
    except ImportError:               # pragma: no cover -- not a POSIX interpreter
        _posix = None
    for name, params in _MUTATORS.items():
        real_fn = getattr(os, name, None)
        if real_fn is None:
            continue
        wrapper = confine(real_fn, name, params)
        if _posix is not None and getattr(_posix, name, None) is real_fn:
            setattr(_posix, name, wrapper)
        setattr(os, name, wrapper)
    real_symlink = getattr(os, "symlink", None)
    if real_symlink is not None:
        wrapper = confine_symlink(real_symlink)
        if _posix is not None and getattr(_posix, "symlink", None) is real_symlink:
            setattr(_posix, "symlink", wrapper)
        os.symlink = wrapper



#: The exact list of (module/class path, attribute) pairs patched to confine process spawning.
#: Used by in-process reversibility mechanisms to snapshot and restore them.
SPAWN_CONFINEMENT_TARGETS = (
    ("os", "system"), ("os", "popen"), ("os", "fork"), ("os", "forkpty"),
    ("os", "posix_spawn"), ("os", "posix_spawnp"), ("os", "startfile"),
    ("os", "execl"), ("os", "execle"), ("os", "execlp"), ("os", "execlpe"),
    ("os", "execv"), ("os", "execve"), ("os", "execvp"), ("os", "execvpe"),
    ("os", "spawnl"), ("os", "spawnle"), ("os", "spawnlp"), ("os", "spawnlpe"),
    ("os", "spawnv"), ("os", "spawnve"), ("os", "spawnvp"), ("os", "spawnvpe"),
    ("posix", "system"), ("posix", "popen"), ("posix", "fork"), ("posix", "forkpty"),
    ("posix", "posix_spawn"), ("posix", "posix_spawnp"),
    ("posix", "execl"), ("posix", "execle"), ("posix", "execlp"), ("posix", "execlpe"),
    ("posix", "execv"), ("posix", "execve"), ("posix", "execvp"), ("posix", "execvpe"),
    ("posix", "spawnl"), ("posix", "spawnle"), ("posix", "spawnlp"), ("posix", "spawnlpe"),
    ("posix", "spawnv"), ("posix", "spawnve"), ("posix", "spawnvp"), ("posix", "spawnvpe"),
    ("subprocess.Popen", "_execute_child"),
    ("_posixsubprocess", "fork_exec"),
    ("_winapi", "CreateProcess"),
)

def _spawn_attributes():
    """``(owner path, owner, attribute)`` for every :data:`SPAWN_CONFINEMENT_TARGETS` entry that exists now.

    ``posix``, ``_posixsubprocess`` and ``_winapi`` are imported first, so both the installer and the snapshot of
    :func:`confined` name what the installer is about to patch.  ``subprocess`` is named only when it is already
    imported (a program imports it itself, after the patch, in most real cases; its ``Popen._execute_child`` is
    then reached through ``_posixsubprocess.fork_exec``, which is patched either way)."""
    try:
        import posix  # noqa: F401
    except ImportError:
        pass
    try:
        import _posixsubprocess  # noqa: F401
    except ImportError:
        pass
    try:
        import _winapi  # noqa: F401
    except ImportError:
        pass
    found = []
    for owner_path, attr_name in SPAWN_CONFINEMENT_TARGETS:
        parts = owner_path.split(".")
        owner = sys.modules.get(parts[0])
        for part in parts[1:]:
            owner = getattr(owner, part, None)
        if owner is not None and callable(getattr(owner, attr_name, None)):
            found.append((owner_path, owner, attr_name))
    return found


def install_spawn_confinement(flags: set[str]) -> None:
    """Confine process spawning by refusing it.

    Process spawning is completely confined inside the program under test:
    every call raises OSError(EPERM) and records the same file effect flags
    as file-system mutators, preventing a program under test from running
    arbitrary commands on the host.
    """
    import errno

    def refuse(name):
        def run(*args, **kwargs):
            flags.add("file_effect")
            flags.add("file_unmodelled")
            raise OSError(errno.EPERM, f"Operation not permitted (process spawning is confined: {name})")
        run.__name__ = name.split(".")[-1]
        run.__qualname__ = run.__name__
        return run

    for owner_path, owner, attr_name in _spawn_attributes():
        setattr(owner, attr_name, refuse(f"{owner_path}.{attr_name}"))


# Audit hooks cannot be removed: keep one dormant hook and restore only its active context.
_NETWORK = {"flags": None, "installed": False}


def install_network_confinement(flags: set[str]) -> None:
    """Refuse CPython socket events before host access, including captured C aliases."""
    if not _NETWORK["installed"]:
        def guard(event, args):
            active = _NETWORK["flags"]
            if active is not None and event.startswith("socket."):
                active.update(("file_effect", "file_unmodelled"))
                raise OSError(errno.EPERM, f"network access is confined: {event}")
        sys.addaudithook(guard)
        _NETWORK["installed"] = True
    _NETWORK["flags"] = flags


def _restore(snapshot) -> None:
    """Put back every attribute in ``snapshot``, then every module-level alias of what stood in its place: the ones
    :func:`_rebind_aliases` made at install, and the ones a module first imported INSIDE the window bound itself
    (``subprocess`` binds ``_posixsubprocess.fork_exec`` as ``_fork_exec``, ``tokenize`` binds ``open``), which
    would otherwise keep refusing or modelling after the window, against a removed scratch ([internal ref])."""
    installed = [(getattr(mod, name, original), original) for mod, name, original in snapshot]
    for mod, name, original in snapshot:
        setattr(mod, name, original)
    _rebind_aliases(installed)


@dataclass(frozen=True)
class Confinement:
    """What :func:`confined` yields: the scratch directory the model runs in, the flags it fills (``file_effect`` /
    ``file_unmodelled`` mean the confined code touched the environment), the declared files, and ([internal ref]) whether
    the block runs in the box (``box``) with the run's touch record (``world``, ``None`` without one)."""

    scratch: str
    flags: set[str]
    files: tuple[str, ...]
    box: bool = False
    world: "WorldRecord | None" = None


#: every attribute ``install_files``, ``install_mutation_confinement`` and ``install_spawn_confinement`` rewrite,
#: as ``(owner, name)``
def _patched_attributes():
    try:
        import posix as _posix
    except ImportError:               # pragma: no cover -- not a POSIX interpreter
        _posix = None
    attrs = [(builtins, "open"), (io, "open"), (_io, "open"), (io, "FileIO"), (_io, "FileIO"),
             (os, "stat"), (os, "lstat"), (os, "readlink"), (os, "access"), (os, "statvfs"),
             (os, "open"), (os, "listdir"), (os, "scandir")]
    if _posix is not None:
        attrs.append((_posix, "open"))
    for name in (*_MUTATORS, "symlink"):
        attrs.append((os, name))
        if _posix is not None:
            attrs.append((_posix, name))
    attrs.extend((owner, name) for _path, owner, name in _spawn_attributes())
    return attrs


THREAD_STARTERS = ("start_new_thread", "start_new", "start_joinable_thread",
                   "_start_joinable_thread", "_start_new_thread")


def thread_starters():
    """Native thread entry points and the aliases captured by threading."""
    import _thread

    return [(threading.Thread, "start"), *((module, name)
            for module in (_thread, threading) for name in THREAD_STARTERS
            if callable(getattr(module, name, None)))]


def refuse_threads(flags):
    """Refuse target threads that could outlive a reversible execution phase."""
    import errno

    saved = [(owner, name, getattr(owner, name)) for owner, name in thread_starters()]

    def refused(*args, **kwargs):
        flags.update(("stack_unmodelled", "file_effect", "file_unmodelled"))
        raise OSError(errno.EPERM, "target thread creation is confined")

    def restore():
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)

    try:
        for owner, name, _original in saved:
            setattr(owner, name, refused)
    except BaseException:
        restore()
        raise
    return restore


#: The temporary directory, resolved when this module is imported: ``tempfile.gettempdir()`` and ``mkdtemp`` create a
#: ``random.Random()`` the first time a process uses them, and inside a run whose ``random`` module is instrumented
#: (sv/_child.py::_patch_random) that unseeded generator is reported as the TARGET's nondeterminism
#: (``unseeded_random_unmodelled``). The scratch directory is therefore named from ``os.urandom`` below.
_TEMP_BASE = tempfile.gettempdir()


def _fresh_scratch() -> str:
    """A new private directory under the temporary directory, without touching the ``random`` module."""
    while True:
        path = os.path.join(_TEMP_BASE, "tharkun-confined-" + os.urandom(8).hex())
        try:
            os.mkdir(path, 0o700)
            return path
        except FileExistsError:
            continue


_CONFINED_LOCK = threading.RLock()
_CONFINED: dict = {"depth": 0}


@contextlib.contextmanager
def confined(files: tuple[str, ...] | list[str] = (), scratch: str | None = None, flags: set[str] | None = None,
             *, threads: bool = True, box: bool = False, world=None):
    """Run the block under the environment model and file-system confinement, then put everything back.

    The REVERSIBLE form of what ``install_files`` (and, through it, ``install_mutation_confinement``) do for the
    whole life of a child or worker process: declared ``files`` are the only readable files, a write outside the
    scratch lands in a scratch-side copy, and every ``os`` call that mutates the file system (``remove``,
    ``rename``, ``mkdir``, ``chmod``, ``truncate``, ``symlink`` ..., and everything built on them:
    ``shutil.rmtree``, ``pathlib.Path.unlink`` ...) outside the scratch raises ``OSError(EROFS)``.  ``files``
    is the program's declared files, the program first (``()`` is fine: the base is then the cwd); ``scratch``
    defaults to a fresh temporary directory this context removes on exit (a scratch you pass is never removed);
    ``flags`` defaults to a new set you read afterwards.  Yields a :class:`Confinement`.  The working directory is
    the scratch while the block runs (the model resolves relative paths and declared files against it) and is the
    previous one afterwards, also when the block raises.

    The installer is the unchanged ``install_files``: this adds a snapshot of every attribute it rewrites
    (``builtins.open``, ``io.open``, ``_io.open``, ``io.FileIO``, ``_io.FileIO``, ``os.stat/open/listdir/scandir``
    and, for every mutator and ``symlink``, ``os.<name>`` AND ``posix.<name>``), taken before installing and put
    back in ``finally`` together with every module-level alias of what was installed (:func:`_restore`).  Re-entrant and thread-safe by a depth counter: a nested ``with`` (same thread or
    another) shares the outermost context's scratch, flags and Confinement without re-snapshotting, and the LAST
    exit restores -- an already-patched attribute is never snapshotted.  A nested context's ``flags`` argument is
    updated with the model's flags when it exits.

    PROCESS-GLOBAL, by construction (``builtins`` and ``os`` are shared, and so is the working directory): while
    any context is active EVERY thread is confined and sees the scratch as its cwd, including whatever else runs
    in this process.  The caller must therefore run the block with NO concurrent Tharkun work in the process (a
    replay of the program under inversion is such a block), and Tharkun's own writes -- a witness, a log, an output
    file -- must use absolute paths captured BEFORE entering, never a path relative to the cwd: a write made
    after the exit goes to the original cwd, and a write from another thread during the window is confined, not
    misplaced (``tests/test_sv_confined.py`` pins both).  Process spawning
    (``os.system``, ``subprocess``, ``os.exec*``, ``os.fork``, ``multiprocessing``) is refused inside the block with
    ``OSError(EPERM)`` (:func:`install_spawn_confinement`) and its attributes are in :func:`_patched_attributes`, so
    the exit puts them back like the rest. ``threads=False`` also refuses new target threads,
    so no background execution can outlive the protected phase; the generic default is unchanged.

    ``box`` ([internal ref], the hybrid search's mocked world) is :func:`install_files`' mode switch: an undeclared path
    answers "absent" without a host call and without ``file_unmodelled``, the import roots answer for real
    (unflagged), and ``tempfile.tempdir`` points into the scratch (like every other attribute, restored on
    exit).  The OUTERMOST context decides; a nested one shares it, and the yielded :class:`Confinement` carries
    ``box`` and the run's :func:`world_record`."""
    declared = tuple(os.fspath(f) for f in files)
    with _CONFINED_LOCK:
        state = _CONFINED
        if state["depth"] == 0:
            owns_scratch = scratch is None
            work = os.path.realpath(_fresh_scratch() if scratch is None else scratch)
            os.makedirs(work, exist_ok=True)
            model_flags: set[str] = set()
            snapshot = [(mod, name, getattr(mod, name)) for mod, name in _patched_attributes() if hasattr(mod, name)]
            try:
                previous_cwd = os.getcwd()
            except OSError:
                previous_cwd = None
            saved_tempdir = tempfile.tempdir
            saved_network_flags = _NETWORK["flags"]
            try:
                install_files(declared, work, model_flags, box=box, **({"world": world} if world is not None else {}))
                os.chdir(work)
                restore_threads = refuse_threads(model_flags) if not threads else None
            except BaseException:
                _NETWORK["flags"] = saved_network_flags
                _restore(snapshot)
                _BOX_WORLD[0] = None
                tempfile.tempdir = saved_tempdir
                if previous_cwd is not None:
                    try:
                        os.chdir(previous_cwd)
                    except OSError:
                        pass
                if owns_scratch:
                    shutil.rmtree(work, ignore_errors=True)
                raise
            state.update(snapshot=snapshot, scratch=work, owns=owns_scratch, flags=model_flags,
                         cwd=previous_cwd, files=declared, thread_refusals=int(not threads),
                         box=box, world=world_record(), tempdir=saved_tempdir,
                         network_flags=saved_network_flags)
            if restore_threads is not None:
                state["restore_threads"] = restore_threads
        elif not threads:
            if not state.get("thread_refusals", 0):
                state["restore_threads"] = refuse_threads(state["flags"])
            state["thread_refusals"] = state.get("thread_refusals", 0) + 1
        state["depth"] += 1
        current = Confinement(state["scratch"], state["flags"], state["files"], state["box"], state["world"])
    try:
        yield current
    finally:
        with _CONFINED_LOCK:
            if not threads:
                state["thread_refusals"] -= 1
                if not state["thread_refusals"]:
                    state.pop("restore_threads")()
            state["depth"] -= 1
            if flags is not None:
                flags.update(state["flags"])
            if state["depth"] == 0:
                _NETWORK["flags"] = state["network_flags"]
                try:
                    _restore(state["snapshot"])
                finally:
                    _BOX_WORLD[0] = None
                    tempfile.tempdir = state["tempdir"]
                    try:
                        if state["cwd"] is not None:
                            os.chdir(state["cwd"])
                    except OSError:
                        pass
                    if state["owns"]:
                        shutil.rmtree(state["scratch"], ignore_errors=True)
                    state.clear()
                    state["depth"] = 0


def references_stdin(source: bytes | str) -> bool:
    """Static, conservative check: does ``source`` reference ``sys.stdin``/
    ``sys.__stdin__`` or call ``input()`` anywhere in the file -- reachable or
    not?  Used to flag stdin as touched (the same "forbids a TRUE verdict"
    signal as ``time``/``unseeded_random_unmodelled``) WITHOUT ever
    substituting a stand-in object for the real ``sys.stdin`` at runtime
    ([internal ref], owner ruling: "Tharkun keeps the REAL sys.stdin: no stand-in
    object, ever" -- a prior dynamic wrapper broke ``sys.stdin.mode``,
    ``sys.stdin is sys.__stdin__`` and ``type(sys.stdin)`` for the target
    program, producing a wrong FALSE on any task that checks any of those).
    Over-approximate on purpose: flagging code that is never reached costs
    only precision (an avoidable UNKNOWN), never soundness, and this must
    run once, before the target executes, when nothing is known yet about
    which lines will run.
    """
    import ast
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"stdin", "__stdin__"}:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input":
            return True
    return False


#: names through which a program reads the Python stack ABOVE its own module frame ([internal ref]): a ``sys._getframe()`` walk, ``f_back``,
#: ``inspect.currentframe``/``getouterframes``, ``traceback.extract_stack``/``format_stack``/``print_stack``/``walk_stack``
_STACK_NAMES = frozenset({"_getframe", "_current_frames", "currentframe", "f_back", "extract_stack", "format_stack",
                          "print_stack", "walk_stack", "getouterframes", "getinnerframes", "getframeinfo"})
#: modules that run code on a stack of their own (a THREAD starts at depth 0, so the recursion limit the child raises for
#: the main thread gives it ~10 extra frames: ``_child.py::run_as_main``) or read frames for a living ([internal ref])
_STACK_MODULES = frozenset({"threading", "_thread", "concurrent", "multiprocessing", "asyncio", "socketserver", "pdb", "bdb",
                            "trace", "cProfile", "profile", "faulthandler", "tracemalloc", "code"})
#: the thread-running ones, whose name as a string (``__import__("thread" + "ing")``) is enough
_THREAD_MODULES = frozenset({"threading", "_thread", "concurrent", "multiprocessing", "asyncio", "socketserver"})
#: module objects whose attributes reach the interpreter's frames: a computed ``getattr`` on one of them, or any computed
#: ``getattr`` in a program that imports one, may be a frame walk the names above never show
_REFLECTED_MODULES = frozenset({"sys", "inspect", "traceback", "gc", "types", "builtins"})


def _fold_str(node):
    """The ``str`` value of a constant expression built from literals (``"_get" + "frame"``, an f-string of literals,
    ``"".join(["_g", "etframe"])``), or ``None``."""
    import ast
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold_str(node.left), _fold_str(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            part = _fold_str(v.value if isinstance(v, ast.FormattedValue) and v.conversion == -1
                             and v.format_spec is None else v)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join"
            and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)
            and len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple))):
        parts = [_fold_str(e) for e in node.args[0].elts]
        return None if any(p is None for p in parts) else node.func.value.value.join(parts)
    return None


#: what turns a NAME into an attribute, a global or a module, with the position of the name (``None``: every argument)
_NAME_RESOLVERS = {"getattr": 1, "hasattr": 1, "getattr_static": 1, "find_class": 1, "attrgetter": None,
                   "methodcaller": 0, "__import__": 0, "import_module": 0, "resolve_name": 0, "locate": 0}
#: ``x.__getattribute__(name)`` / ``object.__getattribute__(x, name)``: the name is the last argument
_GETATTRIBUTE = frozenset({"__getattribute__", "__getattr__"})
#: what runs a string as code
_CODE_RUNNERS = frozenset({"eval", "exec", "compile"})
#: library calls that run a string argument as Python (``timeit.timeit("...")``, ``pdb.run``, ``code.interact``, ...)
_CODE_CALLS = frozenset({"timeit", "repeat", "Timer", "run", "runctx", "runcall", "runcode", "runsource", "run_path",
                         "run_module", "interact", "run_docstring_examples"})
#: library calls that turn string or bytes data into a global lookup (``pickle.loads(b"csys\\norig_argv\\n.")``)
_NAME_DATA_CALLS = frozenset({"loads", "load", "dictConfig", "fileConfig"})
#: what reaches a module's attributes, or any object, without naming them
_ENUMERATORS = frozenset({"vars", "__dict__", "getmembers", "getmembers_static", "get_objects", "get_referents",
                          "get_referrers", "render_doc", "pydoc"})
#: the facilities a string may name to reach any of the above (``getattr(builtins, "getattr")``)
_FORMAT_METHODS = frozenset({"format", "format_map", "vformat"})
_REFLECTIVE = frozenset(_NAME_RESOLVERS) | _GETATTRIBUTE | _CODE_RUNNERS | _ENUMERATORS | _FORMAT_METHODS


def reflective_field_attribute(name):
    """An attribute selector that can escape into unmodeled interpreter/provider state."""
    return name.startswith("__") or name in _REFLECTIVE | _STACK_NAMES | {
        "orig_argv", "f_globals", "f_locals", "gi_frame", "cr_frame", "ag_frame"}


def _format_selector_reason(kind, text, node):
    if kind == "opaque-field":
        return f"format fields the scan cannot read at line {node.lineno}"
    if kind == "field":
        from _string import formatter_field_name_split
        try:
            _, accesses = formatter_field_name_split(text)
            for attribute, name in accesses:
                if attribute and reflective_field_attribute(name):
                    return f"reflective format field {name} at line {node.lineno}"
        except ValueError:
            return f"format field the scan cannot read at line {node.lineno}"
    return None


def format_field_reason(tree, *, filename=None, observed=None, runtime=False):
    """Check active format selectors in this file, including escaped formatter methods."""
    for kind, text, node in _live_strings(tree, filename=filename, observed=observed, runtime=runtime):
        if reason := _format_selector_reason(kind, text, node):
            return reason
    return None


def _str_text(node):
    """The text of a constant ``str`` expression (:func:`_fold_str`) or of a ``bytes`` literal, or ``None``."""
    import ast
    if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
        return node.value.decode("latin-1")
    return _fold_str(node) if isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr, ast.Call)) else None


def _live_strings(tree, *, filename=None, observed=None, runtime=False) -> list:
    """The strings ``tree`` EXECUTES or uses as a NAME, as ``(kind, text, node)`` (owner, 2026-10-29 audit: only the code
    may change a verdict, never a comment or plain string data):

    * ``"code"``: run as Python -- an ``eval`` / ``exec`` / builtin ``compile`` argument, a string annotation (evaluated
      by ``typing.get_type_hints``), a library runner's argument (:data:`_CODE_CALLS`: ``timeit.timeit("...")``, ...);
    * ``"name"``: used as an attribute, module or global name -- a resolver's name argument (:data:`_NAME_RESOLVERS`,
      ``__getattribute__``), a ``__match_args__`` entry, or
      ``pickle.loads``-style data (:data:`_NAME_DATA_CALLS`).
    * ``"field"``: an active ``format`` / ``format_map`` / ``vformat`` field selector, without literal or escaped text;
      ``"opaque-field"``: a template whose selectors cannot be read statically.

    Every other string (data, a message, a docstring) is neither.  A program that imports ``doctest`` runs its
    docstrings, so there every string is a ``"name"``."""
    import ast
    nodes = list(ast.walk(tree))
    parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    str_shadowed = any((isinstance(n, ast.Name) and n.id == 'str' and isinstance(n.ctx, (ast.Store, ast.Del)))
                       or (isinstance(n, ast.arg) and n.arg == 'str')
                       or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == 'str')
                       or (isinstance(n, ast.alias) and (n.asname or n.name.split('.')[0]) == 'str') for n in nodes)
    if any((isinstance(n, ast.Import) and any(a.name.split(".")[0] == "doctest" for a in n.names)) or
           (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "doctest") for n in nodes):
        return [("name", t, n) for n in nodes if (t := _str_text(n)) is not None]
    out = []
    bindings = None

    def templates(node):
        # These are alternatives, not reaching-value proofs; runtime reads still check the actual receiver.
        nonlocal bindings
        if bindings is None:
            bindings = {}
            for n in nodes:
                if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                    parent = parents.get(n)
                    value = parent.value if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) else None
                    if isinstance(parent, ast.AnnAssign) and value is None:
                        continue
                    bindings.setdefault(n.id, []).append(value)
                elif isinstance(n, (ast.arg, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)):
                    name = n.arg if isinstance(n, ast.arg) else n.name
                    bindings.setdefault(name, []).append(None)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    for alias in n.names:
                        bindings.setdefault(alias.asname or alias.name.split('.')[0], []).append(None)
        pending, visited, texts = [node], set(), set()
        while pending:
            value = pending.pop()
            text = _str_text(value)
            if text is not None:
                texts.add(text)
            elif isinstance(value, ast.IfExp):
                pending.extend((value.body, value.orelse))
            elif isinstance(value, ast.Name):
                if value.id in visited:
                    continue
                values = bindings.get(value.id)
                if not values or any(v is None for v in values):
                    return None
                visited.add(value.id)
                pending.extend(values)
            else:
                return None
        return texts or None

    def add(kind, node):
        text = _str_text(node) if node is not None else None
        if text is not None:
            out.append((kind, text, node))

    def receiver_path(node):
        """``("self", "text")`` for ``self.text``, ``("t",)`` for ``t``; ``None`` for any other expression."""
        names = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        return (node.id, *reversed(names)) if isinstance(node, ast.Name) else None
    def fields(node, access=None):
        from string import Formatter
        text = _str_text(node)
        texts = {text} if text is not None else templates(node)
        guarded = isinstance(node, ast.Name) and access is not None
        path = receiver_path(node) if access is not None else None
        if texts is None and runtime and path and observed is not None and filename is not None:
            # An open template (``self.text.format(...)``, a name bound elsewhere): the child reads the ACTUAL string
            # before the call and flags a reflective or unparsable one (``_child.py::_mapping_observer``).  The static
            # tier still names it (below), so a proof stays withheld: any KeyError/IndexError/ValueError/AttributeError
            # the template can raise is a FALSE-search matter only, never a TRUE.
            observed[(os.path.realpath(filename), access.lineno, access.end_lineno,
                      access.col_offset, access.end_col_offset)] = (path, access.attr, 'format', None)
            return
        if texts is None or (runtime and text is None and (not guarded or observed is None or filename is None)):
            out.append(("opaque-field", "", node))
            return
        if guarded and observed is not None and filename is not None:
            observed[(os.path.realpath(filename), access.lineno, access.end_lineno,
                      access.col_offset, access.end_col_offset)] = (node.id, access.attr, 'format', tuple(sorted(texts)))
        pending = list(texts)
        try:
            while pending:
                for _, field, spec, _ in Formatter().parse(pending.pop()):
                    if field is not None:
                        out.append(("field", field, node))
                        if spec:
                            pending.append(spec)
        except ValueError:
            out.append(("opaque-field", "", node))

    for node in nodes:
        if isinstance(node, ast.Call):
            f, args = node.func, node.args
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in _CODE_RUNNERS and args and (name != "compile" or isinstance(f, ast.Name) or (
                    isinstance(f.value, ast.Name) and f.value.id in ("builtins", "__builtins__"))):
                add("code", args[0])
            elif name in _CODE_CALLS:
                for a in [*args, *(k.value for k in node.keywords)]:
                    add("code", a)
            elif name in _NAME_DATA_CALLS:
                for a in args:
                    add("name", a)
            elif name in _NAME_RESOLVERS or name in _GETATTRIBUTE:
                at = len(args) - 1 if name in _GETATTRIBUTE else _NAME_RESOLVERS[name]
                for a in (args if at is None else args[at:at + 1]):
                    add("name", a)
            elif name in ("format", "format_map") and isinstance(f, ast.Attribute):
                unbound = isinstance(f.value, ast.Name) and f.value.id == "str" and not str_shadowed and args
                fields(args[0] if unbound else f.value, None if unbound else f)
            elif name == "vformat" and args:
                fields(args[0])
        elif isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation is not None:
            for c in ast.walk(node.annotation):
                add("code", c)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            for c in ast.walk(node.returns):
                add("code", c)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, (ast.Tuple, ast.List)) and any(
                isinstance(t, ast.Name) and t.id == "__match_args__"
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target])):
            for e in node.value.elts:
                add("name", e)
        if isinstance(node, ast.Attribute) and node.attr in _FORMAT_METHODS:
            call = parents.get(node)
            if not (isinstance(call, ast.Call) and call.func is node):
                fields(node.value if node.attr != "vformat" else node, node if node.attr != 'vformat' else None)
    return out


def _observed_code(call, filename, observed):
    """Sources whose actual builtin/argument the child checks immediately before this CALL."""
    if observed is None or filename is None:
        return ()
    entry = observed.get((os.path.realpath(filename), call.lineno, call.end_lineno,
                          call.col_offset, call.end_col_offset))
    if type(entry) is tuple and len(entry) == 4 and entry[2] == "code":
        return tuple(item[0] if type(item) is tuple else item for item in entry[3])
    return ()


def stack_reason(source: bytes | str, *, filename=None, observed=None) -> "str | None":
    """Why ``source`` may observe a stack the child does not give it, or ``None``: it reads the frames above its module
    frame (``sys._getframe``, ``f_back``, ``inspect.stack``, ``traceback.extract_stack``, ...) however the CODE spells the
    name (an attribute, an import, a string it runs or uses as a name -- :func:`_live_strings`, so ``getattr(sys, "_get" +
    "frame")`` but never a comment or the data ``'no _getframe here'`` -- a computed ``getattr`` next to an import of
    ``sys`` / ``inspect`` / ``traceback``, ``exec`` / ``eval`` of anything but a literal, which is scanned as code), or it runs
    code on a thread (``threading``, ``_thread``, ``concurrent``, ``multiprocessing``, ``asyncio``, ...: a thread starts at
    recursion depth 0, the main thread's limit is raised by the child's frame count).

    Static and over-approximate on purpose, like :func:`references_stdin`: a flagged program that never looks costs only
    precision.  ``python program.py`` runs the module as the BOTTOM frame; the child's sits on top of its own frames, which
    ``_child.py::run_as_main`` shifts out of the recursion limit but cannot hide from ``inspect.stack()``."""
    import ast
    import re
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    imported: set = set()
    inspect_names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                imported.add(top)
                if top == "inspect":
                    inspect_names.add(a.asname or "inspect")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            imported.add(top)
            if top == "inspect":
                inspect_names.add("inspect")
        elif isinstance(node, ast.Call):
            # ``__import__("in" + "spect")`` / ``importlib.import_module("threading")``: the module a literal names
            f = node.func
            if (isinstance(f, ast.Name) and f.id == "__import__") or (
                    isinstance(f, ast.Attribute) and f.attr == "import_module"):
                named = _fold_str(node.args[0]) if node.args else None
                if named is not None:
                    imported.add(named.split(".")[0])
                    if named.split(".")[0] == "inspect":
                        inspect_names.add("inspect")
    for mod in sorted(imported & _STACK_MODULES):
        return f"imports {mod}: code on another thread's stack, or a reader of frames"
    word = re.compile(r"\b(" + "|".join(sorted(_STACK_NAMES)) + r")\b|\binspect\s*\.\s*stack\b|\bfrom\s+inspect\s+import\b[^\n]*\bstack\b")
    reflects = bool(imported & _REFLECTED_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in _STACK_NAMES:
                return f"reads .{node.attr}"
            if node.attr == "stack" and inspect_names:
                return "reads inspect.stack"
            if node.attr == "__dict__" and isinstance(node.value, ast.Name) and node.value.id in _REFLECTED_MODULES:
                return f"reads {node.value.id}.__dict__"
        elif isinstance(node, ast.Name):
            if node.id in _STACK_NAMES:
                return f"names {node.id}"
        elif isinstance(node, ast.alias):
            if node.name in _STACK_NAMES or (node.name in ("stack", "*") and "inspect" in imported):
                return f"imports {node.name}"
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            builtin_call = isinstance(f, ast.Name) or (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                                                       and f.value.id == "builtins")
            if builtin_call and name in ("exec", "eval", "compile") \
                    and not (node.args and _fold_str(node.args[0]) is not None):
                texts = _observed_code(node, filename, observed)
                if not texts:
                    return f"{name} of code the scan cannot read"
                for text in texts:
                    if inner := stack_reason(text):
                        return f"{name} of guarded code: {inner}"
            if name in ("__import__", "import_module") and not (node.args and _fold_str(node.args[0]) is not None):
                return f"{name} of a module the scan cannot name"
            if name == "vars" and len(node.args) == 1 and isinstance(node.args[0], ast.Name) \
                    and node.args[0].id in _REFLECTED_MODULES:
                return f"vars({node.args[0].id})"
            if name in ("getattr", "__getattribute__", "setattr", "delattr") and reflects:
                idx = 1 if name != "__getattribute__" or isinstance(f, ast.Name) else 0
                if len(node.args) <= idx or _fold_str(node.args[idx]) is None:
                    return f"{name} with a computed name next to an import of {sorted(imported & _REFLECTED_MODULES)[0]}"
    # a string counts only where it runs or is used as a name (owner, 2026-10-29 audit): data never changes a verdict
    for kind, text, node in _live_strings(tree, filename=filename, observed=observed, runtime=observed is not None):
        if kind == "opaque-field":
            return f"format fields the scan cannot read at line {node.lineno}"
        if kind == "code":
            inner = stack_reason(text)
            if inner:
                return f"code in a string at line {node.lineno} {inner}"
        elif word.search(text):
            return f"a name that names a frame reader: {text[:40]!r}"
        elif text in _THREAD_MODULES or (text == "stack" and "inspect" in imported):
            return f"a name that names {text}"
    return None


def references_stack(source: bytes | str, *, filename=None, observed=None) -> bool:
    """``stack_reason(source) is not None``."""
    return stack_reason(source, filename=filename, observed=observed) is not None


#: ``sys.orig_argv`` ([internal ref]): interpreter metadata the competition's model does not define (:func:`orig_argv_reason`)
_ORIG_ARGV = "orig_argv"


def orig_argv_reason(source, filename: str | None = None, *, observed: dict | None = None,
                     _depth: int = 0) -> "str | None":
    """Why the CODE of ``source`` may read ``sys.orig_argv``, or ``None`` ([internal ref]; the 2026-10-29 audit: comments and
    string data never count -- ``ast``, and :func:`_live_strings`).

    The competition's model (sv-benchmarks ``python/CONVENTIONS.md``) is "``sys.argv`` is assumed to be any list of
    strings", set IN-PROCESS; it does not define ``sys.orig_argv``.  In the child that is the verifier's own launch line
    (and a forked worker's differs from a fresh child's), so a FALSE that depends on it cannot be confirmed under the
    competition's semantics: ANY read makes the run no witness (the ``orig_argv`` flag,
    ``engine.NEVER_A_WITNESS_FLAGS``) and blocks a static TRUE (:func:`proof_blocker`).

    It counts: the name in code (``x.orig_argv`` on any receiver, a name bound by ``from sys import orig_argv [as y]``,
    ``getattr(x, "orig_argv")``), ``from sys import *``, a ``case C(orig_argv=a)`` pattern; reflection it cannot
    follow -- a resolver (``getattr``, ``__getattribute__``, ``attrgetter``, ``__import__``, ``import_module``, ...) given
    a computed name, used other than by a direct call or imported under another name, ``vars(x)``, ``x.__dict__``,
    ``inspect.getmembers``, ``gc.get_objects``, ``pydoc``, ``eval`` / ``exec`` / ``compile`` of a non-literal (a literal
    is scanned as code), a non-literal ``__match_args__``; and a string the program runs or uses as a name that mentions
    ``orig_argv`` or ``__dict__`` or names a reflective facility.

    ``observed`` (the child, with the program's ``filename``): a read the child checks by OBJECT right before it runs
    (``_child.py::_mapping_observer``) is recorded there, keyed like ``extern.py::native_import_reason``'s mapping reads,
    instead of counting: ``name.__dict__``, ``vars(name)`` and ``getattr(name, "__dict__")`` (a module receiver, ``sys``
    among them, is no witness; a class's mapping keeps its FALSE, R1) and ``getattr(name, <computed>)`` (``sys`` itself
    is no witness).  A program that does not parse counts (it cannot run either)."""
    import ast
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return "unreadable program" if _depth == 0 else None
    nodes = list(ast.walk(tree))
    parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    observed = observed if _depth == 0 and filename is not None else None
    aliases = {a.asname or a.name for n in nodes if isinstance(n, ast.ImportFrom) for a in n.names
               if a.name == _ORIG_ARGV}
    handled = set()

    def observe(node, entry):
        for path in {filename, os.path.abspath(filename), os.path.realpath(filename)}:
            observed[(path, node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)] = entry

    for node in nodes:
        line = getattr(node, "lineno", None) or getattr(parents.get(node), "lineno", "?")
        if ((isinstance(node, ast.Attribute) and node.attr == _ORIG_ARGV) or
                (isinstance(node, ast.Name) and node.id in aliases)):
            if isinstance(node.ctx, ast.Load):
                return f"reads orig_argv at line {line}"
            continue
        if isinstance(node, ast.alias) and node.name == _ORIG_ARGV:
            return f"imports orig_argv at line {line}"
        if isinstance(node, ast.MatchClass) and _ORIG_ARGV in node.kwd_attrs:
            return f"a class pattern reads orig_argv at line {line}"
        if isinstance(node, ast.ImportFrom) and node.module == "sys" and any(a.name == "*" for a in node.names):
            return f"from sys import * at line {line}"
        name = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                else node.name if isinstance(node, ast.alias) else None)
        if name not in _REFLECTIVE and name != "__match_args__":
            continue
        if isinstance(node, (ast.Name, ast.Attribute)) and not isinstance(node.ctx, ast.Load) \
                and name != "__match_args__":
            continue                                        # binding ``getattr = ...`` reads nothing; its calls are judged by object
        call = parents.get(node)
        called = isinstance(call, ast.Call) and call.func is node
        if isinstance(node, ast.alias):
            owner = parents.get(node)
            if name == 'format' and isinstance(owner, ast.ImportFrom) and owner.module == 'builtins':
                continue                                    # scalar formatting has no template-field traversal
            if node.asname and node.asname != name:
                return f"imports {name} as {node.asname} at line {line}"
            if name in _ENUMERATORS:
                return f"imports {name} at line {line}: reaches attributes without naming them"
            continue
        if name == "__match_args__":
            # a class pattern reads the attributes it lists by name: only a literal list is a list of names
            value = call.value if isinstance(call, (ast.Assign, ast.AnnAssign)) else None
            if not (isinstance(value, (ast.Tuple, ast.List)) and all(_str_text(e) is not None for e in value.elts)):
                return f"__match_args__ the scan cannot read at line {line}"
        elif name in _ENUMERATORS:
            if (name == "vars" and called and not call.args and not call.keywords) or (
                    name == "__dict__" and not isinstance(node.ctx, ast.Load)):
                continue                                    # vars() is locals(); a store reads nothing
            if observed is not None and name == "__dict__" and isinstance(node.value, ast.Name) \
                    and isinstance(node.ctx, ast.Load):
                observe(node, node.value.id)
            elif observed is not None and name == "vars" and called and isinstance(node, ast.Name) \
                    and len(call.args) == 1 and isinstance(call.args[0], ast.Name) and not call.keywords:
                observe(call, ("vars", call.args[0].id, "module"))
            else:
                return f"{name} at line {line}: reaches attributes without naming them"
        elif name in _CODE_RUNNERS:
            if name == "compile" and not (isinstance(node, ast.Name) or (
                    isinstance(node.value, ast.Name) and node.value.id in ("builtins", "__builtins__"))):
                continue                                    # re.compile and its kin run no Python
            if not (called and call.args and _str_text(call.args[0]) is not None):
                texts = _observed_code(call, filename, observed) if called else ()
                if not texts:
                    return f"{name} of code the scan cannot read at line {line}"
                for text in texts:
                    if inner := orig_argv_reason(text, _depth=_depth + 1):
                        return f"{name} of guarded code: {inner}"
        elif name in _NAME_RESOLVERS or name in _GETATTRIBUTE:
            if name == "hasattr":
                handled.update(id(a) for a in (call.args if called else ()))
                continue                                    # presence only: orig_argv exists natively too
            if not called or any(isinstance(a, ast.Starred) for a in call.args) or any(
                    k.arg is None for k in call.keywords):
                return f"{name} used other than by a direct call at line {line}"
            at = len(call.args) - 1 if name in _GETATTRIBUTE else _NAME_RESOLVERS[name]
            named = list(call.args) if at is None else call.args[at:at + 1]
            named += [k.value for k in call.keywords if k.arg in ("name", "attr")]
            texts = [_str_text(a) for a in named]
            handled.update(id(a) for a in named)
            receiver = call.args[0] if name == "getattr" and call.args else None
            # the child re-reads these names right before CALL: only names and constants, which run no code in between
            checkable = (observed is not None and isinstance(node, ast.Name) and isinstance(receiver, ast.Name)
                         and all(isinstance(a, (ast.Name, ast.Constant)) for a in call.args))
            if not named or None in texts:
                if name == "getattr" and checkable:
                    if len(named) == 1 and isinstance(named[0], ast.Name):
                        observe(call, ("getattr", receiver.id, "attribute", named[0].id))
                    else:
                        observe(call, ("getattr", receiver.id, "sys"))
                    continue
                return f"{name} with a computed name at line {line}"
            if name == "getattr" and texts[0] == "__dict__" and checkable:
                observe(call, ("getattr", receiver.id, "module"))
            elif any(_ORIG_ARGV in t or "__dict__" in t or t in _REFLECTIVE or (
                    name not in {"__import__", "import_module", "resolve_name", "locate"}
                    and reflective_field_attribute(t)) for t in texts):
                return f"{name} of {texts[0][:40]!r} at line {line}"
    for kind, text, node in _live_strings(tree, filename=filename, observed=observed, runtime=observed is not None):
        if id(node) in handled:
            continue
        if kind in {"field", "opaque-field"}:
            if reason := _format_selector_reason(kind, text, node):
                return reason
        elif kind == "code":
            inner = orig_argv_reason(text, _depth=_depth + 1) if _depth < 4 else "code nested too deep"
            if inner:
                return f"code in a string at line {node.lineno} {inner}"
        elif _ORIG_ARGV in text or "__dict__" in text or text in _REFLECTIVE:
            return f"a name {text[:40]!r} at line {node.lineno}"
    return None


def references_orig_argv(source: bytes | str) -> bool:
    """``orig_argv_reason(source) is not None``."""
    return orig_argv_reason(source) is not None


def references_argv(source: bytes | str) -> bool:
    """Does the CODE of ``source`` name ``sys.argv`` (or ``sys.orig_argv``, which the former text test ``b"argv" in
    source`` matched too): an attribute, a name, an import, ``from sys import *``, a string it runs or uses as a name
    (:func:`_live_strings`)?  A comment ``# argv`` or the data ``"argv"`` is not code (2026-10-29 audit, R2).
    A computed name is left to the child's object checks and to ``determinism.py``'s ``sys`` allowlist."""
    import ast
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    words = ("argv", _ORIG_ARGV)
    for node in ast.walk(tree):
        if ((isinstance(node, ast.Attribute) and node.attr in words) or (isinstance(node, ast.Name) and node.id in words)
                or (isinstance(node, ast.alias) and node.name in words)
                or (isinstance(node, ast.ImportFrom) and node.module == "sys" and any(a.name == "*" for a in node.names))):
            return True
    return any(references_argv(text) if kind == "code" else "argv" in text for kind, text, _ in _live_strings(tree))


def _forms(base: str) -> frozenset[str]:
    """``base`` with the optional text/binary flag in either place (``w``, ``wb``, ``bw``, ...)."""
    out = {base}
    for flag in "bt":
        out |= {base + flag, flag + base}
    return frozenset(out)


#: ``open()`` modes that only WRITE ([internal ref]): exactly one of create-exclusive / write / append,
#: no ``+`` (an update mode reads) and no ``r``.  The flag ``b`` or ``t`` may stand on either side.
OUTPUT_MODES = _forms("w") | _forms("a") | _forms("x")
#: ``open()`` modes that only read (the default, ``r``, is one).
READ_MODES = _forms("r")
#: What ``open()`` itself can raise in a write mode, whatever is written: the checks a proof must
#: not lose when it accepts an output open.  ``OSError`` (and its subclasses: a missing parent
#: directory, a directory, a permission) is the environment, and is not a property class;
#: ``TypeError`` is raised for a path that is not ``str``, ``bytes`` or path-like, and IS the
#: ``dynamic-typing`` class; ``ValueError`` for an embedded NUL in the path or an invalid mode, and
#: its subclass ``UnicodeEncodeError`` for a path that does not encode.  This model does not know a
#: path's type, so it does not discharge them: an analysis that consults it must model them or
#: stay open (``absint`` has no model of ``open`` and aborts at it).
OUTPUT_OPEN_RAISES = ("OSError", "TypeError", "ValueError")

_OPEN_KEYWORDS = frozenset({"file", "mode", "buffering", "encoding", "errors", "newline"})
_OPEN_PARAMETERS = ("file", "mode", "buffering", "encoding", "errors", "newline")


def open_effect(call) -> str:
    """The effect of one ``open(...)`` call node on the file system, from its mode alone:

    * ``"output"``: a constant write-only mode (:data:`OUTPUT_MODES`).  What is written can reach
      the program again only through a later read, and a read is its own call that is classified
      on its own; a write cannot change what the program computes.  Its own failures stay
      (:data:`OUTPUT_OPEN_RAISES`).
    * ``"read"``: a constant read-only mode, or no mode (the default is ``"r"``).
    * ``"unknown"``: everything else: an update mode (``"r+"``, ``"w+"``, ``"a+"``: it reads), a mode
      that is not a constant string, an invalid mode, a file descriptor as the path (a constant
      int), ``opener=`` (runs a callback) or ``closefd=``, a starred or ``**`` argument, an argument
      given twice, or more arguments than ``open`` takes.

    Only the builtin's own signature is understood: the caller decides that ``call`` is the
    builtin (``open`` is not rebound; ``Path.open`` takes the mode first, and ``os.open`` takes
    flags)."""
    import ast

    if len(call.args) > len(_OPEN_PARAMETERS) or any(isinstance(a, ast.Starred) for a in call.args):
        return "unknown"
    given = dict(zip(_OPEN_PARAMETERS, call.args))
    for kw in call.keywords:
        if kw.arg is None or kw.arg not in _OPEN_KEYWORDS or kw.arg in given:
            return "unknown"
        given[kw.arg] = kw.value
    path = given.get("file")
    if path is None or (isinstance(path, ast.Constant) and isinstance(path.value, int)):
        return "unknown"           # no path, or a file descriptor
    mode = given.get("mode")
    if mode is None:
        return "read"
    if not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)):
        return "unknown"
    if mode.value in OUTPUT_MODES:
        return "output"
    return "read" if mode.value in READ_MODES else "unknown"


def _is_bound(tree, name: str) -> bool:
    """Does any construct of ``tree`` bind ``name`` (an assignment, ``def``, ``class``, import,
    parameter, loop or ``with`` target, ``except ... as``, match capture, ``global``/``nonlocal``)?
    A builtin that is bound somewhere may not be the builtin at a given call."""
    import ast

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == name:
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.arg) and node.arg == name:
            return True
        if isinstance(node, ast.alias) and (node.asname or node.name).split(".")[0] == name:
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == name:
            return True
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name:
            return True
        if isinstance(node, ast.MatchMapping) and node.rest == name:
            return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return True
    return False


def proof_blocker(source: str, environment: TaskEnvironment | None = None, *,
                  filename: str = "<program>", world_reads: bool = False) -> str | None:
    """Return the first input effect a proof domain must cover explicitly.

    A write-only ``open()`` (:func:`open_effect` is ``"output"``) is not an input effect and is
    not a blocker: the shared model classifies it as output.  The program may still fail at it
    (:data:`OUTPUT_OPEN_RAISES`), so a proof domain that accepts the call must model those
    failures or stay open at it; every read, update mode, descriptor, rebound or
    non-builtin ``open`` is a blocker as before."""
    import ast
    from tharkun.solve.extern import native_import_reason
    from tharkun.solve.static_code import prepared_code

    source, _ = prepared_code(source, filename)

    native_reason = native_import_reason(source, filename)
    if native_reason:
        return native_reason
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return "unreadable program"
    # New functions capture the active builtin mapping at definition time.  None of
    # the proof domains models replacing that mapping; check before any slicing.
    if _is_bound(tree, "__builtins__"):
        return "__builtins__ binding: unmodelled builtin namespace"
    # sys.orig_argv is real, unmodelled CPython interpreter metadata ([internal ref] MG
    # review, 2026-09-29): it reflects how THIS interpreter process was actually
    # launched (fresh subprocess vs. a reused forked "explore" worker give
    # DIFFERENT values -- neither is the real application argv), never something
    # _child.py sets or could set soundly -- block a static TRUE the same as argv,
    # through every spelling the child's flag sees (an alias, getattr, vars, ...).
    orig = orig_argv_reason(source)
    if orig:
        return f"orig_argv: {orig}: unmodelled interpreter metadata"
    open_rebound = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "argv":
            return f"argv at line {node.lineno}: any list of strings"
        if isinstance(node, ast.ImportFrom) and node.module == "sys" and any(a.name == "argv" for a in node.names):
            return f"argv at line {node.lineno}: any list of strings"
        if isinstance(node, ast.Attribute) and node.attr in {"stdin", "__stdin__"}:
            return f"stdin at line {node.lineno}: arbitrary bytes or EOF"
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if name == "input":
            return f"stdin at line {node.lineno}: arbitrary bytes or EOF"
        if name == "open":
            if isinstance(node.func, ast.Name):
                if open_rebound is None:
                    open_rebound = _is_bound(tree, "open")
                if not open_rebound and open_effect(node) == "output":
                    continue
                if not open_rebound and world_reads:
                    continue
            path = node.args[0] if node.args else None
            if isinstance(path, ast.Constant) and isinstance(path.value, str) and environment:
                effect = environment.read_effect(path.value)
                return f"{effect} file read at line {node.lineno}"
            return f"unknown file read at line {node.lineno}: arbitrary bytes or OSError"
        if name in {"read_text", "read_bytes", "readline", "readlines", "exists",
                    "is_file", "is_dir", "stat", "lstat", "listdir", "scandir",
                    "FileIO", "open_code", "fdopen"}:
            if world_reads and name in {"readline", "readlines"}:
                continue
            return f"file or stdin read at line {node.lineno}: arbitrary bytes or OSError"
        if name == "read" and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
            return f"file descriptor read at line {node.lineno}: arbitrary bytes or OSError"
    return None
