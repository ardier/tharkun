"""``tharkun.solve.extern`` -- "read-and-recurse", not allow-by-name ([internal ref] import boundary).

``solve/determinism.py`` (``DET_MODULES``) and ``solve/census.py`` (``MODULES``) both trust a
small, hand-curated, pre-vetted set of STDLIB modules: anything else makes
``solve/pyresolve.py::Resolver`` refuse (``Top``), and ``solve/absint/interp.py`` builds an
unresolved imported symbol as ``top`` (any call of it: ``Unsupported``). Both refusals share
one root cause: neither prover has ever had a way to go read *another file* and reason about
it the same way it reasons about the program it was handed.

This module is that way, scoped to stay as trustworthy as the stdlib allowlists it sits next
to. A proof is only sound if it reasons about EXACTLY the code that will run at replay time.
For the stdlib allowlists that holds because stdlib behaviour for tharkun's targeted CPython
version is a controlled, versioned assumption already depended on everywhere. The extension
here earns the same property by restricting eligibility to modules found via a directory THE
PROGRAM ITSELF adds to ``sys.path`` -- a literal, static ``sys.path.insert(0, D)`` visible in
the program's own AST (never an ambient system path, never site-packages, never anything a
package index could resolve differently between the analysis run and the replay run). That is
satisfied by construction for the GUUT benchmark harnesses this was built for
(``tharkun_bench/gen_harness.py`` emits exactly this as its harness's own first statement) and
is a general rule, not a one-off carve-out: it is the closest Python analogue of a C SV-COMP
task's own multi-file source -- "code bundled with the task".

What this explicitly does NOT do: it never widens trust to site-packages or anything an
ambient interpreter's own default import machinery would find; it never trusts a C extension
or a namespace package (ambiguous by construction -- refused, not guessed); it never lets a
recursively-read file's OWN ``sys.path`` mutations establish new roots (only the top-level
PROGRAM's own, per (a)/(b) below -- a library mutating ``sys.path`` on its own is not "the
task's own tree" from that library's perspective).

Soundness conditions, cited by letter throughout this module and ``pyresolve.py`` (see
[internal ref] / docs/SV.md "Import boundary" for the full text; MAIN's approval, 2026-09-28):

(a) path literal-ness -- an absolute string literal, or a literal joined to
    ``os.path.dirname(os.path.abspath(__file__))``; never a relative literal, ``os.getcwd()``,
    an env var or any other computed expression; never a literal containing ``..``.
(b) the ``sys.path.insert(0, ...)`` call must be an unconditional, module-level statement,
    executing before the import it enables, in straight-line order.
(c) runtime search-order fidelity: a root shadowing an allowed stdlib name used later in the
    program refuses the WHOLE analysis; imports before the insert keep today's rules
    unchanged; a module already resolved once is reused from cache, never re-resolved to a
    different file within the same analysis (an ambiguous re-import refuses).
(d) imported code is program code: its own top-level statements and sites are analysed, not
    skipped; a library function that can raise a property exception must never yield TRUE.
(e) package semantics: a package import runs ``__init__.py``'s top level too; relative imports
    inside a package resolve against the PACKAGE's own directory; a C extension or an
    ambiguous (``__init__.py``-less) namespace package refuses outright; the cycle guard is
    keyed by resolved absolute file path.
"""

from __future__ import annotations

import ast
import os
import sys
import tokenize

if __package__:
    from .environment import format_field_reason
else:
    from environment import format_field_reason

#: Python source file suffixes recognised as "not compiled" for (e)'s C-extension refusal.
_COMPILED_SUFFIXES = (".so", ".pyd", ".dylib")


class ExternTop(Exception):
    """The extern resolver refuses to trust something; the message says what and where."""


# ------------------------------------------------------------------ (a): literal path forms
def _dotted_call(func: ast.AST, dotted: tuple[str, ...]) -> bool:
    """Does ``func`` spell exactly ``dotted`` (e.g. ``("os", "path", "dirname")``) as a plain
    attribute chain off a bare name?  Purely syntactic -- an alias (``import os as o``) simply
    does not match, which is conservative (falls back to "not eligible", today's refusal)."""
    parts: list[str] = []
    n = func
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if not isinstance(n, ast.Name):
        return False
    parts.append(n.id)
    parts.reverse()
    return tuple(parts) == dotted


def _is_dirname_abspath_file(node: ast.AST) -> bool:
    """``os.path.dirname(os.path.abspath(__file__))`` -- the one non-literal form (a) admits."""
    return (isinstance(node, ast.Call) and _dotted_call(node.func, ("os", "path", "dirname"))
            and len(node.args) == 1 and not node.keywords
            and isinstance(node.args[0], ast.Call)
            and _dotted_call(node.args[0].func, ("os", "path", "abspath"))
            and len(node.args[0].args) == 1 and not node.args[0].keywords
            and isinstance(node.args[0].args[0], ast.Name)
            and node.args[0].args[0].id == "__file__")


def _no_dotdot(p: str) -> bool:
    return ".." not in p.replace("\\", "/").split("/")


def _qualifying_literal(node: ast.AST, file_dir: str | None) -> str | None:
    """The literal directory a ``sys.path.insert(0, node)`` argument names, or None if it is
    not one of the two forms (a) admits (never evaluated -- pattern-matched statically)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        p = node.value
        return os.path.normpath(p) if os.path.isabs(p) and _no_dotdot(p) else None
    if (isinstance(node, ast.Call) and _dotted_call(node.func, ("os", "path", "join"))
            and len(node.args) >= 2 and not node.keywords and _is_dirname_abspath_file(node.args[0])):
        if file_dir is None:
            return None
        parts: list[str] = []
        for a in node.args[1:]:
            if not (isinstance(a, ast.Constant) and isinstance(a.value, str)
                    and _no_dotdot(a.value) and not os.path.isabs(a.value)):
                return None
            parts.append(a.value)
        return os.path.normpath(os.path.join(file_dir, *parts))
    return None


def _is_sys_path_insert(call: ast.AST) -> bool:
    return isinstance(call, ast.Call) and _dotted_call(call.func, ("sys", "path", "insert"))


def is_root_establishing_call(call: ast.AST, file_dir: str | None) -> bool:
    """Is ``call`` the EXACT ``sys.path.insert(0, D)`` shape a qualifying root comes from --
    (a)/(b)? Used by ``absint/interp.py`` to give this one specific call, once recognised this
    precisely, no property-relevant effect (its own import resolution is handled statically,
    off the program's own AST, not by tracking a runtime ``sys.path`` list) -- anything else
    still refuses exactly as before "call of an unknown value"."""
    if not (isinstance(call, ast.Call) and _is_sys_path_insert(call)):
        return False
    if len(call.args) != 2 or call.keywords:
        return False
    if not (isinstance(call.args[0], ast.Constant) and call.args[0].value == 0):
        return False
    return _qualifying_literal(call.args[1], file_dir) is not None


# ------------------------------------------------------------------------- (b)/(c): roots
def root_snapshots(tree: ast.Module, filename: str) -> tuple[dict[int, list[str]], list[str]]:
    """``{id(stmt): [roots visible when stmt executes]}`` for every statement directly in
    ``tree.body`` -- (b): only a module-level, unconditional ``sys.path.insert(0, D)``
    STATEMENT (itself a direct child of the module body -- never inside a def/if/try/for/
    while/class) ever contributes a root, and (c) only to statements that follow it: an
    import earlier in the body keeps today's rules, unchanged, because ``D`` was not yet on
    the path when it ran. Root order is real ``sys.path`` order: each new qualifying insert
    goes to the FRONT (``insert(0, ...)``), so a later insert shadows an earlier one, exactly
    like the real interpreter -- this is the ONLY nesting depth read-and-recurse considers;
    an import nested in a function/conditional is never eligible (todays's rules only)."""
    file_dir = os.path.dirname(filename) if filename and os.path.isabs(filename) else None
    roots: list[str] = []
    snaps: dict[int, list[str]] = {}
    for stmt in tree.body:
        snaps[id(stmt)] = list(roots)
        if isinstance(stmt, ast.Expr) and _is_sys_path_insert(stmt.value):
            call = stmt.value
            if (len(call.args) == 2 and not call.keywords
                    and isinstance(call.args[0], ast.Constant) and call.args[0].value == 0):
                lit = _qualifying_literal(call.args[1], file_dir)
                if lit is not None and os.path.isdir(lit):
                    roots.insert(0, lit)
    return snaps, roots


def shadowed_stdlib_name(roots: list[str], allowed) -> str | None:
    """(c): the first name in ``allowed`` that some root in ``roots`` would shadow (a same-
    named ``.py`` file or package directory) -- the caller refuses the WHOLE analysis rather
    than guess which one a real run would hit. ``roots`` must already be the ones visible at
    the point being checked (see :func:`root_snapshots`) so an import earlier than any
    qualifying insert is never flagged."""
    for root in roots:
        try:
            entries = set(os.listdir(root))
        except OSError:
            continue
        for name in allowed:
            if f"{name}.py" in entries or (name in entries and os.path.isdir(os.path.join(root, name))):
                return name
    return None


# --------------------------------------------------------------------------- (e): file lookup
def _has_compiled_sibling(directory: str, stem: str) -> bool:
    try:
        entries = os.listdir(directory)
    except OSError:
        return False
    for suffix in _COMPILED_SUFFIXES:
        if any(e == stem + suffix or (e.startswith(stem + ".") and e.endswith(suffix)) for e in entries):
            return True
    return False


def find_in_roots(dotted: str, roots: list[str], *, source_aliases: dict | None = None) -> tuple[str, list[str]] | None:
    """Resolve a dotted module name to a ``.py`` file by searching ``roots`` IN ORDER (first
    match wins -- real ``sys.path`` search order). Returns ``(leaf_file, package_inits)`` --
    ``package_inits`` lists every INTERMEDIATE package's ``__init__.py`` real Python runs
    first, outermost to innermost, per (e) (empty for a plain top-level module). Returns None
    if not found in any root (the caller then refuses exactly as today -- this never falls
    back to the ambient interpreter's own ``sys.path``). Raises :class:`ExternTop` for a C
    extension or an ambiguous (``__init__.py``-less) namespace package -- (e), no compromise.
    ``source_aliases`` records actual import filenames for the child's instruction observer, before target code runs.
    """
    parts = dotted.split(".")
    for base in roots:
        cur = base
        ok = True
        inits: list[str] = []
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            py = os.path.join(cur, part + ".py")
            pkg = os.path.join(cur, part)
            init = os.path.join(pkg, "__init__.py")
            # A package directory takes priority over a same-named module FILE in the
            # SAME path entry -- matches CPython's own FileFinder order, confirmed
            # empirically (MG's [internal ref] import-boundary review, PR 86,
            # case c12_pkg_dir_and_pymodule_same_name: a directory containing both
            # thing.py and thing/__init__.py -- native `python P.py` imports the
            # PACKAGE). This used to check ``part + ".py"`` FIRST, the opposite of real
            # Python's order, so a same-named safe module file could mask a dangerous
            # package.
            if os.path.isdir(pkg):
                if os.path.isfile(init):
                    if source_aliases is not None:
                        source_aliases.setdefault(os.path.realpath(init), set()).add(os.path.abspath(init))
                    inits.append(os.path.realpath(init))
                    if last:
                        return os.path.realpath(init), inits[:-1]
                    cur = pkg
                    continue
                raise ExternTop(f"namespace package (no __init__.py): {pkg}")
            if last and os.path.isfile(py):
                if source_aliases is not None:
                    source_aliases.setdefault(os.path.realpath(py), set()).add(os.path.abspath(py))
                return os.path.realpath(py), inits
            if last and _has_compiled_sibling(cur, part):
                raise ExternTop(f"compiled extension module: {part}")
            ok = False
            break
        if not ok:
            continue
    return None


def _task_import_closure(program: str, *, _packages: bool = False) -> tuple[str, ...] | None:
    """Discover closed sources; package graphs are metadata, never the flat proof shortcut."""
    import glob

    program = os.path.abspath(program)
    root = os.path.dirname(program)
    if os.path.realpath(program) != program:
        return None
    files = [program]
    seen = {program}

    def include(module, directory):
        found = find_in_roots(module, [directory])
        if found is None:
            return None
        leaf, inits = found
        if not _packages and (inits or os.path.basename(leaf) == '__init__.py'):
            raise ExternTop('packages are outside the flat source proof shortcut')
        lexical = os.path.join(directory, *module.split('.'))
        expected = os.path.join(lexical, '__init__.py') if os.path.isdir(lexical) else lexical+'.py'
        if leaf != expected or os.path.realpath(expected) != expected:
            raise ExternTop('task source alias or symlink')
        for candidate in [*inits, leaf]:
            if (candidate == program or os.path.commonpath((root, candidate)) != root
                    or os.path.realpath(candidate) != candidate):
                raise ExternTop('task source escapes its import root')
            folder, filename = os.path.split(candidate)
            stem = filename[:-3]
            if (_has_compiled_sibling(folder, stem)
                    or os.path.exists(os.path.join(folder, stem+'.pyc'))
                    or glob.glob(os.path.join(folder, '__pycache__', stem+'.*.pyc'))):
                raise ExternTop('task source has a compiled competitor')
            if candidate not in seen:
                seen.add(candidate)
                files.append(candidate)
        return leaf

    try:
        for path in files:
            with open(path, 'rb') as stream:
                tree = ast.parse(stream.read(), path)
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            prefix = set()
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    prefix.add(node)
                elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                          and type(node.value.value) is str):
                    break
            for node in imports:
                if node not in prefix:
                    return None
                if isinstance(node, ast.Import):
                    modules, directory = [a.name for a in node.names], root
                elif node.module and ((_packages and not node.level and node.module.isidentifier())
                                      or all(a.name != '*' for a in node.names)):
                    modules, directory = [node.module], root
                    if node.level:
                        depth = len(os.path.relpath(os.path.dirname(path), root).split(os.sep))
                        if (path == program or node.level > depth
                                or not os.path.isfile(os.path.join(os.path.dirname(path), '__init__.py'))):
                            return None
                        directory = package_dir_for(node.level, path)
                        if directory is None:
                            return None
                else:
                    return None
                for module in modules:
                    if module == '_sv_verifier' and directory == root and not getattr(node, 'level', 0):
                        continue
                    if (not all(part.isidentifier() for part in module.split('.'))
                            or (directory == root and not getattr(node, 'level', 0)
                                and module.split('.')[0] in sys.stdlib_module_names | {'__main__'})):
                        return None
                    leaf = include(module, directory)
                    if leaf is None:
                        return None
                    if (isinstance(node, ast.ImportFrom) and any(a.name == '*' for a in node.names)
                            and leaf.endswith('/__init__.py')):
                        return None
                    if isinstance(node, ast.ImportFrom) and leaf.endswith('/__init__.py'):
                        with open(leaf, 'rb') as stream:
                            exported = toplevel_public_names(ast.parse(stream.read(), leaf))
                        for alias in node.names:
                            if alias.name not in exported:
                                include(alias.name, os.path.dirname(leaf))
        return tuple(files) if len(files) > 1 else None
    except (OSError, SyntaxError, ValueError, ExternTop):
        return None


def package_dir_for(level: int, from_file: str) -> str | None:
    """The directory relative imports resolve against: ``level`` dots up from ``from_file``'s
    OWN containing directory, per (e) -- "relative imports resolve against the PACKAGE's own
    location, not the top-level program's". Each level up must itself be a package
    (``__init__.py`` present) or this refuses (returns None -> caller raises)."""
    d = os.path.dirname(os.path.realpath(from_file))
    for _ in range(level - 1):
        if not os.path.isfile(os.path.join(d, "__init__.py")):
            return None
        d = os.path.dirname(d)
    return d


# --------------------------------------------------------------- inert package-init shortcut
def _literal_only(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_literal_only(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all((k is None or _literal_only(k)) and _literal_only(v)
                   for k, v in zip(node.keys, node.values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _literal_only(node.operand)
    return False


def toplevel_inert(tree: ast.Module) -> bool:
    """True iff EVERY statement directly in ``tree.body`` is a no-effect-by-construction
    declaration: an import, a ``def``/``class`` (a bare definition -- what it can do only
    matters once/if it is actually called, which dead-code liveness already covers for the
    rest of this project), a docstring, or an assignment of a literal constant. Neither a
    ``def``/``class`` statement nor an import statement can themselves raise a property
    exception or introduce nondeterminism -- ONLY calling into what they define can. A file
    this is true of is therefore safe to treat as contributing NOTHING further, regardless of
    what its own (possibly unmodelled) imports are, exactly like a C SV-COMP task's unused
    header would be -- see docs/SV.md "Import boundary: the inert package-init shortcut".

    This is what makes ``from pkg.sub import name`` provable when ``pkg/__init__.py`` (which
    real Python ALWAYS runs first, package semantics, (e)) itself re-exports OTHER siblings
    via ``from .other import *`` whose own imports are not stdlib-allowlisted (e.g. ``os``) --
    as long as ``__init__.py``'s own top level is inert, those siblings are never even read,
    because nothing downstream can observe them without loading a name FROM ``__init__.py``'s
    own namespace, which this module's demand-driven resolution never needs to do here.
    """
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # a docstring
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            value = stmt.value
            if value is not None and _literal_only(value):
                continue
        return False
    return True


# ------------------------------------------------------------------------------ public names
def toplevel_public_names(tree: ast.Module) -> frozenset[str]:
    """Names a bare ``from <this file> import *`` would bind: every top-level ``def``/
    ``class``/plain-assigned name not starting with ``_``, restricted to a literal ``__all__``
    when the file defines one (real Python semantics)."""
    all_list = None
    names: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, (ast.Import,)):
            for a in stmt.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(stmt, ast.ImportFrom):
            for a in stmt.names:
                if a.name != "*":
                    names.add(a.asname or a.name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    if t.id == "__all__" and stmt.value is not None:
                        lst = _const_str_list(stmt.value)
                        if lst is not None:
                            all_list = lst
                    else:
                        names.add(t.id)
    if all_list is not None:
        return frozenset(all_list)
    return frozenset(n for n in names if not n.startswith("_"))


def _const_str_list(node: ast.AST) -> list[str] | None:
    if isinstance(node, (ast.List, ast.Tuple)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
        return [e.value for e in node.elts]
    return None


# attribute names a static ``getattr(x, "name")`` must not read: the import caches and the reflective escapes this module refuses
_REFLECTIVE_ATTRS = frozenset({"modules", "path", "monitoring", "sys", "import_module", "reload", "__dict__", "__globals__",
                               "__subclasses__", "__getattribute__", "__builtins__", "__import__"})

#: reflective attribute reads that are import state themselves; the ``_STATIC_ONLY_ATTRS`` are ones a native-equivalent
#: child can execute, so only the runtime tier admits them
_REFLECTIVE_OBJECT_ATTRS = frozenset({"import_module", "reload", "__dict__", "__globals__", "__subclasses__",
                                      "__getattribute__"})
_STATIC_ONLY_ATTRS = frozenset({"import_module", "__globals__", "__getattribute__"})

#: interpreter-object reflection: re-importing an evicted module cannot undo a reference graph
_HEAP_ATTRS = frozenset({"get_referents", "get_referrers", "get_objects"})

#: R1 (owner 2026-10-24): ``x.__dict__`` is import state only when ``x`` is a module-bound name or ``sys``; a class's or
#: an instance's mapping is not.  False restores the blanket refusal of every ``.__dict__`` read.
NARROW_DICT_FOR_STATIC = True


def _module_bound_names(tree: ast.Module) -> frozenset[str] | None:
    """The names a file binds to a module (``import``/``from ... import``), over-approximated; None for a ``*`` import,
    where every name is a candidate."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if any(a.name == "*" for a in node.names):
                return None
            names.update(a.asname or a.name for a in node.names)
    return frozenset(names)


def _comprehension_local(node: ast.Name, parents: dict) -> bool:
    """A comprehension target shadows imports, except in its first iterable (the enclosing scope)."""
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.Lambda) and current is parent.body:
            return False
        if isinstance(parent, ast.comprehension) and current is parent.iter:
            owner = parents[parent]
            if parent is owner.generators[0]:
                current = owner
                continue
        if isinstance(parent, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            if any(isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store) and target.id == node.id
                   for gen in parent.generators for target in ast.walk(gen.target)):
                return True
        current = parent
    return False


def _native_modules_key(key: ast.AST) -> bool:
    """A ``sys.modules`` key that names the same entry natively after eviction: a non-stdlib constant, ``X.__module__`` /
    ``X.__name__``, or the program's own ``__name__``.  A stdlib constant may be model-held (a C extension cannot be
    evicted) and any other dynamic key can name a stdlib module."""
    if isinstance(key, ast.Constant):
        return isinstance(key.value, str) and key.value.split(".")[0] not in sys.stdlib_module_names
    return ((isinstance(key, ast.Attribute) and key.attr in {"__module__", "__name__"}) or
            (isinstance(key, ast.Name) and key.id == "__name__"))


def _modules_read_is_native(node: ast.Attribute, parents: dict) -> bool:
    """True when this ``sys.modules`` use reads one native key and nothing else: no iteration, no mutation."""
    parent = parents.get(node)
    if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr == "get":
        call = parents.get(parent)
        if isinstance(call, ast.Call) and call.func is parent and len(call.args) == 1 and not call.keywords:
            return _native_modules_key(call.args[0])
    elif isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Load):
        return _native_modules_key(parent.slice)
    elif (isinstance(parent, ast.Compare) and len(parent.ops) == 1 and len(parent.comparators) == 1 and
          parent.comparators[0] is node and isinstance(parent.ops[0], (ast.In, ast.NotIn))):
        return _native_modules_key(parent.left)
    return False


def _inspected_string_calls(tree: ast.Module):
    """Return direct string-code calls, their sources and parsed bodies, or None.

    The child must still check the actual builtin and source before CALL. Syntactic
    bindings supply alternatives, not a claim that a variable keeps its value.
    Compiled alternatives carry (source, filename, mode); scalar sources stay str/bytes.
    """
    nodes = list(ast.walk(tree))
    parents = {c: n for n in nodes for c in ast.iter_child_nodes(n)}
    bindings = {}
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings.setdefault(target.id, []).append(node.value)

    def sources(node, visiting=frozenset()):
        if isinstance(node, ast.Constant) and type(node.value) in (str, bytes):
            return {node.value}
        if isinstance(node, ast.IfExp):
            return sources(node.body, visiting) | sources(node.orelse, visiting)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "compile":
            if (len(node.args) != 3 or node.keywords
                    or not all(isinstance(a, ast.Constant) and type(a.value) is str for a in node.args[1:])
                    or node.args[2].value not in {"exec", "eval", "single"}):
                raise ValueError("compiler mode, filename or options are not inspected")
            texts = sources(node.args[0], visiting)
            if any(type(text) not in (str, bytes) for text in texts):
                raise ValueError("compile requires inspected string sources")
            return {(text, node.args[1].value, node.args[2].value) for text in texts}
        if isinstance(node, ast.Name) and node.id not in visiting and node.id in bindings:
            result = set()
            for value in bindings[node.id]:
                result |= sources(value, visiting | {node.id})
            return result
        raise ValueError("source is not a finite syntactic string alternative")

    calls = []
    try:
        for node in nodes:
            if not isinstance(node, ast.Name) or node.id not in {"exec", "eval", "compile"}:
                continue
            call = parents.get(node)
            if (not isinstance(call, ast.Call) or call.func is not node
                    or len(call.args) != (3 if node.id == "compile" else 1) or call.keywords
                    or not isinstance(call.args[0], (ast.Name, ast.Constant))):
                return None
            alternatives = sources(call if node.id == "compile" else call.args[0])
            texts = {item[0] for item in alternatives} if node.id == "compile" else alternatives
            bodies = []
            for item in alternatives:
                text, mode = (item[0], item[2]) if type(item) is tuple else (
                    item, "eval" if node.id == "eval" else "exec")
                # eval accepts leading spaces/tabs; exec and the native compiler retain all other semantics.
                if node.id == "eval" and type(item) is not tuple:
                    text = text.lstrip(b" \t" if isinstance(text, bytes) else " \t")
                body = ast.parse(text, mode=mode)
                if format_field_reason(body, runtime=True):
                    return None
                body_nodes = list(ast.walk(body))
                if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in body_nodes):
                    return None
                bodies.extend(body_nodes)
            calls.append((node, call, tuple(texts), bodies))
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
    if not calls:
        return None
    # Dynamic reflection could escape the inspected bodies or expose the tool's observer.
    all_nodes = nodes + [n for _, _, _, body in calls for n in body]
    if any((isinstance(n, ast.Name) and n.id in {
                "getattr", "setattr", "delattr", "globals", "locals", "vars", "__builtins__", "__import__"})
           or (isinstance(n, ast.Attribute) and (n.attr.startswith("__") or n.attr in {
               "_getframe", "currentframe", "f_globals", "f_locals", "f_back", "gi_frame", "cr_frame",
               "ag_frame", "settrace", "setprofile", "addaudithook", "argv", "orig_argv", "stdin"}))
           or (isinstance(n, (ast.Import, ast.ImportFrom)) and any(
               name.split(".")[0] in {"__future__", "builtins", "inspect", "gc", "types", "marshal",
                                      "importlib", "runpy", "code", "codeop"}
               for name in ([a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""])))
           for n in all_nodes):
        return None
    return calls


def native_import_reason(source: bytes | str | ast.Module, filename: str = "<program>", *,
                         reversible: bool = False, runtime: bool = False, mapping_reads: dict | None = None) -> str | None:
    """Refuse observable bootstrap cache state and stdlib shadows, including local imports.

    The model has already imported its own modules; their cache entries cannot reproduce
    a fresh ``python program.py``. Read source only, using the existing import-root rules.

    ``runtime=True`` returns only what re-importing the evicted modules cannot repair (``sv/_child.py``): an
    enumerated or mutated ``sys.modules``, a stdlib or dynamic key, ``reload``, ``exec``/``eval``/``compile``, a ``sys``
    escape, ``sys.monitoring``, ``__subclasses__``, heap reflection and ``import posix``/``nt``. Direct
    ``compile``/``exec``/``eval`` calls with inspected strings or compiled alternatives are search-only when
    ``mapping_reads`` records them for the child's mandatory argument/identity/code observer; all other import
    checks then use the static tier. The rest
    (``__import__``, ``vars``/``globals``/``locals``, ``import_module``, ``sys.path``, ``__globals__``, a stdlib shadow or
    an unreadable import graph) is static-only: a native-equivalent child runs it, so a caller admits it for search and
    withholds TRUE.

    ``mapping_reads`` receives the source positions/names of comprehension mappings admitted only for runtime
    receiver checks. Static proofs still refuse them; the child must flag module or unobserved receivers.
    """
    if not (filename.startswith("<") and filename.endswith(">")):
        filename = os.path.abspath(filename)
    pending = [(source, filename, None)]
    source_aliases = {os.path.realpath(filename): {filename}}
    seen = set()
    inspected = set()
    while pending:
        text, path, inherited = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            tree = text if isinstance(text, ast.Module) else ast.parse(text, path)
        except (SyntaxError, ValueError):
            return "unreadable native import graph"
        nodes = list(ast.walk(tree))
        if reason := format_field_reason(tree, filename=path, observed=mapping_reads, runtime=runtime):
            return f"{reason} in {path}"
        if runtime and path == filename and mapping_reads is not None:
            calls = _inspected_string_calls(tree)
            if calls is not None:
                runtime = False
                for callee, call, texts, body_nodes in calls:
                    inspected.add(id(callee))
                    argument = call.args[0]
                    mapping_reads[(os.path.realpath(path), call.lineno, call.end_lineno,
                                   call.col_offset, call.end_col_offset)] = (
                        callee.id, argument.id if isinstance(argument, ast.Name) else None, "code", texts)
                    nodes.extend(body_nodes)
        snaps, final = root_snapshots(tree, path)
        own = [os.path.dirname(path)] if inherited is None and os.path.isabs(path) else []
        imported = inherited is not None
        inherited = inherited or []
        roots = list(dict.fromkeys(final + inherited + own))
        parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
        if not runtime:
            for node in nodes:
                binding = ((isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                            and node.name == "__getattr__")
                           or (isinstance(node, ast.Name) and node.id == "__getattr__"
                               and isinstance(node.ctx, (ast.Store, ast.Del)))
                           or (isinstance(node, ast.alias) and (node.asname or node.name) == "__getattr__")
                           or (isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar))
                               and node.name == "__getattr__")
                           or (isinstance(node, ast.MatchMapping) and node.rest == "__getattr__"))
                if isinstance(node, ast.Global) and "__getattr__" in node.names:
                    return f"module __getattr__ at {path}:{node.lineno}: import hooks are unmodelled"
                if binding:
                    owner = parents.get(node)
                    while owner is not None and not isinstance(owner, (ast.Module, ast.FunctionDef,
                            ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                        owner = parents.get(owner)
                    if isinstance(owner, ast.Module):
                        return f"module __getattr__ at {path}:{node.lineno}: import hooks are unmodelled"
        sys_names = {a.asname or a.name for n in nodes if isinstance(n, ast.Import)
                     for a in n.names if a.name == "sys"}
        path_names = {a.asname or a.name for n in nodes if isinstance(n, ast.ImportFrom)
                      and n.module == "sys" for a in n.names if a.name == "path"}
        dynamic_names = {"__import__", "exec", "eval", "compile", "globals", "locals", "vars"}
        dynamic_names.update(a.asname or a.name for n in nodes if isinstance(n, ast.ImportFrom)
                             and n.module in {"importlib", "builtins"} for a in n.names
                             if a.name in dynamic_names | {"import_module", "reload"})
        #: a code object survives eviction, so these dynamic names stay refused at the runtime tier
        runtime_names = {"exec", "eval", "compile"}
        runtime_names.update(a.asname or a.name for n in nodes if isinstance(n, ast.ImportFrom)
                             and n.module in {"importlib", "builtins"} for a in n.names
                             if a.name in {"exec", "eval", "compile", "reload"})
        #: the names R1 treats as module-bound (``import x`` / ``from pkg import x``), None when a ``*`` import hides them
        module_names = _module_bound_names(tree)
        imports_sys = any(isinstance(n, (ast.Import, ast.ImportFrom)) and
                          ((isinstance(n, ast.Import) and any(a.name == "sys" for a in n.names)) or
                           (isinstance(n, ast.ImportFrom) and n.module == "sys")) for n in nodes)
        helper_imports = {a.asname or a.name: n.lineno for n in tree.body if isinstance(n, ast.Import)
                          for a in n.names if a.name == "_sv_verifier"}
        helper_line = min(helper_imports.values(), default=float("inf"))
        helper_lookups = set()
        for node in nodes:
            if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and
                    isinstance(node.slice, ast.Constant) and node.slice.value == "_sv_verifier" and
                    helper_line < node.lineno):
                value = node.value
                if (isinstance(value, ast.Attribute) and value.attr == "modules" and
                        isinstance(value.value, ast.Name) and value.value.id in sys_names):
                    helper_lookups.add(id(value))
                elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and
                      value.func.id == "globals" and not value.args and not value.keywords and
                      "_sv_verifier" in helper_imports):
                    helper_lookups.add(id(value.func))
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                    node.func.id == "__import__" and len(node.args) == 1 and not node.keywords and
                    isinstance(node.args[0], ast.Constant) and node.args[0].value == "_sv_verifier"):
                helper_lookups.add(id(node.func))
        type_names = {"type"} | {a.asname or a.name for n in nodes
                      if isinstance(n, ast.ImportFrom) and n.module == "builtins"
                      for a in n.names if a.name == "type"}
        setattr_names = {"setattr"} | {a.asname or a.name for n in nodes
                         if isinstance(n, ast.ImportFrom) and n.module == "builtins"
                         for a in n.names if a.name == "setattr"}
        getattr_names = {"getattr"} | {a.asname or a.name for n in nodes
                         if isinstance(n, ast.ImportFrom) and n.module == "builtins"
                         for a in n.names if a.name == "getattr"}
        for node in nodes:
            if reversible:
                if ((isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__del__") or
                        (isinstance(node, ast.Attribute) and node.attr == "__del__") or
                        (isinstance(node, ast.Constant) and node.value == "__del__")):
                    return f"deferred finalizer at {path}:{node.lineno}: outside reversible execution"
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imported_names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                                      else [node.module or ""])
                    if any(name.split(".")[0] in {
                            "atexit", "weakref", "_weakref", "signal", "multiprocessing", "gc", "logging", "asyncio",
                            "operator", "_operator", "abc", "_py_abc", "enum", "dataclasses", "typing",
                            "types", "_types", "inspect", "pickle", "_pickle", "marshal", "shelve",
                            "runpy", "code", "codeop", "pydoc"} for name in imported_names):
                        return f"deferred callback import at {path}:{node.lineno}: outside reversible execution"
                call = parents.get(node)
                comparison = parents.get(call)
                # Identity consumes the class without exposing a callable metaclass.
                type_identity = (isinstance(node, ast.Name) and node.id == "type" and
                                 isinstance(call, ast.Call) and call.func is node and
                                 len(call.args) == 1 and not isinstance(call.args[0], ast.Starred) and
                                 not call.keywords and isinstance(comparison, ast.Compare) and
                                 all(isinstance(op, (ast.Is, ast.IsNot)) for op in comparison.ops))
                if ((isinstance(node, ast.Name) and node.id in type_names and not type_identity) or
                        (isinstance(node, ast.Attribute) and node.attr in {
                            "type", "__class__", "__mro__", "__bases__", "__base__",
                            "settrace", "setprofile", "addaudithook", "_register_atexit", "register_at_fork"}) or
                        (isinstance(node, ast.ImportFrom) and any(a.name in {
                            "settrace", "setprofile", "addaudithook", "_register_atexit", "register_at_fork"} for a in node.names))):
                    return f"deferred callback/type access at {path}:{node.lineno}: outside reversible execution"
                setattr_ref = ((isinstance(node, ast.Name) and node.id in setattr_names) or
                               (isinstance(node, ast.Attribute) and node.attr == "setattr"))
                getattr_ref = ((isinstance(node, ast.Name) and node.id in getattr_names) or
                               (isinstance(node, ast.Attribute) and node.attr == "getattr"))
                if setattr_ref or getattr_ref:
                    call = parents.get(node)
                    direct = isinstance(call, ast.Call) and call.func is node
                    safe = (direct and not call.keywords and len(call.args) in ((3,) if setattr_ref else (2, 3))
                            and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)
                            and not call.args[1].value.startswith("__")
                            and call.args[1].value not in {"type", "exec", "eval", "compile", "globals", "locals", "vars",
                                                              "import_module", "reload", "settrace", "setprofile",
                                                              "addaudithook", "_register_atexit", "register_at_fork"})
                    if not safe:
                        return f"dynamic deferred attribute at {path}:{node.lineno}: outside reversible execution"
            # The already-imported helper is deliberately modelled, even through its known cache/global lookup.
            if id(node) in helper_lookups or id(node) in inspected:
                continue
            if ((isinstance(node, ast.Attribute) and node.attr == "monitoring") or
                    (isinstance(node, ast.ImportFrom) and node.module == "sys" and
                     any(a.name == "monitoring" for a in node.names)) or
                    (isinstance(node, ast.Import) and any(a.name == "sys.monitoring" for a in node.names)) or
                    (imports_sys and isinstance(node, ast.Constant) and node.value == "monitoring")):
                return f"sys.monitoring at {path}:{node.lineno}: unmodelled resource observer access"
            if ((isinstance(node, ast.Attribute) and node.attr == "modules") or
                    (isinstance(node, ast.ImportFrom) and node.module == "sys" and
                     any(a.name in {"modules", "*", "__dict__", "__getattribute__"} for a in node.names)) or
                    (imports_sys and isinstance(node, ast.Constant) and node.value == "modules")):
                if not (runtime and isinstance(node, ast.Attribute) and _modules_read_is_native(node, parents)):
                    return f"sys.modules at {path}:{node.lineno}: unmodelled bootstrap import cache"
            if ((isinstance(node, ast.Attribute) and node.attr in _HEAP_ATTRS) or
                    (isinstance(node, ast.ImportFrom) and node.module == "gc" and
                     any(a.name in _HEAP_ATTRS for a in node.names))):
                return f"gc referents at {path}:{node.lineno}: unmodelled interpreter object graph"
            if isinstance(node, ast.Name) and node.id in dynamic_names:
                if not runtime or node.id in runtime_names:
                    return f"uses {node.id} at {path}:{node.lineno}: dynamic code/import cache is unmodelled"
            if isinstance(node, ast.Attribute) and node.attr in _REFLECTIVE_OBJECT_ATTRS:
                # R1: a class's or an instance's mapping is not import state, a module's is
                mapping = (node.attr == "__dict__" and NARROW_DICT_FOR_STATIC and
                           not (isinstance(node.value, ast.Name) and
                                (module_names is None or node.value.id in module_names or
                                 node.value.id in sys_names)))
                sys_base = isinstance(node.value, ast.Name) and node.value.id in sys_names
                guarded_mapping = (runtime and not mapping and node.attr == "__dict__" and NARROW_DICT_FOR_STATIC and
                                   isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and
                                   not sys_base and _comprehension_local(node.value, parents))
                if guarded_mapping and mapping_reads is not None:
                    mapping_reads[(os.path.realpath(path), node.lineno, node.end_lineno,
                                   node.col_offset, node.end_col_offset)] = node.value.id
                if not (mapping or guarded_mapping or (runtime and not sys_base and node.attr in _STATIC_ONLY_ATTRS)):
                    return f"reflective {node.attr} at {path}:{node.lineno}: dynamic code/import cache is unmodelled"
            if isinstance(node, ast.Call) and imports_sys and isinstance(node.func, ast.Name) and node.func.id == "getattr":
                helper_attr = (len(node.args) == 2 and not node.keywords and isinstance(node.args[0], ast.Name) and
                               helper_imports.get(node.args[0].id, float("inf")) < node.lineno and
                               isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str) and
                               node.args[1].value.startswith("nondet_"))
                # A static attribute name read off anything but the ``sys`` name is no cache observation: ``sys`` never escapes
                # as a bare name (refused below), the cache names are refused as constants above, and the reflective names
                # are refused here.  ``getattr(type_, '__origin__', type_)`` is introspection, not import state.
                static_attr = (2 <= len(node.args) <= 3 and not node.keywords and
                               isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str) and
                               node.args[1].value not in dynamic_names | _REFLECTIVE_ATTRS and
                               not (isinstance(node.args[0], ast.Name) and node.args[0].id in sys_names))
                # a receiver that is the ``sys`` name reaches ``sys.modules``/``sys.monitoring``: refused at both tiers
                sys_receiver = (bool(node.args) and isinstance(node.args[0], ast.Name) and
                                node.args[0].id in sys_names)
                if (sys_receiver or not runtime) and not (helper_attr or static_attr):
                    return f"sys.modules/import path reflection at {path}:{node.lineno}: unmodelled import cache"
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in sys_names:
                parent = parents.get(node)
                if not (isinstance(parent, ast.Attribute) and parent.value is node):
                    return f"sys object escapes at {path}:{node.lineno}: unmodelled import cache"
            is_path = ((isinstance(node, ast.Attribute) and node.attr == "path" and
                        isinstance(node.value, ast.Name) and node.value.id in sys_names) or
                       (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in path_names))
            if is_path:
                parent = parents.get(node)
                call = parents.get(parent) if isinstance(parent, ast.Attribute) else None
                statement = parents.get(call)
                established = (not imported and isinstance(statement, ast.Expr) and statement in tree.body and
                               is_root_establishing_call(call, os.path.dirname(path) if os.path.isabs(path) else None))
                index_zero = (isinstance(parent, ast.Subscript) and isinstance(parent.ctx, ast.Load) and
                              isinstance(parent.slice, ast.Constant) and parent.slice.value == 0)
                if not runtime and not established and not index_zero:
                    return f"sys.path at {path}:{node.lineno}: unmodelled import path state"
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = ([node.module] if node.module else [])
                names += [".".join(filter(None, (node.module, a.name))) for a in node.names if a.name != "*"]
            else:
                continue
            visible = list(dict.fromkeys(snaps.get(id(node), roots) + inherited + own))
            absolute_visible = visible
            if isinstance(node, ast.ImportFrom) and node.level:
                package = package_dir_for(node.level, path)
                visible = [package] if package else []
            for dotted in names:
                top = dotted.split(".")[0]
                if top in {"posix", "nt"}:
                    return f"native import at {path}:{node.lineno}: the raw OS module is outside the file model"
                if top in sys.builtin_module_names or top == "_sv_verifier":
                    continue
                try:
                    relative = isinstance(node, ast.ImportFrom) and node.level
                    if (not runtime and not relative and top in sys.stdlib_module_names and
                            find_in_roots(top, visible)):
                        return f"stdlib shadow {top!r} at {path}:{node.lineno}: unmodelled bootstrap import cache"
                    found = find_in_roots(dotted, visible, source_aliases=source_aliases)
                except ExternTop as exc:
                    if not runtime:
                        return f"native import at {path}:{node.lineno}: {exc}"
                    continue
                if found is None:
                    continue
                if not runtime and not imported and final and roots != absolute_visible:
                    return f"native import at {path}:{node.lineno}: import roots change after a local import"
                leaf, inits = found
                for target in [*inits, leaf]:
                    if target in seen:
                        continue
                    try:
                        with tokenize.open(target) as fh:
                            child_source = fh.read()
                    except (OSError, UnicodeError, SyntaxError):
                        return f"unreadable native import: {target}"
                    pending.append((child_source, target, absolute_visible))
    if mapping_reads:
        packages = [(os.path.dirname(path), {os.path.dirname(alias) for alias in aliases})
                    for path, aliases in source_aliases.items() if os.path.basename(path) == "__init__.py"]
        for path, aliases in source_aliases.items():
            aliases.add(path)
            pending_aliases = list(aliases)
            while pending_aliases:
                filename = pending_aliases.pop()
                for directory, roots in packages:
                    if filename.startswith(directory + os.sep):
                        for root in roots:
                            alias = root + filename[len(directory):]
                            if alias not in aliases:
                                if len(aliases) >= 64:
                                    return "native import filename aliases exceed 64 files"
                                aliases.add(alias)
                                pending_aliases.append(alias)
        for (path, *position), name in tuple(mapping_reads.items()):
            for alias in source_aliases.get(path, ()):
                mapping_reads[(alias, *position)] = name
    return None
