"""Phase 4: State projection -- build a StateIndex mapping leaf paths to LeafRefs.

Traverses root objects recursively, producing getter/setter closures for
every patchable leaf value.
"""

from __future__ import annotations

from typing import Any

from tharkun.minleaf.types import LeafRef, Path, StateIndex

_PRIMITIVE_TYPES = (int, bool, float, str, type(None))


def build_state_index(
    roots: dict[str, object],
    bytes_policy: str,
    *,
    env: dict[str, object] | None = None,
    max_depth: int = 12,
    max_container_len: int = 4096,
) -> StateIndex:
    """Build a StateIndex mapping every patchable leaf path to a LeafRef."""
    leaves: dict[Path, LeafRef] = {}
    seen: set[int] = set()

    for name in sorted(roots.keys()):
        obj = roots[name]
        path: Path = (name,)
        _traverse(
            obj=obj,
            path=path,
            root_name=name,
            roots=roots,
            env=env,
            leaves=leaves,
            seen=seen,
            bytes_policy=bytes_policy,
            depth=0,
            max_depth=max_depth,
            max_container_len=max_container_len,
        )

    return StateIndex(leaves=leaves, bytes_policy=bytes_policy, roots=roots)


def _traverse(
    *,
    obj: Any,
    path: Path,
    root_name: str,
    roots: dict[str, object],
    env: dict[str, object] | None,
    leaves: dict[Path, LeafRef],
    seen: set[int],
    bytes_policy: str,
    depth: int,
    max_depth: int,
    max_container_len: int,
) -> None:
    """Recursively traverse an object to find all leaf paths."""
    if depth > max_depth:
        return

    # Handle primitives (leaf)
    if isinstance(obj, _PRIMITIVE_TYPES):
        getter, setter = _make_primitive_accessors(
            path, root_name, roots, env
        )
        leaves[path] = LeafRef(
            path=path,
            typ=type(obj),
            getter=getter,
            setter=setter,
        )
        return

    # Handle bytes
    if isinstance(obj, bytes):
        if bytes_policy == "atomic":
            getter, setter = _make_primitive_accessors(
                path, root_name, roots, env
            )
            leaves[path] = LeafRef(
                path=path, typ=bytes, getter=getter, setter=setter,
            )
        else:  # per_byte
            # The bytes object is patchable as a WHOLE as well as byte-by-byte. Without
            # this, a guard like ``token == b"MAGIC"`` is unsolvable under the per_byte
            # policy: b"AB" projects to two int leaves and no per-byte edit makes a
            # 2-byte value equal a 5-byte one. Registering the container keeps the cost
            # model intact (still one distinct path) and makes the length change
            # expressible, exactly as for list/dict.
            whole_get, whole_set = _make_primitive_accessors(path, root_name, roots, env)
            leaves[path] = LeafRef(path=path, typ=bytes, getter=whole_get, setter=whole_set)
            for i in range(min(len(obj), max_container_len)):
                byte_path = path + (i,)
                g, s = _make_byte_accessors(path, i, root_name, roots, env)
                leaves[byte_path] = LeafRef(
                    path=byte_path, typ=int, getter=g, setter=s,
                )
        return

    # Handle set (atomic leaf, no stable indexing)
    if isinstance(obj, (set, frozenset)):
        getter, setter = _make_primitive_accessors(
            path, root_name, roots, env
        )
        leaves[path] = LeafRef(
            path=path, typ=type(obj), getter=getter, setter=setter,
        )
        return

    # Cycle detection for compound objects
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    try:
        # A mutable container is *itself* patchable, not only its elements: a guard on
        # ``len(x)`` is satisfied by replacing the container, which no per-element leaf
        # can express.  Registering the container path alongside its elements keeps the
        # cost model intact (still one distinct path touched) while making that
        # structural change reachable.
        if isinstance(obj, (dict, list)):
            c_get, c_set = _make_primitive_accessors(path, root_name, roots, env)
            leaves[path] = LeafRef(path=path, typ=type(obj), getter=c_get, setter=c_set)

        # Handle dict
        if isinstance(obj, dict):
            for key in sorted(obj.keys(), key=repr):
                if not isinstance(key, (str, int)):
                    continue
                child_path = path + (key,)
                _traverse(
                    obj=obj[key], path=child_path, root_name=root_name,
                    roots=roots, env=env, leaves=leaves, seen=seen,
                    bytes_policy=bytes_policy, depth=depth + 1,
                    max_depth=max_depth, max_container_len=max_container_len,
                )
            return

        # Handle list
        if isinstance(obj, list):
            for i in range(min(len(obj), max_container_len)):
                child_path = path + (i,)
                _traverse(
                    obj=obj[i], path=child_path, root_name=root_name,
                    roots=roots, env=env, leaves=leaves, seen=seen,
                    bytes_policy=bytes_policy, depth=depth + 1,
                    max_depth=max_depth, max_container_len=max_container_len,
                )
            return

        # Handle tuple (read-only)
        if isinstance(obj, tuple):
            for i in range(min(len(obj), max_container_len)):
                child_path = path + (i,)
                _traverse_tuple_element(
                    obj=obj[i], path=child_path, root_name=root_name,
                    roots=roots, env=env, leaves=leaves, seen=seen,
                    bytes_policy=bytes_policy, depth=depth + 1,
                    max_depth=max_depth, max_container_len=max_container_len,
                )
            return

        # Handle objects with __dict__ (dataclasses, plain objects)
        if hasattr(obj, "__dict__"):
            for attr_name in sorted(obj.__dict__.keys()):
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue
                val = getattr(obj, attr_name)
                if callable(val) and not isinstance(val, _PRIMITIVE_TYPES):
                    continue
                child_path = path + (attr_name,)
                _traverse(
                    obj=val, path=child_path, root_name=root_name,
                    roots=roots, env=env, leaves=leaves, seen=seen,
                    bytes_policy=bytes_policy, depth=depth + 1,
                    max_depth=max_depth, max_container_len=max_container_len,
                )
            return
    finally:
        seen.discard(obj_id)


def _traverse_tuple_element(
    *,
    obj: Any,
    path: Path,
    root_name: str,
    roots: dict[str, object],
    env: dict[str, object] | None,
    leaves: dict[Path, LeafRef],
    seen: set[int],
    bytes_policy: str,
    depth: int,
    max_depth: int,
    max_container_len: int,
) -> None:
    """Traverse a tuple element (read-only: setter raises TypeError)."""
    if depth > max_depth:
        return

    if isinstance(obj, _PRIMITIVE_TYPES):
        # Read-only leaf
        def _make_getter(p: Path, rn: str, r: dict) -> Any:
            def getter() -> Any:
                return _resolve_path(p, r)
            return getter

        def _make_setter() -> None:
            def setter(v: Any) -> None:
                raise TypeError("Cannot set value on tuple element")
            return setter

        leaves[path] = LeafRef(
            path=path,
            typ=type(obj),
            getter=_make_getter(path, root_name, roots),
            setter=_make_setter(),
        )
        return

    # For compound objects inside tuples, recurse normally but read-only
    if isinstance(obj, (list, dict)):
        _traverse(
            obj=obj, path=path, root_name=root_name,
            roots=roots, env=env, leaves=leaves, seen=seen,
            bytes_policy=bytes_policy, depth=depth,
            max_depth=max_depth, max_container_len=max_container_len,
        )
        return

    if isinstance(obj, tuple):
        for i in range(min(len(obj), max_container_len)):
            child_path = path + (i,)
            _traverse_tuple_element(
                obj=obj[i], path=child_path, root_name=root_name,
                roots=roots, env=env, leaves=leaves, seen=seen,
                bytes_policy=bytes_policy, depth=depth + 1,
                max_depth=max_depth, max_container_len=max_container_len,
            )
        return


def _resolve_path(path: Path, roots: dict[str, object]) -> Any:
    """Resolve a path to a value by walking from roots."""
    obj = roots[path[0]]
    for seg in path[1:]:
        if isinstance(seg, int):
            if isinstance(obj, (list, tuple, bytes)):
                obj = obj[seg]
            elif isinstance(obj, dict):
                obj = obj[seg]
            else:
                obj = obj[seg]
        else:  # str
            if isinstance(obj, dict):
                obj = obj[seg]
            else:
                obj = getattr(obj, seg)
    return obj


def _set_path(path: Path, roots: dict[str, object], value: Any) -> None:
    """Set a value at a path by walking from roots."""
    if len(path) == 1:
        # Root-level primitive: rebind in roots
        roots[path[0]] = value
        return

    parent = roots[path[0]]
    for seg in path[1:-1]:
        if isinstance(seg, int):
            if isinstance(parent, dict):
                parent = parent[seg]
            else:
                parent = parent[seg]
        else:
            if isinstance(parent, dict):
                parent = parent[seg]
            else:
                parent = getattr(parent, seg)

    last_seg = path[-1]
    if isinstance(last_seg, int):
        if isinstance(parent, dict):
            parent[last_seg] = value
        else:
            parent[last_seg] = value
    else:
        if isinstance(parent, dict):
            parent[last_seg] = value
        else:
            setattr(parent, last_seg, value)


def _make_primitive_accessors(
    path: Path,
    root_name: str,
    roots: dict[str, object],
    env: dict[str, object] | None,
):
    """Create getter/setter closures for a primitive value."""
    p = path  # capture

    def getter() -> Any:
        return _resolve_path(p, roots)

    if len(p) == 1 and env is not None:
        # Root-level primitive: setter rebinds both roots and env
        def setter(value: Any) -> None:
            roots[p[0]] = value
            env[p[0]] = value
    else:
        def setter(value: Any) -> None:
            _set_path(p, roots, value)

    return getter, setter


def _make_byte_accessors(
    parent_path: Path,
    byte_index: int,
    root_name: str,
    roots: dict[str, object],
    env: dict[str, object] | None,
):
    """Create getter/setter closures for a single byte in a bytes object."""
    pp = parent_path
    bi = byte_index

    def getter() -> int:
        bs = _resolve_path(pp, roots)
        return bs[bi]

    if len(pp) == 1 and env is not None:
        # Root-level bytes: setter rebuilds and rebinds both roots and env
        def setter(value: int) -> None:
            bs = _resolve_path(pp, roots)
            new_bs = bytearray(bs)
            new_bs[bi] = value
            new_bs = bytes(new_bs)
            roots[pp[0]] = new_bs
            env[pp[0]] = new_bs
    else:
        def setter(value: int) -> None:
            bs = _resolve_path(pp, roots)
            new_bs = bytearray(bs)
            new_bs[bi] = value
            new_bs = bytes(new_bs)
            _set_path(pp, roots, new_bs)

    return getter, setter
