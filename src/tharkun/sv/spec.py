"""Read an SV-Benchmarks Python property (``.prp``) and task definition (``.yml``) -- stdlib only.

A Python property file has one line per exception class::

    CHECK( init(main()), ! uncaught(IndexError))

The task definition (format 2.0) is YAML; tharkun reads only what it needs
(``input_files`` and the property list) with a line parser so its core stays
dependency-free.  A task whose YAML the line parser cannot read is refused.
"""

from __future__ import annotations

import os
import re

_CHECK = re.compile(r"CHECK\(\s*init\(\w+\(\)\)\s*,\s*!\s*uncaught\((\w+)\)\s*\)")


def read_property(path: str) -> tuple[str, tuple[str, ...]]:
    """``(name, classes)`` for a ``.prp`` file; ``name`` is the file stem."""
    with open(path, encoding="utf-8", newline="") as fh:
        return parse_property(fh.read(), path)


def parse_property(text: str, path: str = "property.prp") -> tuple[str, tuple[str, ...]]:
    """Parse property text, also used to bind independently checked witnesses."""
    classes = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _CHECK.fullmatch(line)
        if not m:
            raise ValueError(f"{path}: not an uncaught(E) property line: {line!r}")
        classes.append(m.group(1))
    if not classes:
        raise ValueError(f"{path}: no CHECK line")
    return os.path.splitext(os.path.basename(path))[0], tuple(classes)


def program_of(path: str) -> str:
    """The first declared input of a task, or a direct .py program."""
    from tharkun.solve.environment import TaskEnvironment
    files = TaskEnvironment.from_task(path).files
    if not files:
        raise ValueError(f"{path}: cannot read input_files")
    return files[0]
