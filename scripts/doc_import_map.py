#!/usr/bin/env python3
"""Measure and verify subsystem AST import maps in documentation against source files.

Verifies:
1. Every file listed in docs/SV.md and docs/ABSINT.md subsystem sections exists on disk.
2. Every Python module in src/tharkun/sv/ and src/tharkun/solve/absint/ is cataloged.
3. The documented SV import and imported-by graphs match measured AST imports.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys
from typing import Dict, List, Set, Tuple


def compute_ast_imports(pkg_dir: pathlib.Path, pkg_prefix: str) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]]]:
    mods = sorted(p.stem for p in pkg_dir.glob("*.py"))
    imports: Dict[str, Set[str]] = {m: set() for m in mods}
    for m in mods:
        source_path = pkg_dir / f"{m}.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                if n.level == 1 and n.module is None:
                    for a in n.names:
                        if a.name in mods:
                            imports[m].add(a.name)
                elif n.level == 1 and n.module and n.module.split(".")[0] in mods:
                    imports[m].add(n.module.split(".")[0])
                elif n.level == 0 and n.module:
                    if n.module in mods:
                        imports[m].add(n.module)
                    if n.module == pkg_prefix:
                        for a in n.names:
                            if a.name in mods:
                                imports[m].add(a.name)
                    elif n.module.startswith(pkg_prefix + "."):
                        parts = n.module[len(pkg_prefix) + 1:].split(".")
                        if parts[0] in mods:
                            imports[m].add(parts[0])
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name in mods:
                        imports[m].add(a.name)
                    if a.name.startswith(pkg_prefix + "."):
                        parts = a.name[len(pkg_prefix) + 1:].split(".")
                        if parts[0] in mods:
                            imports[m].add(parts[0])
        imports[m].discard(m)

    imported_by: Dict[str, List[str]] = {m: sorted(k for k in mods if m in imports[k]) for m in mods}
    return imports, imported_by


def parse_doc_subsystem(doc_text: str, heading_token: str, path_prefix: str) -> Tuple[List[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    if heading_token not in doc_text:
        return [], {}, {}
    sect = doc_text[doc_text.index(heading_token):]
    next_sec = sect.find("\n## ")
    if next_sec != -1:
        sect = sect[:next_sec]

    doc_files: List[str] = []
    doc_imp: Dict[str, Set[str]] = {}
    doc_by: Dict[str, Set[str]] = {}
    cur = None

    pattern_file = re.compile(rf"-\s+`{re.escape(path_prefix)}/([A-Za-z0-9_]+)\.py`\s*$")
    pattern_imp = re.compile(rf"\s+-\s+\*\*Imports \([^)]+\)\*\*:\s*(.*)")
    pattern_by = re.compile(rf"\s+-\s+\*\*Is imported by \([^)]+\)\*\*:\s*(.*)")

    for line in sect.splitlines():
        mf = pattern_file.match(line)
        if mf:
            cur = mf.group(1)
            doc_files.append(cur)
            continue
        mi = pattern_imp.match(line)
        if mi and cur:
            targets = set(re.findall(rf"{re.escape(path_prefix)}/([A-Za-z0-9_]+)\.py", mi.group(1)))
            doc_imp[cur] = targets
            continue
        mb = pattern_by.match(line)
        if mb and cur:
            targets = set(re.findall(rf"{re.escape(path_prefix)}/([A-Za-z0-9_]+)\.py", mb.group(1)))
            doc_by[cur] = targets

    return doc_files, doc_imp, doc_by


def check_docs(root: pathlib.Path, verbose: bool = False) -> bool:
    all_ok = True

    # 1. Check tharkun.sv
    sv_dir = root / "src/tharkun/sv"
    sv_doc = root / "docs/SV.md"
    sv_text = sv_doc.read_text(encoding="utf-8")
    sv_mods = sorted(p.stem for p in sv_dir.glob("*.py"))
    measured_imp, measured_by = compute_ast_imports(sv_dir, "tharkun.sv")

    doc_files, doc_imp, doc_by = parse_doc_subsystem(sv_text, "### Subsystem: `tharkun.sv`", "src/tharkun/sv")

    # Check existence on disk of all doc files
    missing_disk = [m for m in doc_files if not (sv_dir / f"{m}.py").exists()]
    if missing_disk:
        print(f"ERROR: Files in docs/SV.md not found on disk: {missing_disk}", file=sys.stderr)
        all_ok = False

    missing_doc = sorted(set(sv_mods) - set(doc_files))
    if missing_doc:
        print(f"ERROR: Modules in src/tharkun/sv missing from docs/SV.md: {missing_doc}", file=sys.stderr)
        all_ok = False

    # Check edges
    imp_mismatches = [(m, doc_imp.get(m, set()), measured_imp[m]) for m in sv_mods if doc_imp.get(m, set()) != measured_imp[m]]
    by_mismatches = [(m, doc_by.get(m, set()), set(measured_by[m])) for m in sv_mods if doc_by.get(m, set()) != set(measured_by[m])]

    if imp_mismatches:
        print(f"ERROR: Imports mismatches in docs/SV.md: {imp_mismatches}", file=sys.stderr)
        all_ok = False
    if by_mismatches:
        print(f"ERROR: Imported-by mismatches in docs/SV.md: {by_mismatches}", file=sys.stderr)
        all_ok = False

    if verbose:
        print(f"tharkun.sv: {len(sv_mods)} modules checked. Disk missing: {len(missing_disk)}. Doc missing: {len(missing_doc)}. Import mismatches: {len(imp_mismatches)}. Imported-by mismatches: {len(by_mismatches)}.")

    # 2. Check tharkun.solve.absint
    abs_dir = root / "src/tharkun/solve/absint"
    abs_doc = root / "docs/ABSINT.md"
    abs_text = abs_doc.read_text(encoding="utf-8")
    abs_mods = sorted(p.stem for p in abs_dir.glob("*.py"))
    abs_doc_files, _, _ = parse_doc_subsystem(abs_text, "### Subsystem: `tharkun.solve.absint`", "src/tharkun/solve/absint")

    abs_missing_disk = [m for m in abs_doc_files if not (abs_dir / f"{m}.py").exists()]
    if abs_missing_disk:
        print(f"ERROR: Files in docs/ABSINT.md not found on disk: {abs_missing_disk}", file=sys.stderr)
        all_ok = False

    abs_missing_doc = sorted(set(abs_mods) - set(abs_doc_files))
    if abs_missing_doc:
        print(f"ERROR: Modules in src/tharkun/solve/absint missing from docs/ABSINT.md: {abs_missing_doc}", file=sys.stderr)
        all_ok = False

    if verbose:
        print(f"tharkun.solve.absint: {len(abs_mods)} modules checked. Disk missing: {len(abs_missing_disk)}. Doc missing: {len(abs_missing_doc)}.")

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Root of repository")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    ok = check_docs(root, verbose=args.verbose)
    if ok:
        if args.verbose:
            print("All documentation import maps are valid and all files exist.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
