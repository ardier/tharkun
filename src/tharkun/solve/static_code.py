"""Proof-only source preparation; runtime admission and its observers use original source."""
from __future__ import annotations

import ast
import time


def prepared_code(source, filename='<program>', *, deadline=None):
    if not isinstance(source, (str, bytes)):
        return source, None
    spellings = ('eval', 'exec') if isinstance(source, str) else (b'eval', b'exec')
    if not any(s in source for s in spellings) and source.isascii():
        return source, None
    from .contracts import get
    from .lowering.eval_source import lower_module_code

    enabled = set()
    for name, kinds in [('eval', {'Any', 'object'}), ('exec', {'None'}), ('compile', {'types.CodeType | ast.AST'})]:
        contract = get(name)
        if (contract is not None and contract.tested and contract.returns is not None and contract.returns.tested
                and contract.returns.type in kinds
                and all(c.tested and c.condition != 'unknown' for c in contract.raises)):
            enabled.add(name)
    if not enabled:
        return source, None
    if deadline is None:
        deadline = time.process_time() + 3.0
    try:
        lowered = lower_module_code(ast.parse(source, filename), enabled=enabled, deadline=deadline)
        if {'compile', 'exec'} <= enabled:
            from .lowering.compiled_source import lower_compile_snapshots

            snapshots = lower_compile_snapshots(lowered.tree, deadline=deadline)
            if snapshots.changed:
                expanded = lower_module_code(snapshots.tree, enabled=enabled, deadline=deadline, preserve_origins=True)
                expanded.lowered += snapshots.lowered + lowered.lowered
                expanded.synthetic |= snapshots.synthetic
                lowered = expanded
        return (ast.unparse(lowered.tree), lowered) if lowered.changed else (source, None)
    except (SyntaxError, ValueError, RecursionError, OverflowError):
        return source, None


def proof_import_reason(source, filename='<program>'):
    from .extern import native_import_reason

    prepared, _ = prepared_code(source, filename)
    return native_import_reason(prepared, filename)
