"""Safe YAML transport for Python invariant witnesses; parsing is not validation."""
from __future__ import annotations

import yaml

from tharkun.solve.obligations import Invariant

# [internal ref] review (finalizer CPU/memory): libyaml's C scanner/emitter is used when
# available (~15-20x faster than PyYAML's pure-Python implementation on a large
# witness) with IDENTICAL safety guarantees -- CSafeLoader/CSafeDumper are the
# same safe-subset construction/representation, just C-accelerated scanning and
# emitting. Falls back to the pure-Python classes where libyaml isn't built in.
_BaseLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_BaseDumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)


class _Loader(_BaseLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key, value in node.value:
            key = self.construct_object(key, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("non-string or duplicate YAML mapping key")
            result[key] = self.construct_object(value, deep=deep)
        return result


def loads(text: str) -> list[dict]:
    if len(text) > 4 * 1024 * 1024:
        raise ValueError("witness exceeds 4 MiB")
    # Aliases/cycles add no value to this small interchange format.
    if any(isinstance(e, yaml.AliasEvent) for e in yaml.parse(text, Loader=_Loader)):
        raise ValueError("YAML aliases are unsupported")
    entries = yaml.load(text, Loader=_Loader)
    if not isinstance(entries, list) or not entries:
        raise ValueError("expected a nonempty sequence of invariant sets")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"entry_type", "metadata", "content"} or entry["entry_type"] != "invariant_set":
            raise ValueError("invalid invariant_set")
        meta = entry["metadata"]
        if not isinstance(meta, dict) or meta.get("format_version") not in ("2.0", "2.3"):
            raise ValueError("unsupported witness version")
        if not isinstance(entry["content"], list):
            raise ValueError("content must be a sequence")
        for item in entry["content"]:
            Invariant.from_entry(item)
    return entries


def dumps(entries: list[dict]) -> str:
    text = yaml.dump(entries, Dumper=_BaseDumper, sort_keys=False, allow_unicode=True)
    loads(text)
    return text
