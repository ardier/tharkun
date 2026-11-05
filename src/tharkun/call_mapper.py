"""Compatibility shim (#0606, plan #0600): ``tharkun.call_mapper`` moved to ``tharkun.solve.frontend.call_mapper``.

Importing this name yields the moved module itself (the shim aliases it in
``sys.modules``), so every attribute -- public or private -- and every object
identity is preserved.  Permanent: external consumers (eldarion,
eregion_experiments, eregion-synthesis) pin this path.  No logic here.
"""
from tharkun._shims import alias as _alias

_alias(__name__, "tharkun.solve.frontend.call_mapper")
