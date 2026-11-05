"""Module aliasing for the permanent compatibility shims (#0606, plan #0600).

A shim file at an old module path calls ``alias(__name__, "<new home>")``.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def alias(shim_name: str, target: str) -> ModuleType:
    """Make ``shim_name`` resolve to the module ``target`` everywhere.

    Replaces the shim's own entry in ``sys.modules`` with the target module.
    The import system returns whatever ``sys.modules`` holds once a module's
    body has run (the documented replacement idiom), so ``import shim``,
    ``from shim import name`` -- private names included -- and
    ``importlib.import_module("shim")`` all yield the real module, and object
    identity (``shim.config is target.config``) holds by construction.
    """
    module = importlib.import_module(target)
    sys.modules[shim_name] = module
    return module
