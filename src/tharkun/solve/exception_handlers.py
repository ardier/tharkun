"""Static exception identity relations, without execution or handler traversal."""
from __future__ import annotations

import builtins


def builtin_exception_chain(name: str) -> tuple[str, ...]:
    """Canonical builtin identities, independent of names bound in target source."""
    value = getattr(builtins, name, None)
    if not isinstance(value, type) or not issubclass(value, BaseException):
        return ()
    return tuple("builtins." + cls.__name__ for cls in value.__mro__)


def handler_match(class_id: str | None, class_chain_ids, handler: dict, *,
                  class_chain_complete: bool = False) -> str:
    """Return caught/not_caught/maybe_caught for a resolved class relationship.

    This does not prove reachability, handler-binding stability or that a
    finalizer cannot bypass the handler. A negative source-class match needs
    the complete ancestry premise; builtin ancestry is checked independently.
    """
    if not isinstance(handler, dict) or handler.get("unresolved") is not False:
        return "maybe_caught"
    if handler.get("bare") is True:
        return "caught"
    identities = handler.get("class_ids")
    if (not isinstance(identities, (list, tuple)) or not identities
            or any(type(identity) is not str for identity in identities)):
        return "maybe_caught"
    if class_id is None:
        return "caught" if "builtins.BaseException" in identities else "maybe_caught"
    if (type(class_id) is not str or not isinstance(class_chain_ids, (list, tuple))
            or not class_chain_ids or class_chain_ids[0] != class_id
            or any(type(identity) is not str for identity in class_chain_ids)):
        return "maybe_caught"
    builtin_chain = ()
    if class_id.startswith("builtins."):
        builtin_chain = builtin_exception_chain(class_id.removeprefix("builtins."))
        if not builtin_chain or tuple(class_chain_ids) != builtin_chain:
            return "maybe_caught"
    if any(identity in class_chain_ids for identity in identities):
        return "caught"
    return "not_caught" if builtin_chain or class_chain_complete is True else "maybe_caught"
