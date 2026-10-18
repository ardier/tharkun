"""An if/elif chain on bitmask flags — mirrors compute_keys._get_exchange_key_ntlm_v1."""
from chainpkg import deep

EXTENDED = 0x80000
LM_KEY = 0x80
THIRD = 0x400000


def dispatch(flags, payload):
    if flags & EXTENDED:         # a committed value from an earlier solve lands HERE
        return "extended"
    elif flags & LM_KEY:         # the branch that reaches deep.check
        return deep.check(payload[:7])
    elif flags & THIRD:
        return "third"
    return "none"
