"""Reached only through chain.dispatch — mirrors ntlm_auth's des.py."""


def check(key):
    if len(key) != 7:            # width guard: satisfied only by a SHORT payload
        return "SHORT-KEY"       # <- the target
    return "ok"
