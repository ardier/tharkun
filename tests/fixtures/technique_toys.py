"""One guard per technique — each shaped so exactly that technique should crack it."""


# --- static: a plain literal comparison, no execution needed -------------------
def by_static(code):
    if code == 4321:
        return "STATIC"
    return "no"


# --- redqueen: the demanded value exists only at RUNTIME, behind a transform.
# Nothing static can read 0xC0DE out of the source; it has to be copied out of the
# comparison as it executes.
def by_redqueen(token):
    expected = (0x1234 * 3 + 7) & 0xFFFF          # computed, not written
    if int.from_bytes(token[:2], "big") == expected:
        return "REDQUEEN"
    return "no"


# --- minleaf: a guard on ONE leaf deep inside live state. The cheap solvers would
# rebuild the whole structure; the minimal mutation is a single element.
def by_minleaf(cfg):
    if cfg["limits"][2] > 100:
        return "MINLEAF"
    return "no"


# --- Contradictory guard: unreachable. No input satisfies it, and only a sound decider
# can say so rather than merely failing to find one.
def by_dead_guard(n):
    if n > 10 and n < 5:
        return "UNREACHABLE"
    return "no"
