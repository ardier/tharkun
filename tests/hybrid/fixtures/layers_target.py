"""Three call layers. The harness only ever calls top(); the interesting guards live
one and two calls DOWN, in functions the harness never names."""


def top(a, b, c):
    trail = []
    if a == 100:                 # layer 1 guard, in the entry function
        trail.append("L1")
        trail += mid(a, b, c)
    return trail


def mid(a, b, c):
    trail = []
    if b == 200:                 # layer 2 guard, one call down
        trail.append("L2")
        trail += bottom(a, b, c)
    return trail


def bottom(a, b, c):
    trail = []
    if c * c == 900:             # layer 3 guard, two calls down (nonlinear)
        trail.append("L3")
        if a + b + c == 330:     # layer 3, nested
            trail.append("DEEP")
    return trail
