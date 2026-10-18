def f(x):
    if x == 7:          # line 2 — needs an int; a str seed can never satisfy it
        return "HIT"     # line 3 — the reach-search target
    return "no"
