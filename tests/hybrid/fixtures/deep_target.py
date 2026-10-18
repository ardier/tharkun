def parse(magic, kind, size):
    if magic == 1094795585:                 # 0x41414141 — exact 4-byte magic
        if kind * kind == 49 and kind > 0:  # nonlinear
            if size > 1000:
                return "DEEP"               # frontier target
    return "shallow"
