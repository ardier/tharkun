from xfilepkg import inner


def entry(mode, payload):
    if mode == 1234:                # caller-side gate on whether inner runs at all
        return inner.transform(payload)
    return "SKIPPED"
