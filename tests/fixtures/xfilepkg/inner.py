"""Never called by the entry point's caller — reached only THROUGH outer.py."""


def transform(payload):
    if len(payload) > 4:            # answering this with a str kills the guard below
        if payload[0] == 0x7F:      # only satisfiable while payload is bytes
            return "INNER-DEEP"
        return "INNER-SHALLOW"
    return "INNER-SHORT"
