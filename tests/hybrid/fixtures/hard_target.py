# Guards that defeat libFuzzer CMP/value-profile: the RHS constant does not reveal
# the input value (nonlinear square, then an uninvertible modular hash).
def hard(x, y):
    if x * x == 1522756:                                   # x == 1234 (square)
        if (y * 2654435761 + 1) % 4294967291 == 987654:    # modular, uninvertible
            return "CRACKED"                               # depth 2
        return "half"                                      # depth 1
    return "none"                                          # depth 0
