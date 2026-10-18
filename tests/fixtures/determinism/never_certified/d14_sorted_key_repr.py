# PROPS: data-lookup
# REPEAT: 300
class P:
    pass
p, q = P(), P()
first = sorted([p, q], key=repr)[0]
print([0][0 if first is p else 1])
