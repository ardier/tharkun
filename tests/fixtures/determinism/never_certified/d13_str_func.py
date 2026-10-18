# PROPS: data-lookup
# REPEAT: 300
class P:
    pass
x = int(str(P())[-4], 16) % 3
print([0, 0][x])
