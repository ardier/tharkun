# PROPS: data-lookup
# REPEAT: 300
s = frozenset(["a", "b", "c", "d", "e"])
for x in s:
    print({"d": 1}[x])
    break
