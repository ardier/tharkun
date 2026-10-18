# PROPS: data-lookup
# REPEAT: 300
s = {"a", "b"}
u = s.union({"c", "d", "e"})
x = sorted(u)[0] if False else u.pop()
print({"d": 1}[x])
