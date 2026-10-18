# PROPS: data-lookup
# REPEAT: 300
s = {"a", "b", "c", "d", "e"}
t = s
print({"d": 1}[t.pop()])
