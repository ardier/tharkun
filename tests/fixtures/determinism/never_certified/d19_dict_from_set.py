# PROPS: data-lookup
# REPEAT: 300
s = {"a", "b", "c", "d", "e"}
d = dict.fromkeys(s, 0)
k = list(d)[0]
print({"d": 1}[k])
