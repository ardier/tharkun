# PROPS: data-lookup
# REPEAT: 400
# HASHSEED: 2
s = {"a", "b", "c", "d", "e"}
first = next(iter(s))
d = {"d": 1}
print(d[first])
