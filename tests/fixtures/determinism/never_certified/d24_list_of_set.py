# PROPS: data-lookup
# REPEAT: 300
s = {"a", "b", "c", "d", "e"}
box = [s]
print({"d": 1}[list(box[0])[0]])
