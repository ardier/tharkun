# PROPS: data-lookup
# REPEAT: 300
def first(c):
    for x in c:
        return x
s = {"a", "b", "c", "d", "e"}
print({"d": 1}[first(s)])
