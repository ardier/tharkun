# PROPS: data-lookup
# REPEAT: 300
x = "abcdefgh".__hash__() % 7
a = [0] * 6
print(a[x])
