# PROPS: data-lookup
# REPEAT: 300
h = repr(object())
x = int(h[-3], 16)
a = [0] * 15
print(a[x])
