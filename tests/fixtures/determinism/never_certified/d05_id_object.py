# PROPS: data-lookup
# REPEAT: 400
# HASHSEED: 2
x = id(object()) % 4099
a = [0] * 4098
print(a[x])
