# PROPS: data-lookup
# REPEAT: 300
import builtins
x = builtins.id(object()) % 4099
a = [0] * 4098
print(a[x])
