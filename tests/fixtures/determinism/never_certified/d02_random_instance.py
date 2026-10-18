# PROPS: data-lookup
# REPEAT: 400
# HASHSEED: 2
import random
r = random.Random()
x = r.randint(0, 999)
a = [0] * 999
print(a[x])
