# PROPS: data-lookup
# REPEAT: 400
# HASHSEED: 2
import random
x = random.SystemRandom().randint(0, 999)
a = [0] * 999
print(a[x])
