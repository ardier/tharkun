# PROPS: data-lookup
# REPEAT: 400
# HASHSEED: 2
import os
x = os.urandom(1)[0]
a = [0] * 255
print(a[x])
