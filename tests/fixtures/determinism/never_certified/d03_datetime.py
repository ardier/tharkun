# PROPS: data-lookup
# REPEAT: 400
# HASHSEED: 2
import datetime
t = datetime.datetime.now().microsecond
a = [0] * 999999
print(a[t])
