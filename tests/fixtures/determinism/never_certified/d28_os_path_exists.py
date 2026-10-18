# PROPS: data-lookup
import os
d = {True: 1}
print(d[os.path.exists("/nonexistent-marker-28")])
