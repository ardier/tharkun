# PROPS: data-lookup
# REPEAT: 400
# HASHSEED: 2
import os
print(os.environ["SV_NOT_SET_ANYWHERE"] if len(os.environ) % 97 == 3 else 0)
