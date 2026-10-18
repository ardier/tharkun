# PROPS: data-lookup
# REPEAT: 300
o = object()
x = int(f"{o}"[-3], 16)
print([0] * 15 [x] if False else ([0] * 15)[x])
