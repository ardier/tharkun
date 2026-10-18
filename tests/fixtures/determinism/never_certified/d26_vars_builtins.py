# PROPS: data-lookup
# REPEAT: 300
f = vars(__builtins__)["id"] if isinstance(__builtins__, dict) is False else __builtins__["id"]
x = (f(object()) >> 4) % 3
print([0, 0][x])
