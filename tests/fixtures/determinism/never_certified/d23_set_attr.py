# PROPS: data-lookup
# REPEAT: 300
class B:
    def __init__(self):
        self.s = {"a", "b", "c", "d", "e"}
b = B()
print({"d": 1}[next(iter(b.s))])
