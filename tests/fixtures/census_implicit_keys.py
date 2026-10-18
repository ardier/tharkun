class C:
    def keys(self):
        assert hash("x") != 3914910683811788151
        return []

    def __getitem__(self, k):
        return 0

dict(**C())
