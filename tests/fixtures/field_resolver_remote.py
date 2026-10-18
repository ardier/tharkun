"""Cross-file fixture for field_resolver tests.

Classes here are imported by field_resolver_target_crossfile.py to test
cross-module class resolution.
"""


class RemoteConfig:
    host: str
    port: int
    secure: bool

    def __init__(self, host: str = "remote.host", port: int = 9000,
                 secure: bool = True):
        self.host = host
        self.port = port
        self.secure = secure


class RemoteAuth:
    api_key: str
    ttl: int

    def __init__(self, api_key: str = "", ttl: int = 3600):
        self.api_key = api_key
        self.ttl = ttl
