"""Cross-file fixture: imports classes from field_resolver_remote.py."""

from tests.fixtures.field_resolver_remote import RemoteConfig, RemoteAuth


class RemoteClient:
    """Client whose fields are types defined in another file."""
    config: RemoteConfig
    auth: RemoteAuth

    def __init__(self, config: RemoteConfig, auth: RemoteAuth):
        self.config = config
        self.auth = auth

    def is_authenticated(self) -> bool:
        # chain: config -> secure (bool)  — cross-file resolution
        if not self.config.secure:
            return False
        # chain: auth -> ttl (int)  — cross-file resolution
        if self.auth.ttl <= 0:
            return False
        return True
