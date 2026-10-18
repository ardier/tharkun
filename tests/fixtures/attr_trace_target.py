"""Fixture for attr_tracer tests — classes with self.xxx guards."""


class Parser:
    def __init__(self, allow_unicode: bool = False, strict: bool = True):
        self.allow_unicode = allow_unicode
        self.strict = strict
        self.buffer = ""

    def parse(self, data: str) -> str:
        if self.allow_unicode:
            return data.upper()
        return data.encode("ascii", errors="replace").decode()

    def validate(self, text: str) -> bool:
        if self.strict and len(text) == 0:
            return False
        return True


class Renderer:
    def __init__(self):
        self.md = None
        self.tag = "div"

    def setup_markdown(self, md):
        self.md = md

    def render(self, content: str) -> str:
        if self.md and self.md.references:
            return f"<{self.tag}>{content}</{self.tag}>"
        return content


class Config:
    VERSION = "2.0"

    def __init__(self, name: str = "", enabled: bool = False):
        self.name = name
        self.enabled = enabled

    def is_ready(self) -> bool:
        return self.enabled and len(self.name) > 0

    def process(self) -> str:
        if self.is_ready():
            return f"processing {self.name}"
        return "not ready"
