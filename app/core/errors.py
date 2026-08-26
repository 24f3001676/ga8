"""Shared error types mapped to exact HTTP contracts."""


class InvalidInput(Exception):
    """Raised when a request must produce HTTP 400 {"error":"INVALID_INPUT"}."""


class Conflict(Exception):
    """Raised when a request must produce HTTP 409 {"error":"<code>"}."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code
