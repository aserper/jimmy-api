from __future__ import annotations


class JimmyError(Exception):
    """Base exception for all jimmy errors."""


class APIError(JimmyError):
    """Raised when the API returns a non-2xx status."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class AttachmentTooLargeError(JimmyError):
    """Raised when an attachment exceeds the 50KB limit."""

    def __init__(self, size: int, max_size: int) -> None:
        super().__init__(f"Attachment size {size} bytes exceeds limit of {max_size} bytes")
        self.size = size
        self.max_size = max_size
