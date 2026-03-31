"""Unofficial Python client for chatjimmy.ai."""

from jimmy._client import AsyncJimmy, Jimmy
from jimmy._conversation import AsyncConversation, Conversation
from jimmy._exceptions import APIError, AttachmentTooLargeError, JimmyError
from jimmy._models import Attachment, ChatResponse, ChatStats, Message, Model
from jimmy._streaming import AsyncStreamResponse, StreamResponse

__version__ = "0.1.0"

__all__ = [
    "AsyncJimmy",
    "AsyncConversation",
    "AsyncStreamResponse",
    "APIError",
    "Attachment",
    "AttachmentTooLargeError",
    "ChatResponse",
    "ChatStats",
    "Conversation",
    "Jimmy",
    "JimmyError",
    "Message",
    "Model",
    "StreamResponse",
]
