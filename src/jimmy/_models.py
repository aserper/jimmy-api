from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jimmy._types import MessageRole


@dataclass(frozen=True, slots=True)
class ChatStats:
    """Statistics returned by the API after a chat completion."""

    created_at: float
    done: bool
    done_reason: str
    total_duration: float
    ttft: float
    prefill_tokens: int
    prefill_rate: float
    decode_tokens: int
    decode_rate: float
    total_tokens: int
    total_time: float
    roundtrip_time: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatStats:
        return cls(
            created_at=data.get("created_at", 0.0),
            done=data.get("done", False),
            done_reason=data.get("done_reason", ""),
            total_duration=data.get("total_duration", 0.0),
            ttft=data.get("ttft", 0.0),
            prefill_tokens=data.get("prefill_tokens", 0),
            prefill_rate=data.get("prefill_rate", 0.0),
            decode_tokens=data.get("decode_tokens", 0),
            decode_rate=data.get("decode_rate", 0.0),
            total_tokens=data.get("total_tokens", 0),
            total_time=data.get("total_time", 0.0),
            roundtrip_time=data.get("roundtrip_time", 0),
        )


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Response from a chat completion."""

    text: str
    stats: ChatStats | None = None


@dataclass(frozen=True, slots=True)
class Model:
    """An available model."""

    id: str
    object: str
    created: int
    owned_by: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Model:
        return cls(
            id=data["id"],
            object=data.get("object", "model"),
            created=data.get("created", 0),
            owned_by=data.get("owned_by", ""),
        )


@dataclass(frozen=True, slots=True)
class Message:
    """A chat message."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class Attachment:
    """A file attachment for a chat message."""

    name: str
    content: str
    size: int | None = None

    def to_dict(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "size": self.size if self.size is not None else len(self.content.encode()),
            "content": self.content,
        }
