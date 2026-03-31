from __future__ import annotations

from typing import TYPE_CHECKING

from jimmy._models import Attachment, ChatResponse, Message
from jimmy._streaming import AsyncStreamResponse, StreamResponse

if TYPE_CHECKING:
    from jimmy._client import AsyncJimmy, Jimmy


def _build_payload(
    messages: list[Message],
    model: str,
    system_prompt: str,
    top_k: int,
    attachment: Attachment | None,
) -> dict:
    payload: dict = {
        "messages": [m.to_dict() for m in messages],
        "chatOptions": {
            "selectedModel": model,
            "systemPrompt": system_prompt,
            "topK": top_k,
        },
    }
    if attachment:
        payload["attachment"] = attachment.to_dict()
    return payload


class Conversation:
    """A multi-turn conversation that maintains message history.

    Usage::

        conv = client.conversation(system_prompt="Be helpful")
        r1 = conv.send("Hello")
        r2 = conv.send("What did I just say?")
    """

    def __init__(
        self,
        client: Jimmy,
        *,
        model: str,
        system_prompt: str,
        top_k: int,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._top_k = top_k
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        """Return a copy of the conversation history."""
        return list(self._messages)

    def send(
        self,
        message: str,
        *,
        attachment: tuple[str, str] | None = None,
    ) -> ChatResponse:
        """Send a message in this conversation, maintaining history."""
        from jimmy._client import _make_attachment

        self._messages.append(Message(role="user", content=message))
        att = _make_attachment(attachment)
        payload = _build_payload(self._messages, self._model, self._system_prompt, self._top_k, att)
        response = self._client._post_chat(payload)
        self._messages.append(Message(role="assistant", content=response.text))
        return response

    def stream(
        self,
        message: str,
        *,
        attachment: tuple[str, str] | None = None,
    ) -> _ConversationStream:
        """Send a streaming message. Use as a context manager to ensure history is updated."""
        from jimmy._client import _make_attachment

        self._messages.append(Message(role="user", content=message))
        att = _make_attachment(attachment)
        payload = _build_payload(self._messages, self._model, self._system_prompt, self._top_k, att)
        stream = self._client._post_chat_stream(payload)
        return _ConversationStream(stream, self)

    def clear(self) -> None:
        """Clear conversation history."""
        self._messages.clear()


class _ConversationStream:
    """Wrapper that captures streamed text and appends to conversation history."""

    def __init__(self, stream: StreamResponse, conversation: Conversation) -> None:
        self._stream = stream
        self._conversation = conversation
        self._parts: list[str] = []

    def __iter__(self):
        for chunk in self._stream:
            self._parts.append(chunk)
            yield chunk
        # After stream completes, add assistant message to history
        full_text = "".join(self._parts)
        self._conversation._messages.append(Message(role="assistant", content=full_text))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # If iteration wasn't completed, consume remaining and update history
        if not self._parts:
            full_text = self._stream.get_full_text()
            self._conversation._messages.append(Message(role="assistant", content=full_text))
        self._stream.close()

    @property
    def stats(self):
        return self._stream.stats


class AsyncConversation:
    """Async multi-turn conversation that maintains message history."""

    def __init__(
        self,
        client: AsyncJimmy,
        *,
        model: str,
        system_prompt: str,
        top_k: int,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._top_k = top_k
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    async def send(
        self,
        message: str,
        *,
        attachment: tuple[str, str] | None = None,
    ) -> ChatResponse:
        from jimmy._client import _make_attachment

        self._messages.append(Message(role="user", content=message))
        att = _make_attachment(attachment)
        payload = _build_payload(self._messages, self._model, self._system_prompt, self._top_k, att)
        response = await self._client._post_chat(payload)
        self._messages.append(Message(role="assistant", content=response.text))
        return response

    async def stream(
        self,
        message: str,
        *,
        attachment: tuple[str, str] | None = None,
    ) -> _AsyncConversationStream:
        from jimmy._client import _make_attachment

        self._messages.append(Message(role="user", content=message))
        att = _make_attachment(attachment)
        payload = _build_payload(self._messages, self._model, self._system_prompt, self._top_k, att)
        stream = await self._client._post_chat_stream(payload)
        return _AsyncConversationStream(stream, self)

    def clear(self) -> None:
        self._messages.clear()


class _AsyncConversationStream:
    """Async wrapper that captures streamed text and appends to conversation history."""

    def __init__(self, stream: AsyncStreamResponse, conversation: AsyncConversation) -> None:
        self._stream = stream
        self._conversation = conversation
        self._parts: list[str] = []

    async def __aiter__(self):
        async for chunk in self._stream:
            self._parts.append(chunk)
            yield chunk
        full_text = "".join(self._parts)
        self._conversation._messages.append(Message(role="assistant", content=full_text))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        if not self._parts:
            full_text = await self._stream.get_full_text()
            self._conversation._messages.append(Message(role="assistant", content=full_text))
        await self._stream.aclose()

    @property
    def stats(self):
        return self._stream.stats
