from __future__ import annotations

import httpx

from jimmy._constants import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    MAX_ATTACHMENT_SIZE,
    USER_AGENT,
)
from jimmy._conversation import AsyncConversation, Conversation
from jimmy._exceptions import APIError, AttachmentTooLargeError
from jimmy._models import Attachment, ChatResponse, Message, Model
from jimmy._parser import parse_response
from jimmy._streaming import AsyncStreamResponse, StreamResponse


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


def _make_attachment(
    attachment: tuple[str, str] | None,
) -> Attachment | None:
    if attachment is None:
        return None
    name, content = attachment
    size = len(content.encode())
    if size > MAX_ATTACHMENT_SIZE:
        raise AttachmentTooLargeError(size, MAX_ATTACHMENT_SIZE)
    return Attachment(name=name, content=content, size=size)


class Jimmy:
    """Synchronous client for chatjimmy.ai.

    Usage::

        client = Jimmy()
        response = client.chat("Hello!")
        print(response.text)
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "",
        top_k: int = DEFAULT_TOP_K,
        httpx_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._system_prompt = system_prompt
        self._top_k = top_k
        self._client = httpx_client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        self._owns_client = httpx_client is None

    def _resolve(self, model: str | None, system_prompt: str | None, top_k: int | None) -> tuple[str, str, int]:
        return (
            model or self._model,
            system_prompt if system_prompt is not None else self._system_prompt,
            top_k if top_k is not None else self._top_k,
        )

    def _post_chat(self, payload: dict) -> ChatResponse:
        """Internal: POST to /api/chat and return parsed response."""
        resp = self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        if resp.status_code != 200:
            raise APIError(f"Chat API returned {resp.status_code}: {resp.text}", resp.status_code)
        text, stats = parse_response(resp.text)
        return ChatResponse(text=text, stats=stats)

    def _post_chat_stream(self, payload: dict) -> StreamResponse:
        """Internal: POST to /api/chat with streaming."""
        resp = self._client.send(
            self._client.build_request("POST", f"{self._base_url}/api/chat", json=payload),
            stream=True,
        )
        if resp.status_code != 200:
            body = resp.read()
            resp.close()
            raise APIError(f"Chat API returned {resp.status_code}: {body.decode()}", resp.status_code)
        return StreamResponse(resp)

    def chat(
        self,
        message: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        top_k: int | None = None,
        attachment: tuple[str, str] | None = None,
    ) -> ChatResponse:
        """Send a single message and return the full response.

        Args:
            message: The user message.
            model: Override the default model.
            system_prompt: Override the default system prompt.
            top_k: Override the default top_k.
            attachment: Optional ``(filename, content)`` tuple.
        """
        m, sp, tk = self._resolve(model, system_prompt, top_k)
        att = _make_attachment(attachment)
        payload = _build_payload([Message(role="user", content=message)], m, sp, tk, att)
        return self._post_chat(payload)

    def stream(
        self,
        message: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        top_k: int | None = None,
        attachment: tuple[str, str] | None = None,
    ) -> StreamResponse:
        """Send a message and return a streaming response iterator.

        Args:
            message: The user message.
            model: Override the default model.
            system_prompt: Override the default system prompt.
            top_k: Override the default top_k.
            attachment: Optional ``(filename, content)`` tuple.
        """
        m, sp, tk = self._resolve(model, system_prompt, top_k)
        att = _make_attachment(attachment)
        payload = _build_payload([Message(role="user", content=message)], m, sp, tk, att)
        return self._post_chat_stream(payload)

    def models(self) -> list[Model]:
        """List available models."""
        resp = self._client.get(
            f"{self._base_url}/api/models",
            headers={"Cache-Control": "no-store"},
        )
        if resp.status_code != 200:
            raise APIError(f"Models API returned {resp.status_code}: {resp.text}", resp.status_code)
        data = resp.json()
        return [Model.from_dict(m) for m in data.get("data", [])]

    def conversation(
        self,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        top_k: int | None = None,
    ) -> Conversation:
        """Create a new multi-turn conversation."""
        m, sp, tk = self._resolve(model, system_prompt, top_k)
        return Conversation(client=self, model=m, system_prompt=sp, top_k=tk)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Jimmy:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AsyncJimmy:
    """Asynchronous client for chatjimmy.ai.

    Usage::

        async with AsyncJimmy() as client:
            response = await client.chat("Hello!")
            print(response.text)
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "",
        top_k: int = DEFAULT_TOP_K,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._system_prompt = system_prompt
        self._top_k = top_k
        self._client = httpx_client or httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        self._owns_client = httpx_client is None

    def _resolve(self, model: str | None, system_prompt: str | None, top_k: int | None) -> tuple[str, str, int]:
        return (
            model or self._model,
            system_prompt if system_prompt is not None else self._system_prompt,
            top_k if top_k is not None else self._top_k,
        )

    async def _post_chat(self, payload: dict) -> ChatResponse:
        resp = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        if resp.status_code != 200:
            raise APIError(f"Chat API returned {resp.status_code}: {resp.text}", resp.status_code)
        text, stats = parse_response(resp.text)
        return ChatResponse(text=text, stats=stats)

    async def _post_chat_stream(self, payload: dict) -> AsyncStreamResponse:
        resp = await self._client.send(
            self._client.build_request("POST", f"{self._base_url}/api/chat", json=payload),
            stream=True,
        )
        if resp.status_code != 200:
            body = await resp.aread()
            await resp.aclose()
            raise APIError(f"Chat API returned {resp.status_code}: {body.decode()}", resp.status_code)
        return AsyncStreamResponse(resp)

    async def chat(
        self,
        message: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        top_k: int | None = None,
        attachment: tuple[str, str] | None = None,
    ) -> ChatResponse:
        """Send a single message and return the full response."""
        m, sp, tk = self._resolve(model, system_prompt, top_k)
        att = _make_attachment(attachment)
        payload = _build_payload([Message(role="user", content=message)], m, sp, tk, att)
        return await self._post_chat(payload)

    async def stream(
        self,
        message: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        top_k: int | None = None,
        attachment: tuple[str, str] | None = None,
    ) -> AsyncStreamResponse:
        """Send a message and return a streaming response iterator."""
        m, sp, tk = self._resolve(model, system_prompt, top_k)
        att = _make_attachment(attachment)
        payload = _build_payload([Message(role="user", content=message)], m, sp, tk, att)
        return await self._post_chat_stream(payload)

    async def models(self) -> list[Model]:
        """List available models."""
        resp = await self._client.get(
            f"{self._base_url}/api/models",
            headers={"Cache-Control": "no-store"},
        )
        if resp.status_code != 200:
            raise APIError(f"Models API returned {resp.status_code}: {resp.text}", resp.status_code)
        data = resp.json()
        return [Model.from_dict(m) for m in data.get("data", [])]

    async def conversation(
        self,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        top_k: int | None = None,
    ) -> AsyncConversation:
        """Create a new multi-turn conversation."""
        m, sp, tk = self._resolve(model, system_prompt, top_k)
        return AsyncConversation(client=self, model=m, system_prompt=sp, top_k=tk)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncJimmy:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
