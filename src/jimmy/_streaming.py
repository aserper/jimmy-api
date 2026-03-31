from __future__ import annotations

from typing import AsyncIterator, Iterator

import httpx

from jimmy._models import ChatStats
from jimmy._parser import StreamParser


class StreamResponse:
    """Sync iterator that yields text chunks from a streaming chat response.

    Usage::

        with client.stream("Tell me a story") as stream:
            for chunk in stream:
                print(chunk, end="")
        print(stream.stats)

    Or without context manager::

        for chunk in client.stream("Tell me a story"):
            print(chunk, end="")
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._parser = StreamParser()
        self._stats: ChatStats | None = None
        self._consumed = False

    @property
    def stats(self) -> ChatStats | None:
        """Available after iteration completes."""
        return self._stats

    def __iter__(self) -> Iterator[str]:
        try:
            for chunk in self._response.iter_text():
                text = self._parser.feed(chunk)
                if text:
                    yield text
            self._stats = self._parser.finish()
        finally:
            self._consumed = True
            self._response.close()

    def __enter__(self) -> StreamResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._consumed:
            self._response.close()
            self._consumed = True

    def get_full_text(self) -> str:
        """Consume the stream and return the full text."""
        parts: list[str] = []
        for chunk in self:
            parts.append(chunk)
        return "".join(parts)


class AsyncStreamResponse:
    """Async iterator that yields text chunks from a streaming chat response.

    Usage::

        async with client.stream("Tell me a story") as stream:
            async for chunk in stream:
                print(chunk, end="")
        print(stream.stats)
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._parser = StreamParser()
        self._stats: ChatStats | None = None
        self._consumed = False

    @property
    def stats(self) -> ChatStats | None:
        return self._stats

    async def __aiter__(self) -> AsyncIterator[str]:
        try:
            async for chunk in self._response.aiter_text():
                text = self._parser.feed(chunk)
                if text:
                    yield text
            self._stats = self._parser.finish()
        finally:
            self._consumed = True
            await self._response.aclose()

    async def __aenter__(self) -> AsyncStreamResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._consumed:
            await self._response.aclose()
            self._consumed = True

    async def get_full_text(self) -> str:
        """Consume the stream and return the full text."""
        parts: list[str] = []
        async for chunk in self:
            parts.append(chunk)
        return "".join(parts)
