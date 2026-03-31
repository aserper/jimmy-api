from __future__ import annotations

import json

from jimmy._constants import STATS_DELIMITER_END, STATS_DELIMITER_START
from jimmy._models import ChatStats


def parse_response(raw: str) -> tuple[str, ChatStats | None]:
    """Split a raw API response into text content and optional stats."""
    idx = raw.find(STATS_DELIMITER_START)
    if idx == -1:
        return raw, None
    text = raw[:idx]
    stats_raw = raw[idx + len(STATS_DELIMITER_START) :]
    end_idx = stats_raw.find(STATS_DELIMITER_END)
    if end_idx != -1:
        stats_raw = stats_raw[:end_idx]
    return text, ChatStats.from_dict(json.loads(stats_raw))


class StreamParser:
    """Stateful parser for streaming responses.

    Feeds chunks of text and yields only the content portion,
    buffering any partial stats delimiter matches.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._stats: ChatStats | None = None
        self._done = False

    @property
    def stats(self) -> ChatStats | None:
        return self._stats

    def feed(self, chunk: str) -> str:
        """Feed a chunk, return text to yield (may be empty while buffering)."""
        if self._done:
            return ""

        self._buffer += chunk

        # Check if the full stats block is present
        start_idx = self._buffer.find(STATS_DELIMITER_START)
        if start_idx != -1:
            end_idx = self._buffer.find(STATS_DELIMITER_END, start_idx)
            if end_idx != -1:
                # Full stats block found — extract text before it, parse stats
                text = self._buffer[:start_idx]
                stats_raw = self._buffer[
                    start_idx + len(STATS_DELIMITER_START) : end_idx
                ]
                self._stats = ChatStats.from_dict(json.loads(stats_raw))
                self._done = True
                self._buffer = ""
                return text

            # Partial stats block — hold everything from start_idx onwards
            text = self._buffer[:start_idx]
            self._buffer = self._buffer[start_idx:]
            return text

        # Check if the buffer ends with a partial match of the delimiter start
        # e.g. the buffer ends with "<|" or "<|sta" etc.
        delim = STATS_DELIMITER_START
        for i in range(1, len(delim)):
            if self._buffer.endswith(delim[:i]):
                text = self._buffer[: -i]
                self._buffer = self._buffer[-i:]
                return text

        # No delimiter match at all — flush entire buffer
        text = self._buffer
        self._buffer = ""
        return text

    def finish(self) -> ChatStats | None:
        """Called when the stream ends. Parses any remaining buffered stats."""
        if self._buffer and not self._done:
            start_idx = self._buffer.find(STATS_DELIMITER_START)
            if start_idx != -1:
                end_idx = self._buffer.find(STATS_DELIMITER_END, start_idx)
                stats_raw = self._buffer[
                    start_idx
                    + len(STATS_DELIMITER_START) : (
                        end_idx if end_idx != -1 else len(self._buffer)
                    )
                ]
                try:
                    self._stats = ChatStats.from_dict(json.loads(stats_raw))
                except (json.JSONDecodeError, KeyError):
                    pass
            self._buffer = ""
        return self._stats
