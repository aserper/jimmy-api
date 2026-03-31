import json

import httpx
import pytest

from jimmy import APIError, AttachmentTooLargeError, Jimmy, Model
from jimmy._client import _make_attachment
from jimmy._models import Attachment

SAMPLE_STATS = {
    "created_at": 1.0,
    "done": True,
    "done_reason": "stop",
    "total_duration": 0.01,
    "ttft": 0.001,
    "prefill_tokens": 10,
    "prefill_rate": 100.0,
    "decode_tokens": 5,
    "decode_rate": 50.0,
    "total_tokens": 15,
    "total_time": 0.005,
    "roundtrip_time": 12,
}

RESPONSE_TEXT = f'Hello!\n<|stats|>{json.dumps(SAMPLE_STATS)}<|/stats|>\n'
MODELS_RESPONSE = {"object": "list", "data": [{"id": "llama3.1-8B", "object": "model", "created": 1690000000, "owned_by": "Taalas Inc."}]}


class TestMakeAttachment:
    def test_valid(self):
        att = _make_attachment(("test.txt", "hello"))
        assert isinstance(att, Attachment)
        assert att.name == "test.txt"
        assert att.content == "hello"
        assert att.size == 5

    def test_none(self):
        assert _make_attachment(None) is None

    def test_too_large(self):
        with pytest.raises(AttachmentTooLargeError):
            _make_attachment(("big.txt", "x" * 60_000))


class TestJimmyChat:
    def test_chat(self, httpx_mock):
        httpx_mock.add_response(url="https://chatjimmy.ai/api/chat", text=RESPONSE_TEXT)
        client = Jimmy()
        response = client.chat("Hi")
        assert "Hello!" in response.text
        assert response.stats is not None
        assert response.stats.total_tokens == 15

    def test_chat_with_options(self, httpx_mock):
        httpx_mock.add_response(url="https://chatjimmy.ai/api/chat", text=RESPONSE_TEXT)
        client = Jimmy()
        response = client.chat("Hi", model="llama3.1-8B", system_prompt="Be brief", top_k=4)
        assert response.text is not None
        req = httpx_mock.get_request()
        body = json.loads(req.content)
        assert body["chatOptions"]["selectedModel"] == "llama3.1-8B"
        assert body["chatOptions"]["systemPrompt"] == "Be brief"
        assert body["chatOptions"]["topK"] == 4

    def test_chat_with_attachment(self, httpx_mock):
        httpx_mock.add_response(url="https://chatjimmy.ai/api/chat", text=RESPONSE_TEXT)
        client = Jimmy()
        response = client.chat("Summarize", attachment=("f.txt", "data"))
        assert response.text is not None
        req = httpx_mock.get_request()
        body = json.loads(req.content)
        assert body["attachment"]["name"] == "f.txt"
        assert body["attachment"]["content"] == "data"

    def test_chat_api_error(self, httpx_mock):
        httpx_mock.add_response(url="https://chatjimmy.ai/api/chat", status_code=500, text="error")
        client = Jimmy()
        with pytest.raises(APIError) as exc_info:
            client.chat("Hi")
        assert exc_info.value.status_code == 500


class TestJimmyModels:
    def test_models(self, httpx_mock):
        httpx_mock.add_response(url="https://chatjimmy.ai/api/models", json=MODELS_RESPONSE)
        client = Jimmy()
        models = client.models()
        assert len(models) == 1
        assert models[0].id == "llama3.1-8B"
        assert models[0].owned_by == "Taalas Inc."


class TestJimmyContextManager:
    def test_context_manager(self, httpx_mock):
        httpx_mock.add_response(url="https://chatjimmy.ai/api/chat", text=RESPONSE_TEXT)
        with Jimmy() as client:
            response = client.chat("Hi")
            assert response.text is not None
