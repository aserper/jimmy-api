import json

import pytest

from jimmy import Jimmy

SAMPLE_STATS = {
    "created_at": 1.0, "done": True, "done_reason": "stop",
    "total_duration": 0.01, "ttft": 0.001, "prefill_tokens": 10,
    "prefill_rate": 100.0, "decode_tokens": 5, "decode_rate": 50.0,
    "total_tokens": 15, "total_time": 0.005, "roundtrip_time": 12,
}
STATS_SUFFIX = f'<|stats|>{json.dumps(SAMPLE_STATS)}<|/stats|>'


class TestConversation:
    def test_multi_turn(self, httpx_mock):
        httpx_mock.add_response(url="https://chatjimmy.ai/api/chat", text=f"Hi Alice!{STATS_SUFFIX}")
        httpx_mock.add_response(url="https://chatjimmy.ai/api/chat", text=f"Your name is Alice.{STATS_SUFFIX}")

        client = Jimmy()
        conv = client.conversation(system_prompt="Be helpful")

        r1 = conv.send("My name is Alice")
        assert "Alice" in r1.text
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[1].role == "assistant"

        r2 = conv.send("What is my name?")
        assert "Alice" in r2.text
        assert len(conv.messages) == 4

        # Verify second request included full history
        requests = httpx_mock.get_requests()
        second_body = json.loads(requests[1].content)
        assert len(second_body["messages"]) == 3  # user, assistant, user

    def test_clear(self, httpx_mock):
        httpx_mock.add_response(url="https://chatjimmy.ai/api/chat", text=f"Hi!{STATS_SUFFIX}")
        client = Jimmy()
        conv = client.conversation()
        conv.send("Hello")
        assert len(conv.messages) == 2
        conv.clear()
        assert len(conv.messages) == 0

    def test_conversation_with_system_prompt(self, httpx_mock):
        httpx_mock.add_response(url="https://chatjimmy.ai/api/chat", text=f"4{STATS_SUFFIX}")
        client = Jimmy()
        conv = client.conversation(system_prompt="Answer with numbers only", top_k=4)
        conv.send("2+2")
        req = httpx_mock.get_request()
        body = json.loads(req.content)
        assert body["chatOptions"]["systemPrompt"] == "Answer with numbers only"
        assert body["chatOptions"]["topK"] == 4
