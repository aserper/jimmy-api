from jimmy._parser import StreamParser, parse_response


class TestParseResponse:
    def test_no_stats(self):
        text, stats = parse_response("Hello world")
        assert text == "Hello world"
        assert stats is None

    def test_with_stats(self):
        raw = 'Hello!\n<|stats|>{"created_at":1.0,"done":true,"done_reason":"stop","total_duration":0.01,"ttft":0.001,"prefill_tokens":10,"prefill_rate":100.0,"decode_tokens":5,"decode_rate":50.0,"total_tokens":15,"total_time":0.005,"roundtrip_time":12}<|/stats|>'
        text, stats = parse_response(raw)
        assert text == "Hello!\n"
        assert stats is not None
        assert stats.done is True
        assert stats.done_reason == "stop"
        assert stats.total_tokens == 15
        assert stats.decode_rate == 50.0

    def test_stats_without_end_delimiter(self):
        raw = 'Hi<|stats|>{"created_at":1.0,"done":true,"done_reason":"stop","total_duration":0.01,"ttft":0.001,"prefill_tokens":10,"prefill_rate":100.0,"decode_tokens":5,"decode_rate":50.0,"total_tokens":15,"total_time":0.005,"roundtrip_time":12}'
        text, stats = parse_response(raw)
        assert text == "Hi"
        assert stats is not None
        assert stats.total_tokens == 15

    def test_empty_text_with_stats(self):
        raw = '<|stats|>{"created_at":0,"done":true,"done_reason":"stop","total_duration":0,"ttft":0,"prefill_tokens":0,"prefill_rate":0,"decode_tokens":0,"decode_rate":0,"total_tokens":0,"total_time":0,"roundtrip_time":0}<|/stats|>'
        text, stats = parse_response(raw)
        assert text == ""
        assert stats is not None


SAMPLE_STATS_JSON = '{"created_at":1.0,"done":true,"done_reason":"stop","total_duration":0.01,"ttft":0.001,"prefill_tokens":10,"prefill_rate":100.0,"decode_tokens":5,"decode_rate":50.0,"total_tokens":15,"total_time":0.005,"roundtrip_time":12}'


class TestStreamParser:
    def test_single_chunk_with_stats(self):
        parser = StreamParser()
        text = parser.feed(f"Hello world\n<|stats|>{SAMPLE_STATS_JSON}<|/stats|>")
        assert text == "Hello world\n"
        stats = parser.finish()
        assert stats is not None
        assert stats.total_tokens == 15

    def test_multiple_chunks_no_stats(self):
        parser = StreamParser()
        result = []
        result.append(parser.feed("Hello "))
        result.append(parser.feed("world"))
        assert "".join(result) == "Hello world"
        assert parser.finish() is None

    def test_stats_split_across_chunks(self):
        parser = StreamParser()
        result = []
        result.append(parser.feed("Hello"))
        result.append(parser.feed("<|sta"))
        result.append(parser.feed(f"ts|>{SAMPLE_STATS_JSON}<|/stats|>"))
        full = "".join(result)
        assert full == "Hello"
        stats = parser.finish()
        assert stats is not None
        assert stats.done is True

    def test_partial_delimiter_that_isnt(self):
        parser = StreamParser()
        result = []
        result.append(parser.feed("Hello <"))
        result.append(parser.feed("world"))
        full = "".join(result)
        assert "Hello <world" == full

    def test_chunk_ends_with_partial_delimiter(self):
        parser = StreamParser()
        result = []
        result.append(parser.feed("Hi<|"))
        result.append(parser.feed(f"stats|>{SAMPLE_STATS_JSON}<|/stats|>"))
        full = "".join(result)
        assert full == "Hi"
        assert parser.finish() is not None

    def test_stats_in_finish(self):
        parser = StreamParser()
        result = []
        result.append(parser.feed("text"))
        result.append(parser.feed(f"<|stats|>{SAMPLE_STATS_JSON}<|/stats|>"))
        full = "".join(result)
        assert full == "text"
        stats = parser.finish()
        assert stats is not None
