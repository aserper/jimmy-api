# jimmy

Unofficial Python client for [chatjimmy.ai](https://chatjimmy.ai) — an extremely fast LLM powered by Llama 3.1 8B.

## Installation

```bash
pip install git+https://github.com/amitbend/jimmy-api.git
```

## Quick Start

```python
from jimmy import Jimmy

client = Jimmy()

# Simple chat
response = client.chat("What is the capital of France?")
print(response.text)
print(response.stats.decode_rate, "tokens/sec")
```

## Features

### Chat with options

```python
response = client.chat(
    "Explain quantum computing",
    system_prompt="Explain like I'm 5",
    top_k=4,
)
```

### Streaming

```python
for chunk in client.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Multi-turn conversations

```python
conv = client.conversation(system_prompt="You are a math tutor")
conv.send("What is calculus?")
conv.send("Give me an example")  # remembers context
print(conv.messages)  # full history
```

### File attachments

```python
response = client.chat(
    "Summarize this file",
    attachment=("notes.txt", "Contents of the file here..."),
)
```

### List models

```python
for model in client.models():
    print(f"{model.id} by {model.owned_by}")
```

### Async support

```python
import asyncio
from jimmy import AsyncJimmy

async def main():
    async with AsyncJimmy() as client:
        response = await client.chat("Hello!")
        print(response.text)

        async for chunk in await client.stream("Tell me a joke"):
            print(chunk, end="")

asyncio.run(main())
```

## Configuration

```python
client = Jimmy(
    base_url="https://chatjimmy.ai",  # default
    model="llama3.1-8B",              # default
    system_prompt="",                  # default
    top_k=8,                           # default
    timeout=120.0,                     # seconds
)
```

All options can be overridden per-call:

```python
client.chat("Hi", model="llama3.1-8B", system_prompt="Be concise", top_k=4)
```

## Response object

```python
response = client.chat("Hi")
response.text          # str — the generated text
response.stats         # ChatStats | None
response.stats.ttft    # time to first token (seconds)
response.stats.decode_rate    # tokens/second
response.stats.total_tokens   # total tokens used
response.stats.total_time     # total generation time
```

## License

MIT
