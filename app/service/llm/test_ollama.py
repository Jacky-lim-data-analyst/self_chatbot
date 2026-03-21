"""Quick smoke-test for OllamaLLM"""

import asyncio

from app.models import LLMConfig, Message
from .ollama import OllamaLLM

MESSAGES = [Message(role="user", content="Say hello in one sentence")]

async def test_stream_chat(llm: OllamaLLM) -> None:
    print("--stream chat--")
    async for chunk in llm.stream_chat(MESSAGES):
        print(chunk, end="", flush=True)
    print("\n")

async def test_chat(llm: OllamaLLM) -> None:
    print("--chat--")
    reply = await llm.chat(MESSAGES)
    print(reply, "\n")

async def main() -> None:
    config = LLMConfig(model="qwen3.5:latest", system_prompt="Be your true self")

    async with OllamaLLM(config) as llm:
        await test_stream_chat(llm)
        await test_chat(llm)

if __name__ == "__main__":
    asyncio.run(main())
