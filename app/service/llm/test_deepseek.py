"""Quick smoke-test for DeepSeekLLM"""

import asyncio

from app.models import LLMConfig, Message
from .deepseek import DeeepSeekLLM
from .factory import create_llm

MESSAGES = [Message(role="user", content="Who are you?")]


async def test_stream_chat(llm: DeeepSeekLLM) -> None:
    print("--stream chat--")
    async for chunk in llm.stream_chat(MESSAGES):
        print(chunk, end="", flush=True)
    print("\n")


async def test_chat(llm: DeeepSeekLLM) -> None:
    print("--chat--")
    reply = await llm.chat(MESSAGES)
    print(reply, "\n")


async def main() -> None:
    config = LLMConfig(model="deepseek-chat", system_prompt="Be your true self")
    deepseek_llm = create_llm(provider="deepseek", config=config)

    async with deepseek_llm as llm:
        await test_stream_chat(llm)
        await test_chat(llm)


if __name__ == "__main__":
    asyncio.run(main())
