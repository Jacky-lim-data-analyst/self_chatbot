"""Deepseek chat model implementation"""

from typing import AsyncIterator

from app.models import LLMConfig, Message

from .base import BaseLLM
from openai import AsyncClient
from settings import settings

class DeeepSeekLLM(BaseLLM):
    """DeepSeek model"""
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # self.config was defined
        # define the client
        self._client: AsyncClient = AsyncClient(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url=settings.deepseek_base_url
        )

    async def stream_chat(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield text chunks from DeepSeek's streaming chat endpoint."""
        payload = self._build_messages(messages)

        stream = await self._client.chat.completions.create(
            model=self.config.model,
            messages=payload,
            temperature=self.config.temperature,
            stream=True,
            **self.config.extra
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def chat(self, messages: list[Message]) -> str:
        """Return the full response by accumulating the stream."""
        return await self.stream_to_string(messages)
    
    async def close(self) -> None:
        """Close the underlying httpx async client."""
        await self._client.close()

