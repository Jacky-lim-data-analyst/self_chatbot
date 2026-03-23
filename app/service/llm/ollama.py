"""Ollama chat model implementation"""

from typing import AsyncIterator

from ollama import AsyncClient
from app.models import LLMConfig, Message
from .base import BaseLLM
from settings import settings


class OllamaLLM(BaseLLM):
    """Ollama model"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # self.config was defined
        # define the client
        # print(settings.ollama_host)
        self._client: AsyncClient = AsyncClient(host=settings.ollama_host)

    async def stream_chat(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield text chunks from Ollama's streaming chat endpoint."""
        payload = self._build_messages(messages)

        response = await self._client.chat(
            self.config.model,
            messages=payload,
            stream=True,
            options={"temperature": self.config.temperature},
            **self.config.extra,
        )

        async for chunk in response:
            content = chunk["message"]["content"]
            if content:
                yield content

    async def chat(self, messages: list[Message]) -> str:
        """Return the full response by accumulating the stream."""
        return await self.stream_to_string(messages)

    async def close(self) -> None:
        """Close the underlying httpx async client. No way to delete the ollama async client"""
        pass
