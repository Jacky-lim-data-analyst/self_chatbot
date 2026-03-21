"""
Base LLM class defining the shared interface for all LLM providers.
Both Ollama and DeepSeek implementation use asnyc clients
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Any

from app.models import Message, LLMConfig


class BaseLLM(ABC):
    """Abstract base class for all LLM provider clients

    Each subclass implement:
        - stream_chat(): yields text chunks from a streaming response
        - chat(): returns full response as a string

    Usage pattern (async context manager):
        async with MyLLM(config) as llm:
            async for chunk in llm.stream_chat(messages):
                print(chunk, end="", flush=True)"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[Any] = None

    # ------------------------------------------------------------------
    # Abstract interface — every provider must implement these
    # ------------------------------------------------------------------
    @abstractmethod
    async def stream_chat(self, messages: list[Message]) -> AsyncIterator[str]:
        """
        Stream a chat response token-by-token.

        Args:
            messages: Conversation history (system / user / assistant turns).

        Yields:
            str: Successive text chunks as they arrive from the model.

        Example:
            async for chunk in llm.stream_chat(messages):
                print(chunk, end="", flush=True)
        """
        ...

    @abstractmethod
    async def chat(self, messages: list[Message]) -> str:
        """
        Return the complete chat response as a single string.

        Implementations may accumulate stream_chat() internally or call
        a non-streaming endpoint directly.

        Args:
            messages: Conversation history.

        Returns:
            str: The full model reply.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release underlying async client resources."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------
    def _build_messages(self, messages: list[Message]) -> list[dict]:
        """
        Convert Message dataclasses → plain dicts expected by most SDKs.
        Prepends the system prompt from config if present and not already
        the first message.
        """
        result: list[dict] = []

        if self.config.system_prompt:
            if not messages or messages[0].role != "system":
                result.append(
                    {"role": "system", "content": self.config.system_prompt}
                )

        result.extend({"role": m.role, "content": m.content} for m in messages)

        return result

    async def stream_to_string(self, messages: list[Message]) -> str:
        """Convenience: drain stream_chat() into a single string"""
        chunks: list[str] = []
        async for chunk in self.stream_chat(messages):
            chunks.append(chunk)

        return "".join(chunks)

    # ------------------------------------------------------------------
    # Async context-manager support
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "BaseLLM":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model!r})"
