"""
LLM factory & registry
======================
Centralises creation of all LLM provider instances.

Usage
-----
# 1. Simple factory function
llm = create_llm("ollama", config)
llm = create_llm("deepseek", config)

# 4. Async context-manager (auto-closes the client)
async with create_llm("deepseek", config) as llm:
    async for chunk in llm.stream_chat(messages):
        print(chunk, end="", flush=True)

# 5. Convenience builder — no LLMConfig boilerplate
llm = build_llm("ollama", model="llama3.2", system_prompt="You are helpful.")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from app.models import LLMConfig
from .base import BaseLLM
from .ollama import OllamaLLM
from .deepseek import DeeepSeekLLM

if TYPE_CHECKING:
    pass  # keep import sections clean for type checkers

# ---------
# TYPE ALIAS
# --------
LLMClass = Type[BaseLLM]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class LLMRegistry:
    """A mutable map of provider-name -> LLM class

    The default registry ships with Ollama and DeepSeek pre-registered.
    Additional providers can be added at runtime via `register()` or the
    `@registry.provider(name)` decorator.
    """

    def __init__(self) -> None:
        self._providers: dict[str, LLMClass] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self, name: str, cls: LLMClass, *, aliases: list[str] | None = None
    ) -> None:
        """
        Register *cls* under *name* (and any optional *aliases*).

        Args:
            name:    Primary key used when calling `create(name, ...)`.
            cls:     A concrete subclass of BaseLLM.
            aliases: Extra names that map to the same class
                     (e.g. ["ds", "deep-seek"] for DeeepSeekLLM).

        Raises:
            TypeError:  If *cls* is not a subclass of BaseLLM.
            ValueError: If *name* is already registered.
        """
        if not (isinstance(cls, type) and issubclass(cls, BaseLLM)):
            raise TypeError(f"{cls!r} must be subclass of BaseLLM")
        for key in [name] + (aliases or []):
            key = key.lower()
            if key in self._providers:
                raise ValueError(
                    f"Provider '{key}' is already registered"
                    "Call unregister first if you want to replace it"
                )
            self._providers[key] = cls

    def unregister(self, name: str) -> None:
        """Remove a provider and (its aliases) from registry"""
        key = name.lower()
        self._providers.pop(key, None)

    def provider(self, name: str, *, aliases: list[str] | None = None):
        """
        Class decorator - registers the decorated class under *name*

        Example:

            @registry.provider("groq", aliases=["groq-ai"])
            class GroqLLM(BaseLLM):
                ..."""

        def decorator(cls: LLMClass) -> LLMClass:
            self.register(name, cls, aliases=aliases)
            return cls

        return decorator

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def available(self) -> list[str]:
        """Sorted list of all registered provider names"""
        return sorted(self._providers)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._providers

    def __repr__(self) -> str:
        return f"LLMRegistry(providers={self.available})"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    def create(self, provider: str, config: LLMConfig) -> BaseLLM:
        """
                Instantiate an LLM for *provider* with the given *config*.

        Args:
            provider: Case-insensitive name (e.g. "ollama", "deepseek").
            config:   Fully constructed LLMConfig.

        Returns:
            A concrete BaseLLM instance.

        Raises:
            KeyError: If the provider is not registered."""
        key = provider.lower()
        try:
            cls = self._providers[key]
        except KeyError:
            raise KeyError(
                f"Unknown LLM provider '{provider}'. Available: {self.available}"
            )
        return cls(config)

    def build(
        self,
        provider: str,
        *,
        model: str,
        system_prompt: str,
        temperature: float = 0.7,
        **extra,
    ) -> BaseLLM:
        """
        Convenience factory — build a config inline instead of pre-constructing
        an LLMConfig object.

        Args:
            provider:      Provider name (e.g. "ollama").
            model:         Model identifier (e.g. "llama3.2", "deepseek-chat").
            system_prompt: Optional system instruction.
            temperature:   Sampling temperature (default 0.7).
            **extra:       Any extra kwargs forwarded to LLMConfig.extra.

        Returns:
            A configured BaseLLM instance.
        """
        config = LLMConfig(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            extra=extra,
        )
        return self.create(provider, config)


# ---------------------------------------------------------------------------
# Default global registry (pre-populated)
# ---------------------------------------------------------------------------
_default_registry = LLMRegistry()
_default_registry.register("ollama", OllamaLLM, aliases=["ollama-local"])
_default_registry.register("deepseek", DeeepSeekLLM, aliases=["ds"])


# ---------------------------------------------------------------------------
# Module-level convenience helpers (delegate to _default_registry)
# ---------------------------------------------------------------------------
def create_llm(provider: str, config: LLMConfig) -> BaseLLM:
    """Shortcut for `_default_registry.create(provider, config)`."""
    return _default_registry.create(provider, config)


def build_llm(
    provider: str,
    *,
    model: str,
    system_prompt: str = "You are a honest assistant",
    temperature: float = 0.7,
    **extra,
) -> BaseLLM:
    """Shortcut for `_default_registry.build(...)`."""
    return _default_registry.build(
        provider,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        **extra,
    )


def register_provider(name: str, cls: LLMClass, *, aliases: list[str] | None = None):
    """
    Register a custom provider in the global registry
    """
    _default_registry.register(name, cls, aliases=aliases)


def available_providers() -> list[str]:
    """Return all provider names registered in the global registry."""
    return _default_registry.available
