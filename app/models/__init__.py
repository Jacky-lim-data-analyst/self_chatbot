"""
Centralized data models for the chatbot app
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------
# message data
# --------
@dataclass
class Message:
    """Represents a single chat message"""

    role: str  # "system", "user", "assistant"
    content: str


# -----------
# LLM configurations
# -----------
@dataclass
class LLMConfig:
    """Common configurations shared across all LLM providers"""

    model: str
    temperature: float = 1.0
    system_prompt: Optional[str] = None
    extra: dict = field(default_factory=dict)  # provider-specific overrides
