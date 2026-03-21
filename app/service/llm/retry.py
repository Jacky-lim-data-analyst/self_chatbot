"""
Exponential-backoff retry wrapper for BaseLLM.stream_chat

1. **RetryLLM wrapper class** — wraps any existing BaseLLM instance:
    llm = build_llm("deepseek", model="deepseek-chat")
    llm = RetryLLM(llm)

2. **retry_stream decorator** — wrap a single stream_chat method ad-hoc:
        class MyLLM(BaseLLM):
            @retry_stream()
            async def stream_chat(self, messages):
                ...
"""

from __future__ import annotations

import asyncio 
import functools
