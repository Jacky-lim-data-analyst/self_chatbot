"""
chat.py - streaming chat router
==================================
Mounts at /chat and exposes a single POST endpoint that streams LLM
responses back to the client as Server-Sent Events (SSE).

Endpoint
---
POST /chat/stream

Request body  (ChatRequest)
    provider   : str          – e.g. "ollama", "deepseek"
    model      : str          – e.g. "qwen3.5:latest", "deepseek-chat"
    messages   : list[Message]
    temperature: float = 0.7  – sampling temperature
    system_prompt: str = ""   – optional system instruction override

Response
---
text/event-stream (SSE)
    data: <chunk>  - one or more text chunks
    data: [DONE] - terminal sentinel

Error responses
    422  Unprocessable Entity – invalid request body (FastAPI default)
    400  Bad Request          – unknown provider / model error
    500  Internal Server Error – unexpected runtime failure
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.models import LLMConfig, Message
from app.service.llm.factory import create_llm, available_providers

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    provider: str = Field(..., examples=["ollama", "deepseek"])
    model: str = Field(..., examples=["qwen3.5:latest", "deepseek-chat"])
    messages: list[Message] = Field(..., min_length=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    system_prompt: str = Field("You are a honest and truthful assistant.", min_length=0)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------
def _sse(data: str) -> str:
    """Format a single SSE frame"""
    return f"data: {data}\n\n"


async def _stream_sse(request: ChatRequest) -> AsyncIterator[str]:
    """
    Core generator: instantiates the requested LLM, streams chunks, then
    emits the [DONE] sentinel.  The LLM client is always closed via the
    async context-manager even when the client disconnects mid-stream.
    """
    config = LLMConfig(
        model=request.model,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
    )

    async with create_llm(request.provider, config) as llm:
        async for chunk in llm.stream_chat(request.messages):
            # Encode chunk as JSON so special characters are safe in SSE
            yield _sse(json.dumps(chunk))
            # Yield control back to the event loop so the response is flushed
            await asyncio.sleep(0)

    yield _sse("[DONE]")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@router.post(
    "/stream",
    summary="Stream a chat response (SSE)",
    response_description="Server-sent events stream of text chunks",
)
async def stream_chat(body: ChatRequest) -> StreamingResponse:
    """
    Stream an LLM response for the supplied conversation history.

    - **provider** must be one of the registered providers (see `/providers`).
    - **messages** must contain at least one entry.
    - Chunks arrive as `data: <json-encoded-string>` SSE frames.
    - The stream ends with a `data: [DONE]` sentinel frame.
    """
    if body.provider not in available_providers():
        raise HTTPException(
            status_code=400,
            detail={
                f"Unknown provider '{body.provider}'Available: {available_providers()}"
            },
        )

    try:
        return StreamingResponse(
            _stream_sse(body),
            media_type="text/event-stream",
            headers={
                # Prevent proxies / nginx from buffering the stream
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )
    except KeyError as exc:
        print(f"Provider lookup failed: {exc}")
        raise HTTPException(status_code=400, detail=(str(exc))) from exc
    except Exception as exc:
        print("Unexpected error in stream_chat")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


# ---------------------------------------------------------------------------
# Introspection helper
# ---------------------------------------------------------------------------
@router.get("/providers", summary="List registered LLM providers")
async def list_providers() -> dict:
    """Return all provider names currently registered in the global registry."""
    return {"providers": available_providers()}
