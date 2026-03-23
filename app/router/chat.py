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
    conv_id    : int | None = None - if set, the history is loaded from DB,
                                    prepended to `messages`, and both the user turn
                                    and assistant reply are persisted

Context window management
    The combined list (DB history + new messages) is truncated to the latest
    MAX_CONTEXT_MESSAGES entries before being sent for inference

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
from app.service.database import get_conversation, get_messages, add_message
from app.util.logging import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])

# maximum number of messages forwarded to LLM
MAX_CONTEXT_MESSAGES = 10

logger = get_logger("chat_endpoint")

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    provider: str = Field(..., examples=["ollama", "deepseek"])
    model: str = Field(..., examples=["qwen3.5:latest", "deepseek-chat"])
    messages: list[Message] = Field(..., min_length=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    system_prompt: str = Field("You are a honest and truthful assistant.", min_length=0)
    conv_id: int | None = Field(
        default=None,
        description=(
            "Optional conversation ID. When provided, previous messages are "
            "fetched from the database and prepended to `messages` before "
            "inference. The new user turn and the assistant reply are also "
            "persisted back to the database."
        ),
    )


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------
def _sse(data: str) -> str:
    """Format a single SSE frame"""
    return f"data: {data}\n\n"


async def _stream_sse(
    request: ChatRequest, assembled_messages: list[Message]
) -> AsyncIterator[str]:
    """
    Core generator: instantiates the requested LLM, streams chunks, then
    emits the [DONE] sentinel.  The LLM client is always closed via the
    async context-manager even when the client disconnects mid-stream.
    If `request.conv_id` is set, the fully accumulated assistant reply is
    persisted to the database after the stream ends.

    Args:
        request:            The original ChatRequest (used for LLM config and
                            conv_id).
        assembled_messages: The final message list to send to the LLM
                            (DB history + new user turn, already truncated).
    """
    config = LLMConfig(
        model=request.model,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
    )

    accumulated_reply: list[str] = []

    async with create_llm(request.provider, config) as llm:
        async for chunk in llm.stream_chat(assembled_messages):
            accumulated_reply.append(chunk)
            # Encode chunk as JSON so special characters are safe in SSE
            yield _sse(json.dumps(chunk))
            # Yield control back to the event loop so the response is flushed
            await asyncio.sleep(0)

    # persist the assistant reply once the stream is fully consumed
    if request.conv_id is not None and accumulated_reply:
        full_reply = "".join(accumulated_reply)
        add_message(
            conv_id=request.conv_id,
            role="assistant",
            content=full_reply,
            model=request.model,
        )

    yield _sse("[DONE]")


# ---------------------------------------------------------------------------
# Context-window assembly
# ---------------------------------------------------------------------------
def _assemble_messages(
    body: ChatRequest,
) -> list[Message]:
    """
    Build the final message list for LLM inference:

    1. If `conv_id` is provided, load DB history and convert each row to a
       `Message` object.
    2. Append the new user turn(s) from the request.
    3. Truncate to the last MAX_CONTEXT_MESSAGES entries.

    The new user turn is also persisted to the database at this point so it
    appears in future history fetches regardless of whether the stream
    succeeds.

    Args:
        body: The incoming ChatRequest.

    Returns:
        The assembled, truncated list of Message objects ready for inference.

    Raises:
        HTTPException 404: if conv_id is provided but not found in the DB.
    """
    if body.conv_id is None:
        # stateless call - use the request message as is
        return body.messages

    # validate conversation exists
    if not get_conversation(body.conv_id):
        raise HTTPException(
            status_code=404, detail=f"Conversation id {body.conv_id} not found"
        )

    # load existing history from the database
    db_rows = get_messages(body.conv_id)
    history: list[Message] = [
        Message(role=row["role"], content=row["content"]) for row in db_rows
    ]

    # persist the new user turn before inference
    for msg in body.messages:
        add_message(conv_id=body.conv_id, role=msg.role, content=msg.content)

    # combine DB history + current request body
    combined = history + body.messages

    return combined[-MAX_CONTEXT_MESSAGES:]


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
    - **conv_id** is optional. When supplied, previous messages are fetched
      from the database and prepended to `messages`. Both the new user turn
      and the final assistant reply are saved back to the database. The
      combined history is truncated to the latest 10 messages before inference.
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

    assembled = _assemble_messages(body)

    try:
        return StreamingResponse(
            _stream_sse(body, assembled),
            media_type="text/event-stream",
            headers={
                # Prevent proxies / nginx from buffering the stream
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )
    except KeyError as exc:
        logger.error(f"Provider lookup failed: {exc}")
        raise HTTPException(status_code=400, detail=(str(exc))) from exc
    except Exception as exc:
        logger.error("Unexpected error in stream_chat")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


# ---------------------------------------------------------------------------
# Introspection helper
# ---------------------------------------------------------------------------
@router.get("/providers", summary="List registered LLM providers")
async def list_providers() -> dict:
    """Return all provider names currently registered in the global registry."""
    return {"providers": available_providers()}
