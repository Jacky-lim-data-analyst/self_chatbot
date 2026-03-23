"""
FASTAPI router:
Exposes endpoints for conversation and message management
delegating all persistence to database.py
"""

from fastapi import APIRouter, Depends, HTTPException, status
import sqlite3
from pydantic import BaseModel, Field

from app.service.database import (
    get_db,  # For FastAPI dependency
    create_conversation,
    add_message,
    delete_conversation,
    list_conversations,
    get_messages,
    get_conversation,
)

router = APIRouter(prefix="/api", tags=["chatbot"])

# ── Request / Response Schemas ─────────────────────────────────────────────────


class CreateConversationRequest(BaseModel):
    title: str = Field(
        default="New conversation",
        description="Display title for the conversation",
        min_length=1,
        examples=["My first chat"],
    )


class AddMessageRequest(BaseModel):
    role: str = Field(
        ...,
        description="One of 'user', 'assistant' or 'system'",
        examples=["user"],
    )
    content: str = Field(..., description="Text body of the message", min_length=1)
    model: str | None = Field(
        default=None, description="Optional model identifier, e.g. deepseek-chat"
    )


class MessageResponse(BaseModel):
    id: int
    conv_id: int
    role: str
    content: str
    model: str | None
    created_at: str


class ConversationResponse(BaseModel):
    conv_id: int
    title: str
    created_at: str


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
def new_conversation(
    body: CreateConversationRequest = CreateConversationRequest(),
    conn: sqlite3.Connection = Depends(get_db),
) -> ConversationResponse:
    """
    Create a new conversation with an optional title.
    If no title is provided, it defaults to `'New Conversation'`.

    Returns the newly created conversation row.
    """
    conv = create_conversation(title=body.title)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Conversation cannot be created",
        )
    return ConversationResponse(**conv)


@router.post(
    "/conversations/{conv_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert a message into a conversation",
)
def insert_message(
    conv_id: int,
    body: AddMessageRequest,
    conn: sqlite3.Connection = Depends(get_db),
) -> MessageResponse:
    """
    Append a new message to an existing conversation.

    - **conv_id**: the conversation to post the message into.
    - **role**: must be one of `user`, `assistant`, or `system`.
    - **content**: non-empty message text.
    - **model**: optional model tag (stored for auditing / display purposes).

    Raises **404** if the conversation does not exist.
    Raises **422** if `role` is invalid.
    """
    # condition: conversation must exist
    if not get_conversation(conv_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conv_id} not found",
        )

    try:
        msg = add_message(
            conv_id=conv_id, role=body.role, content=body.content, model=body.model
        )
    except ValueError as exc:
        # invalid role value
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    if not msg:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Message could not be inserted",
        )

    return MessageResponse(**msg)


@router.delete(
    "/conversations/{conv_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation and all its messages",
)
def remove_conversation(
    conv_id: int, conn: sqlite3.Connection = Depends(get_db)
) -> None:
    """
    Permanently delete a conversation **and** every message it contains
    (cascade is handled by the SQLite `ON DELETE CASCADE` foreign key).

    Raises **404** if the conversation does not exist.
    """
    deleted = delete_conversation(conv_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conv_id} not found",
        )


@router.get(
    "/conversations",
    response_model=list[ConversationResponse],
    summary="List all conversations",
)
def list_all_conversations(
    conn: sqlite3.Connection = Depends(get_db),
) -> list[ConversationResponse]:
    """
    Return every conversation stored in the database,
    ordered by most-recently created first.
    """
    rows = list_conversations()
    return [ConversationResponse(**row) for row in rows]


@router.get(
    "/conversations/{conv_id}/messages",
    response_model=list[MessageResponse],
    summary="Get all messages in a conversation",
)
def get_conversation_messages(
    conv_id: int, conn: sqlite3.Connection = Depends(get_db)
) -> list[MessageResponse]:
    """
    Retrieve the full message thread for a conversation,
    ordered chronologically (oldest first).

    Raises **404** if the conversation does not exist.
    """
    if not get_conversation(conv_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conv_id} not found",
        )

    messages = get_messages(conv_id)
    return [MessageResponse(**msg) for msg in messages]
