"""
database.py
SQLite database module for a personal chatbot app.
Handles connection, schema creation and CRUD operations
for Conversation and Message entities
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator  # yield

from app.util.logging import get_logger

logger = get_logger("database")

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH = "chatbot.db"

# ── Schema ────────────────────────────────────────────────────────────────────
# data definition language
DDL = """
CREATE TABLE IF NOT EXISTS conversation (
    conv_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL DEFAULT 'New Conversation',
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS message (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     INTEGER NOT NULL,
    role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content     TEXT    NOT NULL,
    model       TEXT,
    created_at  TEXT    NOT NULL,
    FOREIGN KEY (conv_id) REFERENCES conversation (conv_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_message_conv_id ON message (conv_id);
"""

# ── Connection ─────────────────────────────────────────────────────────────────


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Open a raw SQLite connection with sensible defaults.
    Prefer the `db_cursor` context manager for everyday use.
    """
    conn = sqlite3.connect(db_path)  # open connection
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")  # enforce FK constraints (data integrity)
    conn.execute("PRAGMA journal_mode = WAL")  # better concurrent read performance
    return conn


@contextmanager
def db_cursor(db_path: str = DB_PATH) -> Generator[sqlite3.Cursor, None, None]:
    """
    Context manager that yields a cursor, commits on success,
    and rolls back + re-raises on any exception.

    Usage:
        with db_cursor() as cur:
            cur.execute(...)
    """
    conn = get_db_connection(db_path)
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialisation ─────────────────────────────────────────────────────────────
def init_db(db_path: str = DB_PATH) -> None:
    """Create tables and indexes if they do not already exist."""
    conn = get_db_connection(db_path)
    try:
        conn.executescript(DDL)
        conn.commit()
        logger.info(f"Database initialized at '{db_path}'")
    finally:
        conn.close()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _now() -> str:
    """Return the current UTC time as an ISO-8601 string"""
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row else None


# ── Conversation CRUD ──────────────────────────────────────────────────────────
def create_conversation(title: str = "New conversation") -> dict:
    """
    Insert a new conversation row
    Returns the newly created conversation as a dict.
    """
    sql = "INSERT INTO conversation (title, created_at) VALUES (?, ?)"
    with db_cursor() as cur:
        cur.execute(sql, (title, _now()))
        conv_id = cur.lastrowid

    if conv_id is None:
        return {}

    conv = get_conversation(conv_id)
    return conv if conv else {}


def get_conversation(conv_id: int) -> dict | None:
    """Fetch a single conversation by its primary key."""
    sql = "SELECT * FROM conversation WHERE conv_id = ?"
    with db_cursor() as cur:
        cur.execute(sql, (conv_id,))
        return _row_to_dict(cur.fetchone())


def list_conversations() -> list[dict]:
    """Return all conversations ordered by most-recently created first"""
    sql = "SELECT * FROM conversation ORDER BY created_at DESC"
    with db_cursor() as cur:
        cur.execute(sql)
        return [dict(row) for row in cur.fetchall()]


def update_conversation_title(conv_id: int, title: str) -> dict | None:
    """Rename a conversation. Returns the updated row, or None if not found."""
    sql = "UPDATE conversation SET title = ? WHERE conv_id = ?"
    with db_cursor() as cur:
        cur.execute(sql, (title, conv_id))

    return get_conversation(conv_id)


def delete_conversation(conv_id: int) -> bool:
    """
    Delete a conversation and all its associated messages
    (cascade is handled by the FK ON DELETE CASCADE).

    Returns True if a row was deleted, False if conv_id was not found.
    """
    sql = "DELETE FROM conversation WHERE conv_id = ?"
    with db_cursor() as cur:
        cur.execute(sql, (conv_id,))
        return cur.rowcount > 0


# ── Message CRUD ───────────────────────────────────────────────────────────────


def add_message(
    conv_id: int, role: str, content: str, model: str | None = None
) -> dict:
    """Append a message to an existing conversation

    Args:
        conv_id:  Parent conversation ID.
        role:     One of 'user', 'assistant', or 'system'.
        content:  Text body of the message.
        model:    Optional model identifier (e.g. 'deepseek', 'ollama').

    Returns the newly inserted message as a dict.

    Raises:
        ValueError: if `role` is not one of the accepted values.
        sqlite3.IntegrityError: if conv_id does not reference an existing conversation."""
    valid_roles = {"user", "system", "assistant"}
    if role not in valid_roles:
        raise ValueError(f"role must be one of {valid_roles}, got {role}")

    sql = """
    INSERT INTO message (conv_id, role, content, model, created_at)
    VALUES (?, ?, ?, ?, ?)
"""

    with db_cursor() as cur:
        cur.execute(sql, (conv_id, role, content, model, _now()))
        msg_id = cur.lastrowid

    if msg_id is None:
        return {}

    msg = get_message(msg_id)
    return msg if msg else {}


def get_message(msg_id: int) -> dict | None:
    """Fetch a single message by its primary key"""
    sql = "SELECT * FROM message WHERE id = ?"
    with db_cursor() as cur:
        cur.execute(sql, (msg_id,))
        return _row_to_dict(cur.fetchone())


def get_messages(conv_id: int) -> list[dict]:
    """
    Fetch all messages that belong to a conversation,
    ordered chronologically (oldest first).

    This is the primary method for reconstructing a conversation thread.
    """
    sql = """
    SELECT * FROM message
    WHERE conv_id = ?
    ORDER BY created_at ASC
"""

    with db_cursor() as cur:
        cur.execute(sql, (conv_id,))
        return [dict(row) for row in cur.fetchall()]


def delete_message(msg_id: int) -> bool:
    """Delete a single message. Returns True if deleted, False if not found."""
    sql = "DELETE FROM message WHERE id = ?"
    with db_cursor() as cur:
        cur.execute(sql, (msg_id,))
        return cur.rowcount > 0


# ── FastAPI dependency ─────────────────────────────────────────────────────────
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """
    FastAPI dependency that yields a connection per request
    and guarantees it is closed afterwards.

    Usage in a route:
        @app.get("/conversations")
        def list_convs(conn: sqlite3.Connection = Depends(get_db)):
            ...
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()

    # create conversation
    conv = create_conversation("my first chat")
    print("Created:", conv)

    # Add messages
    add_message(conv["conv_id"], "user", "Hello, who are you?")
    add_message(
        conv["conv_id"],
        "assistant",
        "I am your personal assistant.",
        model="claude-sonnet-4-20250514",
    )
    add_message(conv["conv_id"], "user", "Great, let's get started!")

    # fetch the full thread
    messages = get_messages(conv["conv_id"])
    print(f"\n{len(messages)} messages in conversation {conv['conv_id']}:")
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content']}")

    # delete the conversation (cascades to messages)
    deleted = delete_conversation(conv["conv_id"])
    print(f"\nDeleted conversation: {deleted}")
    print("Messages after delete:", get_messages(conv["conv_id"]))
