"""
Unit and integration tests for the /chat router

Strategy
--------
* Unit tests exercise the pure helper functions (_assemble_messages, _sse)
  by importing them directly and patching database calls.
* Integration tests hit the live TestClient for the SSE endpoint and parse
  the event stream to make assertions on the chunks delivered.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch


FACTORY = "app.service.llm.factory"
DB = "app.service.database"
CHAT_MOD = "app.routers.chat"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_sse(raw: str) -> list[str]:
    """Turns a raw SSE response body into a list of data values
    Strips the 'data: ' prefix and ignores blank lines"""
    events = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            events.append(line[len("data: ") :])

    return events


def make_llm_context(chunks: list[str]):
    """
    Build an async context-manager that yields an LLM stub whose
    stream_chat() produces the given text chunks.
    """
    llm_stub = MagicMock()
    llm_stub.stream_chat = AsyncMock(return_value=aiter_from(chunks))

    @asynccontextmanager
    async def _ctx(provider, config):
        yield llm_stub

    return _ctx


async def aiter_from(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# GET /chat/providers
# ---------------------------------------------------------------------------


class TestListProviders:
    def test_returns_registered_providers(self, client):
        with patch(
            f"{FACTORY}.available_providers", return_value=["ollama", "deepseek"]
        ):
            response = client.get("/chat/providers")

        assert response.status_code == 200
        assert set(response.json()["providers"]) == {"ollama", "deepseek"}

    def test_returns_empty_list_when_no_providers(self, client):
        with patch(f"{FACTORY}.available_providers", return_value=[]):
            response = client.get("/chat/providers")

        assert response.status_code == 200
        assert response.json()["providers"] == []
