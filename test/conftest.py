"""
conftest.py – shared fixtures for the entire test suite.

Stub strategy
-------------
We only stub the *leaf* modules the routers depend on that we don't want
to execute during tests (database, llm factory, models, logging).
The real `app` package spine is left untouched on sys.modules.

Correct module paths (from the actual project layout):
  app.router        (not app.routers)
  app.models        (package with __init__.py)
  app.service.database
  app.service.llm.factory
  app.util.logging
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Stub factory
# ---------------------------------------------------------------------------


def _stub(name: str, **attrs) -> types.ModuleType:
    """Register a fake module in sys.modules only if not already loaded."""
    if name in sys.modules:
        mod = sys.modules[name]
    else:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = sys.modules.get(parts[0])
            if parent is not None:
                setattr(parent, parts[1], mod)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# Stub leaf dependencies (parent namespaces before children)
_stub(
    "app.models",
    LLMConfig=MagicMock,
    Message=lambda role, content: MagicMock(role=role, content=content),
)
_stub("app.util")
_stub(
    "app.util.logging",
    get_logger=lambda _: MagicMock(),
)
_stub("app.service")
_stub("app.service.llm")
_stub(
    "app.service.llm.factory",
    create_llm=MagicMock(),
    available_providers=MagicMock(return_value=["ollama", "deepseek"]),
)
_stub(
    "app.service.database",
    get_db=MagicMock(),
    get_db_connection=MagicMock(),
    create_conversation=MagicMock(),
    add_message=MagicMock(),
    delete_conversation=MagicMock(),
    list_conversations=MagicMock(),
    get_messages=MagicMock(),
    get_conversation=MagicMock(),
)

# ---------------------------------------------------------------------------
# Import routers — correct path is app.router (no trailing 's')
# ---------------------------------------------------------------------------

from app.router.chat import router as chat_router  # noqa: E402
from app.router.conversation import router as conv_router  # noqa: E402
from app.router.health import router as health_router  # noqa: E402


# ---------------------------------------------------------------------------
# App + client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def app() -> FastAPI:
    application = FastAPI()
    application.include_router(chat_router)
    application.include_router(conv_router)
    application.include_router(health_router)
    return application


@pytest.fixture(scope="session")
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Reusable data factories
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_message() -> dict:
    return {"role": "user", "content": "Hello!"}


@pytest.fixture
def sample_conversation() -> dict:
    return {"conv_id": 1, "title": "Test chat", "created_at": "2024-01-01T00:00:00"}


@pytest.fixture
def sample_db_message() -> dict:
    return {
        "id": 1,
        "conv_id": 1,
        "role": "user",
        "content": "Hello!",
        "model": None,
        "created_at": "2024-01-01T00:00:00",
    }
