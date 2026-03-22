""" 
FastAPI application entry point
============================================
Creates and configures the FastAPI app, mounts all routers, and
exposes the ASGI callable used by uvicorn / gunicorn.

Run (development)
-----------------
    uvicorn app.main:app --reload --port 8000
 
Run (production)
----------------
    gunicorn app.main:app -k uvicorn.workers.UvicornWorker --workers 4
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.router.health import router as health_router
from app.router.chat import router as chat_router

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.service.llm.factory import available_providers
    print("LLM Gateway starting up")
    print(f"Registered providers: {available_providers()}")
    # needs to initialize database

    yield   # app run here

    print("LLM gateway shutting down")

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Gateway",
        description=(
            "Unified streaming interface for multiple LLM providers "
            "Ollama, Deepseek, any custom provider registered at runtime"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # ----------------------------------------------------------------
    # Middleware
    # ----------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------------
    # Routers
    # ----------------------------------------------------------------
    app.include_router(health_router)
    app.include_router(chat_router)

    return app

app = create_app()
