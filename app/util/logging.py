"""
Custom logging module for the personal chatbot FastAPI backend

Features:
    - Colored console output (DEBUG + CRITICAL)
    - Daily-rotating log files (kept for 7 days)
    - Named child loggers per module (db, api, health)
    - Request/response middleware logger for FastAPI
    - One line setup: call configure_logging() in main.py

Usage:
    # main.py
    from logger import configure_logging, request_logging_middleware
    configure_logging()
    app.middleware("http")(request_logging_middleware)
 
    # any other module
    from logger import get_logger
    log = get_logger("db")
    log.info("Database initialised")
"""

import logging
import sys
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# ── Config ─────────────────────────────────────────────────────────────────────
APP_LOGGER_NAME = "chatbot"
LOG_DIR = Path(".logs")
LOG_FILE = LOG_DIR / "chatbot.log"
LOG_LEVEL_CONSOLE = logging.DEBUG
LOG_LEVEL_FILE = logging.INFO

# ── ANSI colour map ────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_COLOURS = {
    logging.DEBUG:    "\033[36m",   # cyan
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}

class _ColoredFormatter(logging.Formatter):
    """Formatter that prepends a coloured level badge to every console line."""

    _FMT = "{colour}{bold}[{level:<8}]{reset} {dim}{name}{reset}  {msg}"

    def format(self, record: logging.LogRecord) -> str:
        color = _COLOURS.get(record.levelno, _RESET)
        badge = self._FMT.format(
            colour=color,
            bold=_BOLD,
            level=record.levelname,
            reset=_RESET,
            dim="\033[2m]",
            name=record.name,
            msg=""
        )
        # Let the parent build the full message (exc_info etc.) then prepend badge
        original = super().format(record)
        return badge + original
    
# Plain format for log files (no ANSI codes)
_FILE_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ── Setup ──────────────────────────────────────────────────────────────────────

def configure_logging() -> None:
    """
    Call once at application startup (e.g. in main.py lifespan or module level).
    Idempotent — safe to call multiple times.
    """
    root = logging.getLogger(APP_LOGGER_NAME)
    if root.handlers:
        return
    
    root.setLevel(logging.DEBUG)   # handlers filter individually

    # ── Console handler ──────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(LOG_LEVEL_CONSOLE)
    console.setFormatter(
        _ColoredFormatter(
            fmt="%(asctime)s %(message)s",
            datefmt="%H:%M:%S"
        )
    )
    root.addHandler(console)

    # ── Rotating file handler ────────────────────────────────────────────────
    LOG_DIR.mkdir(exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        filename=LOG_FILE,
        when="midnight",    # rotate at midnight
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setLevel(LOG_LEVEL_FILE)
    file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    root.addHandler(file_handler)

    # ── Silence noisy third-party loggers ────────────────────────────────────
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info("Logging configured (console=%s, file=%s)",
              logging.getLevelName(LOG_LEVEL_CONSOLE), LOG_FILE)
    
# ── Public helper ──────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger namespaced under the app root.
 
    Args:
        name: Short label for the subsystem, e.g. "db", "api", "health".
 
    Example:
        log = get_logger("db")
        log.debug("Opened connection to %s", DB_PATH)
    """
    return logging.getLogger(f"{APP_LOGGER_NAME}.{name}")

# ── FastAPI request / response middleware ──────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request and its outcome.
 
    Attaches to FastAPI via:
        app.add_middleware(RequestLoggingMiddleware)
 
    Sample output:
        → GET  /health/          (no body)
        ← 200  GET  /health/     12.3 ms
        ← 422  POST /messages/   8.1 ms   [Unprocessable Entity]
    """
    _log = get_logger("http")

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()

        # Log the incoming request
        body_hint = "" if request.method in ("GET", "DELETE", "HEAD") else "(body)"
        self._log.debug("→ %-6s %s  %s", request.method, request.url.path, body_hint)

        response: Response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1_000
        level = logging.INFO if response.status_code < 400 else logging.WARNING

        if response.status_code >= 500:
            level = logging.ERROR

        self._log.log(
            level,
            "← %d  %-6s %s  %.1f ms",
            response.status_code,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        return response
    
# ── Convenience: log unhandled exceptions ─────────────────────────────────────
def log_exception(exc: Exception, context: str = "") -> None:
    """
    Log an unexpected exception with full traceback.
    Use inside except blocks where you want a record but still re-raise.
 
    Example:
        try:
            risky_operation()
        except Exception as e:
            log_exception(e, context="add_message")
            raise
    """
    _exc_log = get_logger("exception")
    _exc_log.exception("Unhandled exception%s: %s",
                       f" in {context}" if context else "", exc)
    