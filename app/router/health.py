from fastapi import APIRouter
from datetime import datetime, timezone
from app.service.database import get_db_connection

router = APIRouter(prefix="/health", tags=["health"])


def _check_db() -> tuple[str, str | None]:
    """
    Attempt a lightweight SELECT 1 against the database.
    Returns ("ok", None) on success, or ("error", <message>) on failure.
    """
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        return "ok", None
    except Exception as exc:
        return "error", str(exc)


@router.get("/")
def liveness():
    db_status, db_error = _check_db()
    overall = "ok" if db_status == "ok" else "degraded"

    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "database": {
                "status": db_status,
                **({"error": db_error} if db_error else {}),
            }
        },
    }
