"""
test_health.py – unit tests for GET /health/
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock


DB_PATH = "app.router.health.get_db_connection"


class TestHealthLiveness:
    """GET /health/ – liveness probe"""

    def test_returns_ok_when_db_healthy(self, client):
        mock_conn = MagicMock()
        mock_conn.execute.return_value = None

        with patch(DB_PATH, return_value=mock_conn):
            response = client.get("/health/")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["checks"]["database"]["status"] == "ok"
        assert "error" not in body["checks"]["database"]

    def test_returns_degraded_when_db_fails(self, client):
        with patch(DB_PATH, side_effect=Exception("connection refused")):
            response = client.get("/health/")

        assert response.status_code == 200  # HTTP is still 200 – it's a health payload
        body = response.json()
        assert body["status"] == "degraded"
        assert body["checks"]["database"]["status"] == "error"
        assert "connection refused" in body["checks"]["database"]["error"]

    def test_response_contains_timestamp(self, client):
        mock_conn = MagicMock()
        with patch(DB_PATH, return_value=mock_conn):
            response = client.get("/health/")

        body = response.json()
        assert "timestamp" in body
        # ISO-8601 UTC timestamp ends with +00:00
        assert body["timestamp"].endswith("+00:00")

    def test_db_connection_closed_after_check(self, client):
        mock_conn = MagicMock()
        with patch(DB_PATH, return_value=mock_conn):
            client.get("/health/")

        mock_conn.close.assert_called_once()

    def test_db_execute_called_with_select_1(self, client):
        mock_conn = MagicMock()
        with patch(DB_PATH, return_value=mock_conn):
            client.get("/health/")

        mock_conn.execute.assert_called_once_with("SELECT 1")
