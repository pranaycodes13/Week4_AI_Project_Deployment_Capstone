"""Comprehensive automated tests for structured JSON logging and request correlation middleware."""

import concurrent.futures
import io
import json
import logging
import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.logging_utils import (
    JsonFormatter,
    is_valid_request_id,
)
from src.api.service import ModelNotLoadedError, WineModelService


@pytest.fixture
def test_client() -> TestClient:
    """Fixture providing configured TestClient with active model service."""
    app = create_app()
    with TestClient(app) as client:
        yield client


# ==============================================================================
# 1. Request ID Validation & Normalization Unit Tests
# ==============================================================================


def test_is_valid_request_id_valid_cases() -> None:
    """Verify valid request IDs matching alphanumeric, hyphen, and underscore pattern."""
    assert is_valid_request_id("abc-123_XYZ") is True
    assert is_valid_request_id(str(uuid.uuid4())) is True
    assert is_valid_request_id("a" * 64) is True
    assert is_valid_request_id("1") is True


def test_is_valid_request_id_invalid_cases() -> None:
    """Verify rejection of invalid request IDs to prevent log/header injection."""
    assert is_valid_request_id("") is False
    assert is_valid_request_id(None) is False
    assert is_valid_request_id("a" * 65) is False  # Oversized (> 64)
    assert is_valid_request_id("bad id with spaces") is False
    assert is_valid_request_id("req\nid\ninjection") is False
    assert is_valid_request_id("req;DROP TABLE;") is False
    assert is_valid_request_id("<script>alert(1)</script>") is False
    assert is_valid_request_id("id@domain.com") is False


# ==============================================================================
# 2. Response Header & Request ID Middleware Integration Tests
# ==============================================================================


def test_missing_request_id_generates_valid_uuid(test_client: TestClient) -> None:
    """Verify a missing X-Request-ID header triggers automatic generation of a valid UUID."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    req_id = response.headers["X-Request-ID"]
    # Must be parseable as a valid UUID4
    parsed_uuid = uuid.UUID(req_id, version=4)
    assert str(parsed_uuid) == req_id


def test_valid_client_request_id_is_preserved(test_client: TestClient) -> None:
    """Verify a valid incoming X-Request-ID is preserved and returned."""
    custom_id = "client-trace-id-12345_XYZ"
    response = test_client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id


def test_invalid_request_id_is_safely_replaced(test_client: TestClient) -> None:
    """Verify an invalid request ID with control/special chars is safely replaced with a UUID."""
    malicious_id = "bad\r\ninjection <script>"
    response = test_client.get("/health", headers={"X-Request-ID": malicious_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != malicious_id

    # Replaced with valid UUID
    parsed_uuid = uuid.UUID(response.headers["X-Request-ID"], version=4)
    assert str(parsed_uuid) == response.headers["X-Request-ID"]


def test_oversized_request_id_is_safely_replaced(test_client: TestClient) -> None:
    """Verify an oversized request ID (>64 chars) is replaced with a standard UUID."""
    oversized_id = "x" * 128
    response = test_client.get("/health", headers={"X-Request-ID": oversized_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != oversized_id

    parsed_uuid = uuid.UUID(response.headers["X-Request-ID"], version=4)
    assert str(parsed_uuid) == response.headers["X-Request-ID"]


# ==============================================================================
# 3. Error Responses Include X-Request-ID
# ==============================================================================


def test_validation_error_422_contains_request_id(test_client: TestClient) -> None:
    """Verify 422 Unprocessable Content returns X-Request-ID header."""
    response = test_client.post("/predict", json={"malformed": "body"})
    assert response.status_code == 422
    assert "X-Request-ID" in response.headers
    uuid.UUID(response.headers["X-Request-ID"], version=4)


def test_not_found_404_contains_request_id(test_client: TestClient) -> None:
    """Verify 404 Not Found returns X-Request-ID header."""
    response = test_client.get("/nonexistent-endpoint-route")
    assert response.status_code == 404
    assert "X-Request-ID" in response.headers
    uuid.UUID(response.headers["X-Request-ID"], version=4)


def test_method_not_allowed_405_contains_request_id(test_client: TestClient) -> None:
    """Verify 405 Method Not Allowed returns X-Request-ID header."""
    response = test_client.get("/predict")
    assert response.status_code == 405
    assert "X-Request-ID" in response.headers
    uuid.UUID(response.headers["X-Request-ID"], version=4)


def test_service_unavailable_503_contains_request_id() -> None:
    """Verify 503 Service Unavailable returns X-Request-ID header."""
    mock_service = MagicMock(spec=WineModelService)
    mock_service.predict_single.side_effect = ModelNotLoadedError("Model missing.")
    app = create_app(service_instance=mock_service)
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
            "magnesium": 100.0, "total_phenols": 2.65, "flavanoids": 2.76,
            "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38,
            "hue": 1.05, "od280/od315_of_diluted_wines": 3.4, "proline": 1050.0
        })
        assert response.status_code == 503
        assert "X-Request-ID" in response.headers


def test_internal_server_error_500_sanitized_with_request_id() -> None:
    """Verify 500 Internal Server Error returns X-Request-ID and sanitized message without traceback."""
    mock_service = MagicMock(spec=WineModelService)
    mock_service.predict_single.side_effect = RuntimeError("Database crashed unexpectedly /etc/passwd")
    app = create_app(service_instance=mock_service)
    with TestClient(app, raise_server_exceptions=False) as client:
        custom_id = "trace-500-error-check"
        response = client.post(
            "/predict",
            json={
                "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
                "magnesium": 100.0, "total_phenols": 2.65, "flavanoids": 2.76,
                "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38,
                "hue": 1.05, "od280/od315_of_diluted_wines": 3.4, "proline": 1050.0
            },
            headers={"X-Request-ID": custom_id},
        )
        assert response.status_code == 500
        assert response.headers["X-Request-ID"] == custom_id
        body = response.json()
        assert body["error"] == "Internal Server Error"
        assert body["detail"] == "An internal server error occurred while processing the request."
        # Confirm internal paths/crash details are NOT exposed to the client
        assert "Database crashed" not in json.dumps(body)
        assert "/etc/passwd" not in json.dumps(body)


# ==============================================================================
# 4. Concurrency & Context Isolation Tests
# ==============================================================================


def test_concurrent_requests_receive_independent_ids(test_client: TestClient) -> None:
    """Verify concurrent requests get independent, isolated request IDs without leakage."""
    def make_request(index: int) -> tuple[int, str]:
        expected_id = f"worker-trace-{index:04d}"
        resp = test_client.get("/health", headers={"X-Request-ID": expected_id})
        return index, resp.headers.get("X-Request-ID", "")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(make_request, i) for i in range(25)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Confirm every thread's request ID matches exactly its own expected ID
    for index, returned_id in results:
        assert returned_id == f"worker-trace-{index:04d}"

    # Confirm all 25 returned IDs are strictly unique
    all_ids = [r[1] for r in results]
    assert len(set(all_ids)) == 25


def test_concurrent_generated_ids_are_unique(test_client: TestClient) -> None:
    """Verify concurrent requests without client IDs generate strictly unique UUIDs."""
    def make_request(_: int) -> str:
        resp = test_client.get("/health")
        return resp.headers.get("X-Request-ID", "")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(make_request, i) for i in range(30)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Every request must have a non-empty, unique UUID
    assert len(results) == 30
    assert len(set(results)) == 30
    for uid in results:
        uuid.UUID(uid, version=4)


# ==============================================================================
# 5. Structured JSON Logging Tests
# ==============================================================================


def test_json_formatter_structure_and_iso_timestamp() -> None:
    """Verify JsonFormatter outputs valid JSON with ISO-8601 UTC timestamp and correlation ID."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Processing test wine sample",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-12345"
    record.event = "sample_event"

    formatted = formatter.format(record)
    log_data = json.loads(formatted)

    assert log_data["level"] == "INFO"
    assert log_data["logger"] == "test_logger"
    assert log_data["message"] == "Processing test wine sample"
    assert log_data["request_id"] == "req-12345"
    assert log_data["event"] == "sample_event"

    # Verify timestamp is ISO-8601 UTC parseable
    parsed_dt = datetime.fromisoformat(log_data["timestamp"])
    assert parsed_dt.tzinfo is not None


def test_access_log_emitted_as_json_with_attributes() -> None:
    """Verify access logger emits valid JSON with method, path, status_code, and duration_ms."""
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(JsonFormatter())

    access_logger = logging.getLogger("wine_api.access")
    access_logger.handlers.clear()
    access_logger.addHandler(handler)
    access_logger.setLevel(logging.INFO)

    app = create_app()
    with TestClient(app) as client:
        custom_id = "trace-access-log-test"
        response = client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.status_code == 200

    log_output = log_stream.getvalue()
    assert log_output.strip() != ""

    # Parse JSON log line
    log_entry = json.loads(log_output.strip().splitlines()[-1])
    assert log_entry["event"] == "http_request"
    assert log_entry["request_id"] == custom_id
    assert log_entry["method"] == "GET"
    assert log_entry["path"] == "/health"
    assert log_entry["status_code"] == 200
    assert isinstance(log_entry["duration_ms"], (int, float))
    assert log_entry["duration_ms"] >= 0


def test_query_parameters_excluded_from_access_log_path() -> None:
    """Verify query parameters are excluded from logged path to avoid sensitive query leaks."""
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(JsonFormatter())

    access_logger = logging.getLogger("wine_api.access")
    access_logger.handlers.clear()
    access_logger.addHandler(handler)
    access_logger.setLevel(logging.INFO)

    app = create_app()
    with TestClient(app) as client:
        client.get("/health?token=secret_api_token_12345&user=admin")

    log_output = log_stream.getvalue()
    log_entry = json.loads(log_output.strip().splitlines()[-1])

    # Path must only be /health, zero sensitive query parameter contents
    assert log_entry["path"] == "/health"
    assert "secret_api_token_12345" not in json.dumps(log_entry)
