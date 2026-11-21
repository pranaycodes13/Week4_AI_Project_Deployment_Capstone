"""Comprehensive automated test suite for Streamlit frontend API client and validation."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from src.frontend.api_client import (
    ApiConnectionError,
    ApiResponseError,
    ApiServerError,
    ApiServiceUnavailableError,
    ApiTimeoutError,
    ApiValidationError,
    WineApiClient,
    resolve_api_base_url,
    validate_feature_dict,
)


@pytest.fixture
def valid_features() -> dict[str, float]:
    """Provide a valid 13-feature input dictionary."""
    return {
        "alcohol": 13.2,
        "malic_acid": 1.78,
        "ash": 2.14,
        "alcalinity_of_ash": 11.2,
        "magnesium": 100.0,
        "total_phenols": 2.65,
        "flavanoids": 2.76,
        "nonflavanoid_phenols": 0.26,
        "proanthocyanins": 1.28,
        "color_intensity": 4.38,
        "hue": 1.05,
        "od280/od315_of_diluted_wines": 3.4,
        "proline": 1050.0,
    }


# ==============================================================================
# 1. Feature Validation Logic Tests
# ==============================================================================


def test_validate_feature_dict_success(valid_features: dict[str, float]) -> None:
    """Verify validation passes for valid dictionary."""
    validated = validate_feature_dict(valid_features)
    assert len(validated) == 13
    assert validated["alcohol"] == 13.2
    assert validated["od280/od315_of_diluted_wines"] == 3.4


def test_validate_feature_dict_missing_feature(valid_features: dict[str, float]) -> None:
    """Verify validation fails when a feature is omitted."""
    del valid_features["proline"]
    with pytest.raises(ApiValidationError, match="Missing required feature"):
        validate_feature_dict(valid_features)


def test_validate_feature_dict_extra_feature(valid_features: dict[str, float]) -> None:
    """Verify validation fails when extra fields are included."""
    valid_features["vintage_year"] = 2019
    with pytest.raises(ApiValidationError, match="Unexpected extra feature"):
        validate_feature_dict(valid_features)


def test_validate_feature_dict_rejects_boolean(valid_features: dict[str, float]) -> None:
    """Verify validation strictly rejects boolean inputs."""
    valid_features["alcohol"] = True  # type: ignore[assignment]
    with pytest.raises(ApiValidationError, match="cannot be a boolean"):
        validate_feature_dict(valid_features)


def test_validate_feature_dict_rejects_nan_and_inf(valid_features: dict[str, float]) -> None:
    """Verify validation rejects NaN and Infinity."""
    valid_features["ash"] = float("nan")
    with pytest.raises(ApiValidationError, match="cannot be NaN or Infinite"):
        validate_feature_dict(valid_features)

    valid_features["ash"] = float("inf")
    with pytest.raises(ApiValidationError, match="cannot be NaN or Infinite"):
        validate_feature_dict(valid_features)


def test_validate_feature_dict_rejects_non_numeric(valid_features: dict[str, float]) -> None:
    """Verify validation rejects non-convertible strings."""
    valid_features["hue"] = "invalid_string"  # type: ignore[assignment]
    with pytest.raises(ApiValidationError, match="must be a numeric value"):
        validate_feature_dict(valid_features)


# ==============================================================================
# 2. WineApiClient Endpoint & Error Handling Tests (Mocked Transport)
# ==============================================================================


def test_client_url_trailing_slash_normalization() -> None:
    """Verify client normalizes trailing slashes safely."""
    client = WineApiClient(base_url="http://localhost:8000///")
    assert client.base_url == "http://localhost:8000"


def test_resolve_api_base_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolve_api_base_url returns default 127.0.0.1:8000 when no overrides exist."""
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("API_URL", raising=False)
    assert resolve_api_base_url() == "http://127.0.0.1:8000"


def test_resolve_api_base_url_from_api_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolve_api_base_url respects API_BASE_URL env var with trailing slashes."""
    monkeypatch.setenv("API_BASE_URL", "http://wine-api:8000///")
    monkeypatch.delenv("API_URL", raising=False)
    assert resolve_api_base_url() == "http://wine-api:8000"


def test_resolve_api_base_url_from_api_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolve_api_base_url falls back to API_URL env var when API_BASE_URL is absent."""
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setenv("API_URL", "http://backup-host:9000/")
    assert resolve_api_base_url() == "http://backup-host:9000"


def test_resolve_api_base_url_env_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify API_BASE_URL takes precedence over API_URL."""
    monkeypatch.setenv("API_BASE_URL", "http://primary-api:8000")
    monkeypatch.setenv("API_URL", "http://secondary-api:8000")
    assert resolve_api_base_url() == "http://primary-api:8000"


def test_resolve_api_base_url_session_override_highest_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify explicit session override takes highest priority over all environment variables."""
    monkeypatch.setenv("API_BASE_URL", "http://wine-api:8000")
    assert resolve_api_base_url("http://user-sidebar-override:8000///") == "http://user-sidebar-override:8000"


def test_client_context_manager_lifecycle() -> None:
    """Verify client cleanly enters and exits context manager."""
    with WineApiClient(base_url="http://localhost:8000") as client:
        assert client.base_url == "http://localhost:8000"
        assert not client.client.is_closed
    # Client closed upon exit
    assert client._client.is_closed


def test_health_success() -> None:
    """Verify successful health endpoint query."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(
            200,
            json={
                "status": "healthy",
                "model_loaded": True,
                "model_ready": True,
                "model_name": "Support Vector Classifier (SVC)",
                "version": "1.0.0",
            },
        )

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://testserver", client=mock_client)
    res = api.health()

    assert res["status"] == "healthy"
    assert res["model_loaded"] is True
    assert res["model_ready"] is True


def test_model_info_success() -> None:
    """Verify successful model-info endpoint query."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/model-info"
        return httpx.Response(
            200,
            json={
                "model_name": "Wine Classification Pipeline",
                "model_type": "sklearn.svm.SVC",
                "pipeline_steps": ["StandardScaler", "SVC"],
                "feature_count": 13,
                "feature_names": ["alcohol"],
                "target_class_labels": [0, 1, 2],
                "target_class_names": ["class_0", "class_1", "class_2"],
                "hyperparameters": {"C": 10.0},
                "cross_validation_metrics": {"mean_cv_macro_f1": 0.9927},
                "holdout_test_metrics": {"accuracy": 0.9444},
                "scikit_learn_version": "1.7.2",
                "joblib_version": "1.5.2",
            },
        )

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://testserver", client=mock_client)
    res = api.model_info()

    assert res["model_name"] == "Wine Classification Pipeline"
    assert res["hyperparameters"]["C"] == 10.0
    assert res["holdout_test_metrics"]["accuracy"] == 0.9444


def test_predict_single_success(valid_features: dict[str, float]) -> None:
    """Verify single prediction call and response structure."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        return httpx.Response(
            200,
            json={
                "predicted_class": 0,
                "predicted_class_name": "class_0",
                "probabilities": {"class_0": 0.9859, "class_1": 0.0075, "class_2": 0.0066},
                "model_name": "Support Vector Classifier (SVC)",
                "model_version": "1.0.0",
            },
        )

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://testserver", client=mock_client)
    res = api.predict(valid_features)

    assert res["predicted_class"] == 0
    assert res["predicted_class_name"] == "class_0"
    assert res["probabilities"]["class_0"] == 0.9859


def test_predict_batch_success(valid_features: dict[str, float]) -> None:
    """Verify batch prediction preserves ordering and parses correctly."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict/batch"
        return httpx.Response(
            200,
            json={
                "predictions": [
                    {
                        "sample_index": 0,
                        "predicted_class": 0,
                        "predicted_class_name": "class_0",
                        "probabilities": {"class_0": 0.9859, "class_1": 0.0075, "class_2": 0.0066},
                    },
                    {
                        "sample_index": 1,
                        "predicted_class": 1,
                        "predicted_class_name": "class_1",
                        "probabilities": {"class_0": 0.01, "class_1": 0.98, "class_2": 0.01},
                    },
                ],
                "total_records": 2,
                "successful_predictions": 2,
                "model_name": "Support Vector Classifier (SVC)",
            },
        )

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://testserver", client=mock_client)
    res = api.predict_batch([valid_features, valid_features])

    assert res["total_records"] == 2
    assert len(res["predictions"]) == 2
    assert res["predictions"][0]["sample_index"] == 0
    assert res["predictions"][1]["sample_index"] == 1


def test_predict_batch_limits(valid_features: dict[str, float]) -> None:
    """Verify batch request rejects empty list and >100 samples."""
    api = WineApiClient(base_url="http://testserver")

    with pytest.raises(ApiValidationError, match="cannot be empty"):
        api.predict_batch([])

    oversized = [valid_features] * 101
    with pytest.raises(ApiValidationError, match="exceeds maximum limit of 100"):
        api.predict_batch(oversized)


def test_http_422_validation_error(valid_features: dict[str, float]) -> None:
    """Verify HTTP 422 maps to ApiValidationError with formatted details."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "error": "Input Validation Error",
                "detail": [
                    {"location": "body -> alcohol", "message": "Input should be a valid number"}
                ],
            },
        )

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://testserver", client=mock_client)

    with pytest.raises(ApiValidationError) as exc_info:
        api.predict(valid_features)
    assert "body -> alcohol" in str(exc_info.value)
    assert exc_info.value.status_code == 422


def test_http_503_service_unavailable() -> None:
    """Verify HTTP 503 maps to ApiServiceUnavailableError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"error": "Service Unavailable", "detail": "Model artifact is currently not loaded."},
        )

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://testserver", client=mock_client)

    with pytest.raises(ApiServiceUnavailableError, match="Model artifact is currently not loaded"):
        api.health()


def test_http_500_internal_server_error() -> None:
    """Verify HTTP 500 maps to ApiServerError with sanitized message."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": "Internal Server Error", "detail": "Internal error occurred."},
        )

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://testserver", client=mock_client)

    with pytest.raises(ApiServerError, match="Internal server error occurred in backend"):
        api.health()


def test_connection_error_handling() -> None:
    """Verify network connection failures raise ApiConnectionError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by target host")

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://unreachable-host:8000", client=mock_client)

    with pytest.raises(ApiConnectionError, match="Cannot connect to Wine Classification API"):
        api.health()


def test_timeout_handling() -> None:
    """Verify read/connect timeouts raise ApiTimeoutError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Server took too long to respond")

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://slow-host:8000", client=mock_client)

    with pytest.raises(ApiTimeoutError, match="timed out"):
        api.health()


def test_timeout_configuration_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default timeout values (15s connect, 90s read) and env overrides."""
    # 1. Default instance matches 15s connect and 90s read
    api_default = WineApiClient(base_url="http://testserver")
    assert api_default.timeout.connect == 15.0
    assert api_default.timeout.read == 90.0

    # 2. Explicit constructor parameters override module defaults
    api_custom = WineApiClient(base_url="http://testserver", connect_timeout=10.0, request_timeout=45.0)
    assert api_custom.timeout.connect == 10.0
    assert api_custom.timeout.read == 45.0

    # 3. Environment variable overrides dynamically respected
    monkeypatch.setenv("API_CONNECT_TIMEOUT", "20.0")
    monkeypatch.setenv("API_REQUEST_TIMEOUT", "120.0")

    api_env = WineApiClient(base_url="http://testserver")
    assert api_env.timeout.connect == 20.0
    assert api_env.timeout.read == 120.0


def test_invalid_json_handling() -> None:
    """Verify invalid JSON response raises ApiResponseError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>Bad Gateway</body></html>")

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://testserver", client=mock_client)

    with pytest.raises(ApiResponseError, match="invalid JSON"):
        api.health()


def test_unexpected_response_structure() -> None:
    """Verify non-dictionary or missing key structure raises ApiResponseError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "array"])

    mock_client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    api = WineApiClient(base_url="http://testserver", client=mock_client)

    with pytest.raises(ApiResponseError, match="Unexpected response structure"):
        api.health()


# ==============================================================================
# 3. Streamlit Application AppTest Execution
# ==============================================================================


def test_streamlit_app_loads_cleanly() -> None:
    """Verify Streamlit app initializes and mounts without runtime exceptions."""
    at = AppTest.from_file("../src/frontend/app.py", default_timeout=10)
    at.run()
    assert len(at.exception) == 0


def test_frontend_does_not_claim_guaranteed_calibration() -> None:
    """Verify frontend code and UI do not claim guaranteed probability calibration."""
    app_source = Path("src/frontend/app.py").read_text(encoding="utf-8")
    assert "guaranteed calibrated" not in app_source.lower()
    assert "calibrated probability certainty" not in app_source.lower()
    assert "not guaranteed to be empirically calibrated" in app_source
