"""HTTP client for communicating with the Wine Classification & Prediction FastAPI service."""

import contextlib
import json
import math
import os
from types import TracebackType
from typing import Any, Self

import httpx

from src.config import WINE_FEATURE_NAMES

DEFAULT_API_BASE_URL: str = "http://127.0.0.1:8000"
DEFAULT_CONNECT_TIMEOUT: float = 15.0
DEFAULT_REQUEST_TIMEOUT: float = 90.0


def resolve_api_base_url(session_override: str | None = None) -> str:
    """Resolve API base URL respecting precedence: session override -> API_BASE_URL -> API_URL -> st.secrets -> local default.

    Args:
        session_override: Optional explicit URL override (e.g., from Streamlit session state or user input).

    Returns:
        str: Sanitized API base URL with trailing slashes stripped.
    """
    if session_override and session_override.strip():
        return session_override.strip().rstrip("/")
    env_url = os.getenv("API_BASE_URL") or os.getenv("API_URL")
    if env_url and env_url.strip():
        return env_url.strip().rstrip("/")

    # Fallback to Streamlit Community Cloud st.secrets if configured
    with contextlib.suppress(KeyError, AttributeError, FileNotFoundError):
        import streamlit as st

        secret_url = st.secrets.get("API_BASE_URL") or st.secrets.get("API_URL")
        if secret_url and str(secret_url).strip():
            return str(secret_url).strip().rstrip("/")

    return DEFAULT_API_BASE_URL


class ApiClientError(Exception):
    """Base exception for all API client interactions."""

    def __init__(self, message: str, status_code: int | None = None, detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail


class ApiConnectionError(ApiClientError):
    """Raised when connection to backend fails or is refused."""


class ApiTimeoutError(ApiClientError):
    """Raised when backend request times out."""


class ApiValidationError(ApiClientError):
    """Raised when request payload fails schema validation (HTTP 422)."""


class ApiServiceUnavailableError(ApiClientError):
    """Raised when model or service is unavailable (HTTP 503)."""


class ApiServerError(ApiClientError):
    """Raised when backend experiences an internal error (HTTP 500)."""


class ApiResponseError(ApiClientError):
    """Raised when backend returns unexpected response structures or non-200 codes."""


def validate_feature_dict(features: dict[str, Any]) -> dict[str, float]:
    """Validate a feature dictionary against the authoritative 13-feature contract.

    Args:
        features: Dictionary of candidate features.

    Returns:
        dict[str, float]: Validated dictionary containing exactly the 13 required floats.

    Raises:
        ApiValidationError: If features are missing, extra, non-numeric, bool, NaN, or Inf.
    """
    if not isinstance(features, dict):
        raise ApiValidationError("Features payload must be a dictionary.")

    provided_keys = set(features.keys())
    expected_keys = set(WINE_FEATURE_NAMES)

    missing = expected_keys - provided_keys
    if missing:
        raise ApiValidationError(f"Missing required feature(s): {sorted(missing)}")

    extra = provided_keys - expected_keys
    if extra:
        raise ApiValidationError(f"Unexpected extra feature(s) forbidden: {sorted(extra)}")

    validated: dict[str, float] = {}
    for key in WINE_FEATURE_NAMES:
        val = features[key]
        if isinstance(val, bool):
            raise ApiValidationError(f"Feature '{key}' cannot be a boolean.")
        try:
            float_val = float(val)
        except (ValueError, TypeError):
            raise ApiValidationError(f"Feature '{key}' must be a numeric value, got '{val}'.") from None

        if math.isnan(float_val) or math.isinf(float_val):
            raise ApiValidationError(f"Feature '{key}' cannot be NaN or Infinite.")

        validated[key] = float_val

    return validated


class WineApiClient:
    """Reusable HTTP client interface for FastAPI Wine Classification backend."""

    def __init__(
        self,
        base_url: str | None = None,
        connect_timeout: float | None = None,
        request_timeout: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize API client.

        Args:
            base_url: Base URL of FastAPI service. Defaults to env var API_BASE_URL or http://127.0.0.1:8000.
            connect_timeout: Seconds to wait for socket connection. Defaults to API_CONNECT_TIMEOUT or 15.0s.
            request_timeout: Seconds to wait for response read. Defaults to API_REQUEST_TIMEOUT or 90.0s.
            client: Optional injected httpx.Client (useful for unit testing).
        """
        self.base_url = resolve_api_base_url(base_url)
        c_timeout = (
            connect_timeout
            if connect_timeout is not None
            else float(os.getenv("API_CONNECT_TIMEOUT", str(DEFAULT_CONNECT_TIMEOUT)))
        )
        r_timeout = (
            request_timeout
            if request_timeout is not None
            else float(os.getenv("API_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT)))
        )
        self.timeout = httpx.Timeout(timeout=r_timeout, connect=c_timeout)
        self._custom_client = client
        self._client: httpx.Client | None = client

    @property
    def client(self) -> httpx.Client:
        """Lazily initialize and return reusable httpx client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client is not None and not self._client.is_closed and self._custom_client is None:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def _handle_response_error(self, response: httpx.Response) -> None:
        """Parse HTTP error status codes into structured exceptions."""
        status_code = response.status_code
        detail: Any = None
        error_msg = f"HTTP {status_code} Error"

        try:
            body = response.json()
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("error")
                if isinstance(detail, list):
                    # Formatted Pydantic validation error list
                    formatted_errors = []
                    for item in detail:
                        if isinstance(item, dict):
                            loc = item.get("location") or item.get("field") or "field"
                            msg = item.get("message") or item.get("msg") or "Invalid value"
                            formatted_errors.append(f"{loc}: {msg}")
                        else:
                            formatted_errors.append(str(item))
                    error_msg = "; ".join(formatted_errors)
                elif detail:
                    error_msg = str(detail)
        except (ValueError, TypeError, json.JSONDecodeError):
            error_msg = response.text or f"Server returned status {status_code}"

        if status_code == 422:
            raise ApiValidationError(error_msg, status_code=status_code, detail=detail)
        if status_code == 503:
            raise ApiServiceUnavailableError(error_msg, status_code=status_code, detail=detail)
        if status_code >= 500:
            raise ApiServerError(
                "Internal server error occurred in backend service.",
                status_code=status_code,
                detail=detail,
            )
        raise ApiResponseError(error_msg, status_code=status_code, detail=detail)

    def health(self) -> dict[str, Any]:
        """Check service health and model readiness status.

        Returns:
            dict[str, Any]: Health status payload containing model_loaded and model_ready.

        Raises:
            ApiConnectionError: If server cannot be reached.
            ApiTimeoutError: If request times out.
            ApiClientError: If response status is non-200 or payload malformed.
        """
        try:
            response = self.client.get("/health")
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ApiConnectionError(
                f"Cannot connect to Wine Classification API at {self.base_url}. Please ensure the FastAPI server is running."
            ) from exc
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException) as exc:
            raise ApiTimeoutError(f"Health check request to {self.base_url} timed out.") from exc

        if response.status_code != 200:
            self._handle_response_error(response)

        try:
            data = response.json()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ApiResponseError("Backend returned invalid JSON for health check.") from exc

        if not isinstance(data, dict) or "status" not in data:
            raise ApiResponseError("Unexpected response structure from /health.")

        return data

    def model_info(self) -> dict[str, Any]:
        """Retrieve model metadata, architecture details, and holdout evaluation metrics.

        Returns:
            dict[str, Any]: Model information dictionary.
        """
        try:
            response = self.client.get("/model-info")
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ApiConnectionError(
                f"Cannot connect to Wine Classification API at {self.base_url}."
            ) from exc
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException) as exc:
            raise ApiTimeoutError("Model info request timed out.") from exc

        if response.status_code != 200:
            self._handle_response_error(response)

        try:
            data = response.json()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ApiResponseError("Backend returned invalid JSON for model-info.") from exc

        if not isinstance(data, dict) or "model_name" not in data:
            raise ApiResponseError("Unexpected response structure from /model-info.")

        return data

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Submit a single wine record for classification.

        Args:
            features: Dictionary containing all 13 required constituents.

        Returns:
            dict[str, Any]: PredictionResponse dictionary.
        """
        validated_payload = validate_feature_dict(features)

        try:
            response = self.client.post("/predict", json=validated_payload)
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ApiConnectionError(
                f"Cannot connect to Wine Classification API at {self.base_url}."
            ) from exc
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException) as exc:
            raise ApiTimeoutError("Prediction request timed out.") from exc

        if response.status_code != 200:
            self._handle_response_error(response)

        try:
            data = response.json()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ApiResponseError("Backend returned invalid JSON for prediction.") from exc

        if not isinstance(data, dict) or "predicted_class" not in data:
            raise ApiResponseError("Unexpected response structure from /predict.")

        return data

    def predict_batch(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit a batch of up to 100 wine samples for bulk inference.

        Args:
            samples: List of dictionaries, each containing the 13 required features.

        Returns:
            dict[str, Any]: BatchPredictionResponse dictionary.
        """
        if not isinstance(samples, list):
            raise ApiValidationError("Batch payload must be a list of sample dictionaries.")

        if len(samples) == 0:
            raise ApiValidationError("Batch request cannot be empty (minimum 1 sample required).")

        if len(samples) > 100:
            raise ApiValidationError(
                f"Batch request exceeds maximum limit of 100 samples (received {len(samples)})."
            )

        validated_samples = [validate_feature_dict(s) for s in samples]

        try:
            response = self.client.post("/predict/batch", json={"samples": validated_samples})
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ApiConnectionError(
                f"Cannot connect to Wine Classification API at {self.base_url}."
            ) from exc
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException) as exc:
            raise ApiTimeoutError("Batch prediction request timed out.") from exc

        if response.status_code != 200:
            self._handle_response_error(response)

        try:
            data = response.json()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ApiResponseError("Backend returned invalid JSON for batch prediction.") from exc

        if not isinstance(data, dict) or "predictions" not in data:
            raise ApiResponseError("Unexpected response structure from /predict/batch.")

        return data
