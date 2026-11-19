"""Comprehensive integration and unit test suite for FastAPI Wine Classification API."""

import copy
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.service import ModelIncompatibleError, WineModelService
from src.config import WINE_FEATURE_NAMES, WINE_TARGET_NAMES
from src.schemas.wine_schema import WineFeatures

# Canonical valid wine sample matching the training dataset feature order
VALID_WINE_SAMPLE: dict[str, Any] = {
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

# Known Class 0 sample (high proline, high flavanoids)
CLASS_0_SAMPLE: dict[str, Any] = {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127.0,
    "total_phenols": 2.80,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280/od315_of_diluted_wines": 3.92,
    "proline": 1065.0,
}

# Known Class 1 sample (lower alcohol, lower proline, good OD)
CLASS_1_SAMPLE: dict[str, Any] = {
    "alcohol": 12.37,
    "malic_acid": 0.94,
    "ash": 1.36,
    "alcalinity_of_ash": 10.6,
    "magnesium": 88.0,
    "total_phenols": 1.98,
    "flavanoids": 0.57,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 0.42,
    "color_intensity": 1.95,
    "hue": 1.05,
    "od280/od315_of_diluted_wines": 1.82,
    "proline": 520.0,
}

# Known Class 2 sample (high malic, high color, low OD280/OD315, low flavanoids)
CLASS_2_SAMPLE: dict[str, Any] = {
    "alcohol": 13.11,
    "malic_acid": 1.90,
    "ash": 2.17,
    "alcalinity_of_ash": 21.0,
    "magnesium": 88.0,
    "total_phenols": 2.30,
    "flavanoids": 0.92,
    "nonflavanoid_phenols": 0.32,
    "proanthocyanins": 1.01,
    "color_intensity": 7.10,
    "hue": 0.72,
    "od280/od315_of_diluted_wines": 1.58,
    "proline": 625.0,
}


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a TestClient with lifespan context managing model lifecycle."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def unloaded_client() -> TestClient:
    """Fixture providing a TestClient where the model is not loaded."""
    mock_service = WineModelService()
    # Explicitly uninitialized service (is_ready == False)
    mock_service._is_loaded = False
    mock_service._pipeline = None
    app = create_app(service_instance=mock_service)
    # Bypass lifespan loading by using TestClient directly without entering lifespan context
    return TestClient(app, raise_server_exceptions=False)


# ==============================================================================
# Health Check Endpoint Tests (/health)
# ==============================================================================


def test_health_check_healthy(client: TestClient) -> None:
    """Test /health returns 200 and healthy status when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_ready"] is True
    assert "Support Vector Classifier" in data["model_name"]
    assert "version" in data


def test_health_check_unhealthy(unloaded_client: TestClient) -> None:
    """Test /health returns 503 when model is not loaded."""
    response = unloaded_client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False
    assert data["model_ready"] is False


# ==============================================================================
# Model Info Endpoint Tests (/model-info)
# ==============================================================================


def test_model_info_success(client: TestClient) -> None:
    """Test /model-info returns 200 with complete architectural metadata and metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()

    assert "Support Vector Classifier" in data["model_name"]
    assert "SVC" in data["model_type"]
    assert data["pipeline_steps"] == ["StandardScaler", "SVC"]
    assert data["feature_count"] == 13
    assert data["feature_names"] == WINE_FEATURE_NAMES
    assert data["target_class_labels"] == [0, 1, 2]
    assert data["target_class_names"] == WINE_TARGET_NAMES

    # Hyperparameters
    params = data["hyperparameters"]
    assert params["C"] == 10.0
    assert params["kernel"] == "rbf"
    assert params["gamma"] == "scale"
    assert params["probability"] is True

    # Metrics
    cv_metrics = data["cross_validation_metrics"]
    assert cv_metrics["mean_cv_macro_f1"] == pytest.approx(0.9927, rel=1e-3)
    assert cv_metrics["std_cv_macro_f1"] == pytest.approx(0.0147, rel=1e-2)

    holdout_metrics = data["holdout_test_metrics"]
    assert holdout_metrics["accuracy"] == pytest.approx(0.9444, rel=1e-3)
    assert holdout_metrics["macro_f1"] == pytest.approx(0.9407, rel=1e-3)
    assert holdout_metrics["roc_auc_ovr_macro"] == pytest.approx(1.0, rel=1e-3)

    # Scikit-learn and Joblib versions
    assert "scikit_learn_version" in data
    assert "joblib_version" in data


def test_model_info_no_path_leakage(client: TestClient) -> None:
    """Ensure /model-info does not leak absolute host filesystem paths."""
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.text
    assert "C:\\" not in body
    assert "pranayyy" not in body
    assert "Desktop" not in body


def test_model_info_unloaded_model(unloaded_client: TestClient) -> None:
    """Test /model-info returns 503 when model is not loaded."""
    response = unloaded_client.get("/model-info")
    assert response.status_code == 503
    assert response.json()["error"] == "Service Unavailable"


# ==============================================================================
# Single Prediction Endpoint Tests (/predict)
# ==============================================================================


def test_predict_single_success(client: TestClient) -> None:
    """Test /predict returns 200 and valid schema for valid input."""
    response = client.post("/predict", json=VALID_WINE_SAMPLE)
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in [0, 1, 2]
    assert data["predicted_class_name"] in WINE_TARGET_NAMES
    assert data["predicted_class_name"] == f"class_{data['predicted_class']}"
    assert "Support Vector Classifier" in data["model_name"]

    # Probabilities validation
    probs = data["probabilities"]
    assert set(probs.keys()) == set(WINE_TARGET_NAMES)
    for prob in probs.values():
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    # Probabilities must sum to ~1.0
    prob_sum = sum(probs.values())
    assert prob_sum == pytest.approx(1.0, abs=1e-4)


def test_predict_classes_consistency(client: TestClient) -> None:
    """Test predictions on distinctive wine profiles match expected classes."""
    resp_0 = client.post("/predict", json=CLASS_0_SAMPLE)
    assert resp_0.status_code == 200
    assert resp_0.json()["predicted_class"] == 0

    resp_1 = client.post("/predict", json=CLASS_1_SAMPLE)
    assert resp_1.status_code == 200
    assert resp_1.json()["predicted_class"] == 1

    resp_2 = client.post("/predict", json=CLASS_2_SAMPLE)
    assert resp_2.status_code == 200
    assert resp_2.json()["predicted_class"] == 2


# ==============================================================================
# Input Validation & Bad Request Tests (/predict)
# ==============================================================================


def test_predict_missing_feature(client: TestClient) -> None:
    """Test /predict rejects request with missing required feature (422)."""
    invalid_sample = copy.deepcopy(VALID_WINE_SAMPLE)
    del invalid_sample["alcohol"]
    response = client.post("/predict", json=invalid_sample)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Input Validation Error"


def test_predict_extra_feature_forbidden(client: TestClient) -> None:
    """Test /predict rejects request with unallowed extra features (422)."""
    invalid_sample = copy.deepcopy(VALID_WINE_SAMPLE)
    invalid_sample["unauthorized_additive"] = 4.2
    response = client.post("/predict", json=invalid_sample)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Input Validation Error"


def test_predict_string_as_numeric(client: TestClient) -> None:
    """Test /predict rejects non-numeric string values (422)."""
    invalid_sample = copy.deepcopy(VALID_WINE_SAMPLE)
    invalid_sample["alcohol"] = "high"
    response = client.post("/predict", json=invalid_sample)
    assert response.status_code == 422


def test_predict_rejects_boolean(client: TestClient) -> None:
    """Test /predict rejects boolean values passed as float (422)."""
    invalid_sample = copy.deepcopy(VALID_WINE_SAMPLE)
    invalid_sample["alcohol"] = True
    response = client.post("/predict", json=invalid_sample)
    assert response.status_code == 422


def test_predict_rejects_none_null(client: TestClient) -> None:
    """Test /predict rejects null values (422)."""
    invalid_sample = copy.deepcopy(VALID_WINE_SAMPLE)
    invalid_sample["proline"] = None
    response = client.post("/predict", json=invalid_sample)
    assert response.status_code == 422


def test_predict_rejects_nan_and_infinity(client: TestClient) -> None:
    """Test /predict rejects NaN and Infinity floating point values (422)."""
    # 1. Validate directly on Pydantic schema
    sample_kwargs = copy.deepcopy(VALID_WINE_SAMPLE)
    sample_kwargs["alcohol"] = float("nan")
    with pytest.raises(Exception, match="NaN and Infinity values are not permitted"):
        WineFeatures.model_validate(sample_kwargs)

    # 2. Validate via HTTP endpoint
    raw_payload = json.dumps(VALID_WINE_SAMPLE).replace("13.2", "NaN")
    resp_nan = client.post(
        "/predict",
        content=raw_payload,
        headers={"Content-Type": "application/json"},
    )
    assert resp_nan.status_code == 422


def test_predict_batch_invalid_payload_type(client: TestClient) -> None:
    """Test /predict/batch rejects integer or string primitives (422)."""
    resp_int = client.post("/predict/batch", json=12345)
    assert resp_int.status_code == 422

    resp_str = client.post("/predict/batch", json="not a batch")
    assert resp_str.status_code == 422



def test_predict_empty_body(client: TestClient) -> None:
    """Test /predict rejects empty JSON body (422)."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_malformed_json(client: TestClient) -> None:
    """Test /predict rejects malformed non-JSON payload (422)."""
    response = client.post(
        "/predict",
        content="invalid json raw string",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# ==============================================================================
# Batch Prediction Endpoint Tests (/predict/batch)
# ==============================================================================


def test_predict_batch_dict_format(client: TestClient) -> None:
    """Test /predict/batch with object format {'samples': [...]} returns 200."""
    payload = {"samples": [CLASS_0_SAMPLE, CLASS_1_SAMPLE, CLASS_2_SAMPLE]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 3
    assert data["successful_predictions"] == 3
    assert "Support Vector Classifier" in data["model_name"]

    predictions = data["predictions"]
    assert len(predictions) == 3
    assert predictions[0]["sample_index"] == 0
    assert predictions[0]["predicted_class"] == 0
    assert predictions[1]["sample_index"] == 1
    assert predictions[1]["predicted_class"] == 1
    assert predictions[2]["sample_index"] == 2
    assert predictions[2]["predicted_class"] == 2


def test_predict_batch_raw_list_format(client: TestClient) -> None:
    """Test /predict/batch with raw list format [...] returns 200."""
    payload = [CLASS_0_SAMPLE, CLASS_1_SAMPLE]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2


def test_predict_batch_empty_list_rejected(client: TestClient) -> None:
    """Test /predict/batch rejects empty batch with 422."""
    response = client.post("/predict/batch", json={"samples": []})
    assert response.status_code == 422

    response_list = client.post("/predict/batch", json=[])
    assert response_list.status_code == 422


def test_predict_batch_exceeding_max_limit(client: TestClient) -> None:
    """Test /predict/batch rejects batch > 100 samples with 422."""
    oversized = [VALID_WINE_SAMPLE for _ in range(101)]
    response = client.post("/predict/batch", json={"samples": oversized})
    assert response.status_code == 422

    response_list = client.post("/predict/batch", json=oversized)
    assert response_list.status_code == 422


def test_predict_batch_partial_invalid_fails_atomically(client: TestClient) -> None:
    """Test that a batch with one corrupted sample fails entirely (422)."""
    corrupted = copy.deepcopy(VALID_WINE_SAMPLE)
    del corrupted["alcohol"]
    payload = [CLASS_0_SAMPLE, corrupted, CLASS_2_SAMPLE]

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


# ==============================================================================
# HTTP Protocol & Routing Tests
# ==============================================================================


def test_not_found_route(client: TestClient) -> None:
    """Test unknown route returns 404."""
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404


def test_method_not_allowed(client: TestClient) -> None:
    """Test GET method on /predict returns 405 Method Not Allowed."""
    response = client.get("/predict")
    assert response.status_code == 405


# ==============================================================================
# Error Handling & Resilience Tests
# ==============================================================================


def test_predict_when_model_unloaded(unloaded_client: TestClient) -> None:
    """Test /predict returns 503 when model is unloaded."""
    response = unloaded_client.post("/predict", json=VALID_WINE_SAMPLE)
    assert response.status_code == 503
    data = response.json()
    assert data["error"] == "Service Unavailable"


def test_predict_when_model_corrupted() -> None:
    """Test 503 error handling when model raises ModelIncompatibleError."""
    mock_service = WineModelService()
    mock_service.predict_single = MagicMock(side_effect=ModelIncompatibleError("Corrupted"))
    app = create_app(service_instance=mock_service)
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.post("/predict", json=VALID_WINE_SAMPLE)
        assert response.status_code == 503
        assert response.json()["error"] == "Service Unavailable"


def test_generic_internal_error_sanitization() -> None:
    """Test that unexpected internal exceptions return 500 without leaking stack trace."""
    mock_service = WineModelService()
    mock_service.predict_single = MagicMock(side_effect=RuntimeError("Internal database explosion at C:\\secret\\file.py"))
    app = create_app(service_instance=mock_service)
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.post("/predict", json=VALID_WINE_SAMPLE)
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal Server Error"
        assert "explosion" not in data["detail"]
        assert "secret" not in data["detail"]
