"""Unit and integration tests for model artifact serialization, loading, and inference verification."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.api.service import (
    ModelIncompatibleError,
    ModelNotLoadedError,
    WineModelService,
)
from src.config import (
    METRICS_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    WINE_FEATURE_COUNT,
    WINE_TARGET_NAMES,
)
from src.evaluation.evaluate import (
    build_and_fit_champion_pipeline,
    load_evaluation_data,
    verify_reloaded_artifact,
)
from src.schemas.wine_schema import WineFeatures


def test_model_artifact_files_exist() -> None:
    """Verify that serialized model and metadata JSON files exist and are non-empty."""
    assert MODEL_ARTIFACT_PATH.exists(), f"Model artifact missing at {MODEL_ARTIFACT_PATH}"
    assert MODEL_ARTIFACT_PATH.stat().st_size > 0, "Model artifact file is empty"

    assert MODEL_METADATA_PATH.exists(), f"Metadata JSON missing at {MODEL_METADATA_PATH}"
    assert MODEL_METADATA_PATH.stat().st_size > 0, "Metadata JSON file is empty"

    assert METRICS_PATH.exists(), f"Metrics JSON missing at {METRICS_PATH}"
    assert METRICS_PATH.stat().st_size > 0, "Metrics JSON file is empty"


def test_reloaded_pipeline_structure() -> None:
    """Verify that reloaded object is an sklearn Pipeline with StandardScaler and SVC steps."""
    pipeline = joblib.load(MODEL_ARTIFACT_PATH)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert "scaler" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["classifier"], SVC)


def test_reloaded_inference_consistency() -> None:
    """Verify reloaded pipeline predictions and probabilities match in-memory model."""
    X_train, y_train, X_test, _, _ = load_evaluation_data()

    # In-memory pipeline
    fresh_pipeline = build_and_fit_champion_pipeline(X_train, y_train)
    in_mem_preds = fresh_pipeline.predict(X_test)
    in_mem_proba = fresh_pipeline.predict_proba(X_test)

    # Reloaded pipeline
    reloaded_pipeline = joblib.load(MODEL_ARTIFACT_PATH)
    reloaded_preds = reloaded_pipeline.predict(X_test)
    reloaded_proba = reloaded_pipeline.predict_proba(X_test)

    np.testing.assert_array_equal(
        reloaded_preds,
        in_mem_preds,
        err_msg="Reloaded predictions do not match fresh in-memory predictions!",
    )
    np.testing.assert_allclose(
        reloaded_proba,
        in_mem_proba,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Reloaded probabilities do not match fresh in-memory probabilities!",
    )


def test_verify_reloaded_artifact_helper() -> None:
    """Verify verify_reloaded_artifact function executes and passes all 8 checks."""
    _, _, X_test, _, _ = load_evaluation_data()
    pipeline = joblib.load(MODEL_ARTIFACT_PATH)
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)

    passed = verify_reloaded_artifact(MODEL_ARTIFACT_PATH, X_test, preds, proba)
    assert passed is True


def test_model_metadata_completeness() -> None:
    """Verify model_metadata.json contains all required provenance and environment fields."""
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = [
        "model_name",
        "model_type",
        "pipeline_steps",
        "hyperparameters",
        "training_sample_count",
        "holdout_test_sample_count",
        "feature_count",
        "feature_names",
        "target_class_labels",
        "target_class_names",
        "phase3_cv_macro_f1_mean",
        "phase3_cv_macro_f1_std",
        "holdout_accuracy",
        "holdout_macro_f1",
        "python_version",
        "scikit_learn_version",
        "joblib_version",
        "model_artifact_filename",
        "serialized_at_utc",
    ]
    for key in required_keys:
        assert key in meta, f"Missing required key '{key}' in model_metadata.json"

    assert meta["feature_count"] == WINE_FEATURE_COUNT
    assert len(meta["feature_names"]) == WINE_FEATURE_COUNT
    assert meta["target_class_labels"] == [0, 1, 2]
    assert meta["target_class_names"] == WINE_TARGET_NAMES
    assert meta["training_sample_count"] == 142
    assert meta["holdout_test_sample_count"] == 36


def test_probability_distribution_properties() -> None:
    """Verify probability outputs are valid distributions (non-negative and sum to 1.0)."""
    _, _, X_test, _, _ = load_evaluation_data()
    pipeline = joblib.load(MODEL_ARTIFACT_PATH)
    proba = pipeline.predict_proba(X_test)

    assert (proba >= 0.0).all(), "Negative probability detected"
    assert (proba <= 1.0).all(), "Probability > 1.0 detected"

    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(X_test)), atol=1e-5)


def test_service_load_missing_files_raises(tmp_path: Path) -> None:
    """Verify WineModelService raises ModelNotLoadedError when model files are missing."""
    missing_art = tmp_path / "nonexistent_model.joblib"
    missing_meta = tmp_path / "nonexistent_meta.json"
    service = WineModelService(artifact_path=missing_art, metadata_path=missing_meta)

    assert service.is_ready is False
    with pytest.raises(ModelNotLoadedError, match="Model artifact file not found"):
        service.load_model()
    assert service.is_ready is False


def test_service_load_missing_metadata_raises(tmp_path: Path) -> None:
    """Verify WineModelService raises ModelNotLoadedError when metadata file is missing."""
    valid_obj = Pipeline([("scaler", StandardScaler()), ("classifier", SVC())])
    art_path = tmp_path / "valid_model.joblib"
    joblib.dump(valid_obj, art_path)
    missing_meta = tmp_path / "missing_meta.json"

    service = WineModelService(artifact_path=art_path, metadata_path=missing_meta)
    with pytest.raises(ModelNotLoadedError, match="Model metadata file not found"):
        service.load_model()
    assert service.is_ready is False


def test_service_load_corrupted_pipeline_raises(tmp_path: Path) -> None:
    """Verify WineModelService raises ModelIncompatibleError when artifact is not a Pipeline."""
    art_path = tmp_path / "corrupt.joblib"
    meta_path = tmp_path / "meta.json"
    joblib.dump({"not": "a pipeline"}, art_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"model_name": "fake", "feature_names": ["a"] * 13}, f)

    service = WineModelService(artifact_path=art_path, metadata_path=meta_path)
    with pytest.raises(ModelIncompatibleError, match="not a valid scikit-learn Pipeline"):
        service.load_model()
    assert service.is_ready is False


def test_service_load_pipeline_wrong_steps_raises(tmp_path: Path) -> None:
    """Verify WineModelService raises ModelIncompatibleError when required steps are missing."""
    incomplete_pipe = Pipeline([("scaler", StandardScaler())])
    art_path = tmp_path / "incomplete.joblib"
    meta_path = tmp_path / "meta.json"
    joblib.dump(incomplete_pipe, art_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"model_name": "fake", "feature_names": ["a"] * 13}, f)

    service = WineModelService(artifact_path=art_path, metadata_path=meta_path)
    with pytest.raises(ModelIncompatibleError, match="missing required steps"):
        service.load_model()
    assert service.is_ready is False


def test_service_unloaded_inference_and_empty_batch_raises() -> None:
    """Verify inference calls on uninitialized service raise appropriate errors."""
    service = WineModelService()
    assert service.is_ready is False

    valid_features = WineFeatures(
        alcohol=13.0,
        malic_acid=2.0,
        ash=2.0,
        alcalinity_of_ash=15.0,
        magnesium=100.0,
        total_phenols=2.5,
        flavanoids=2.5,
        nonflavanoid_phenols=0.3,
        proanthocyanins=1.5,
        color_intensity=5.0,
        hue=1.0,
        od280_od315_of_diluted_wines=3.0,
        proline=1000.0,
    )

    with pytest.raises(ModelNotLoadedError):
        service.predict_single(valid_features)

    with pytest.raises(ModelNotLoadedError):
        service.predict_batch([valid_features])

    # Test empty batch validation when pipeline is set
    service._is_loaded = True
    service._pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", SVC())])
    with pytest.raises(ValueError, match="Batch request cannot be empty"):
        service.predict_batch([])

