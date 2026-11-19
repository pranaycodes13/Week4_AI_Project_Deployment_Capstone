"""Unit and integration tests for model training, cross-validation, and selection."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BEST_MODEL_METADATA_JSON,
    CV_COMPARISON_CSV,
    CV_COMPARISON_JSON,
    CV_FOLDS,
    HP_SEARCH_RESULTS_CSV,
    RANDOM_STATE,
    TRAINING_RESULTS_DIR,
    WINE_FEATURE_COUNT,
    X_TEST_FILE,
    Y_TEST_FILE,
)
from src.training.train import (
    evaluate_candidate_models,
    get_candidate_models,
    load_training_data,
    select_best_model,
)


def test_candidate_models_registered() -> None:
    """Requirement 1: Verify all three candidate classifiers are registered."""
    candidates = get_candidate_models()
    expected_names = {"logistic_regression", "random_forest", "svc"}
    assert set(candidates.keys()) == expected_names


def test_every_candidate_is_pipeline() -> None:
    """Requirement 2: Every classifier candidate must be an sklearn Pipeline instance."""
    candidates = get_candidate_models()
    for name, model in candidates.items():
        assert isinstance(model, Pipeline), f"Candidate '{name}' must be an sklearn Pipeline"


def test_preprocessing_inside_pipeline() -> None:
    """Requirement 3: Verify StandardScaler is encapsulated inside every candidate Pipeline."""
    candidates = get_candidate_models()
    for name, pipeline in candidates.items():
        assert "scaler" in pipeline.named_steps, f"'{name}' pipeline missing 'scaler' step"
        assert isinstance(
            pipeline.named_steps["scaler"], StandardScaler
        ), f"'{name}' scaler must be StandardScaler"
        assert "classifier" in pipeline.named_steps, f"'{name}' pipeline missing 'classifier' step"


def test_cross_validation_configuration() -> None:
    """Requirement 4: Verify cross-validation uses exactly 5 stratified folds."""
    assert CV_FOLDS == 5
    assert RANDOM_STATE == 42


def test_training_uses_only_train_partition() -> None:
    """Requirement 5: Verify training data loader pulls strictly 142 samples of training data."""
    X_train, y_train = load_training_data()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert len(X_train) == 142
    assert len(y_train) == 142
    assert X_train.shape[1] == WINE_FEATURE_COUNT
    assert set(y_train.unique()) == {0, 1, 2}


def test_holdout_test_not_accessed_during_training() -> None:
    """Requirement 6: Verify training pipeline does not read holdout test files."""
    real_read_csv = pd.read_csv

    accessed_paths: list[str] = []

    def mock_read_csv(filepath_or_buffer, *args, **kwargs):
        path_str = str(filepath_or_buffer)
        accessed_paths.append(path_str)
        if str(X_TEST_FILE) in path_str or str(Y_TEST_FILE) in path_str:
            raise PermissionError(
                f"Data leakage detected! Training accessed holdout test file: {path_str}"
            )
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    with patch("pandas.read_csv", side_effect=mock_read_csv):
        X_train, y_train = load_training_data()
        assert len(X_train) == 142
        assert len(y_train) == 142

    for accessed in accessed_paths:
        assert str(X_TEST_FILE) not in accessed
        assert str(Y_TEST_FILE) not in accessed


def test_cv_results_contain_expected_metrics() -> None:
    """Requirement 7: Verify baseline CV output contains all required primary and secondary metrics."""
    X_train, y_train = load_training_data()
    candidates = get_candidate_models()
    # Test on a single fast candidate to keep test execution snappy
    single_candidate = {"logistic_regression": candidates["logistic_regression"]}

    results = evaluate_candidate_models(X_train, y_train, single_candidate, cv_folds=3)
    lr_results = results["logistic_regression"]

    required_keys = [
        "fold_f1_macro",
        "mean_f1_macro",
        "std_f1_macro",
        "fold_accuracy",
        "mean_accuracy",
        "std_accuracy",
        "fold_precision_macro",
        "mean_precision_macro",
        "std_precision_macro",
        "fold_recall_macro",
        "mean_recall_macro",
        "std_recall_macro",
    ]
    for key in required_keys:
        assert key in lr_results, f"Missing metric key '{key}' in CV results"


def test_fold_scores_and_statistics_valid() -> None:
    """Requirement 8: Verify fold scores are valid probabilities and mean/std mathematically match."""
    X_train, y_train = load_training_data()
    candidates = {"svc": get_candidate_models()["svc"]}
    results = evaluate_candidate_models(X_train, y_train, candidates, cv_folds=3)
    svc_res = results["svc"]

    assert len(svc_res["fold_f1_macro"]) == 3
    for score in svc_res["fold_f1_macro"]:
        assert 0.0 <= score <= 1.0

    calc_mean = float(np.mean(svc_res["fold_f1_macro"]))
    calc_std = float(np.std(svc_res["fold_f1_macro"]))
    np.testing.assert_allclose(svc_res["mean_f1_macro"], calc_mean, rtol=1e-5)
    np.testing.assert_allclose(svc_res["std_f1_macro"], calc_std, rtol=1e-5)


def test_model_selection_uses_mean_cv_macro_f1() -> None:
    """Requirement 9: Model selection strictly picks the highest mean CV Macro F1 candidate."""
    mock_tuned = {
        "model_a": {
            "best_f1_macro_mean": 0.85,
            "best_f1_macro_std": 0.02,
            "best_params": {"param": 1},
        },
        "model_b": {
            "best_f1_macro_mean": 0.95,
            "best_f1_macro_std": 0.01,
            "best_params": {"param": 2},
        },
        "model_c": {
            "best_f1_macro_mean": 0.90,
            "best_f1_macro_std": 0.03,
            "best_params": {"param": 3},
        },
    }
    decision = select_best_model(mock_tuned)
    assert decision["selected_model"] == "model_b"
    assert decision["mean_cv_macro_f1"] == 0.95
    assert decision["ranking"][0]["model_name"] == "model_b"
    assert decision["ranking"][1]["model_name"] == "model_c"
    assert decision["ranking"][2]["model_name"] == "model_a"


def test_selected_model_metadata_valid() -> None:
    """Requirement 10: Selected model metadata contains full provenance attributes."""
    mock_tuned = {
        "svc": {
            "best_f1_macro_mean": 0.99,
            "best_f1_macro_std": 0.01,
            "best_params": {"classifier__C": 10.0},
        }
    }
    decision = select_best_model(mock_tuned)
    assert "selected_model" in decision
    assert "best_params" in decision
    assert "mean_cv_macro_f1" in decision
    assert "std_cv_macro_f1" in decision
    assert "model_selection_rationale" in decision
    assert len(decision["ranking"]) > 0


def test_training_reproducibility() -> None:
    """Requirement 11: Repeated baseline CV with same seed produces deterministic results."""
    X_train, y_train = load_training_data()
    candidates1 = {"logistic_regression": get_candidate_models(random_state=42)["logistic_regression"]}
    candidates2 = {"logistic_regression": get_candidate_models(random_state=42)["logistic_regression"]}

    res1 = evaluate_candidate_models(X_train, y_train, candidates1, cv_folds=3, random_state=42)
    res2 = evaluate_candidate_models(X_train, y_train, candidates2, cv_folds=3, random_state=42)

    np.testing.assert_allclose(
        res1["logistic_regression"]["fold_f1_macro"],
        res2["logistic_regression"]["fold_f1_macro"],
        rtol=1e-5,
    )


def test_no_trained_model_serialized_in_phase_3() -> None:
    """Requirement 12: Zero trained model binaries (*.joblib, *.pkl) produced by Phase 3 training module."""
    # Training results directory must not contain binary model artifacts
    training_joblib = list(TRAINING_RESULTS_DIR.glob("*.joblib"))
    training_pkl = list(TRAINING_RESULTS_DIR.glob("*.pkl"))
    assert len(training_joblib) == 0, f"Found unexpected serialized models in training results: {training_joblib}"
    assert len(training_pkl) == 0, f"Found unexpected serialized models in training results: {training_pkl}"

    # Verify that none of the Phase 3 training artifact files are model binaries
    for file_path in TRAINING_RESULTS_DIR.iterdir():
        assert not file_path.name.endswith((".joblib", ".pkl")), (
            f"Training artifact {file_path.name} is a serialized model binary!"
        )


def test_saved_artifacts_validity() -> None:
    """Verify saved training artifacts exist on disk and parse properly."""
    assert CV_COMPARISON_JSON.exists(), "cv_comparison.json must exist"
    assert CV_COMPARISON_CSV.exists(), "cv_comparison.csv must exist"
    assert BEST_MODEL_METADATA_JSON.exists(), "best_model_metadata.json must exist"
    assert HP_SEARCH_RESULTS_CSV.exists(), "hyperparameter_search_results.csv must exist"

    with open(CV_COMPARISON_JSON, "r", encoding="utf-8") as f:
        comp_json = json.load(f)
    assert comp_json["dataset_name"] == "wine"
    assert "selection_decision" in comp_json

    comp_df = pd.read_csv(CV_COMPARISON_CSV)
    assert len(comp_df) == 3
    assert "model_name" in comp_df.columns
    assert "tuned_mean_f1_macro" in comp_df.columns

    with open(BEST_MODEL_METADATA_JSON, "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    assert meta_json["selected_model"] == "svc"
    assert meta_json["primary_metric"] == "f1_macro"
