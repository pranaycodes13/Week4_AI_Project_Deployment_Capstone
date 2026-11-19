"""Unit and integration tests for holdout test evaluation and metrics calculation."""

import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSIFICATION_REPORT_JSON,
    CONFUSION_MATRIX_CSV,
    CONFUSION_MATRIX_JSON,
    EVALUATION_SUMMARY_MD,
    FINAL_METRICS_JSON,
    TEST_PREDICTIONS_CSV,
    WINE_CLASS_COUNT,
    WINE_FEATURE_COUNT,
    WINE_TARGET_NAMES,
)
from src.evaluation.evaluate import (
    build_and_fit_champion_pipeline,
    evaluate_holdout_set,
    get_champion_hyperparameters,
    load_evaluation_data,
)


def test_evaluation_data_dimensions() -> None:
    """Verify train and test partitions match expected sample and feature dimensions."""
    X_train, y_train, X_test, y_test, dataset_meta = load_evaluation_data()

    assert len(X_train) == 142
    assert len(y_train) == 142
    assert len(X_test) == 36
    assert len(y_test) == 36

    assert X_train.shape[1] == WINE_FEATURE_COUNT
    assert X_test.shape[1] == WINE_FEATURE_COUNT
    assert dataset_meta["sample_count"] == 178


def test_feature_names_and_ordering() -> None:
    """Verify column names and ordering in train and test sets strictly match metadata."""
    X_train, _, X_test, _, dataset_meta = load_evaluation_data()
    expected_names = dataset_meta["feature_names"]

    assert list(X_train.columns) == expected_names
    assert list(X_test.columns) == expected_names


def test_champion_hyperparameters() -> None:
    """Verify champion hyperparameters match Phase 3 selections."""
    hp = get_champion_hyperparameters()
    assert hp["C"] == 10.0
    assert hp["kernel"] == "rbf"
    assert hp["gamma"] == "scale"
    assert hp["probability"] is True
    assert hp["random_state"] == 42


def test_pipeline_fitting_training_data_only() -> None:
    """Verify pipeline fits strictly on training partition and scaler statistics match train."""
    X_train, y_train, _, _, _ = load_evaluation_data()
    pipeline = build_and_fit_champion_pipeline(X_train, y_train)

    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline.named_steps["scaler"], "mean_")
    assert hasattr(pipeline.named_steps["classifier"], "support_")

    train_mean = X_train.values.mean(axis=0)
    scaler_mean = pipeline.named_steps["scaler"].mean_
    np.testing.assert_allclose(scaler_mean, train_mean, rtol=1e-5)


def test_holdout_metric_bounds() -> None:
    """Verify all computed classification metrics lie strictly within [0.0, 1.0]."""
    X_train, y_train, X_test, y_test, _ = load_evaluation_data()
    pipeline = build_and_fit_champion_pipeline(X_train, y_train)
    eval_results = evaluate_holdout_set(pipeline, X_test, y_test)

    metric_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]
    for key in metric_keys:
        val = eval_results[key]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0, f"Metric {key}={val} outside [0, 1]"


def test_confusion_matrix_properties() -> None:
    """Verify confusion matrix shape (3x3), class row sums, and total count."""
    X_train, y_train, X_test, y_test, _ = load_evaluation_data()
    pipeline = build_and_fit_champion_pipeline(X_train, y_train)
    eval_results = evaluate_holdout_set(pipeline, X_test, y_test)

    cm = eval_results["confusion_matrix"]
    assert len(cm) == WINE_CLASS_COUNT
    assert all(len(row) == WINE_CLASS_COUNT for row in cm)

    total_samples = sum(sum(row) for row in cm)
    assert total_samples == len(X_test) == 36

    # Verify per-class support row sums (12, 14, 10)
    class_counts = y_test.value_counts().sort_index().tolist()
    for idx, expected_count in enumerate(class_counts):
        assert sum(cm[idx]) == expected_count


def test_classification_report_structure() -> None:
    """Verify classification report contains all target class labels and summary keys."""
    X_train, y_train, X_test, y_test, _ = load_evaluation_data()
    pipeline = build_and_fit_champion_pipeline(X_train, y_train)
    eval_results = evaluate_holdout_set(pipeline, X_test, y_test)

    report_dict = eval_results["classification_report_dict"]
    for class_name in WINE_TARGET_NAMES:
        assert class_name in report_dict
        for stat in ("precision", "recall", "f1-score", "support"):
            assert stat in report_dict[class_name]

    assert "macro avg" in report_dict
    assert "weighted avg" in report_dict


def test_roc_auc_calculation() -> None:
    """Verify multi-class One-vs-Rest ROC-AUC is valid and error-free."""
    X_train, y_train, X_test, y_test, _ = load_evaluation_data()
    pipeline = build_and_fit_champion_pipeline(X_train, y_train)
    eval_results = evaluate_holdout_set(pipeline, X_test, y_test)

    assert eval_results["roc_auc_calculation_error"] is None
    assert eval_results["roc_auc_ovr_macro"] is not None
    assert 0.5 <= eval_results["roc_auc_ovr_macro"] <= 1.0


def test_evaluation_artifacts_exist() -> None:
    """Verify all required evaluation result artifacts are generated and valid."""
    required_files = [
        FINAL_METRICS_JSON,
        CLASSIFICATION_REPORT_JSON,
        CONFUSION_MATRIX_CSV,
        CONFUSION_MATRIX_JSON,
        TEST_PREDICTIONS_CSV,
        EVALUATION_SUMMARY_MD,
    ]
    for file_path in required_files:
        assert file_path.exists(), f"Missing evaluation artifact: {file_path}"
        assert file_path.stat().st_size > 0, f"Artifact {file_path} is empty"

    with open(FINAL_METRICS_JSON, "r", encoding="utf-8") as f:
        final_meta = json.load(f)
    assert final_meta["model_name"] == "Support Vector Classifier (SVC)"
    assert "metrics" in final_meta

    preds_df = pd.read_csv(TEST_PREDICTIONS_CSV)
    assert len(preds_df) == 36
    assert list(preds_df.columns) == [
        "true_label",
        "predicted_label",
        "prob_class_0",
        "prob_class_1",
        "prob_class_2",
    ]

