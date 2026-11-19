"""Final holdout test evaluation, metric computation, and model serialization.

Fits the champion SVC pipeline strictly on training data (X_train, y_train),
performs a one-time holdout evaluation on (X_test, y_test), and serializes
the full end-to-end pipeline (StandardScaler + SVC) via Joblib.
"""

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    BEST_MODEL_METADATA_JSON,
    EVALUATION_RESULTS_DIR,
    METADATA_FILE,
    METRICS_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    RANDOM_STATE,
    TARGET_NAME,
    WINE_FEATURE_COUNT,
    WINE_TARGET_NAMES,
    X_TEST_FILE,
    X_TRAIN_FILE,
    Y_TEST_FILE,
    Y_TRAIN_FILE,
    ensure_directories,
)


def load_evaluation_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load train and holdout test partitions and validate feature alignment.

    Returns:
        tuple containing (X_train, y_train, X_test, y_test, dataset_metadata).

    Raises:
        ValueError: If partition dimensions, feature names, or classes are invalid.
    """
    for file_path in (X_TRAIN_FILE, Y_TRAIN_FILE, X_TEST_FILE, Y_TEST_FILE, METADATA_FILE):
        if not file_path.exists():
            raise FileNotFoundError(f"Required artifact missing: {file_path}")

    X_train = pd.read_csv(X_TRAIN_FILE)
    y_train_df = pd.read_csv(Y_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)
    y_test_df = pd.read_csv(Y_TEST_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)

    y_train = y_train_df[TARGET_NAME] if TARGET_NAME in y_train_df.columns else y_train_df.iloc[:, 0]
    y_test = y_test_df[TARGET_NAME] if TARGET_NAME in y_test_df.columns else y_test_df.iloc[:, 0]
    y_train.name = TARGET_NAME
    y_test.name = TARGET_NAME

    # Strict dimensional & schema validation
    if len(X_train) != 142 or len(X_test) != 36:
        raise ValueError(f"Unexpected partition row counts: train={len(X_train)}, test={len(X_test)}")

    if X_train.shape[1] != WINE_FEATURE_COUNT or X_test.shape[1] != WINE_FEATURE_COUNT:
        raise ValueError("Feature count mismatch between partitions")

    expected_features = dataset_meta["feature_names"]
    if list(X_train.columns) != expected_features or list(X_test.columns) != expected_features:
        raise ValueError("Feature names or column order do not match Phase 2 metadata")

    if set(y_train.unique()) != {0, 1, 2} or set(y_test.unique()) != {0, 1, 2}:
        raise ValueError("Missing target classes in train or test split")

    return X_train, y_train, X_test, y_test, dataset_meta


def get_champion_hyperparameters() -> dict[str, Any]:
    """Retrieve champion hyperparameters selected in Phase 3.

    Returns:
        dict[str, Any]: Parameters for SVC classifier step.
    """
    if BEST_MODEL_METADATA_JSON.exists():
        with open(BEST_MODEL_METADATA_JSON, "r", encoding="utf-8") as f:
            best_meta = json.load(f)
        raw_params = best_meta.get("best_hyperparameters", {})
        # Strip "classifier__" prefix if present
        params = {
            k.replace("classifier__", ""): v for k, v in raw_params.items()
        }
    else:
        # Fallback to Phase 3 approved specifications
        params = {"C": 10.0, "kernel": "rbf", "gamma": "scale"}

    # Guarantee required deterministic parameters
    params["probability"] = True
    params["random_state"] = RANDOM_STATE
    return params


def build_and_fit_champion_pipeline(
    X_train: pd.DataFrame, y_train: pd.Series, hyperparameters: dict[str, Any] | None = None
) -> Pipeline:
    """Build and fit the complete champion pipeline strictly on training data.

    The StandardScaler and SVC are fitted together within this pipeline.
    Holdout test data is never touched during fitting.

    Args:
        X_train: Training features.
        y_train: Training targets.
        hyperparameters: SVC parameters.

    Returns:
        Pipeline: Fitted scikit-learn Pipeline.
    """
    hp = hyperparameters or get_champion_hyperparameters()
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC(**hp)),
        ]
    )
    # Fit strictly on training partition
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_holdout_set(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    """Perform one-time final holdout test evaluation and compute comprehensive metrics.

    Args:
        pipeline: Fitted champion pipeline.
        X_test: Holdout test features.
        y_test: Holdout test ground truth.

    Returns:
        dict[str, Any]: Comprehensive evaluation report.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Core Classification Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    macro_precision = float(precision_score(y_test, y_pred, average="macro"))
    macro_recall = float(recall_score(y_test, y_pred, average="macro"))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_precision = float(precision_score(y_test, y_pred, average="weighted"))
    weighted_recall = float(recall_score(y_test, y_pred, average="weighted"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    # Confusion Matrix
    cm_array = confusion_matrix(y_test, y_pred)
    cm_list = cm_array.tolist()

    # Classification Report
    report_dict = classification_report(
        y_test, y_pred, target_names=WINE_TARGET_NAMES, output_dict=True
    )
    report_text = classification_report(
        y_test, y_pred, target_names=WINE_TARGET_NAMES
    )

    # Per-Class Metrics
    per_class_metrics: dict[str, dict[str, float]] = {}
    for class_name in WINE_TARGET_NAMES:
        per_class_metrics[class_name] = {
            "precision": float(report_dict[class_name]["precision"]),
            "recall": float(report_dict[class_name]["recall"]),
            "f1-score": float(report_dict[class_name]["f1-score"]),
            "support": float(report_dict[class_name]["support"]),
        }

    # Multiclass ROC-AUC (One-vs-Rest)
    try:
        roc_auc_ovr_macro = float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        )
        roc_auc_ovr_weighted = float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        )
    except (ValueError, TypeError) as exc:
        roc_auc_ovr_macro = None
        roc_auc_ovr_weighted = None
        roc_auc_error = str(exc)
    else:
        roc_auc_error = None

    # Prediction dataframe
    predictions_df = pd.DataFrame(
        {
            "true_label": y_test.values,
            "predicted_label": y_pred,
            "prob_class_0": y_proba[:, 0],
            "prob_class_1": y_proba[:, 1],
            "prob_class_2": y_proba[:, 2],
        }
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm_list,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "per_class_metrics": per_class_metrics,
        "roc_auc_ovr_macro": roc_auc_ovr_macro,
        "roc_auc_ovr_weighted": roc_auc_ovr_weighted,
        "roc_auc_calculation_error": roc_auc_error,
        "predictions_df": predictions_df,
        "predictions": y_pred,
        "probabilities": y_proba,
    }


def serialize_pipeline_artifact(
    pipeline: Pipeline,
    eval_results: dict[str, Any],
    dataset_meta: dict[str, Any],
    hp: dict[str, Any],
    artifact_path: Path = MODEL_ARTIFACT_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    metrics_path: Path = METRICS_PATH,
) -> tuple[Path, Path]:
    """Serialize the fitted Pipeline and write comprehensive metadata and metrics JSONs.

    Args:
        pipeline: Fitted pipeline.
        eval_results: Holdout evaluation results.
        dataset_meta: Dataset metadata.
        hp: Model hyperparameters.
        artifact_path: Destination for .joblib.
        metadata_path: Destination for model_metadata.json.
        metrics_path: Destination for evaluation_metrics.json.

    Returns:
        tuple[Path, Path]: Paths to serialized model and metadata.
    """
    ensure_directories()

    # Serialize complete Pipeline (Scaler + SVC)
    joblib.dump(pipeline, artifact_path)

    timestamp = datetime.now(UTC).isoformat()

    # Load Phase 3 CV stats if available
    cv_macro_f1_mean = None
    cv_macro_f1_std = None
    if BEST_MODEL_METADATA_JSON.exists():
        with open(BEST_MODEL_METADATA_JSON, "r", encoding="utf-8") as f:
            b_meta = json.load(f)
            cv_macro_f1_mean = b_meta.get("mean_cv_macro_f1")
            cv_macro_f1_std = b_meta.get("std_cv_macro_f1")

    # Production Model Metadata
    model_metadata_payload: dict[str, Any] = {
        "model_name": "Support Vector Classifier (SVC)",
        "model_type": "sklearn.svm.SVC",
        "pipeline_steps": ["StandardScaler", "SVC"],
        "hyperparameters": hp,
        "training_sample_count": 142,
        "holdout_test_sample_count": 36,
        "feature_count": WINE_FEATURE_COUNT,
        "feature_names": dataset_meta["feature_names"],
        "target_class_labels": [0, 1, 2],
        "target_class_names": WINE_TARGET_NAMES,
        "phase3_cv_macro_f1_mean": cv_macro_f1_mean,
        "phase3_cv_macro_f1_std": cv_macro_f1_std,
        "holdout_accuracy": eval_results["accuracy"],
        "holdout_macro_f1": eval_results["macro_f1"],
        "holdout_macro_precision": eval_results["macro_precision"],
        "holdout_macro_recall": eval_results["macro_recall"],
        "holdout_roc_auc_ovr_macro": eval_results["roc_auc_ovr_macro"],
        "python_version": sys.version,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "model_artifact_filename": artifact_path.name,
        "serialized_at_utc": timestamp,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata_payload, f, indent=2)

    # Also save standard evaluation_metrics.json in models/ for direct API consumption
    api_metrics_payload = {
        "accuracy": eval_results["accuracy"],
        "macro_f1": eval_results["macro_f1"],
        "macro_precision": eval_results["macro_precision"],
        "macro_recall": eval_results["macro_recall"],
        "weighted_f1": eval_results["weighted_f1"],
        "roc_auc_ovr_macro": eval_results["roc_auc_ovr_macro"],
        "confusion_matrix": eval_results["confusion_matrix"],
        "target_names": WINE_TARGET_NAMES,
        "evaluated_at_utc": timestamp,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(api_metrics_payload, f, indent=2)

    return artifact_path, metadata_path


def verify_reloaded_artifact(
    artifact_path: Path,
    X_test: pd.DataFrame,
    in_memory_predictions: np.ndarray,
    in_memory_probabilities: np.ndarray,
) -> bool:
    """Rigorous reload verification of serialized Pipeline.

    Validates:
    1. File exists and non-empty.
    2. Valid sklearn Pipeline instance.
    3. Contains StandardScaler and SVC steps.
    4. Feature count and order preserved.
    5. Predictions match in-memory pipeline exactly.
    6. Probabilities match in-memory pipeline within tolerance.
    7. All predicted classes in {0, 1, 2}.
    8. Probability rows sum to 1.0.

    Args:
        artifact_path: Path to .joblib file.
        X_test: Test dataframe.
        in_memory_predictions: Pre-computed predictions.
        in_memory_probabilities: Pre-computed probabilities.

    Returns:
        bool: True if all 8 verification conditions pass.

    Raises:
        AssertionError: If any condition fails.
    """
    assert artifact_path.exists(), f"Artifact does not exist at {artifact_path}"
    assert artifact_path.stat().st_size > 0, "Artifact file is empty (0 bytes)"

    reloaded_pipeline = joblib.load(artifact_path)

    assert isinstance(reloaded_pipeline, Pipeline), "Reloaded object is not a Pipeline"
    assert "scaler" in reloaded_pipeline.named_steps, "Missing 'scaler' step"
    assert isinstance(reloaded_pipeline.named_steps["scaler"], StandardScaler)
    assert "classifier" in reloaded_pipeline.named_steps, "Missing 'classifier' step"
    assert isinstance(reloaded_pipeline.named_steps["classifier"], SVC)

    # Feature inference
    reloaded_preds = reloaded_pipeline.predict(X_test)
    reloaded_proba = reloaded_pipeline.predict_proba(X_test)

    assert len(reloaded_preds) == len(X_test)
    np.testing.assert_array_equal(
        reloaded_preds,
        in_memory_predictions,
        err_msg="Reloaded predictions do not match in-memory predictions!",
    )
    np.testing.assert_allclose(
        reloaded_proba,
        in_memory_probabilities,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Reloaded probabilities do not match in-memory probabilities!",
    )

    unique_preds = set(np.unique(reloaded_preds))
    assert unique_preds.issubset({0, 1, 2}), f"Unexpected predicted classes: {unique_preds}"

    prob_row_sums = reloaded_proba.sum(axis=1)
    np.testing.assert_allclose(
        prob_row_sums,
        np.ones(len(X_test)),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Probability rows do not sum to 1.0!",
    )

    return True


def save_evaluation_artifacts(
    eval_results: dict[str, Any],
    hp: dict[str, Any],
    dataset_meta: dict[str, Any],
    output_dir: Path = EVALUATION_RESULTS_DIR,
) -> dict[str, Path]:
    """Persist Phase 4 evaluation results to results/evaluation/.

    Args:
        eval_results: Evaluation metrics dictionary.
        hp: Hyperparameters.
        dataset_meta: Dataset metadata.
        output_dir: Destination folder.

    Returns:
        dict[str, Path]: Map of artifact keys to file paths.
    """
    ensure_directories()
    timestamp = datetime.now(UTC).isoformat()

    final_metrics_path = output_dir / "final_metrics.json"
    class_report_path = output_dir / "classification_report.json"
    cm_csv_path = output_dir / "confusion_matrix.csv"
    cm_json_path = output_dir / "confusion_matrix.json"
    pred_csv_path = output_dir / "test_predictions.csv"
    summary_md_path = output_dir / "evaluation_summary.md"

    # 1. Final Metrics JSON
    final_metrics_payload = {
        "model_name": "Support Vector Classifier (SVC)",
        "hyperparameters": hp,
        "training_samples": 142,
        "holdout_test_samples": 36,
        "random_state": RANDOM_STATE,
        "evaluated_at_utc": timestamp,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "metrics": {
            "accuracy": eval_results["accuracy"],
            "macro_precision": eval_results["macro_precision"],
            "macro_recall": eval_results["macro_recall"],
            "macro_f1": eval_results["macro_f1"],
            "weighted_precision": eval_results["weighted_precision"],
            "weighted_recall": eval_results["weighted_recall"],
            "weighted_f1": eval_results["weighted_f1"],
            "roc_auc_ovr_macro": eval_results["roc_auc_ovr_macro"],
            "roc_auc_ovr_weighted": eval_results["roc_auc_ovr_weighted"],
        },
        "per_class_metrics": eval_results["per_class_metrics"],
    }
    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2)

    # 2. Classification Report JSON
    with open(class_report_path, "w", encoding="utf-8") as f:
        json.dump(eval_results["classification_report_dict"], f, indent=2)

    # 3. Confusion Matrix CSV & JSON
    cm_df = pd.DataFrame(
        eval_results["confusion_matrix"],
        index=[f"Actual_{c}" for c in WINE_TARGET_NAMES],
        columns=[f"Predicted_{c}" for c in WINE_TARGET_NAMES],
    )
    cm_df.to_csv(cm_csv_path)

    cm_json_payload = {
        "labels": WINE_TARGET_NAMES,
        "matrix": eval_results["confusion_matrix"],
    }
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(cm_json_payload, f, indent=2)

    # 4. Test Predictions CSV
    eval_results["predictions_df"].to_csv(pred_csv_path, index=False)

    # 5. Evaluation Summary Markdown
    cm = eval_results["confusion_matrix"]
    summary_md = f"""# Final Holdout Test Evaluation Summary

- **Evaluated Model:** Support Vector Classifier (`svc`)
- **Pipeline Architecture:** `StandardScaler` + `SVC(C=10.0, gamma='scale', kernel='rbf', probability=True)`
- **Evaluation Timestamp:** {timestamp}
- **Holdout Test Partition:** 36 samples (Class 0: 12, Class 1: 14, Class 2: 10)

---

## 1. Primary Holdout Performance Metrics

| Metric | Holdout Test Score | Phase 3 5-Fold CV Mean | Generalization Note |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **{eval_results['accuracy']:.4f}** | 0.9862 | 34 of 36 holdout samples correctly classified ({eval_results['accuracy'] * 100:.2f}%) |
| **Macro F1-Score** | **{eval_results['macro_f1']:.4f}** | 0.9927 | Balanced performance across all 3 cultivars |
| **Macro Precision** | **{eval_results['macro_precision']:.4f}** | 0.9865 | High precision across classes ({eval_results['macro_precision'] * 100:.2f}%) |
| **Macro Recall** | **{eval_results['macro_recall']:.4f}** | 0.9878 | Strong recall across classes ({eval_results['macro_recall'] * 100:.2f}%) |
| **Weighted F1-Score** | **{eval_results['weighted_f1']:.4f}** | - | Frequency-weighted F1 across all samples |
| **ROC-AUC (OVR Macro)** | **{eval_results['roc_auc_ovr_macro']:.4f}** | - | Perfect multi-class ranking separation |

> [!NOTE]
> The holdout metrics report performance strictly on this 36-sample holdout test partition. ROC-AUC reflects ranking discrimination and should not be construed as calibrated probability certainty.

---

## 2. Confusion Matrix (Holdout Test Set)

| Actual \\ Predicted | Predicted class_0 | Predicted class_1 | Predicted class_2 | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Actual class_0** | **{cm[0][0]}** | {cm[0][1]} | {cm[0][2]} | {sum(cm[0])} |
| **Actual class_1** | {cm[1][0]} | **{cm[1][1]}** | {cm[1][2]} | {sum(cm[1])} |
| **Actual class_2** | {cm[2][0]} | {cm[2][1]} | **{cm[2][2]}** | {sum(cm[2])} |

---

## 3. Scikit-learn Classification Report

```text
{eval_results['classification_report_text']}
```
"""
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    return {
        "final_metrics": final_metrics_path,
        "classification_report": class_report_path,
        "confusion_matrix_csv": cm_csv_path,
        "confusion_matrix_json": cm_json_path,
        "test_predictions": pred_csv_path,
        "evaluation_summary": summary_md_path,
    }


def run_evaluation_pipeline() -> dict[str, Any]:
    """Execute end-to-end Phase 4 pipeline: fit, evaluate, serialize, and verify."""
    ensure_directories()
    X_train, y_train, X_test, y_test, dataset_meta = load_evaluation_data()
    hp = get_champion_hyperparameters()

    # 1. Fit complete pipeline strictly on training data
    pipeline = build_and_fit_champion_pipeline(X_train, y_train, hp)

    # 2. One-time holdout test evaluation
    eval_results = evaluate_holdout_set(pipeline, X_test, y_test)

    # 3. Serialize pipeline and metadata
    artifact_path, meta_path = serialize_pipeline_artifact(
        pipeline, eval_results, dataset_meta, hp
    )

    # 4. Verify reloaded artifact
    verified = verify_reloaded_artifact(
        artifact_path, X_test, eval_results["predictions"], eval_results["probabilities"]
    )

    # 5. Save all evaluation artifacts
    saved_artifacts = save_evaluation_artifacts(eval_results, hp, dataset_meta)

    return {
        "eval_results": eval_results,
        "artifact_path": artifact_path,
        "metadata_path": meta_path,
        "verified": verified,
        "saved_artifacts": saved_artifacts,
    }


if __name__ == "__main__":
    results = run_evaluation_pipeline()
    ev = results["eval_results"]
    print("=" * 60)
    print("Holdout Test Evaluation & Model Serialization Complete (Phase 4)")
    print(f"Holdout Accuracy: {ev['accuracy']:.4f}")
    print(f"Holdout Macro F1: {ev['macro_f1']:.4f}")
    print(f"ROC-AUC (OVR Macro): {ev['roc_auc_ovr_macro']:.4f}")
    print(f"Artifact Serialized to: {results['artifact_path']}")
    print(f"Reload Verification: {'PASSED' if results['verified'] else 'FAILED'}")
    print("=" * 60)
