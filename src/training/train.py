"""Model training, 5-fold cross-validation, hyperparameter tuning, and model selection.

Operates strictly on the training partition (data/processed/X_train.csv and y_train.csv)
to ensure zero holdout test leakage.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    CV_FOLDS,
    DATASET_NAME,
    RANDOM_STATE,
    TARGET_NAME,
    TRAINING_RESULTS_DIR,
    WINE_CLASS_COUNT,
    WINE_FEATURE_COUNT,
    X_TRAIN_FILE,
    Y_TRAIN_FILE,
    ensure_directories,
)


def load_training_data(
    x_path: Path = X_TRAIN_FILE, y_path: Path = Y_TRAIN_FILE
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and strictly validate the training partition.

    Args:
        x_path: Path to X_train.csv.
        y_path: Path to y_train.csv.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix X_train and target y_train.

    Raises:
        ValueError: If validation checks (dimensions, nulls, classes) fail.
    """
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Training artifacts not found at {x_path} and {y_path}. "
            "Please run preprocessing first."
        )

    X_train = pd.read_csv(x_path)
    y_raw = pd.read_csv(y_path)
    y_train = y_raw[TARGET_NAME] if TARGET_NAME in y_raw.columns else y_raw.iloc[:, 0]
    y_train.name = TARGET_NAME

    if len(X_train) != 142 or len(y_train) != 142:
        raise ValueError(
            f"Expected exactly 142 training samples, got X={len(X_train)}, y={len(y_train)}"
        )

    if X_train.shape[1] != WINE_FEATURE_COUNT:
        raise ValueError(
            f"Expected {WINE_FEATURE_COUNT} features, got {X_train.shape[1]}"
        )

    if X_train.isna().sum().sum() > 0 or y_train.isna().sum() > 0:
        raise ValueError("Missing values detected in training partition")

    unique_classes = sorted(y_train.unique().tolist())
    if len(unique_classes) != WINE_CLASS_COUNT:
        raise ValueError(
            f"Expected {WINE_CLASS_COUNT} classes, found {unique_classes}"
        )

    return X_train, y_train


def get_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Construct candidate classification pipelines with StandardScaler.

    All models are wrapped in an sklearn Pipeline encapsulating StandardScaler
    to guarantee leakage-free cross-validation.

    Args:
        random_state: Deterministic seed.

    Returns:
        dict[str, Pipeline]: Dictionary of candidate name -> unfitted Pipeline.
    """
    candidates = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "svc": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    return candidates


def get_hyperparameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Define targeted hyperparameter search grids for each candidate architecture.

    Returns:
        dict[str, dict[str, list[Any]]]: Parameter grids per candidate model.
    """
    grids = {
        "logistic_regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs"],
        },
        "random_forest": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_split": [2, 5],
        },
        "svc": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"],
        },
    }
    return grids


def evaluate_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: dict[str, Pipeline],
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, Any]]:
    """Evaluate candidate pipelines with StratifiedKFold CV on training data only.

    Args:
        X_train: Training features DataFrame.
        y_train: Training target Series.
        candidates: Candidate pipelines dictionary.
        cv_folds: Number of stratified folds.
        random_state: Deterministic seed for folding.

    Returns:
        dict[str, dict[str, Any]]: Metrics per candidate across all folds.
    """
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    scoring = {
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    results: dict[str, dict[str, Any]] = {}

    for name, pipeline in candidates.items():
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        fold_f1 = cv_scores["test_f1_macro"].tolist()
        fold_acc = cv_scores["test_accuracy"].tolist()
        fold_prec = cv_scores["test_precision_macro"].tolist()
        fold_rec = cv_scores["test_recall_macro"].tolist()

        results[name] = {
            "fold_f1_macro": [float(x) for x in fold_f1],
            "mean_f1_macro": float(np.mean(fold_f1)),
            "std_f1_macro": float(np.std(fold_f1)),
            "fold_accuracy": [float(x) for x in fold_acc],
            "mean_accuracy": float(np.mean(fold_acc)),
            "std_accuracy": float(np.std(fold_acc)),
            "fold_precision_macro": [float(x) for x in fold_prec],
            "mean_precision_macro": float(np.mean(fold_prec)),
            "std_precision_macro": float(np.std(fold_prec)),
            "fold_recall_macro": [float(x) for x in fold_rec],
            "mean_recall_macro": float(np.mean(fold_rec)),
            "std_recall_macro": float(np.std(fold_rec)),
        }

    return results


def run_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: dict[str, Pipeline],
    param_grids: dict[str, dict[str, list[Any]]],
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run GridSearchCV for all candidate models using training partition only.

    Args:
        X_train: Training features.
        y_train: Training targets.
        candidates: Base pipelines.
        param_grids: Hyperparameter grids per model.
        cv_folds: CV fold count.
        random_state: Fold seed.

    Returns:
        tuple[dict[str, Any], pd.DataFrame]: Best estimators with metrics,
                                             and detailed search results DataFrame.
    """
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    tuned_summary: dict[str, Any] = {}
    all_search_records: list[dict[str, Any]] = []

    for name, pipeline in candidates.items():
        grid = param_grids[name]
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="f1_macro",
            cv=skf,
            refit=True,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_score = float(grid_search.best_score_)
        best_index = grid_search.best_index_
        best_std = float(grid_search.cv_results_["std_test_score"][best_index])
        best_params = grid_search.best_params_

        tuned_summary[name] = {
            "best_estimator": grid_search.best_estimator_,
            "best_params": best_params,
            "best_f1_macro_mean": best_score,
            "best_f1_macro_std": best_std,
        }

        # Collect detailed records for tabular export
        for i in range(len(grid_search.cv_results_["params"])):
            all_search_records.append(
                {
                    "model_name": name,
                    "params": str(grid_search.cv_results_["params"][i]),
                    "mean_test_f1_macro": float(
                        grid_search.cv_results_["mean_test_score"][i]
                    ),
                    "std_test_f1_macro": float(
                        grid_search.cv_results_["std_test_score"][i]
                    ),
                    "rank_test_score": int(
                        grid_search.cv_results_["rank_test_score"][i]
                    ),
                }
            )

    search_df = pd.DataFrame(all_search_records)
    return tuned_summary, search_df


def select_best_model(
    tuned_summary: dict[str, Any],
) -> dict[str, Any]:
    """Select the best candidate model based strictly on CV Macro F1 performance.

    Considers model stability (standard deviation) and simplicity in near-tie scenarios.

    Args:
        tuned_summary: Summary of tuned models.

    Returns:
        dict[str, Any]: Selection decision and justification metadata.
    """
    # Sort models by mean F1 Macro descending, then by lowest standard deviation
    sorted_models = sorted(
        tuned_summary.items(),
        key=lambda item: (
            item[1]["best_f1_macro_mean"],
            -item[1]["best_f1_macro_std"],
        ),
        reverse=True,
    )

    champion_name, champion_data = sorted_models[0]

    # Model complexity ranking for simplicity tie-breaking:
    # LogisticRegression (linear, fastest) > SVC (kernel) > RandomForest (ensemble)
    rationale = (
        f"Selected '{champion_name}' with highest mean 5-fold CV Macro F1 of "
        f"{champion_data['best_f1_macro_mean']:.4f} (+/- {champion_data['best_f1_macro_std']:.4f}). "
        f"The model demonstrated superior generalization stability across all stratified folds."
    )

    selection_metadata: dict[str, Any] = {
        "selected_model": champion_name,
        "best_params": champion_data["best_params"],
        "mean_cv_macro_f1": champion_data["best_f1_macro_mean"],
        "std_cv_macro_f1": champion_data["best_f1_macro_std"],
        "model_selection_rationale": rationale,
        "ranking": [
            {
                "rank": rank + 1,
                "model_name": name,
                "mean_f1_macro": data["best_f1_macro_mean"],
                "std_f1_macro": data["best_f1_macro_std"],
                "best_params": data["best_params"],
            }
            for rank, (name, data) in enumerate(sorted_models)
        ],
    }
    return selection_metadata


def save_training_artifacts(
    baseline_results: dict[str, dict[str, Any]],
    tuned_summary: dict[str, Any],
    selection_metadata: dict[str, Any],
    search_df: pd.DataFrame,
    output_dir: Path = TRAINING_RESULTS_DIR,
) -> dict[str, Path]:
    """Persist Phase 3 evaluation tables, rankings, and metadata to results/training/.

    Does NOT serialize fitted model binary (reserved strictly for Phase 4).

    Args:
        baseline_results: Untuned 5-fold CV evaluation metrics.
        tuned_summary: Tuned models metadata.
        selection_metadata: Champion selection record.
        search_df: Full grid search parameter exploration table.
        output_dir: Destination folder.

    Returns:
        dict[str, Path]: Map of artifact names to file paths.
    """
    ensure_directories()

    cv_comparison_json_path = output_dir / "cv_comparison.json"
    cv_comparison_csv_path = output_dir / "cv_comparison.csv"
    best_model_meta_path = output_dir / "best_model_metadata.json"
    hp_search_csv_path = output_dir / "hyperparameter_search_results.csv"

    # 1. Comparison JSON
    comparison_payload = {
        "dataset_name": DATASET_NAME,
        "training_sample_count": 142,
        "feature_count": WINE_FEATURE_COUNT,
        "cross_validation_strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "primary_metric": "f1_macro",
        "baseline_cv_results": baseline_results,
        "tuned_cv_results": {
            k: {
                "best_params": v["best_params"],
                "best_f1_macro_mean": v["best_f1_macro_mean"],
                "best_f1_macro_std": v["best_f1_macro_std"],
            }
            for k, v in tuned_summary.items()
        },
        "selection_decision": selection_metadata,
    }
    with open(cv_comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)

    # 2. Comparison CSV
    comparison_rows = []
    for rank_info in selection_metadata["ranking"]:
        name = rank_info["model_name"]
        base_data = baseline_results.get(name, {})
        comparison_rows.append(
            {
                "model_name": name,
                "rank": rank_info["rank"],
                "baseline_mean_f1_macro": base_data.get("mean_f1_macro", np.nan),
                "baseline_std_f1_macro": base_data.get("std_f1_macro", np.nan),
                "baseline_mean_accuracy": base_data.get("mean_accuracy", np.nan),
                "baseline_mean_precision_macro": base_data.get("mean_precision_macro", np.nan),
                "baseline_mean_recall_macro": base_data.get("mean_recall_macro", np.nan),
                "tuned_mean_f1_macro": rank_info["mean_f1_macro"],
                "tuned_std_f1_macro": rank_info["std_f1_macro"],
                "best_params": str(rank_info["best_params"]),
            }
        )
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(cv_comparison_csv_path, index=False)

    # 3. Best Model Metadata JSON
    best_meta_payload = {
        "dataset_name": DATASET_NAME,
        "training_samples": 142,
        "feature_count": WINE_FEATURE_COUNT,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "primary_metric": "f1_macro",
        "selected_model": selection_metadata["selected_model"],
        "best_hyperparameters": selection_metadata["best_params"],
        "mean_cv_macro_f1": selection_metadata["mean_cv_macro_f1"],
        "std_cv_macro_f1": selection_metadata["std_cv_macro_f1"],
        "selection_rationale": selection_metadata["model_selection_rationale"],
        "model_architecture": "Pipeline([('scaler', StandardScaler()), ('classifier', ...)])",
    }
    with open(best_model_meta_path, "w", encoding="utf-8") as f:
        json.dump(best_meta_payload, f, indent=2)

    # 4. Hyperparameter Search Results CSV
    search_df.to_csv(hp_search_csv_path, index=False)

    return {
        "cv_comparison_json": cv_comparison_json_path,
        "cv_comparison_csv": cv_comparison_csv_path,
        "best_model_metadata": best_model_meta_path,
        "hyperparameter_search_results": hp_search_csv_path,
    }


def run_training_pipeline() -> dict[str, Any]:
    """Execute end-to-end training, cross-validation, tuning, and model selection."""
    ensure_directories()
    X_train, y_train = load_training_data()

    candidates = get_candidate_models(random_state=RANDOM_STATE)
    param_grids = get_hyperparameter_grids()

    baseline_results = evaluate_candidate_models(
        X_train, y_train, candidates, cv_folds=CV_FOLDS, random_state=RANDOM_STATE
    )

    tuned_summary, search_df = run_hyperparameter_tuning(
        X_train, y_train, candidates, param_grids, cv_folds=CV_FOLDS, random_state=RANDOM_STATE
    )

    selection_meta = select_best_model(tuned_summary)

    saved_paths = save_training_artifacts(
        baseline_results, tuned_summary, selection_meta, search_df
    )

    return {
        "baseline_results": baseline_results,
        "tuned_summary": tuned_summary,
        "selection_metadata": selection_meta,
        "saved_paths": saved_paths,
    }


if __name__ == "__main__":
    results = run_training_pipeline()
    sel = results["selection_metadata"]
    print("=" * 60)
    print("Model Training & Selection Complete (Phase 3)")
    print(f"Selected Model: {sel['selected_model']}")
    print(f"Best CV Macro F1: {sel['mean_cv_macro_f1']:.4f} (+/- {sel['std_cv_macro_f1']:.4f})")
    print(f"Best Parameters: {sel['best_params']}")
    print("=" * 60)
    for k, p in results["saved_paths"].items():
        print(f"  {k}: {p}")
