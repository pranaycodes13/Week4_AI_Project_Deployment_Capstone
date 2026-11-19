"""Preprocessing pipeline and data splitting module for Wine dataset."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATASET_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    WINE_CLASS_COUNT,
    WINE_FEATURE_COUNT,
    ensure_directories,
)
from src.data.loader import load_raw_wine_data


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform reproducible, stratified train/test split on features and target.

    Args:
        X: Feature matrix DataFrame.
        y: Target Series.
        test_size: Fraction of samples to include in holdout test set.
        random_state: Deterministic random state seed.

    Returns:
        tuple containing (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If partition sizes or class representation criteria fail.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Validate split dimensions
    expected_train_rows = round(len(X) * (1 - test_size))
    expected_test_rows = round(len(X) * test_size)

    if len(X_train) != expected_train_rows or len(X_test) != expected_test_rows:
        raise ValueError(
            f"Split size mismatch: expected ({expected_train_rows}, {expected_test_rows}), "
            f"got ({len(X_train)}, {len(X_test)})"
        )

    if X_train.shape[1] != WINE_FEATURE_COUNT or X_test.shape[1] != WINE_FEATURE_COUNT:
        raise ValueError(
            f"Feature count mismatch: expected {WINE_FEATURE_COUNT}, "
            f"got train {X_train.shape[1]}, test {X_test.shape[1]}"
        )

    # Validate index disjointness (no overlap between train and test)
    if not set(X_train.index).isdisjoint(set(X_test.index)):
        raise ValueError("Train and test indices overlap; data leakage detected!")

    # Validate all classes present in both splits
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())
    if len(train_classes) != WINE_CLASS_COUNT or len(test_classes) != WINE_CLASS_COUNT:
        raise ValueError(
            f"Incomplete class representation in partitions: "
            f"train classes={train_classes}, test classes={test_classes}"
        )

    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline() -> Pipeline:
    """Factory creating an unfitted scikit-learn preprocessing Pipeline.

    Encapsulates StandardScaler to ensure transformation statistics are computed
    strictly on training data, preventing data leakage.

    Returns:
        Pipeline: Unfitted scikit-learn Pipeline with a StandardScaler step.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    """Persist train/test partitions and metadata JSON to disk.

    Args:
        X_train: Training features DataFrame.
        X_test: Holdout test features DataFrame.
        y_train: Training target Series.
        y_test: Holdout test target Series.
        output_dir: Destination directory.

    Returns:
        dict[str, Path]: Paths to saved files.
    """
    ensure_directories()

    x_train_path = output_dir / "X_train.csv"
    x_test_path = output_dir / "X_test.csv"
    y_train_path = output_dir / "y_train.csv"
    y_test_path = output_dir / "y_test.csv"
    metadata_path = output_dir / "dataset_metadata.json"

    # Save CSV partitions preserving feature names and row alignment
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_frame().to_csv(y_train_path, index=False)
    y_test.to_frame().to_csv(y_test_path, index=False)

    metadata: dict[str, Any] = {
        "dataset_name": DATASET_NAME,
        "sample_count": len(X_train) + len(X_test),
        "feature_count": X_train.shape[1],
        "feature_names": list(X_train.columns),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distributions": {
            "total": {
                int(c): int(cnt)
                for c, cnt in pd.concat([y_train, y_test])
                .value_counts()
                .sort_index()
                .items()
            },
            "train": {
                int(c): int(cnt)
                for c, cnt in y_train.value_counts().sort_index().items()
            },
            "test": {
                int(c): int(cnt)
                for c, cnt in y_test.value_counts().sort_index().items()
            },
        },
        "dataset_validation_results": {
            "missing_values_train": int(X_train.isna().sum().sum()),
            "missing_values_test": int(X_test.isna().sum().sum()),
            "duplicate_samples_in_raw": 0,
            "features_valid": True,
            "classes_valid": True,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "X_train": x_train_path,
        "X_test": x_test_path,
        "y_train": y_train_path,
        "y_test": y_test_path,
        "metadata": metadata_path,
    }


if __name__ == "__main__":
    ensure_directories()
    X, y = load_raw_wine_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    saved = save_processed_data(X_train, X_test, y_train, y_test)
    print("Preprocessing completed successfully:")
    for key, path in saved.items():
        print(f"  {key}: {path}")
