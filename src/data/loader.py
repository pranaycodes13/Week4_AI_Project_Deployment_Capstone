"""Wine dataset ingestion, validation, and inspection module."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_wine

from src.config import (
    RAW_DATA_FILE,
    TARGET_NAME,
    WINE_CLASS_COUNT,
    WINE_FEATURE_COUNT,
    ensure_directories,
)


def load_raw_wine_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the scikit-learn Wine dataset and validate integrity.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.

    Raises:
        ValueError: If validation checks (dimensions, nulls, duplicates) fail.
    """
    wine_bunch = load_wine(as_frame=True)
    X: pd.DataFrame = wine_bunch.data.copy()
    y: pd.Series = wine_bunch.target.copy()
    y.name = TARGET_NAME

    # Validation Checks
    if X.shape != (178, WINE_FEATURE_COUNT):
        raise ValueError(f"Expected shape (178, {WINE_FEATURE_COUNT}), got {X.shape}")

    if len(y) != 178:
        raise ValueError(f"Expected target length 178, got {len(y)}")

    unique_classes = set(y.unique())
    if len(unique_classes) != WINE_CLASS_COUNT:
        raise ValueError(
            f"Expected {WINE_CLASS_COUNT} classes, found {len(unique_classes)}: {unique_classes}"
        )

    missing_in_X = int(X.isna().sum().sum())
    missing_in_y = int(y.isna().sum())
    if missing_in_X > 0 or missing_in_y > 0:
        raise ValueError(
            f"Missing values detected: X has {missing_in_X}, y has {missing_in_y}"
        )

    duplicate_count = int(X.duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"Duplicate samples detected in X: {duplicate_count}")

    return X, y


def save_raw_wine_data(
    X: pd.DataFrame, y: pd.Series, output_path: Path = RAW_DATA_FILE
) -> Path:
    """Save raw wine dataset snapshot including target to CSV.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        output_path: Destination path for CSV file.

    Returns:
        Path: Path to saved raw CSV file.
    """
    ensure_directories()
    raw_df = X.copy()
    raw_df[y.name] = y.values
    raw_df.to_csv(output_path, index=False)
    return output_path


def inspect_dataset(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Compute comprehensive inspection statistics for features and target.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Returns:
        dict[str, Any]: Inspection metrics including shape, types, missing counts,
                        distribution, and summary statistics.
    """
    describe_df = X.describe()
    summary_stats = describe_df.to_dict()

    min_values = {col: float(X[col].min()) for col in X.columns}
    max_values = {col: float(X[col].max()) for col in X.columns}

    inspection_results: dict[str, Any] = {
        "shape": {"rows": int(X.shape[0]), "columns": int(X.shape[1])},
        "feature_names": list(X.columns),
        "data_types": {col: str(dtype) for col, dtype in X.dtypes.items()},
        "missing_values_per_column": {col: int(X[col].isna().sum()) for col in X.columns},
        "missing_values_target": int(y.isna().sum()),
        "duplicate_rows_count": int(X.duplicated().sum()),
        "target_class_distribution": {
            int(cls): int(count) for cls, count in y.value_counts().sort_index().items()
        },
        "descriptive_statistics": summary_stats,
        "feature_min_values": min_values,
        "feature_max_values": max_values,
    }
    return inspection_results


if __name__ == "__main__":
    ensure_directories()
    X, y = load_raw_wine_data()
    saved_path = save_raw_wine_data(X, y)
    inspection = inspect_dataset(X, y)
    print(f"Dataset successfully ingested and saved to {saved_path}")
    print(f"Shape: {inspection['shape']}")
    print(f"Target distribution: {inspection['target_class_distribution']}")
    print(f"Duplicates: {inspection['duplicate_rows_count']}")

