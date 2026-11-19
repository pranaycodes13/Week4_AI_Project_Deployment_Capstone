"""Unit and integration tests for Wine data ingestion and inspection."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from src.config import (
    TARGET_NAME,
    WINE_CLASS_COUNT,
    WINE_FEATURE_COUNT,
)
from src.data.loader import inspect_dataset, load_raw_wine_data, save_raw_wine_data


def test_data_loader_dimensions() -> None:
    """Validate dataset shape matches 178 samples and 13 numerical features."""
    X, y = load_raw_wine_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (178, WINE_FEATURE_COUNT)
    assert len(y) == 178


def test_data_loader_feature_names() -> None:
    """Ensure original scikit-learn feature names are fully preserved."""
    X, _ = load_raw_wine_data()
    expected_names = list(load_wine().feature_names)
    assert list(X.columns) == expected_names
    assert len(X.columns) == WINE_FEATURE_COUNT


def test_data_loader_target_classes() -> None:
    """Ensure target contains exactly 3 distinct classes (0, 1, 2)."""
    _, y = load_raw_wine_data()
    unique_classes = sorted(y.unique().tolist())
    assert len(unique_classes) == WINE_CLASS_COUNT
    assert unique_classes == [0, 1, 2]
    assert y.name == TARGET_NAME


def test_data_loader_numerical_types() -> None:
    """Verify that all features are numerical floating-point values."""
    X, y = load_raw_wine_data()
    for col in X.columns:
        assert np.issubdtype(X[col].dtype, np.floating)
    assert np.issubdtype(y.dtype, np.integer)


def test_data_loader_no_missing_values() -> None:
    """Verify zero NaN, null, or missing entries exist across features and target."""
    X, y = load_raw_wine_data()
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0


def test_data_loader_no_duplicate_rows() -> None:
    """Verify no duplicate sample feature rows exist in the raw dataset."""
    X, _ = load_raw_wine_data()
    assert X.duplicated().sum() == 0


def test_data_loader_target_distribution() -> None:
    """Verify ground truth class distribution counts match standard wine dataset."""
    _, y = load_raw_wine_data()
    distribution = y.value_counts().to_dict()
    assert distribution[0] == 59
    assert distribution[1] == 71
    assert distribution[2] == 48


def test_save_and_reload_raw_data(tmp_path) -> None:
    """Verify saving to CSV and reloading preserves data schema, values, and target."""
    X, y = load_raw_wine_data()
    test_path = tmp_path / "wine_test_raw.csv"
    saved_path = save_raw_wine_data(X, y, output_path=test_path)

    assert saved_path.exists()
    reloaded_df = pd.read_csv(saved_path)
    assert reloaded_df.shape == (178, WINE_FEATURE_COUNT + 1)
    assert TARGET_NAME in reloaded_df.columns
    assert list(reloaded_df.columns[:-1]) == list(X.columns)
    assert reloaded_df[TARGET_NAME].tolist() == y.tolist()


def test_inspect_dataset_structure() -> None:
    """Verify inspect_dataset produces all required descriptive metadata keys."""
    X, y = load_raw_wine_data()
    metrics = inspect_dataset(X, y)

    required_keys = [
        "shape",
        "feature_names",
        "data_types",
        "missing_values_per_column",
        "missing_values_target",
        "duplicate_rows_count",
        "target_class_distribution",
        "descriptive_statistics",
        "feature_min_values",
        "feature_max_values",
    ]
    for key in required_keys:
        assert key in metrics

    assert metrics["shape"]["rows"] == 178
    assert metrics["shape"]["columns"] == 13
    assert metrics["duplicate_rows_count"] == 0
    assert metrics["target_class_distribution"] == {0: 59, 1: 71, 2: 48}


def test_load_raw_wine_data_corrupted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify load_raw_wine_data raises ValueError when underlying dataset is corrupted."""
    real_load = load_wine

    # 1. Corrupted shape (dropped feature column)
    def mock_dropped_col(**kwargs):
        bunch = real_load(**kwargs)
        bunch.data = bunch.data.iloc[:, :-1]
        return bunch

    monkeypatch.setattr("src.data.loader.load_wine", mock_dropped_col)
    with pytest.raises(ValueError, match="Expected shape"):
        load_raw_wine_data()

    # 2. Corrupted target length
    def mock_dropped_target(**kwargs):
        bunch = real_load(**kwargs)
        bunch.target = bunch.target.iloc[:-1]
        return bunch

    monkeypatch.setattr("src.data.loader.load_wine", mock_dropped_target)
    with pytest.raises(ValueError, match="Expected target length"):
        load_raw_wine_data()

    # 3. Missing values in features
    def mock_missing_vals(**kwargs):
        bunch = real_load(**kwargs)
        df = bunch.data.copy()
        df.iloc[0, 0] = np.nan
        bunch.data = df
        return bunch

    monkeypatch.setattr("src.data.loader.load_wine", mock_missing_vals)
    with pytest.raises(ValueError, match="Missing values detected"):
        load_raw_wine_data()

