"""Unit and integration tests for preprocessing pipeline and data partitioning."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    WINE_CLASS_COUNT,
    WINE_FEATURE_COUNT,
)
from src.data.loader import load_raw_wine_data
from src.preprocessing.pipeline import (
    create_preprocessing_pipeline,
    save_processed_data,
    split_data,
)


def test_stratified_split_sizes() -> None:
    """Verify split proportions yield exactly 142 train and 36 test samples."""
    X, y = load_raw_wine_data()
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    assert len(X_train) == 142
    assert len(X_test) == 36
    assert len(y_train) == 142
    assert len(y_test) == 36
    assert X_train.shape[1] == WINE_FEATURE_COUNT
    assert X_test.shape[1] == WINE_FEATURE_COUNT


def test_stratified_split_class_representation() -> None:
    """Verify all 3 target classes are represented in train and test partitions."""
    X, y = load_raw_wine_data()
    _, _, y_train, y_test = split_data(X, y)

    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())

    assert len(train_classes) == WINE_CLASS_COUNT
    assert len(test_classes) == WINE_CLASS_COUNT
    assert train_classes == {0, 1, 2}
    assert test_classes == {0, 1, 2}

    # Verify stratified proportions (approx 33% class 0, 40% class 1, 27% class 2)
    assert y_train.value_counts()[0] == 47
    assert y_train.value_counts()[1] == 57
    assert y_train.value_counts()[2] == 38
    assert y_test.value_counts()[0] == 12
    assert y_test.value_counts()[1] == 14
    assert y_test.value_counts()[2] == 10


def test_split_reproducibility() -> None:
    """Ensure identical splits when using the fixed random_state seed."""
    X, y = load_raw_wine_data()
    X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=42)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


def test_split_indices_disjoint() -> None:
    """Ensure train and test sample indices do not overlap (zero data leakage)."""
    X, y = load_raw_wine_data()
    X_train, X_test, _, _ = split_data(X, y)

    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    assert train_indices.isdisjoint(test_indices)
    assert len(train_indices.intersection(test_indices)) == 0
    assert len(train_indices) + len(test_indices) == 178


def test_feature_ordering_preserved() -> None:
    """Ensure original column ordering is strictly maintained across partitions."""
    X, y = load_raw_wine_data()
    original_cols = list(X.columns)
    X_train, X_test, _, _ = split_data(X, y)

    assert list(X_train.columns) == original_cols
    assert list(X_test.columns) == original_cols


def test_pipeline_contains_standard_scaler() -> None:
    """Verify that preprocessing pipeline consists of StandardScaler."""
    pipeline = create_preprocessing_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert "scaler" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_pipeline_is_unfitted() -> None:
    """Verify factory returns an unfitted pipeline without cached mean/scale attributes."""
    pipeline = create_preprocessing_pipeline()
    scaler = pipeline.named_steps["scaler"]
    assert not hasattr(scaler, "mean_")
    assert not hasattr(scaler, "scale_")
    assert not hasattr(scaler, "n_samples_seen_")


def test_pipeline_fit_transform_capabilities() -> None:
    """Verify pipeline fits exclusively on training data and transforms test data."""
    X, y = load_raw_wine_data()
    X_train, X_test, _, _ = split_data(X, y)

    pipeline = create_preprocessing_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    # Transformed data dimensions
    assert X_train_scaled.shape == (142, WINE_FEATURE_COUNT)
    assert X_test_scaled.shape == (36, WINE_FEATURE_COUNT)

    # Standard scaling properties on train: mean approx 0, std approx 1
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-7)


def test_test_set_never_used_to_fit_scaler() -> None:
    """Verify scaler statistics (mean_ and scale_) depend purely on X_train."""
    X, y = load_raw_wine_data()
    X_train, _, _, _ = split_data(X, y)

    pipeline = create_preprocessing_pipeline()
    pipeline.fit(X_train)

    train_scaler_mean = pipeline.named_steps["scaler"].mean_.copy()
    expected_train_mean = X_train.values.mean(axis=0)

    # The fitted mean must match X_train mean exactly
    np.testing.assert_allclose(train_scaler_mean, expected_train_mean, rtol=1e-5)

    # And must NOT equal the full dataset mean (verifying test was excluded)
    full_mean = X.values.mean(axis=0)
    assert not np.allclose(train_scaler_mean, full_mean, atol=1e-3)


def test_save_processed_data_roundtrip(tmp_path) -> None:
    """Verify processed partitions and metadata JSON can be saved and loaded accurately."""
    X, y = load_raw_wine_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    saved_files = save_processed_data(
        X_train, X_test, y_train, y_test, output_dir=tmp_path
    )

    for key, path in saved_files.items():
        assert path.exists(), f"File {key} at {path} should exist"

    # Reload partitions
    loaded_x_train = pd.read_csv(saved_files["X_train"])
    loaded_x_test = pd.read_csv(saved_files["X_test"])
    loaded_y_train = pd.read_csv(saved_files["y_train"])
    loaded_y_test = pd.read_csv(saved_files["y_test"])

    assert loaded_x_train.shape == (142, WINE_FEATURE_COUNT)
    assert loaded_x_test.shape == (36, WINE_FEATURE_COUNT)
    assert len(loaded_y_train) == 142
    assert len(loaded_y_test) == 36
    assert list(loaded_x_train.columns) == list(X.columns)

    # Metadata validation
    with open(saved_files["metadata"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["dataset_name"] == "wine"
    assert meta["train_samples"] == 142
    assert meta["test_samples"] == 36
    assert meta["dataset_validation_results"]["missing_values_train"] == 0


def test_split_data_feature_mismatch_raises() -> None:
    """Verify split_data raises ValueError when feature count is mismatched."""
    X, y = load_raw_wine_data()
    bad_X = X.iloc[:, :-1]  # 12 columns instead of 13
    with pytest.raises(ValueError, match="Feature count mismatch"):
        split_data(bad_X, y)

