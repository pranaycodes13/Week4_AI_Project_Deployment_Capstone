"""Foundation tests for project configuration, directories, and parameters."""

from pathlib import Path

from src.config import (
    CV_FOLDS,
    DATA_DIR,
    DOCS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_DATA_DIR,
    TARGET_NAME,
    TEST_SIZE,
    WINE_CLASS_COUNT,
    WINE_FEATURE_COUNT,
    ensure_directories,
)


def test_project_root_resolution() -> None:
    """Verify that PROJECT_ROOT resolves to a valid existing directory containing src/."""
    assert PROJECT_ROOT.exists(), "PROJECT_ROOT must exist on the filesystem"
    assert PROJECT_ROOT.is_dir(), "PROJECT_ROOT must be a directory"
    assert (PROJECT_ROOT / "src").exists(), "PROJECT_ROOT must contain the 'src' directory"
    assert (PROJECT_ROOT / "requirements.txt").exists(), "PROJECT_ROOT must contain 'requirements.txt'"


def test_configuration_constants() -> None:
    """Verify ML and dataset configuration constants are within valid bounds."""
    assert RANDOM_STATE == 42
    assert isinstance(TEST_SIZE, float)
    assert 0.0 < TEST_SIZE < 1.0
    assert TEST_SIZE == 0.20
    assert CV_FOLDS == 5
    assert TARGET_NAME == "target"
    assert WINE_FEATURE_COUNT == 13
    assert WINE_CLASS_COUNT == 3


def test_expected_directory_paths() -> None:
    """Verify directory paths are properly structured relative to PROJECT_ROOT."""
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RAW_DATA_DIR == DATA_DIR / "raw"
    assert PROCESSED_DATA_DIR == DATA_DIR / "processed"
    assert MODELS_DIR == PROJECT_ROOT / "models"
    assert NOTEBOOKS_DIR == PROJECT_ROOT / "notebooks"
    assert DOCS_DIR == PROJECT_ROOT / "docs"


def test_directory_creation() -> None:
    """Verify ensure_directories() creates all required folders without error."""
    ensure_directories()
    expected_dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        NOTEBOOKS_DIR,
        DOCS_DIR,
    ]
    for directory in expected_dirs:
        assert directory.exists(), f"Directory {directory} should exist after ensure_directories()"
        assert directory.is_dir(), f"Path {directory} should be a directory"


def test_no_hardcoded_user_paths() -> None:
    """Ensure project root path does not contain static machine-bound assumptions."""
    assert isinstance(PROJECT_ROOT, Path)
    # Verify that PROJECT_ROOT is absolute and anchored to the codebase
    assert PROJECT_ROOT.is_absolute()
