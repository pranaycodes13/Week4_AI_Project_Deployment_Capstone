"""Configuration module for Wine Classification & Prediction API.

Defines project paths, dataset parameters, model hyperparameters,
and directory management utilities.
"""

from pathlib import Path

# Base Paths (dynamically resolved relative to repository root)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
TRAINING_RESULTS_DIR: Path = RESULTS_DIR / "training"
EVALUATION_RESULTS_DIR: Path = RESULTS_DIR / "evaluation"

# Specific File Paths
RAW_DATA_FILE: Path = RAW_DATA_DIR / "wine.csv"
X_TRAIN_FILE: Path = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_FILE: Path = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_FILE: Path = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_FILE: Path = PROCESSED_DATA_DIR / "y_test.csv"
METADATA_FILE: Path = PROCESSED_DATA_DIR / "dataset_metadata.json"

# Training Results Paths
CV_COMPARISON_JSON: Path = TRAINING_RESULTS_DIR / "cv_comparison.json"
CV_COMPARISON_CSV: Path = TRAINING_RESULTS_DIR / "cv_comparison.csv"
BEST_MODEL_METADATA_JSON: Path = TRAINING_RESULTS_DIR / "best_model_metadata.json"
HP_SEARCH_RESULTS_CSV: Path = TRAINING_RESULTS_DIR / "hyperparameter_search_results.csv"

# Evaluation Results Paths
FINAL_METRICS_JSON: Path = EVALUATION_RESULTS_DIR / "final_metrics.json"
CLASSIFICATION_REPORT_JSON: Path = EVALUATION_RESULTS_DIR / "classification_report.json"
CONFUSION_MATRIX_CSV: Path = EVALUATION_RESULTS_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_JSON: Path = EVALUATION_RESULTS_DIR / "confusion_matrix.json"
TEST_PREDICTIONS_CSV: Path = EVALUATION_RESULTS_DIR / "test_predictions.csv"
EVALUATION_SUMMARY_MD: Path = EVALUATION_RESULTS_DIR / "evaluation_summary.md"

# Reproducibility & Splitting Parameters
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# Dataset Metadata
DATASET_NAME: str = "wine"
TARGET_NAME: str = "target"
WINE_FEATURE_COUNT: int = 13
WINE_CLASS_COUNT: int = 3
WINE_TARGET_NAMES: list[str] = ["class_0", "class_1", "class_2"]
WINE_FEATURE_NAMES: list[str] = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]

# Model Pipeline Artifact Names
PIPELINE_FILENAME: str = "best_wine_pipeline.joblib"
METRICS_FILENAME: str = "evaluation_metrics.json"
MODEL_METADATA_FILENAME: str = "model_metadata.json"
MODEL_ARTIFACT_PATH: Path = MODELS_DIR / PIPELINE_FILENAME
METRICS_PATH: Path = MODELS_DIR / METRICS_FILENAME
MODEL_METADATA_PATH: Path = MODELS_DIR / MODEL_METADATA_FILENAME


def ensure_directories() -> None:
    """Create project directories if they do not already exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        NOTEBOOKS_DIR,
        DOCS_DIR,
        RESULTS_DIR,
        TRAINING_RESULTS_DIR,
        EVALUATION_RESULTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
