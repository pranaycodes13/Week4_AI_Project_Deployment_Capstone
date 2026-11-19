"""Pydantic validation schemas for Wine Classification & Prediction API requests and responses."""

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WineFeatures(BaseModel):
    """Input features schema validating all 13 chemical constituents in strict order.

    Disallows extra keys, booleans, NaN, and infinite values.
    Handles the 12th feature alias 'od280/od315_of_diluted_wines'.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "alcohol": 13.2,
                "malic_acid": 1.78,
                "ash": 2.14,
                "alcalinity_of_ash": 11.2,
                "magnesium": 100.0,
                "total_phenols": 2.65,
                "flavanoids": 2.76,
                "nonflavanoid_phenols": 0.26,
                "proanthocyanins": 1.28,
                "color_intensity": 4.38,
                "hue": 1.05,
                "od280/od315_of_diluted_wines": 3.4,
                "proline": 1050.0,
            }
        },
    )

    alcohol: float = Field(..., description="Alcohol content (% vol)")
    malic_acid: float = Field(..., description="Malic acid (g/L)")
    ash: float = Field(..., description="Ash content (g/L)")
    alcalinity_of_ash: float = Field(..., description="Alcalinity of ash")
    magnesium: float = Field(..., description="Magnesium (mg/L)")
    total_phenols: float = Field(..., description="Total phenols")
    flavanoids: float = Field(..., description="Flavanoids")
    nonflavanoid_phenols: float = Field(..., description="Nonflavanoid phenols")
    proanthocyanins: float = Field(..., description="Proanthocyanins")
    color_intensity: float = Field(..., description="Color intensity")
    hue: float = Field(..., description="Hue")
    od280_od315_of_diluted_wines: float = Field(
        ...,
        alias="od280/od315_of_diluted_wines",
        validation_alias="od280/od315_of_diluted_wines",
        serialization_alias="od280/od315_of_diluted_wines",
        description="OD280/OD315 of diluted wines (protein content indicator)",
    )
    proline: float = Field(..., description="Proline (mg/L)")

    @field_validator("*", mode="before")
    @classmethod
    def reject_bool_nan_and_infinities(cls, v: Any) -> Any:
        """Reject booleans, NaN, and infinite floating-point values."""
        if isinstance(v, bool):
            # Pydantic requires ValueError to convert into RequestValidationError (422)
            raise ValueError("Boolean values are not permitted as numeric inputs")  # noqa: TRY004
        if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("NaN and Infinity values are not permitted")
        return v

    def to_feature_dict(self) -> dict[str, float]:
        """Convert validated record into an exact feature dictionary matching model schema."""
        return {
            "alcohol": self.alcohol,
            "malic_acid": self.malic_acid,
            "ash": self.ash,
            "alcalinity_of_ash": self.alcalinity_of_ash,
            "magnesium": self.magnesium,
            "total_phenols": self.total_phenols,
            "flavanoids": self.flavanoids,
            "nonflavanoid_phenols": self.nonflavanoid_phenols,
            "proanthocyanins": self.proanthocyanins,
            "color_intensity": self.color_intensity,
            "hue": self.hue,
            "od280/od315_of_diluted_wines": self.od280_od315_of_diluted_wines,
            "proline": self.proline,
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema supporting up to 100 wine samples."""

    model_config = ConfigDict(extra="forbid")

    samples: list[WineFeatures] = Field(
        ...,
        description="Non-empty list of Wine feature samples (max 100)",
        min_length=1,
        max_length=100,
    )


class PredictionResponse(BaseModel):
    """Single prediction response schema."""

    model_config = ConfigDict(extra="forbid")

    predicted_class: int = Field(..., description="Predicted class integer (0, 1, or 2)")
    predicted_class_name: str = Field(..., description="Predicted cultivar name (e.g. class_0)")
    probabilities: dict[str, float] = Field(
        ..., description="Predicted class probabilities mapping class name to probability"
    )
    model_name: str = Field(..., description="Model identifier")
    model_version: str | None = Field(default=None, description="Model artifact version")


class BatchPredictionItem(BaseModel):
    """Single item result inside a batch response."""

    model_config = ConfigDict(extra="forbid")

    sample_index: int = Field(..., description="Zero-based index of sample in the batch request")
    predicted_class: int = Field(..., description="Predicted class integer (0, 1, or 2)")
    predicted_class_name: str = Field(..., description="Predicted cultivar name")
    probabilities: dict[str, float] = Field(..., description="Predicted class probabilities")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""

    model_config = ConfigDict(extra="forbid")

    predictions: list[BatchPredictionItem] = Field(
        ..., description="List of predictions preserving input sequence"
    )
    total_records: int = Field(..., description="Total input records processed")
    successful_predictions: int = Field(..., description="Count of successfully generated predictions")
    model_name: str = Field(..., description="Model identifier")


class HealthResponse(BaseModel):
    """Service health and readiness response schema."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Overall service status ('healthy' or 'unhealthy')")
    model_loaded: bool = Field(..., description="Whether model artifact is loaded in memory")
    model_ready: bool = Field(..., description="Whether model is ready for inference")
    model_name: str | None = Field(default=None, description="Loaded model identifier")
    version: str = Field(default="1.0.0", description="API version")


class ModelInfoResponse(BaseModel):
    """Model metadata and evaluation summary response schema."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(..., description="Model descriptive name")
    model_type: str = Field(..., description="Underlying model class")
    pipeline_steps: list[str] = Field(..., description="Sequential steps in the inference pipeline")
    feature_count: int = Field(..., description="Total number of input features (13)")
    feature_names: list[str] = Field(..., description="Ordered list of required feature names")
    target_class_labels: list[int] = Field(..., description="Target class integer labels")
    target_class_names: list[str] = Field(..., description="Target cultivar names")
    hyperparameters: dict[str, Any] = Field(..., description="Tuned model hyperparameters")
    cross_validation_metrics: dict[str, Any] = Field(..., description="Phase 3 5-fold CV metrics")
    holdout_test_metrics: dict[str, Any] = Field(
        ..., description="Phase 4 evaluation metrics on specific 36-sample holdout test partition"
    )
    scikit_learn_version: str = Field(..., description="Scikit-learn version used for training")
    joblib_version: str = Field(..., description="Joblib version used for serialization")
