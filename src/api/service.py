"""Model service module providing singleton artifact loading, validation, and inference."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    PROJECT_ROOT,
    WINE_FEATURE_COUNT,
    WINE_TARGET_NAMES,
)
from src.schemas.wine_schema import (
    BatchPredictionItem,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    WineFeatures,
)


class ModelServiceError(Exception):
    """Base exception for model service operations."""


class ModelNotLoadedError(ModelServiceError):
    """Raised when inference is requested but model is not loaded."""


class ModelIncompatibleError(ModelServiceError):
    """Raised when loaded artifact fails integrity or architecture validation."""


class WineModelService:
    """Centralized inference service managing model lifecycle and predictions."""

    def __init__(
        self,
        artifact_path: Path = MODEL_ARTIFACT_PATH,
        metadata_path: Path = MODEL_METADATA_PATH,
    ) -> None:
        self.artifact_path = artifact_path
        self.metadata_path = metadata_path
        self._pipeline: Pipeline | None = None
        self._metadata: dict[str, Any] | None = None
        self._is_loaded: bool = False
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        """Check whether service is fully initialized and ready for inference."""
        return self._is_loaded and self._pipeline is not None

    def load_model(self) -> None:
        """Load and validate the serialized sklearn Pipeline and metadata.

        Resolves paths relative to PROJECT_ROOT and validates steps and feature order.

        Raises:
            ModelNotLoadedError: If artifact or metadata files are missing.
            ModelIncompatibleError: If artifact fails pipeline verification.
        """
        # Ensure paths are absolute and anchored to project root
        art_path = self.artifact_path if self.artifact_path.is_absolute() else PROJECT_ROOT / self.artifact_path
        meta_path = self.metadata_path if self.metadata_path.is_absolute() else PROJECT_ROOT / self.metadata_path

        if not art_path.exists():
            self._is_loaded = False
            self._load_error = "Model artifact file not found on disk."
            raise ModelNotLoadedError("Model artifact file not found.")

        if not meta_path.exists():
            self._is_loaded = False
            self._load_error = "Model metadata file not found on disk."
            raise ModelNotLoadedError("Model metadata file not found.")

        try:
            # 1. Load metadata
            with open(meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            # 2. Load serialized Pipeline
            loaded_obj = joblib.load(art_path)

            # 3. Architectural validation
            if not isinstance(loaded_obj, Pipeline):
                raise ModelIncompatibleError("Loaded object is not a valid scikit-learn Pipeline.")

            step_names = [name for name, _ in loaded_obj.steps]
            if "scaler" not in step_names or "classifier" not in step_names:
                raise ModelIncompatibleError(
                    f"Pipeline missing required steps ('scaler', 'classifier'). Found: {step_names}"
                )

            if not isinstance(loaded_obj.named_steps["scaler"], StandardScaler):
                raise ModelIncompatibleError("First step in pipeline must be StandardScaler.")

            if not isinstance(loaded_obj.named_steps["classifier"], SVC):
                raise ModelIncompatibleError("Classifier step in pipeline must be SVC.")

            # 4. Feature schema validation
            expected_features = self._metadata.get("feature_names", [])
            if len(expected_features) != WINE_FEATURE_COUNT:
                raise ModelIncompatibleError(
                    f"Expected {WINE_FEATURE_COUNT} features in metadata, found {len(expected_features)}."
                )

            # 5. Pipeline feature check
            if hasattr(loaded_obj, "feature_names_in_"):
                pipeline_features = list(loaded_obj.feature_names_in_)
                if pipeline_features != expected_features:
                    raise ModelIncompatibleError(
                        "Pipeline feature_names_in_ does not match metadata feature order."
                    )

            self._pipeline = loaded_obj
            self._is_loaded = True
            self._load_error = None

        except (ModelNotLoadedError, ModelIncompatibleError) as e:
            self._pipeline = None
            self._is_loaded = False
            self._load_error = str(e)
            raise
        except Exception as e:
            self._pipeline = None
            self._is_loaded = False
            self._load_error = "Failed to load model artifact due to an internal error."
            raise ModelIncompatibleError("Model artifact loading failed due to corruption or mismatch.") from e

    def predict_single(self, features: WineFeatures) -> PredictionResponse:
        """Run inference on a single validated WineFeatures record.

        Args:
            features: Validated wine feature inputs.

        Returns:
            PredictionResponse: Predicted class, cultivar name, and probabilities.

        Raises:
            ModelNotLoadedError: If model is not loaded.
        """
        if not self.is_ready or self._pipeline is None:
            raise ModelNotLoadedError("Model is not loaded or ready for inference.")

        feature_dict = features.to_feature_dict()
        df = pd.DataFrame([feature_dict])

        pred_class_idx = int(self._pipeline.predict(df)[0])
        probabilities_arr = self._pipeline.predict_proba(df)[0]

        class_names = self._metadata.get("target_class_names", WINE_TARGET_NAMES) if self._metadata else WINE_TARGET_NAMES
        predicted_class_name = class_names[pred_class_idx]

        probabilities_dict = {
            class_names[i]: float(probabilities_arr[i])
            for i in range(len(class_names))
        }

        model_name = self._metadata.get("model_name", "Support Vector Classifier (SVC)") if self._metadata else "SVC"

        return PredictionResponse(
            predicted_class=pred_class_idx,
            predicted_class_name=predicted_class_name,
            probabilities=probabilities_dict,
            model_name=model_name,
            model_version="1.0.0",
        )

    def predict_batch(self, samples: list[WineFeatures]) -> BatchPredictionResponse:
        """Run batch inference on multiple validated WineFeatures records.

        Args:
            samples: List of validated WineFeatures (1 to 100).

        Returns:
            BatchPredictionResponse: List of individual predictions preserving order.

        Raises:
            ModelNotLoadedError: If model is not loaded.
        """
        if not self.is_ready or self._pipeline is None:
            raise ModelNotLoadedError("Model is not loaded or ready for inference.")

        if not samples:
            raise ValueError("Batch request cannot be empty.")

        feature_dicts = [sample.to_feature_dict() for sample in samples]
        df = pd.DataFrame(feature_dicts)

        predictions_arr = self._pipeline.predict(df)
        probabilities_mat = self._pipeline.predict_proba(df)

        class_names = self._metadata.get("target_class_names", WINE_TARGET_NAMES) if self._metadata else WINE_TARGET_NAMES
        model_name = self._metadata.get("model_name", "Support Vector Classifier (SVC)") if self._metadata else "SVC"

        items: list[BatchPredictionItem] = []
        for i in range(len(samples)):
            pred_class = int(predictions_arr[i])
            probs = {
                class_names[c]: float(probabilities_mat[i][c])
                for c in range(len(class_names))
            }
            items.append(
                BatchPredictionItem(
                    sample_index=i,
                    predicted_class=pred_class,
                    predicted_class_name=class_names[pred_class],
                    probabilities=probs,
                )
            )

        return BatchPredictionResponse(
            predictions=items,
            total_records=len(samples),
            successful_predictions=len(items),
            model_name=model_name,
        )

    def get_health(self) -> HealthResponse:
        """Provide service liveness, readiness, and model status."""
        model_name = self._metadata.get("model_name", "Support Vector Classifier (SVC)") if self._metadata and self._is_loaded else None
        status = "healthy" if self.is_ready else "unhealthy"

        return HealthResponse(
            status=status,
            model_loaded=self._is_loaded,
            model_ready=self.is_ready,
            model_name=model_name,
            version="1.0.0",
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Retrieve model metadata, architecture, and holdout evaluation metrics.

        Excludes absolute local filesystem paths and internal secrets.

        Raises:
            ModelNotLoadedError: If model is not loaded.
        """
        if not self.is_ready or self._metadata is None:
            raise ModelNotLoadedError("Model metadata is not loaded or available.")

        meta = self._metadata

        return ModelInfoResponse(
            model_name=meta.get("model_name", "Support Vector Classifier (SVC)"),
            model_type=meta.get("model_type", "sklearn.svm.SVC"),
            pipeline_steps=meta.get("pipeline_steps", ["StandardScaler", "SVC"]),
            feature_count=meta.get("feature_count", WINE_FEATURE_COUNT),
            feature_names=meta.get("feature_names", []),
            target_class_labels=meta.get("target_class_labels", [0, 1, 2]),
            target_class_names=meta.get("target_class_names", WINE_TARGET_NAMES),
            hyperparameters=meta.get("hyperparameters", {}),
            cross_validation_metrics={
                "strategy": "Stratified 5-Fold Cross-Validation (Training Partition Only)",
                "mean_cv_macro_f1": meta.get("phase3_cv_macro_f1_mean"),
                "std_cv_macro_f1": meta.get("phase3_cv_macro_f1_std"),
            },
            holdout_test_metrics={
                "partition_scope": "Evaluated strictly on unseen 36-sample holdout test partition",
                "accuracy": meta.get("holdout_accuracy"),
                "macro_f1": meta.get("holdout_macro_f1"),
                "macro_precision": meta.get("holdout_macro_precision"),
                "macro_recall": meta.get("holdout_macro_recall"),
                "roc_auc_ovr_macro": meta.get("holdout_roc_auc_ovr_macro"),
            },
            scikit_learn_version=meta.get("scikit_learn_version", "unknown"),
            joblib_version=meta.get("joblib_version", "unknown"),
        )


# Global singleton instance
model_service = WineModelService()
