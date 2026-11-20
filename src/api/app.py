"""FastAPI application factory and route handlers for Wine Classification service."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.api.logging_utils import (
    CorrelationAndAccessLogMiddleware,
    get_current_request_id,
    setup_logging,
)
from src.api.service import (
    ModelIncompatibleError,
    ModelNotLoadedError,
    ModelServiceError,
    WineModelService,
    model_service,
)
from src.schemas.wine_schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    WineFeatures,
)

logger = logging.getLogger("wine_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager handling application startup and shutdown."""
    setup_logging()
    # Attempt to load model artifact on startup
    try:
        app.state.model_service.load_model()
        logger.info(
            "Wine classification model loaded successfully.",
            extra={"event": "startup"},
        )
    except (ModelServiceError, FileNotFoundError, OSError) as exc:
        logger.warning(
            f"Model could not be loaded at startup: {exc}",
            extra={"event": "startup_warning"},
        )
    yield
    logger.info(
        "Wine classification service shutting down.",
        extra={"event": "shutdown"},
    )


def create_app(service_instance: WineModelService | None = None) -> FastAPI:
    """Application factory creating configured FastAPI app.

    Args:
        service_instance: Optional WineModelService injection (useful for tests).

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Wine Classification & Prediction API",
        description=(
            "Production-ready Machine Learning prediction service for classifying Italian wines "
            "into 3 cultivars based on 13 chemical constituents."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Attach service instance to application state
    app.state.model_service = service_instance or model_service

    # Attach correlation and structured access logging middleware
    app.add_middleware(CorrelationAndAccessLogMiddleware)

    # --- Exception Handlers ---

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request schema validation errors with clean structured response."""
        req_id = getattr(request.state, "request_id", None) or get_current_request_id()
        errors: list[dict[str, Any]] = []
        for err in exc.errors():
            loc = " -> ".join(str(part) for part in err.get("loc", []))
            errors.append(
                {
                    "location": loc,
                    "field": err.get("loc", [])[-1] if err.get("loc") else "body",
                    "message": err.get("msg", "Invalid input value"),
                    "type": err.get("type", "value_error"),
                }
            )
        headers = {"X-Request-ID": req_id} if req_id else None
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "Input Validation Error",
                "detail": errors,
            },
            headers=headers,
        )

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(
        request: Request, exc: ModelNotLoadedError
    ) -> JSONResponse:
        """Handle missing or unloaded model requests."""
        req_id = getattr(request.state, "request_id", None) or get_current_request_id()
        logger.warning(
            f"Model not loaded for request {request.method} {request.url.path}: {exc}",
            extra={"event": "model_not_loaded", "request_id": req_id},
        )
        headers = {"X-Request-ID": req_id} if req_id else None
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Service Unavailable",
                "detail": "Model artifact is currently not loaded or unavailable for inference.",
            },
            headers=headers,
        )

    @app.exception_handler(ModelIncompatibleError)
    async def model_incompatible_handler(
        request: Request, exc: ModelIncompatibleError
    ) -> JSONResponse:
        """Handle corrupted or incompatible model artifacts."""
        req_id = getattr(request.state, "request_id", None) or get_current_request_id()
        logger.error(
            f"Model incompatible for request {request.method} {request.url.path}: {exc}",
            extra={"event": "model_incompatible", "request_id": req_id},
        )
        headers = {"X-Request-ID": req_id} if req_id else None
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Service Unavailable",
                "detail": "Model artifact is incompatible or damaged.",
            },
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Sanitize unexpected server exceptions, preventing stack trace or path exposure."""
        req_id = getattr(request.state, "request_id", None) or get_current_request_id()
        logger.error(
            f"Internal server error: {exc}",
            exc_info=exc,
            extra={"event": "internal_server_error", "request_id": req_id},
        )
        headers = {"X-Request-ID": req_id} if req_id else None
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An internal server error occurred while processing the request.",
            },
            headers=headers,
        )

    # --- Routes ---

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Liveness and Readiness Probe",
        description="Check whether the API service is alive and the model artifact is loaded and ready.",
        tags=["System"],
    )
    def health_check() -> Any:
        service: WineModelService = app.state.model_service
        health = service.get_health()
        if not health.model_ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health.model_dump(),
            )
        return health

    @app.get(
        "/model-info",
        response_model=ModelInfoResponse,
        summary="Retrieve Model Metadata and Performance Metrics",
        description="Inspect model architecture, 13 required features, class mappings, and holdout metrics.",
        tags=["System"],
    )
    def get_model_info() -> ModelInfoResponse:
        service: WineModelService = app.state.model_service
        try:
            return service.get_model_info()
        except ModelNotLoadedError:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Service Unavailable",
                    "detail": "Model metadata is unavailable because the model is not loaded.",
                },
            )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Classify a Single Wine Sample",
        description="Submit 13 chemical constituents to predict the cultivar (class_0, class_1, class_2) with probabilities.",
        tags=["Inference"],
    )
    def predict_sample(features: WineFeatures) -> PredictionResponse:
        service: WineModelService = app.state.model_service
        return service.predict_single(features)

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        summary="Classify Multiple Wine Samples in Batch",
        description="Submit a batch of up to 100 wine samples for bulk inference.",
        tags=["Inference"],
    )
    def predict_batch_samples(
        payload: Annotated[
            Any,
            Body(
                description="List of wine samples (e.g. [{...}, {...}]) or object with 'samples' key",
            ),
        ],
    ) -> BatchPredictionResponse:
        # Support both raw list: [{...}, {...}] and wrapped: {"samples": [{...}, {...}]}
        samples_list: list[WineFeatures] = []
        if isinstance(payload, list):
            if len(payload) == 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Batch request cannot be empty (min 1 sample required).",
                )
            if len(payload) > 100:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Batch size exceeds maximum limit of 100 samples (received {len(payload)}).",
                )
            for idx, item in enumerate(payload):
                try:
                    samples_list.append(WineFeatures.model_validate(item))
                except (ValueError, TypeError, ValidationError) as val_exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail=f"Validation error at batch index {idx}: {val_exc}",
                    )
        elif isinstance(payload, dict):
            try:
                batch_req = BatchPredictionRequest.model_validate(payload)
                samples_list = batch_req.samples
            except (ValueError, TypeError, ValidationError) as dict_exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Batch validation error: {dict_exc}",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Batch payload must be a JSON array or an object with 'samples' list.",
            )

        service: WineModelService = app.state.model_service
        return service.predict_batch(samples_list)

    return app


# Default app instance for ASGI servers (e.g. uvicorn src.api.app:app)
app = create_app()
