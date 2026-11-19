"""FastAPI application and route handlers."""

from src.api.app import app, create_app
from src.api.service import WineModelService, model_service

__all__ = ["WineModelService", "app", "create_app", "model_service"]
