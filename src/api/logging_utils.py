"""Structured JSON logging configuration, contextvars, and request correlation middleware."""

import contextvars
import json
import logging
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Context variable storing correlation ID across async execution contexts
correlation_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Request ID validation pattern: alphanumeric, hyphen, underscore, 1 to 64 chars
REQUEST_ID_REGEX: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def is_valid_request_id(request_id: str | None) -> bool:
    """Validate incoming request ID format (alphanumeric, -, _, length 1-64).

    Rejects empty strings, control characters, special characters, and oversized inputs
    to prevent log and header injection vulnerabilities.
    """
    if not request_id or not isinstance(request_id, str):
        return False
    return bool(REQUEST_ID_REGEX.match(request_id))


def get_current_request_id() -> str | None:
    """Retrieve current request ID from contextvar."""
    return correlation_id_ctx.get()


class JsonFormatter(logging.Formatter):
    """Custom logging formatter emitting valid, single-line structured JSON records."""

    def format(self, record: logging.LogRecord) -> str:
        record_time = datetime.fromtimestamp(record.created, tz=UTC).isoformat()

        log_entry: dict[str, Any] = {
            "timestamp": record_time,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Resolve request correlation ID
        req_id = getattr(record, "request_id", None) or correlation_id_ctx.get()
        if req_id:
            log_entry["request_id"] = req_id

        # Attach event name if specified
        event = getattr(record, "event", None)
        if event:
            log_entry["event"] = event

        # Attach structured request/access attributes if available
        for key in ("method", "path", "status_code", "duration_ms", "client_ip"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        # Include sanitized internal exception stack trace when exc_info is present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root and application loggers with structured JSON formatting on stdout."""
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is invoked multiple times
    has_json_handler = any(
        isinstance(getattr(h, "formatter", None), JsonFormatter)
        for h in root_logger.handlers
    )

    if not has_json_handler:
        # Clear existing stream handlers to prevent duplicate lines
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(JsonFormatter())
        stream_handler.setLevel(numeric_level)
        root_logger.addHandler(stream_handler)

    # Suppress duplicate default Uvicorn access logs in favor of structured access events
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers.clear()
    uvicorn_access.propagate = False


class CorrelationAndAccessLogMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware managing request IDs and structured access logging."""

    def __init__(self, app: Any, logger_name: str = "wine_api.access") -> None:
        super().__init__(app)
        self.access_logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()

        # 1. Accept valid incoming X-Request-ID or generate cryptographically random UUID
        raw_req_id = request.headers.get("x-request-id")
        if is_valid_request_id(raw_req_id):
            request_id = str(raw_req_id)
        else:
            request_id = str(uuid.uuid4())

        # 2. Attach request_id to request.state and task-isolated contextvar
        request.state.request_id = request_id
        token = correlation_id_ctx.set(request_id)

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            client_ip = request.client.host if request.client else None
            self.access_logger.error(
                f"{request.method} {request.url.path} 500 in {duration_ms}ms",
                extra={
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                    "client_ip": client_ip,
                },
            )
            raise
        else:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            client_ip = request.client.host if request.client else None

            # 3. Ensure X-Request-ID header is always present in response
            response.headers["X-Request-ID"] = request_id

            # 4. Emit single structured access log event
            self.access_logger.info(
                f"{request.method} {request.url.path} {response.status_code} in {duration_ms}ms",
                extra={
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "client_ip": client_ip,
                },
            )
            return response
        finally:
            # 5. Clean up request-scoped context
            correlation_id_ctx.reset(token)
