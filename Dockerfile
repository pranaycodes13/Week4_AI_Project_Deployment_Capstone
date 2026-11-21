# ==============================================================================
# Production Dockerfile for Wine Classification & Prediction API
# Base: Python 3.13 slim (official Debian-based lightweight image)
# Security: Non-root execution, zero secrets, minimal attack surface
# ==============================================================================

FROM python:3.13-slim

# Set environment variables for optimized container runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Set explicit application directory
WORKDIR /app

# Create dedicated non-root application user and group (UID 10001)
RUN groupadd -r -g 10001 appuser && \
    useradd -r -u 10001 -g appuser -d /app -s /sbin/nologin -c "Application Runner" appuser

# Install production dependencies before copying full source for layer caching
COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip check

# Copy application source code and serialized model artifacts with non-root ownership
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser run.py README.md ./

# Create runtime directories expected by config.py and ensure permissions
RUN mkdir -p /app/data /app/results && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose documented application port
EXPOSE 8000

# Container liveness & model readiness health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; res = urllib.request.urlopen('http://127.0.0.1:8000/health'); sys.exit(0 if res.getcode() == 200 else 1)"

# Production ASGI server entry point (supports dynamic cloud PORT)
CMD ["sh", "-c", "exec uvicorn src.api.app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]

