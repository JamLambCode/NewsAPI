# Multi-stage build for the News Intelligence API
FROM python:3.11-slim AS builder

# Install uv for faster package management
RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy dependency specs
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache .


# Final runtime image
FROM python:3.11-slim

# Install runtime dependencies for torch (libgomp for CPU inference)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set up application user (non-root for security)
RUN useradd --create-home --shell /bin/bash holocron && \
    mkdir -p /app && \
    chown -R holocron:holocron /app

WORKDIR /app

# Copy application code
COPY --chown=holocron:holocron src/ ./src/
COPY --chown=holocron:holocron scripts/ ./scripts/
COPY --chown=holocron:holocron relationships.yaml ./
COPY --chown=holocron:holocron pyproject.toml ./

USER holocron

# Activate venv in PATH
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health', timeout=5.0).raise_for_status()"

# Default command
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]

