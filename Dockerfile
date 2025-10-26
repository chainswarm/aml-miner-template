# Multi-stage build for AML Miner Template
# Python 3.13-slim base image for optimal size and performance

# ============================================================================
# Build Stage: Install dependencies and compile packages
# ============================================================================
FROM python:3.13-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
# Use --no-cache-dir to reduce image size
# Install to /install directory for easy copying
RUN pip install --no-cache-dir --prefix=/install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================================
# Runtime Stage: Minimal image with only necessary components
# ============================================================================
FROM python:3.13-slim

# Metadata
LABEL maintainer="AML Miner Team"
LABEL description="AML Miner Template - Alert scoring, ranking, and cluster assessment"
LABEL version="1.0.0"

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Disable pip version check
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# Set Python path
ENV PYTHONPATH=/app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
# -m creates home directory, -u specifies UID
RUN useradd -m -u 1000 miner && \
    mkdir -p /app /app/trained_models /app/logs /app/data && \
    chown -R miner:miner /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=miner:miner aml_miner ./aml_miner
COPY --chown=miner:miner pyproject.toml ./

# Copy trained models directory structure
# Models can be mounted as volume at runtime if not present at build time
COPY --chown=miner:miner trained_models ./trained_models/

# Create .env file placeholder
RUN touch .env && chown miner:miner .env

# Switch to non-root user
USER miner

# Expose port
EXPOSE 8000

# Health check
# Check API health endpoint every 30 seconds
# Wait 40 seconds before starting checks (allows startup time)
# Timeout after 3 seconds
# Fail after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Run API server
# Use exec form for proper signal handling
CMD ["python", "-m", "aml_miner.api.server"]

# Alternative commands (can be overridden):
# Training: docker run <image> python scripts/train_models.py --data-dir ./data --output-dir ./trained_models
# Validation: docker run <image> python scripts/validate_submission.py --batch-path ./data/batch_001