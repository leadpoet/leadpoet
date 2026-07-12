# ============================================================
# LeadPoet Trustless Gateway - Dockerfile
# ============================================================
# Multi-stage build for minimal production image
# Non-root user for security
# Health check for orchestration

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies to user directory
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim

ARG BUILD_ID=unknown
ARG GITHUB_SHA=unknown
ARG GITHUB_REF_NAME=unknown
ARG BUILD_TIME_UTC=unknown

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 gateway

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/gateway/.local

# Copy gateway application code
COPY gateway/ ./gateway/

# Gateway runtime imports shared canonical, Research Lab, qualification, and
# fulfillment-verification packages from the repository root. Keep the Docker
# smoke image aligned with the production /home/ec2-user package layout.
COPY leadpoet_canonical/ ./leadpoet_canonical/
COPY leadpoet_verifier/ ./leadpoet_verifier/
COPY research_lab/ ./research_lab/
COPY qualification/ ./qualification/
COPY validator_models/ ./validator_models/
COPY Leadpoet/ ./Leadpoet/
COPY schemas/ ./schemas/

# Do not bake local .env files into images. Configure the container with
# runtime environment variables or mount a GATEWAY_ENV_FILE fallback.

# Set ownership to non-root user
RUN chown -R gateway:gateway /app

# Switch to non-root user
USER gateway

# Add local bin to PATH
ENV PATH=/home/gateway/.local/bin:$PATH

# Set Python to run in unbuffered mode (logs appear immediately)
ENV PYTHONUNBUFFERED=1
ENV BUILD_ID=${BUILD_ID}
ENV GITHUB_SHA=${GITHUB_SHA}
ENV GITHUB_REF_NAME=${GITHUB_REF_NAME}
ENV BUILD_TIME_UTC=${BUILD_TIME_UTC}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
