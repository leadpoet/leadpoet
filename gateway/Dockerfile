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

# Copy .env file (ensure it's in .dockerignore if sensitive)
# In production, use environment variables instead
COPY .env .env

# Set ownership to non-root user
RUN chown -R gateway:gateway /app

# Switch to non-root user
USER gateway

# Add local bin to PATH
ENV PATH=/home/gateway/.local/bin:$PATH

# Set Python to run in unbuffered mode (logs appear immediately)
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]

