FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including build tools for compiling Rust packages like bittensor-drand)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy entire codebase first (setup.py needs README.md)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Create validator_weights directory for state persistence
RUN mkdir -p validator_weights

# Expose ports (if needed for HTTP server)
EXPOSE 8093 8094

# Entry point: Run validator
# Use exec form to properly handle signals
ENTRYPOINT ["python3", "neurons/validator.py"]

