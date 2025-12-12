FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Copy entire codebase
COPY . .

# Create validator_weights directory for state persistence
RUN mkdir -p validator_weights

# Expose ports (if needed for HTTP server)
EXPOSE 8093 8094

# Entry point: Run validator
# Use exec form to properly handle signals
ENTRYPOINT ["python3", "neurons/validator.py"]

