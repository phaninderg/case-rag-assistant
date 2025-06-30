FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in the same layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements*.txt dev-requirements.txt ./

# Install all Python dependencies in a single layer to reduce image size
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        pip install --no-cache-dir torch torchvision torchaudio && \
        pip install --no-cache-dir -r requirements-linux.txt; \
    else \
        pip install --no-cache-dir torch torchvision torchaudio && \
        pip install --no-cache-dir -r requirements.txt; \
    fi && \
    find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.10/site-packages -name "__pycache__" -delete

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/embeddings /app/data/cases /app/offload \
    && chmod -R 777 /app/data /app/offload

# Expose port
EXPOSE 8000

# Set entrypoint with increased timeout
CMD ["gunicorn", "src.app:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300", "--graceful-timeout", "300"]
