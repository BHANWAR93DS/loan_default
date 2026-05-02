FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY app.py .
COPY model.py .
COPY drift.py .
COPY explain.py .

# Copy ALL model artifacts and data
COPY data/ ./data/

# Copy MLflow tracking DB if exists
COPY mlruns/ ./mlruns/

# Expose Flask port
EXPOSE 5000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV GIT_PYTHON_REFRESH=quiet

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" \
    || exit 1

# Start Flask
CMD ["python", "app.py"]