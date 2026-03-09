# ── Stage 1: Base image ───────────────────────────────────────────────────────
FROM python:3.10-slim AS base

# System deps for building C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Dependencies ─────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Application ──────────────────────────────────────────────────────
FROM deps AS app

# Copy project source
COPY src/   ./src/
COPY api/   ./api/
COPY data/  ./data/
COPY models/ ./models/
COPY train.py    .
COPY evaluate.py .

# Create chart output directory
RUN mkdir -p /app/data/charts

# Expose FastAPI port
EXPOSE 8000

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default command — start the API server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
