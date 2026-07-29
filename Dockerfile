# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install all dependencies in a separate layer so the final image stays lean.
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed for ONNX runtime and native Python package compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download the spaCy model into the install prefix
RUN python -m spacy download en_core_web_sm

# ── Stage 2: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY api/     ./api/
COPY src/      ./src/
COPY models/  ./models/

# Expose FastAPI port
EXPOSE 8000

# Run the API server (0.0.0.0 so it's reachable outside the container)
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
