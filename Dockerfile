FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application code first (needed for pip install .)
COPY pyproject.toml .
COPY app/ app/
COPY migrations/ migrations/

# Python deps
RUN pip install --no-cache-dir . uvicorn[standard] fastapi

# Create output directory
RUN mkdir -p /app/out

# Railway injects PORT at runtime
ENV PORT=8000
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]
