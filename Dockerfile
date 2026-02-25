# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile  –  thin application layer
#
# All Python dependencies live in the base image (Dockerfile.base).
# Cloud Build builds the base first (with layer caching), then builds this
# image on top of it.  Locally you can just `docker build .` and it falls
# back to plain python:3.11-slim (deps will be installed at build time).
# ─────────────────────────────────────────────────────────────────────────────

# Cloud Build injects the pre-built base image tag via --build-arg.
# Local builds fall back to python:3.11-slim (slower but self-contained).
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# If running without the pre-built base (local dev), install deps here.
# When BASE_IMAGE is the pre-built image this layer is a no-op because
# pip will find everything already installed and skip quickly.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# Copy application source
COPY . .

# Cloud Run injects $PORT at runtime; 8080 is the default.
EXPOSE 8080

# Healthcheck so Docker/Cloud Run can verify the container is alive.
# agent.py starts an HTTP server on $PORT that returns 200 OK.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/ || exit 1

CMD ["python", "agent.py", "start"]