FROM python:3.11-slim

WORKDIR /app

# System deps required by chromadb, sentence-transformers, and tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --no-cache-dir keeps the image smaller and avoids disk-pressure during build
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT; fall back to 8080 for local runs
EXPOSE 8080

CMD ["python", "agent.py", "start"]