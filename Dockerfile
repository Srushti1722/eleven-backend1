# ---- Base image ----
FROM python:3.11-slim

# ---- Environment ----
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# ---- System deps ----
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Install Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy app code ----
COPY . .

# ---- Cloud Run health port ----
EXPOSE 8080

# ---- Start LiveKit Agent ----
CMD ["python", "agent.py"]
