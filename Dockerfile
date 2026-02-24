FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run requirement
ENV PORT=8080

# IMPORTANT: no extra args, no "start"
CMD ["python", "agent.py"]
