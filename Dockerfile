ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}
WORKDIR /app
COPY requirements.txt .

# FIX: removed 2>/dev/null || true so install errors are visible and fatal
# Also force reinstall google packages to fix namespace conflicts
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --force-reinstall google-generativeai

COPY . .
RUN python agent.py download-files
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/ || exit 1
CMD ["python", "agent.py"]
