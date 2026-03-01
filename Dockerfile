ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# livekit pulls in the bare 'google 1.0.0' stub as a transitive dependency.
# This empty package poisons the google namespace and breaks google-generativeai.
# Uninstall it AFTER everything else is installed so pip doesn't re-add it.
RUN pip uninstall -y google 2>/dev/null || true

# Verify the fix — build fails here if something re-added the stub
RUN python -c "from google import genai; print('google.genai OK')"

COPY . .

RUN python agent.py download-files

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/ || exit 1

CMD ["python", "agent.py"]
