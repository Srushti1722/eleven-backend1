ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt .

# Step 1: Install everything normally
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Nuke ALL google-related packages to clear namespace conflicts
RUN pip uninstall -y \
    google \
    google-generativeai \
    google-ai-generativelanguage \
    google-cloud-aiplatform \
    google-api-core \
    google-auth \
    googleapis-common-protos \
    2>/dev/null || true

# Step 3: Reinstall google packages cleanly in the correct dependency order
RUN pip install --no-cache-dir \
    google-auth \
    google-api-core \
    googleapis-common-protos \
    google-generativeai

# Step 4: Verify the import works — build will FAIL here if still broken
RUN python -c "from google import genai; print('✅ google.genai import OK')"

COPY . .

RUN python agent.py download-files

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/ || exit 1

CMD ["python", "agent.py"]
