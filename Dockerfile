# Cloud Build passes the pre-built base image via --build-arg.
# Falls back to python:3.11-slim for local `docker build` without the arg.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy only the application code — all deps are already in the base image
COPY . .

# Cloud Run injects $PORT; fall back to 8080 for local runs
EXPOSE 8080

CMD ["python", "agent.py", "start"]