FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy workspace-level files needed for uv sync
COPY pyproject.toml uv.lock ./
COPY apps/api/pyproject.toml apps/api/
COPY apps/chatbot_ui/pyproject.toml apps/chatbot_ui/

# Install only the api workspace member and its dependencies
RUN uv sync --frozen --package api

# Copy application source
COPY apps/api/ apps/api/

ENV PYTHONPATH=/app/apps/api/src

EXPOSE 8000

CMD ["uv", "run", "--package", "api", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "apps/api"]
