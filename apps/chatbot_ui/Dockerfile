FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy workspace-level files needed for uv sync
COPY pyproject.toml uv.lock ./
COPY apps/api/pyproject.toml apps/api/
COPY apps/chatbot_ui/pyproject.toml apps/chatbot_ui/

# Install only the chatbot-ui workspace member and its dependencies
RUN uv sync --frozen --package chatbot-ui

# Copy application source
COPY apps/chatbot_ui/ apps/chatbot_ui/

EXPOSE 8501

CMD ["uv", "run", "--package", "chatbot-ui", "streamlit", "run", "apps/chatbot_ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
