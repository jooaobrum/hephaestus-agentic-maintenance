FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /db_mcp_server

# Copy workspace-level files needed for uv sync
COPY pyproject.toml uv.lock ./
COPY apps/db_mcp_server/pyproject.toml apps/db_mcp_server/README.md apps/db_mcp_server/

# Install only the api workspace member and its dependencies
RUN uv sync --frozen --package db-mcp-server

# Copy application source
COPY apps/db_mcp_server/ apps/db_mcp_server/

ENV PYTHONPATH=/db_mcp_server/apps/db_mcp_server/src/db_mcp_server

EXPOSE 8001

CMD ["uv", "run", "--package", "db-mcp-server", "python", "apps/db_mcp_server/main.py"]
