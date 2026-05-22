# Hephaestus: Agentic Maintenance Assistant

Hephaestus is an enterprise-grade, RAG-powered agentic maintenance assistant designed to accelerate Root Cause Analysis (RCA) and institutional knowledge capture for industrial maintenance teams. It combines multi-agent orchestration via **LangGraph**, hybrid retrieval-augmented generation (RAG) in **Qdrant**, real-time sensor integration via a **FastMCP DB Server**, and structured knowledge persistence in **PostgreSQL**.

![Hephaestus pitch](docs/pitch.png)

---

## Demo

### Root Cause Analysis — Troubleshooting Agent

![Troubleshooting agent demo](docs/troubleshooting.gif)

### Historical Analysis — Summarizer Agent

![Summarizer agent demo](docs/summarizer.gif)

---

## Architecture

![System architecture](docs/architecture.png)

The system is a multi-agent supervisor graph:

- **Coordinator** — routes incoming queries to the right specialist
- **Troubleshooting agent** — iterative RCA loop: gathers evidence, discriminates hypotheses, converges on root cause
- **Summarizer agent** — retrieves and summarizes past interventions, builds reusable Known Case Templates
- **FastMCP DB Server** — exposes PostgreSQL sensor data and maintenance records as MCP tools
- **Qdrant** — hybrid (dense + sparse) retrieval for CM interventions and procedure manuals

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| Agent Orchestration | LangGraph (multi-agent supervisor with PostgreSQL checkpointer) |
| Backend API | FastAPI (SSE streaming) |
| Frontend UI | Streamlit |
| Vector Database | Qdrant (hybrid sparse + dense via RRF) |
| Relational Database | PostgreSQL 16 (sensor data + LangGraph checkpointer) |
| LLM | OpenAI (embeddings, generation, agent logic) |
| PDF Extraction | Mistral AI (OCR for procedure guides) |
| Observability | LangSmith + RAGAS |

---

## Environment Setup

### 1. Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package and project manager
- **Docker & Docker Compose** — for containerized services

Install `uv` if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install dependencies

```bash
git clone <repo-url>
cd hephaestus-agentic-maintenance
uv sync
```

`uv sync` creates a `.venv` at the project root and installs all workspace members (`apps/api`, `apps/chatbot_ui`, `apps/db_mcp_server`) and shared notebook dependencies.

### 3. Configure environment variables

Create a `.env` file in the **project root**. This file is read by both local scripts and Docker services.

```env
# --- OpenAI (required: embeddings, generation, agents) ---
OPENAI_API_KEY="your-openai-api-key"

# --- Mistral AI (required: PDF OCR for procedure guides) ---
MISTRAL_API_KEY="your-mistral-api-key"

# --- Cohere (optional: post-retrieval reranking) ---
CO_API_KEY="your-cohere-api-key"

# --- LangSmith (highly recommended: tracing & evaluation) ---
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your-langsmith-api-key"
LANGSMITH_PROJECT="hephaestus-agentic-maintenance"

# --- Service URLs (optional: defaults are pre-wired for the Docker stack) ---
# QDRANT_URL="http://localhost:6333"
# POSTGRES_URL="postgresql+psycopg://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
```

> **Required keys to get started:** `OPENAI_API_KEY` and `MISTRAL_API_KEY`. LangSmith keys are optional but strongly recommended for observability.

---

## Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Copy and fill in the .env file (see above)
cp .env.example .env   # edit with your keys

# 3. Ingest data into Qdrant & PostgreSQL
make run-pipeline

# 4. Start all services (API, UI, vector DB, relational DB, MCP server)
make run-docker-compose
```

Open [http://localhost:8501](http://localhost:8501) to interact with the assistant.

---

## Data Ingestion Pipelines

All pipelines are orchestrated by `scripts/orchestrate_pipeline.py`.

### Interventions & fault clustering

Loads historical corrective maintenance (CM) records into PostgreSQL, vectorizes them into Qdrant, and runs UMAP/HDBSCAN clustering to build a fault-mode knowledge graph.

```bash
make run-interventions
# or
uv run python scripts/orchestrate_pipeline.py interventions

# Force-reload CSV even if tables are already populated:
uv run python scripts/orchestrate_pipeline.py interventions --force-db-load
```

### Troubleshooting procedures

Parses PDF maintenance manuals with Mistral OCR, extracts semantic chunks, enriches them with LLM context, and indexes them as hybrid sparse + dense vectors in Qdrant.

```bash
make run-procedures
# or
uv run python scripts/orchestrate_pipeline.py procedures
```

### Run all pipelines

```bash
make run-pipeline
# or
uv run python scripts/orchestrate_pipeline.py all
```

---

## Running the Services Stack

```bash
make run-docker-compose
```

| Service | Port | URL | Purpose |
| :--- | :--- | :--- | :--- |
| Streamlit Chat UI | `8501` | [http://localhost:8501](http://localhost:8501) | Conversational frontend |
| FastAPI Gateway | `8000` | [http://localhost:8000/docs](http://localhost:8000/docs) | REST API + SSE streaming |
| FastMCP DB Server | `8001` | `http://localhost:8001` | Exposes PostgreSQL data to agents |
| Qdrant Vector DB | `6333` | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) | Hybrid retrieval |
| PostgreSQL 16 | `5433` | `localhost:5433` | Sensor data + LangGraph state |

```bash
# Stream logs from all containers
docker compose logs -f

# Tear down the stack
docker compose down
```

---

## Observability & Evaluation

### LangSmith tracing

All retriever queries, LLM calls, and agent nodes are wrapped with `@traceable`. With a valid `LANGSMITH_API_KEY` in `.env`, every turn streams nested execution graphs, latency, and token usage to your LangSmith dashboard automatically.

### RAGAS evaluation

Validates the RAG engine against benchmark datasets across four metrics: Context Precision, Context Recall, Faithfulness, and Answer Relevancy.

```bash
make run-evals
# or
PYTHONPATH="apps/api/src" uv run python apps/api/evals/eval.py
```

### Agent evaluation

```bash
make run-agent-evals
# or
PYTHONPATH="apps/api/src" uv run python apps/api/evals/eval_agents.py
```

---

## Project Structure

```
hephaestus-agentic-maintenance/
├── apps/
│   ├── api/              # FastAPI backend — RAG pipelines, LangGraph orchestration, REST routes
│   ├── chatbot_ui/       # Streamlit conversational UI
│   └── db_mcp_server/    # FastMCP DB tool server (exposes PostgreSQL to agents)
├── scripts/              # Ingestion, parsing, and pipeline orchestration scripts
├── notebooks/            # Data exploration, clustering, and evaluation experiments
├── data/                 # Source datasets, PDFs, intermediate outputs (gitignored)
├── docs/                 # Architecture diagrams and demo assets
├── docker-compose.yml
├── Makefile              # Shortcut commands for common workflows
└── pyproject.toml        # uv workspace root
```
