# Hephaestus: Agentic Maintenance Assistant 🛠🔥

Hephaestus is an enterprise-grade, RAG-powered agentic maintenance assistant designed to accelerate Root Cause Analysis (RCA) and institutional knowledge capture for industrial maintenance teams. It combines multi-agent orchestration via **LangGraph**, hybrid retrieval-augmented generation (RAG) in **Qdrant**, real-time sensor integration via a **FastMCP DB Server**, and structured knowledge persistence in **PostgreSQL**.

---

## 🛠 Tech Stack

- **Agent Orchestration:** LangGraph (StateGraph with multi-turn persistence)
- **Backend API:** FastAPI (supporting server-sent events for streaming outputs)
- **Frontend UI:** Streamlit (interactive chat interface)
- **Vector Database:** Qdrant (hybrid sparse + dense vector search via Reciprocal Rank Fusion)
- **Relational Database:** PostgreSQL 16 (for sensor readings, component RUL, and LangGraph checkpointer state)
- **LLM Integrations:** OpenAI (embeddings, generation, and agent logic)
- **PDF Extraction:** Mistral AI (high-fidelity OCR for troubleshooting procedure guides)
- **Observability & Evaluation:** LangSmith & RAGAS

---

## 🚀 Environment Setup

### 1. Prerequisites
Ensure you have the following installed on your system:
- **Python 3.12+** (highly recommended to manage with [uv](https://github.com/astral-sh/uv))
- **Docker & Docker Compose** (for running the containerized database & backend services)

### 2. Install Dependencies
Hephaestus utilizes a `uv` workspace layout. To sync all workspace member dependencies (API, UI, MCP Server) and create the local virtual environment, run:
```bash
uv sync
```

### 3. Configure the Environment (`.env`)
Create a `.env` file in the root directory of the project. This file is required for both the local python scripts and the docker services. Below is a copy-pasteable configuration template with explanations:

```env
# --- OpenAI API Configuration (Required for embeddings, generation, & agents) ---
OPENAI_API_KEY="your-openai-api-key"

# --- Mistral AI Configuration (Required for PDF OCR extraction) ---
MISTRAL_API_KEY="your-mistral-api-key"

# --- Cohere API Configuration (Optional: for post-retrieval reranking) ---
CO_API_KEY="your-cohere-api-key"

# --- LangSmith Observability & Tracing (Highly Recommended) ---
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your-langsmith-api-key"
LANGSMITH_PROJECT="hephaestus-agentic-maintenance"

# --- Database URL Overrides (Optional - defaults are pre-configured for Docker Stack) ---
# QDRANT_URL="http://localhost:6333"
# POSTGRES_URL="postgresql+psycopg://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
```

---

## 🗂 Running the Data Ingestion Pipelines

The data ingestion pipeline parses source PDF manuals, enriches documents with LLM context, ingests corrective maintenance records into Postgres/Qdrant, and runs clustering algorithms to generate a structured failure-mode knowledge graph.

All pipelines are orchestrated via the central runner script: `scripts/orchestrate_pipeline.py`.

### 1. Interventions & Fault Clustering Pipeline
This pipeline prepares all historical Corrective Maintenance (CM) records. It performs:
1. Database schema initialization (`scripts/ingestion/init_db.py`).
2. CM CSV validation and load to PostgreSQL (`scripts/ingestion/ingest_csv_to_postgres.py`).
3. CM intervention vectorization and hybrid ingestion to Qdrant (`scripts/ingestion/ingest_interventions_to_qdrant.py`).
4. Automated fault clustering using UMAP/HDBSCAN and known-issues graph generation (`scripts/ingestion/build_known_issues_graph.py`).

**Run via Makefile:**
```bash
make run-interventions
```
**Run via uv directly:**
```bash
uv run python scripts/orchestrate_pipeline.py interventions
```
*Note: To force-reload the CSV into Postgres even if tables are already populated, add the `--force-db-load` flag:*
```bash
uv run python scripts/orchestrate_pipeline.py interventions --force-db-load
```

### 2. Troubleshooting Procedures Pipeline
This pipeline processes PDFs of maintenance manuals and troubleshooting guides:
1. Parses PDF guides via Mistral OCR (`scripts/ingestion/parse_procedures_pdf.py`).
2. Extracts semantic chunks and enriches them with LLM-generated context (`scripts/ingestion/extract_procedures.py`).
3. Indexes enriched chunks into Qdrant hybrid sparse + dense vectors (`scripts/ingestion/ingest_procedures.py`).

**Run via Makefile:**
```bash
make run-procedures
```
**Run via uv directly:**
```bash
uv run python scripts/orchestrate_pipeline.py procedures
```

### 3. Orchestrate All Pipelines
To spin up all offline data tasks and ingest both interventions and procedures sequentially:
**Run via Makefile:**
```bash
make run-pipeline
```
**Run via uv directly:**
```bash
uv run python scripts/orchestrate_pipeline.py all
```

---

## 🚀 Running the Services Stack

Hephaestus provides a completely containerized services stack that can be launched using Docker Compose.

To compile, link, and start the active services in the background:
```bash
make run-docker-compose
```

This command orchestrates and starts the following components:

| Service | Port | Endpoint / UI | Purpose |
| :--- | :--- | :--- | :--- |
| **Streamlit Chat UI** | `8501` | [http://localhost:8501](http://localhost:8501) | Conversational frontend for operators & technicians |
| **FastAPI Gateway** | `8000` | [http://localhost:8000/docs](http://localhost:8000/docs) | REST API & SSE streaming orchestration |
| **FastMCP DB Server** | `8001` | `http://localhost:8001` | Exposes PostgreSQL sensor & life data to agents |
| **Qdrant Vector DB** | `6333` | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) | Sparse/Dense hybrid retrieval search |
| **PostgreSQL 16** | `5433` | `localhost:5433` | Relational storage for sensors and LangGraph state persistence |

To inspect container logs:
```bash
docker compose logs -f
```

To tear down the stack and stop all running containers:
```bash
docker compose down
```

---

## 📊 Observability & RAGAS Evaluations

### LangSmith Tracing
All retriever queries, LLM operations, and agent nodes are wrapped with `@traceable` annotations. Once your `.env` contains a valid `LANGSMITH_API_KEY`, every conversational turn will stream nested execution graphs, latency profiling, and token usages to your LangSmith dashboard.

### RAGAS Evaluation Pipeline
To validate the performance of the retrieval-augmented generation engine against established benchmark datasets, Hephaestus runs a RAGAS evaluation suite computing:
- **Context Precision** (measuring if retrieved context is highly relevant to queries)
- **Context Recall** (verifying if all expected reference information was retrieved)
- **Faithfulness** (ensuring generation is strictly grounded in retrieved evidence)
- **Answer Relevancy** (verifying if answers directly address original user queries)

**Run via Makefile:**
```bash
make run-evals
```
**Run via uv directly:**
```bash
PYTHONPATH="apps/api/src" uv run python apps/api/evals/eval.py
```

---

## 📂 Project Structure

- `apps/api/`: FastAPI backend holding the RAG pipelines, LangGraph state orchestration, and REST routes.
- `apps/chatbot_ui/`: Streamlit conversational chat user interface.
- `apps/db_mcp_server/`: FastMCP-based DB tool connector server.
- `scripts/`: Collection of ingestion, parser, and pipeline scripts.
- `notebooks/`: Data science exploration, clustering development, and model performance experiments.
- `data/`: Directory for source datasets, PDFs, and intermediate outputs.
- `Makefile`: Quick command shortcuts for common workflows.
