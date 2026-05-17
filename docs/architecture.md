# Hephaestus — Agentic Maintenance Intelligence Platform
## Solution Architecture

---

## 1. Executive Summary

**Hephaestus** is an enterprise-grade, AI-powered maintenance intelligence platform designed to accelerate root cause analysis (RCA) and institutional knowledge capture for industrial maintenance teams. It combines multi-agent orchestration, hybrid retrieval-augmented generation (RAG), real-time sensor integration, and structured knowledge persistence to reduce mean time to resolution (MTTR) and prevent recurring failures.

The platform is built on proven, production-ready open-source components, deployed as a containerized microservices architecture, and natively integrated with industry-standard observability and evaluation tooling.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              HEPHAESTUS PLATFORM                                │
│                                                                                 │
│  ┌──────────────┐     ┌──────────────────────────────────────────────────────┐  │
│  │              │     │                  AGENT LAYER                         │  │
│  │  PRESENTATION│────▶│  ┌────────────┐   ┌─────────────┐  ┌─────────────┐  │  │
│  │    LAYER     │SSE  │  │Coordinator │──▶│ Summarizer  │  │Troubleshoot │  │  │
│  │              │◀────│  │   Agent    │   │   Agent     │  │   Agent     │  │  │
│  │  Streamlit   │     │  └────────────┘   └─────────────┘  └─────────────┘  │  │
│  │  Chat UI     │     │        │               │                  │          │  │
│  │  :8501       │     │        └───────────────┴──────────────────┘          │  │
│  │              │     │                        │                             │  │
│  └──────────────┘     │               ┌────────▼────────┐                   │  │
│                       │               │   Tool Layer    │                   │  │
│  ┌──────────────┐     │               │  Retrieval +    │                   │  │
│  │              │     │               │  Sensor + DB    │                   │  │
│  │  EVALUATION  │     │               └────────┬────────┘                   │  │
│  │    LAYER     │     └────────────────────────┼────────────────────────────┘  │
│  │              │                              │                               │
│  │  RAGAS +     │     ┌────────────────────────┼────────────────────────────┐  │
│  │  LangSmith   │     │           DATA & PERSISTENCE LAYER                  │  │
│  │              │     │  ┌─────────┐  ┌────────▼──────┐  ┌──────────────┐  │  │
│  └──────────────┘     │  │ Qdrant  │  │  PostgreSQL   │  │  OpenAI API  │  │  │
│                       │  │ Vector  │  │  Relational   │  │  Embeddings  │  │  │
│                       │  │   DB    │  │      DB       │  │ + Generation │  │  │
│                       │  └─────────┘  └───────────────┘  └──────────────┘  │  │
│                       └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Architecture

### 3.1 Presentation Layer

```
┌────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                    │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │            Streamlit Chat UI  :8501              │  │
│  │                                                  │  │
│  │  ┌─────────────────┐  ┌────────────────────────┐ │  │
│  │  │  Chat Interface │  │   Session Management   │ │  │
│  │  │  - Message hist │  │  - thread_id (UUID)    │ │  │
│  │  │  - Example quer │  │  - New session button  │ │  │
│  │  │  - Feedback UI  │  │  - SSE stream reader   │ │  │
│  │  └────────┬────────┘  └────────────────────────┘ │  │
│  │           │                                       │  │
│  │     SSE Events:                                   │  │
│  │     {"event": "status",    "data": "..."}         │  │
│  │     {"event": "tool_calls","data": "..."}         │  │
│  │     {"event": "answer",    "data": "..."}         │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                              │
│              POST /multiagent/stream                   │
│              POST /submit_feedback/                    │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
```

---

### 3.2 API Gateway Layer

```
┌────────────────────────────────────────────────────────┐
│                  API GATEWAY LAYER                     │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │              FastAPI  :8000                      │  │
│  │                                                  │  │
│  │  POST /multiagent/stream                         │  │
│  │  ┌──────────────────────────────────────────┐    │  │
│  │  │  RAGRequest { query, thread_id }         │    │  │
│  │  │       ↓                                  │    │  │
│  │  │  stream_multiagent_pipeline()            │    │  │
│  │  │       ↓                                  │    │  │
│  │  │  Server-Sent Events (SSE)                │    │  │
│  │  └──────────────────────────────────────────┘    │  │
│  │                                                  │  │
│  │  POST /submit_feedback/                          │  │
│  │  ┌──────────────────────────────────────────┐    │  │
│  │  │  FeedbackRequest { trace_id, score, text}│    │  │
│  │  │       ↓                                  │    │  │
│  │  │  LangSmith Feedback Collection           │    │  │
│  │  └──────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │          FastMCP DB Server  :8001                │  │
│  │  MCP Tools: get_sensor_catalog()                 │  │
│  │             get_sensor_readings()                │  │
│  │             get_remaining_life()                 │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

### 3.3 Agent Orchestration Layer

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      AGENT ORCHESTRATION LAYER                             │
│                         LangGraph StateGraph                               │
│                                                                            │
│                    ┌─────────────────────┐                                 │
│         START ────▶│   Coordinator Agent  │                                 │
│                    │  (gpt-4o-mini)       │                                 │
│                    │                     │                                 │
│                    │  Intent: Route to   │                                 │
│                    │  specialist or      │                                 │
│                    │  answer directly    │                                 │
│                    └──────────┬──────────┘                                 │
│                               │                                            │
│              ┌────────────────┼────────────────┐                           │
│              │                │                │                           │
│              ▼                ▼                ▼                           │
│    ┌──────────────────┐       │    ┌───────────────────────────────────┐  │
│    │ Summarizer Agent │       │    │       Troubleshooting Agent       │  │
│    │  (gpt-4o-mini)   │  END  │    │         (gpt-4o-mini)             │  │
│    │                  │◀──────┘    │                                   │  │
│    │ Purpose:         │            │  RCA Mental Model:                │  │
│    │ "What happened?" │            │  1. ANCHOR   — SetInvestigation   │  │
│    │                  │            │  2. ORIENT   — list sections      │  │
│    │  ┌─────────────┐ │            │  3. GATHER   — parallel retrieval │  │
│    │  │Tool Loop    │ │            │  4. HYPOTHESIZE — rank hypotheses │  │
│    │  │list_ids     │ │            │  5. CONFIRM  — user agreement     │  │
│    │  │summarize    │ │            │  6. SAVE     — persist RCA case   │  │
│    │  │build_tmpl   │ │            │                                   │  │
│    │  │save_tmpl    │ │            │  Hypothesis Ledger:               │  │
│    │  └─────────────┘ │            │  ┌──────────────────────────────┐ │  │
│    │                  │            │  │ ID | Statement | Confidence  │ │  │
│    │  Output:         │            │  │    | Status    | Sources     │ │  │
│    │  Known-case      │            │  └──────────────────────────────┘ │  │
│    │  templates for   │            │                                   │  │
│    │  future use      │            │  Source Types:                    │  │
│    └──────────────────┘            │  [PROC_REF] Procedure manuals    │  │
│                                    │  [GRAPH]    Known-issues graph   │  │
│                                    │  [INT]      Past interventions   │  │
│                                    │  [USER]     Technician feedback  │  │
│                                    └───────────────────────────────────┘  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    State Persistence (PostgresSaver)                 │  │
│  │  thread_id → MultiAgentState { messages, coordinator, summarizer,   │  │
│  │              troubleshooting } persisted per checkpoint              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.4 Tool Execution Layer

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          TOOL EXECUTION LAYER                              │
│                                                                            │
│  ┌─────────────────────────┐   ┌──────────────────────────────────────┐   │
│  │   RETRIEVAL TOOLS       │   │         DATABASE TOOLS               │   │
│  │                         │   │                                      │   │
│  │  CM History:            │   │  Sensors:                            │   │
│  │  get_formatted_         │   │  get_sensor_catalog_tool()           │   │
│  │    cm_context()         │   │  get_sensor_readings_tool()          │   │
│  │  get_recent_formatted_  │   │  get_sensor_timeline_tool()          │   │
│  │    cm_context()         │   │  get_threshold_events_tool()         │   │
│  │  list_intervention_     │   │  get_sensor_anomaly_summary()        │   │
│  │    ids_by_date()        │   │                                      │   │
│  │  summarize_             │   │  Components:                         │   │
│  │    intervention()       │   │  get_remaining_life_tool()           │   │
│  │                         │   │                                      │   │
│  │  Procedures:            │   │  Fleet:                              │   │
│  │  get_formatted_         │   │  check_machine_exists()              │   │
│  │    procedure_context()  │   │  get_fleet_impact_for_symptom()      │   │
│  │  list_procedure_        │   │  list_available_machines()           │   │
│  │    sections()           │   │                                      │   │
│  │                         │   │  Known Cases:                        │   │
│  │  Knowledge Graph:       │   │  get_known_case_templates()          │   │
│  │  query_known_issues_    │   │  save_known_case_template()          │   │
│  │    graph()              │   │  save_confirmed_rca_case()           │   │
│  │  list_known_issue_      │   │                                      │   │
│  │    categories()         │   │  RCA Ledger:                         │   │
│  │                         │   │  SetInvestigation()                  │   │
│  └────────────┬────────────┘   │  ConfirmRootCause()                  │   │
│               │                │  SaveCase()                          │   │
│               │                └────────────────┬─────────────────────┘   │
│               │                                 │                         │
│               └────────────────┬────────────────┘                         │
│                                ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Embedding Pipeline (per retrieval call)                │   │
│  │                                                                     │   │
│  │  Query Text ──▶ text-embedding-3-small ──▶ 1536-dim vector          │   │
│  │       ↓                                         ↓                  │   │
│  │  BM25 Tokens                            Semantic Vector            │   │
│  │       └───────────────────┬─────────────────────┘                  │   │
│  │                           ▼                                         │   │
│  │               Qdrant Hybrid Search (RRF Fusion)                     │   │
│  │               Top-N=20 candidates → Top-K=10 results               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.5 Data & Persistence Layer

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       DATA & PERSISTENCE LAYER                             │
│                                                                            │
│  ┌──────────────────────────────┐   ┌────────────────────────────────────┐ │
│  │      QDRANT VECTOR DB        │   │        POSTGRESQL 16               │ │
│  │         :6333 / :6334        │   │            :5433                   │ │
│  │                              │   │                                    │ │
│  │  Collection:                 │   │  Schema: maintenance               │ │
│  │  cm_interventions_hybrid     │   │  ┌──────────────────────────────┐  │ │
│  │  ┌──────────────────────┐    │   │  │ sensor_catalog               │  │ │
│  │  │ Maintenance Records  │    │   │  │ sensor_readings (time-series)│  │ │
│  │  │ - fault_code         │    │   │  │ remaining_life (RUL/cond.)   │  │ │
│  │  │ - related_interv.    │    │   │  │ known_case_templates         │  │ │
│  │  │ - event_description  │    │   │  │ confirmed_rca_cases          │  │ │
│  │  │ - comments           │    │   │  └──────────────────────────────┘  │ │
│  │  │ Hybrid: BM25 + emb.  │    │   │                                    │ │
│  │  └──────────────────────┘    │   │  Schema: langgraph                 │ │
│  │                              │   │  ┌──────────────────────────────┐  │ │
│  │  Collection:                 │   │  │ checkpoints (agent state)    │  │ │
│  │  procedures_hybrid           │   │  │ checkpoint_blobs             │  │ │
│  │  ┌──────────────────────┐    │   │  │ checkpoint_writes            │  │ │
│  │  │ Procedure Manuals    │    │   │  └──────────────────────────────┘  │ │
│  │  │ - fault_codes        │    │   │                                    │ │
│  │  │ - troubleshoot steps │    │   │  Key: thread_id → full agent       │ │
│  │  │ - known-issue graph  │    │   │  state persisted between turns     │ │
│  │  │ Hybrid: BM25 + emb.  │    │   └────────────────────────────────────┘ │
│  │  └──────────────────────┘    │                                          │
│  └──────────────────────────────┘                                          │
│                                                                            │
│  ┌──────────────────────────────┐   ┌────────────────────────────────────┐ │
│  │        OPENAI API            │   │       COHERE API (optional)        │ │
│  │                              │   │                                    │ │
│  │  Embeddings:                 │   │  Reranking:                        │ │
│  │  text-embedding-3-small      │   │  rerank-v4.0-pro                   │ │
│  │  (1536 dims)                 │   │  (post-retrieval reranking,        │ │
│  │                              │   │   currently inactive)              │ │
│  │  Generation:                 │   │                                    │ │
│  │  gpt-4o-mini (agents)        │   └────────────────────────────────────┘ │
│  │  gpt-5.1 (structured output) │                                          │
│  │  gpt-5.4-nano (tool summaries│                                          │
│  └──────────────────────────────┘                                          │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.6 Ingestion Layer

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION LAYER                                  │
│                    (Notebooks 02 → 09, offline pipeline)                   │
│                                                                            │
│  ┌─────────────────────────┐         ┌─────────────────────────────────┐   │
│  │   RAW DATA SOURCES      │         │       INGESTION PIPELINE        │   │
│  │                         │         │                                 │   │
│  │  ┌───────────────────┐  │         │  Step 1: Load & Validate        │   │
│  │  │ interventions.csv │  │─────┐   │  - Parse fault codes (B-NNN,   │   │
│  │  │ (CM records)      │  │     │   │    H-NNN format)               │   │
│  │  └───────────────────┘  │     │   │  - Validate required fields    │   │
│  │                         │     │   │                                 │   │
│  │  ┌───────────────────┐  │     ▼   │  Step 2: Chunk & Enrich        │   │
│  │  │ Procedure PDFs    │  │  ┌─────┐│  - Build composite text:       │   │
│  │  │ (maintenance      │  │  │     ││  [FAULT_CODE] + [REL_INT] +    │   │
│  │  │  manuals, parsed  │  │  │Proc.││  [EVENT] + [COMMENTS]          │   │
│  │  │  via Mistral AI)  │  │  │ CSV ││                                 │   │
│  │  └───────────────────┘  │  └─────┘│  Step 3: Embed                  │   │
│  │                         │     │   │  text-embedding-3-small →       │   │
│  │  ┌───────────────────┐  │     │   │  1536-dim vectors               │   │
│  │  │ Sensor Data       │  │     │   │                                 │   │
│  │  │ (PostgreSQL, live)│  │     │   │  Step 4: Index (Hybrid)         │   │
│  │  └───────────────────┘  │     │   │  BM25 sparse + dense vectors    │   │
│  │                         │     │   │  → Qdrant upsert                │   │
│  │  ┌───────────────────┐  │     │   │                                 │   │
│  │  │ Known Issue Graph │  │     ▼   │                                 │   │
│  │  │ (structured JSON) │  │  ┌─────┐│  Step 5: Validate              │   │
│  │  └───────────────────┘  │  │Qdrant││  - Spot-check retrieval       │   │
│  └─────────────────────────┘  └─────┘│  - Evaluate precision@K        │   │
│                                       └─────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.7 Observability & Evaluation Layer

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY & EVALUATION LAYER                        │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      LANGSMITH PLATFORM                              │  │
│  │                                                                      │  │
│  │  ┌────────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │  │
│  │  │    TRACING         │  │   DATASETS       │  │   FEEDBACK      │  │  │
│  │  │                    │  │                  │  │                 │  │  │
│  │  │ @traceable on:     │  │ hephaestus-      │  │ User thumbs     │  │  │
│  │  │ - _embed_text()    │  │ rag-eval         │  │ up/down →       │  │  │
│  │  │ - retrieve funcs   │  │                  │  │ POST            │  │  │
│  │  │ - agent nodes      │  │ Q&A pairs with   │  │ /submit_        │  │  │
│  │  │ - tool calls       │  │ ground truth     │  │ feedback/       │  │  │
│  │  │                    │  │ chunk IDs for    │  │                 │  │  │
│  │  │ Full lineage per   │  │ precision/recall │  │ trace_id →      │  │  │
│  │  │ request (nested    │  │ evaluation       │  │ LangSmith run   │  │  │
│  │  │ spans, latency,    │  │                  │  │                 │  │  │
│  │  │ token usage)       │  └──────────────────┘  └─────────────────┘  │  │
│  │  └────────────────────┘                                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       RAGAS EVALUATION SUITE                         │  │
│  │                   (apps/api/evals/eval.py + Notebook 05)             │  │
│  │                                                                      │  │
│  │  ┌────────────────────────┐   ┌────────────────────────────────────┐ │  │
│  │  │   RETRIEVAL METRICS    │   │       GENERATION METRICS           │ │  │
│  │  │                        │   │                                    │ │  │
│  │  │  IDBasedContextPrecision│   │  Faithfulness                     │ │  │
│  │  │  "Are retrieved chunks  │   │  "Does the answer stay within     │ │  │
│  │  │   relevant to query?"   │   │   the bounds of retrieved context?"│ │  │
│  │  │  Baseline: 0.29         │   │  Baseline: 0.90                   │ │  │
│  │  │                        │   │                                    │ │  │
│  │  │  IDBasedContextRecall  │   │  AnswerRelevancy                   │ │  │
│  │  │  "Were the expected    │   │  "Is the answer relevant           │ │  │
│  │  │   chunks retrieved?"   │   │   to the original question?"       │ │  │
│  │  │  Baseline: 0.99        │   │  Baseline: 0.63                    │ │  │
│  │  └────────────────────────┘   └────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  │  Evaluation LLM: gpt-5.4-mini    Embeddings: text-embedding-3-small  │  │
│  │  Results → LangSmith experiment "rag-eval-baseline"                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. End-to-End Request Flow

```
 MAINTENANCE TECHNICIAN
        │
        │  "Machine XYZ-3 is vibrating abnormally — what's causing it?"
        │
        ▼
 ┌─────────────────┐
 │  Streamlit UI   │  Creates/reuses thread_id → POST /multiagent/stream
 │  :8501          │
 └────────┬────────┘
          │  SSE connection opens
          ▼
 ┌─────────────────┐
 │  FastAPI        │  Validates RAGRequest → calls stream_multiagent_pipeline()
 │  :8000          │
 └────────┬────────┘
          │
          ▼
 ┌──────────────────────────────────────────────────────────┐
 │  LangGraph StateGraph                                    │
 │                                                          │
 │  1. Load checkpoint from PostgreSQL (thread_id)          │
 │                                                          │
 │  2. COORDINATOR AGENT                                    │
 │     → Prompt: "Is this diagnostic or historical?"        │
 │     → Routes to: troubleshooting_agent                   │
 │     → Emits: status event "Routing to RCA specialist"    │
 │                                                          │
 │  3. TROUBLESHOOTING AGENT (RCA Loop)                     │
 │                                                          │
 │     Turn 1 — ANCHOR & ORIENT                             │
 │     ├─ SetInvestigation(machine="XYZ-3",                 │
 │     │                   symptom="abnormal vibration")    │
 │     ├─ list_procedure_sections("XYZ-3")                  │
 │     │    → Qdrant [procedures_hybrid]                    │
 │     └─ list_known_issue_categories("XYZ-3")              │
 │          → Qdrant [procedures_hybrid]                    │
 │                                                          │
 │     Turn 2 — GATHER (parallel)                           │
 │     ├─ get_formatted_procedure_context(                  │
 │     │    query, file_name="XYZ3-manual.pdf", top_k=5)   │
 │     │    → Qdrant hybrid search → [PROC_REF] sources    │
 │     ├─ query_known_issues_graph(                         │
 │     │    "vibration", machine="XYZ-3")                   │
 │     │    → Qdrant → [GRAPH] sources                     │
 │     ├─ get_recent_formatted_cm_context(                  │
 │     │    query, machine="XYZ-3", days_span=7, top_k=5)  │
 │     │    → Qdrant + PostgreSQL date filter → [INT]      │
 │     └─ get_sensor_readings_tool(                         │
 │          "XYZ-3", start_date, end_date, tag="vib_rms")   │
 │          → PostgreSQL sensor_readings                    │
 │                                                          │
 │     Turn 3 — HYPOTHESIZE                                 │
 │     ├─ Hypothesis A (PROC_REF): "Bearing wear"    0.60   │
 │     ├─ Hypothesis B (GRAPH):    "Imbalance"       0.50   │
 │     └─ Hypothesis C (INT):      "Coupling fault"  0.45   │
 │     → Emits: answer event with full ledger               │
 │                                                          │
 │  4. TECHNICIAN FEEDBACK                                  │
 │     "Bearing wear is most likely — sensor shows spike"   │
 │     → confidence A → 0.87 (LIKELY)                       │
 │     → Emits: updated ledger                              │
 │                                                          │
 │  5. CONFIRM & SAVE                                       │
 │     ├─ ConfirmRootCause(hypothesis_id="A")               │
 │     └─ SaveCase(machine, symptom, root_cause, actions)   │
 │          → PostgreSQL confirmed_rca_cases                │
 │                                                          │
 │  6. Save checkpoint → PostgreSQL                         │
 └──────────────────────────────────────────────────────────┘
          │
          │  SSE stream: status → tool_calls → answer events
          ▼
 ┌─────────────────┐
 │  Streamlit UI   │  Renders answer + feedback buttons
 └────────┬────────┘
          │  User clicks 👍 / 👎
          ▼
 ┌─────────────────┐
 │  FastAPI        │  POST /submit_feedback/ → LangSmith
 │  :8000          │  Records quality signal for future eval
 └─────────────────┘
```

---

## 5. Infrastructure & Deployment

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    DOCKER COMPOSE DEPLOYMENT                               │
│                         hephaestus-network                                 │
│                                                                            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │   qdrant     │   │  postgres    │   │     api      │   │ streamlit  │  │
│  │              │   │              │   │              │   │    -app    │  │
│  │  :6333 REST  │   │  :5433       │   │  :8000       │   │  :8501     │  │
│  │  :6334 gRPC  │   │              │   │              │   │            │  │
│  │              │   │  langgraph_  │   │  FastAPI +   │   │  Streamlit │  │
│  │  Vol:        │   │  db          │   │  LangGraph   │   │  Chat UI   │  │
│  │  .qdrant_    │   │              │   │              │   │            │  │
│  │  data/       │   │  Vol:        │   │  Hot-reload  │   │  Deps:     │  │
│  │              │   │  postgres_   │   │  (src mount) │   │  api svc   │  │
│  │  Health:     │   │  data/       │   │              │   │            │  │
│  │  TCP :6333   │   │              │   │  Deps:       │   │            │  │
│  └──────────────┘   └──────────────┘   │  qdrant ✓   │   └────────────┘  │
│                                        │  postgres ✓  │                   │
│  ┌──────────────┐                      └──────────────┘                   │
│  │  db_mcp_     │                                                         │
│  │  server      │                                                         │
│  │  :8001       │                                                         │
│  │              │                                                         │
│  │  FastMCP     │                                                         │
│  │  MCP tools   │                                                         │
│  │  over HTTP   │                                                         │
│  └──────────────┘                                                         │
│                                                                            │
│  make run-docker-compose                                                   │
│  → uv sync                                                                 │
│  → docker compose up --build -d                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Technology Stack Summary

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Orchestration** | LangGraph | ≥0.4.0 | Multi-agent state machine, conditional routing |
| **State Persistence** | langgraph-checkpoint-postgres | ≥3.0.5 | Multi-turn conversation state |
| **LLM — Agents** | OpenAI gpt-4o-mini | Latest | Coordinator, Summarizer, Troubleshooting agents |
| **LLM — Structured** | OpenAI gpt-5.1 | Latest | Structured output (KnownIssue schema) |
| **LLM — Tools** | OpenAI gpt-5.4-nano | Latest | Lightweight tool-level summarization |
| **Embeddings** | text-embedding-3-small | Latest | 1536-dim semantic vectors |
| **Vector Search** | Qdrant | Latest | Hybrid BM25 + semantic (RRF fusion) |
| **Relational DB** | PostgreSQL 16 | 16 | Sensor data, RCA cases, LangGraph checkpoints |
| **API Framework** | FastAPI | ≥0.135.2 | REST + SSE endpoint layer |
| **Frontend** | Streamlit | ≥1.52.2 | Chat interface with streaming |
| **MCP Server** | FastMCP | ≥3.2.4 | Database tool exposure via MCP protocol |
| **Observability** | LangSmith | ≥0.7.22 | Tracing, datasets, feedback collection |
| **Evaluation** | RAGAS | ≥0.4.3 | Context precision/recall, faithfulness, relevancy |
| **PDF Parsing** | Mistral AI | ≥2.4.0 | OCR and text extraction from procedure PDFs |
| **Structured Output** | Instructor | ≥1.14.5 | Guaranteed Pydantic schema compliance |
| **Package Manager** | uv | Latest | Workspace dependency management |
| **Container** | Docker Compose | 3.8 | Service orchestration |
| **Python** | CPython | ≥3.12 | Runtime |

---

## 7. Data Model Overview

### Qdrant Collections

| Collection | Index Type | Content | Key Fields |
|---|---|---|---|
| `cm_interventions_hybrid` | BM25 + Dense | Corrective maintenance records | fault_code, machine_id, event, comments, intervention_id |
| `procedures_hybrid` | BM25 + Dense | Procedure manuals + known-issue graph | section, fault_code, steps, file_name, machine_family |

### PostgreSQL Tables (schema: `maintenance`)

| Table | Purpose | Key Fields |
|---|---|---|
| `sensor_catalog` | Sensor metadata and thresholds | machine, tag, unit, warn_lo/hi, crit_lo/hi, fault_correlation |
| `sensor_readings` | Time-series sensor data | machine, timestamp, tag, value, status |
| `remaining_life` | Component condition and RUL | machine, component, rul_days, condition, remaining_pct |
| `known_case_templates` | LLM-generated failure patterns | template_id, symptom_name, root_causes, affected_machines |
| `confirmed_rca_cases` | Validated root cause analyses | case_id, machine_id, symptom, root_cause, actions |

### LangGraph Checkpoint Tables (schema: `langgraph`)

| Table | Purpose |
|---|---|
| `checkpoints` | Full agent state per thread_id + checkpoint |
| `checkpoint_blobs` | Binary state blobs |
| `checkpoint_writes` | Incremental state writes |

---

## 8. Key Architectural Decisions

### 1. Multi-Agent Specialization over Monolithic Agent
A single generalist agent cannot maintain both the "historical summary" and "active RCA investigation" mental models simultaneously. Coordinator routing enables clean separation of concerns, allowing each specialist to optimize its tool selection and prompt structure independently.

### 2. Hypothesis Ledger Pattern
Rather than emitting a single free-text diagnosis, the Troubleshooting Agent maintains a structured ledger of competing hypotheses with explicit confidence scores and source traceability. This enables incremental refinement through user feedback and provides a clear audit trail for each confirmed root cause.

### 3. Hybrid Search (BM25 + Semantic)
Maintenance queries frequently use domain-specific terminology (fault codes, part numbers, machine IDs) that purely semantic models mishandle. BM25 preserves keyword precision; semantic search handles paraphrasing. RRF fusion combines both signals without requiring threshold tuning.

### 4. State Persistence via PostgresSaver
Multi-turn RCA investigations span multiple user messages. PostgreSQL-backed LangGraph checkpoints allow the agent to resume an active investigation after network interruptions, page reloads, or system restarts — critical for long diagnostic sessions.

### 5. Structured Feedback Loop (LangSmith)
User thumbs-up/down signals are captured against specific LangSmith trace IDs. This creates a labelled dataset of successful and failed diagnoses that can directly feed future RAGAS evaluations and prompt fine-tuning.

---

## 9. Evaluation Results (Baseline)

| Metric | Description | Baseline Score |
|---|---|---|
| **IDBasedContextPrecision** | Retrieved chunks relevant to expected answer | 0.29 |
| **IDBasedContextRecall** | Expected chunks successfully retrieved | 0.99 |
| **Faithfulness** | Answer stays within retrieved context | 0.90 |
| **AnswerRelevancy** | Answer addresses the original question | 0.63 |

**Key insight:** Near-perfect recall (0.99) confirms the hybrid retrieval captures all relevant material. Precision (0.29) and answer relevancy (0.63) are improvement targets, addressed by reranking (Cohere integration in progress) and prompt refinement.

---

## 10. Glossary

| Term | Definition |
|---|---|
| **RCA** | Root Cause Analysis — structured process to identify the underlying cause of a failure |
| **RAG** | Retrieval-Augmented Generation — LLM answers grounded in retrieved documents |
| **CM** | Corrective Maintenance — reactive maintenance performed after a failure |
| **RUL** | Remaining Useful Life — estimated operational time before component replacement |
| **RRF** | Reciprocal Rank Fusion — algorithm to merge ranked lists from multiple search strategies |
| **MCP** | Model Context Protocol — standard for exposing tools to LLM agents |
| **SSE** | Server-Sent Events — HTTP-based unidirectional streaming protocol |
| **BM25** | Best Match 25 — probabilistic keyword-based ranking algorithm |
| **RAGAS** | Retrieval Augmented Generation Assessment — evaluation framework for RAG systems |
| **LangGraph** | Graph-based agent orchestration framework from LangChain |
| **thread_id** | UUID identifying a user session; maps to a unique LangGraph checkpoint chain |
