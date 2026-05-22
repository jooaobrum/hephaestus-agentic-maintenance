from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_PATH = _ROOT / "configs" / "config.yml"


class _Models(BaseModel):
    generation: str
    fallback: str
    embedding: str
    keyword: str
    reranking: str
    evaluation: str


class _LLM(BaseModel):
    temperature: float
    timeout_seconds: int
    max_retries: int
    streaming: bool


class _QdrantCollections(BaseModel):
    cm_interventions: str
    procedures: str
    known_issues: str


class _Qdrant(BaseModel):
    collections: _QdrantCollections


class _DatabaseTables(BaseModel):
    known_case_templates: str
    confirmed_rca_cases: str
    sensor_catalog: str
    sensor_readings: str
    interventions: str
    remaining_life: str


class _Database(BaseModel):
    schema_name: str = Field(alias="schema")
    tables: _DatabaseTables

    def qualified(self, table: str) -> str:
        return f"{self.schema_name}.{getattr(self.tables, table)}"


class _RRFWeights(BaseModel):
    cm: list[float]
    procedures: list[float]


class _Retrieval(BaseModel):
    default_top_k: int
    default_top_n: int
    procedures_top_k: int
    known_issues_search_limit: int
    known_issues_scroll_limit: int
    similar_machines_limit: int
    rrf_weights: _RRFWeights


class _Agents(BaseModel):
    default_workspace_id: str
    recursion_limit: int
    answer_dedup_key_length: int
    transfer_message_scan_length: int


class _SSE(BaseModel):
    node_status: dict[str, str]
    token_nodes: list[str]
    answer_nodes: list[str]


class _SourceRefs(BaseModel):
    intervention_prefix: str
    procedure_prefix: str


class _Sensors(BaseModel):
    status_warning: str
    status_critical: str
    default_days_span: int
    date_window_labels: list[int]


class _Paths(BaseModel):
    workspaces_dir: str
    prompts_subdir: str
    registry_subdir: str


class _UI(BaseModel):
    feedback_timeout_seconds: int
    stream_timeout_seconds: int


class _Evaluation(BaseModel):
    dataset_name: str


class AppConfig(BaseModel):
    models: _Models
    llm: _LLM
    qdrant: _Qdrant
    database: _Database
    retrieval: _Retrieval
    agents: _Agents
    sse: _SSE
    source_refs: _SourceRefs
    sensors: _Sensors
    paths: _Paths
    ui: _UI
    evaluation: _Evaluation


class Secrets(BaseSettings):
    OPENAI_API_KEY: str
    CO_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    PG_URL: str = (
        "postgresql://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
    )

    model_config = SettingsConfigDict(
        env_file=(_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )


def _load_app_config() -> AppConfig:
    with open(_CONFIG_PATH, "r") as f:
        return AppConfig(**yaml.safe_load(f))


app = _load_app_config()
secrets = Secrets()

WORKSPACES_DIR: Path = _ROOT / app.paths.workspaces_dir
