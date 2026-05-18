import yaml
from pydantic import BaseModel

from core.config import config


class Workspace(BaseModel):
    workspace_id: str
    glossary: str = ""
    allowed_tools: frozenset[str] = frozenset()
    filters: dict = {}


def load_workspace(workspace_id: str) -> Workspace:
    path = config.WORKSPACES_DIR / f"{workspace_id}.yml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Workspace(**data)
