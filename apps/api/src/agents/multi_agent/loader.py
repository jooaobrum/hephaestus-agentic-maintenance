from pathlib import Path

import yaml
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

import agents.tools.tools_auth as _tools_module

_REGISTRY_DIR = Path(__file__).parent / "registry"

# Reflect all @tool-decorated objects from tools_auth into a name→tool map
_TOOL_REGISTRY: dict[str, BaseTool] = {
    name: obj for name, obj in vars(_tools_module).items() if isinstance(obj, BaseTool)
}

# Set of valid agent names for stream.py mode guard
AGENT_NAMES: set[str] = {f.stem for f in _REGISTRY_DIR.glob("*.yaml")}


def _resolve_tools(tool_names: list[str]) -> list[BaseTool]:
    resolved = []
    for name in tool_names:
        if name not in _TOOL_REGISTRY:
            raise ValueError(
                f"Tool '{name}' not found in tools_auth module. "
                f"Available: {sorted(_TOOL_REGISTRY)}"
            )
        resolved.append(_TOOL_REGISTRY[name])
    return resolved


def _load_registry(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    name_from_meta = cfg["metadata"]["name"]
    name_from_file = path.stem
    if name_from_meta != name_from_file:
        raise ValueError(
            f"Registry mismatch: file is '{name_from_file}.yaml' "
            f"but metadata.name is '{name_from_meta}'"
        )
    return cfg


def build_agent(name: str, llm, checkpointer=None):
    cfg = _load_registry(_REGISTRY_DIR / f"{name}.yaml")
    tools = _resolve_tools(cfg["tools"])
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=cfg["prompt"],
        name=name,
        checkpointer=checkpointer,
    )


def build_all_agents(llm) -> list:
    return [build_agent(f.stem, llm) for f in sorted(_REGISTRY_DIR.glob("*.yaml"))]
