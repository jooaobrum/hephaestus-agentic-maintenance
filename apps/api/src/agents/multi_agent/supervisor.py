from pathlib import Path

from langgraph_supervisor import create_supervisor

from agents.core.llm import make_llm
from agents.core.prompts import load_prompt
from agents.multi_agent.loader import build_all_agents

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def build_supervisor(checkpointer=None):
    llm = make_llm()
    agents = build_all_agents(llm)
    coordinator_prompt = load_prompt(_PROMPTS_DIR / "coordinator.yml", "coordinator")
    return create_supervisor(
        model=llm,
        agents=agents,
        prompt=coordinator_prompt,
    ).compile(checkpointer=checkpointer)
